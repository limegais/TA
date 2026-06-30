from flask import Flask, render_template, jsonify, request, Response, session, redirect, url_for
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import json
from datetime import datetime, timedelta, timezone
from collections import deque
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
import random
import math
import os
import tempfile
import subprocess
import urllib.request
import urllib.error

# ==================== SBMS API CONFIG ====================
# External SBMS server for device control (Outlet, AC, Lamp, Master)
sbms_config = {
    'server_url': 'https://iotlab-uns.com/neo-sbms',      # e.g. http://192.168.1.10 or https://sbms.example.com
    'token': None,
    'user': None,
    'connected': False,
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartroom-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Authentication Credentials (role-based)
USERS = {
    'admin': {'password': 'admin', 'role': 'admin'},
    'user':  {'password': 'user',  'role': 'user'},
}

# Device Status Tracking
device_last_seen = {
    'esp32_ac': {'last_seen': None, 'status': 'offline'},
    'esp32_lamp': {'last_seen': None, 'status': 'offline'},
    'camera': {'last_seen': None, 'status': 'offline'}
}

# Alert Rules
alert_rules = {
    'high_temp': {'threshold': 35, 'enabled': True, 'triggered': False},
    'high_humidity': {'threshold': 80, 'enabled': True, 'triggered': False},
    'no_person_timeout': {'ac_timeout_minutes': 5, 'lamp_timeout_minutes': 5, 'enabled': True, 'last_person_seen': None}
}
active_alerts = deque(maxlen=50)

# Timezone warning (set at startup by check_timezone())
_system_tz_warn = ''   # empty = OK; non-empty = displayed in dashboard banner

def check_timezone():
    """Check RPi system timezone. Warn if not Asia/Jakarta."""
    global _system_tz_warn
    try:
        result = subprocess.run(
            ['timedatectl', 'show', '--property=Timezone', '--value'],
            capture_output=True, text=True, timeout=3
        )
        tz = result.stdout.strip()
        if tz and tz != 'Asia/Jakarta':
            _system_tz_warn = f"System timezone is '{tz}', expected 'Asia/Jakarta'. Run: sudo timedatectl set-timezone Asia/Jakarta"
            print(f"[WARN] {_system_tz_warn}")
        else:
            _system_tz_warn = ''
            if tz:
                print(f"  [TZ] Timezone OK: {tz}")
    except FileNotFoundError:
        pass   # Not on Linux/RPi — skip silently
    except Exception as e:
        print(f"[WARN] Timezone check failed: {e}")
GOOGLE_FORM_URL = "https://docs.google.com/forms/"
occupancy_feedback = deque(maxlen=200)

# Energy Phase: 'before' / 'after' / 'idle'
energy_phase = 'idle'
energy_recording = {
    'before': {'active': False, 'start': None, 'end': None},
    'after': {'active': False, 'start': None, 'end': None}
}

# Lamp Energy Phase (separate recording from AC)
lamp_phase = 'idle'
lamp_recording = {
    'before': {'active': False, 'start': None, 'end': None},
    'after': {'active': False, 'start': None, 'end': None}
}

# Lamp power estimation constants (ESP32 Lamp has no PZEM — we estimate from brightness)
LAMP_RATED_WATT = 30.0   # Total watts of both lamps at 100% brightness (2 × 15W LED)
LAMP_VOLTAGE = 220.0     # Voltage nominal Indonesia
_lamp_energy_kwh = 0.0   # kWh accumulator for lamp estimation
_lamp_energy_last_ts = 0.0

# Lock for all recording-state globals (energy_phase, energy_recording, lamp_phase, lamp_recording, _lamp_energy_kwh)
_recording_lock = threading.Lock()

# Persist energy_recording to disk so it survives server restart
ENERGY_RECORDING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'energy_recording.json')
ENERGY_RECORDING_BAK  = ENERGY_RECORDING_FILE + '.bak'

def _validate_recording_entry(entry):
    """Return a clean recording entry dict — reject bad types from corrupted JSON."""
    if not isinstance(entry, dict):
        return {'active': False, 'start': None, 'end': None}
    return {
        'active': bool(entry.get('active', False)),
        'start':  entry.get('start') if isinstance(entry.get('start'), str) else None,
        'end':    entry.get('end')   if isinstance(entry.get('end'),   str) else None,
    }

def save_energy_recording():
    """
    Atomically persist recording state to disk.
    Strategy: write to a temp file in the same directory, fsync, then rename
    (rename is atomic on POSIX/ext4 — what the RPi runs).
    A .bak copy of the previous good file is kept for disaster recovery.
    """
    with _recording_lock:
        payload = {
            'phase':          energy_phase,
            'recording':      energy_recording,
            'lamp_phase':     lamp_phase,
            'lamp_recording': lamp_recording,
            'lamp_energy_kwh': round(_lamp_energy_kwh, 6),
            'saved_at':       datetime.utcnow().isoformat() + 'Z',
        }
        try:
            dir_path = os.path.dirname(ENERGY_RECORDING_FILE)
            # Write to a sibling temp file so the rename stays on the same filesystem
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as fh:
                    json.dump(payload, fh, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())   # force kernel buffer → disk
            except Exception:
                os.unlink(tmp_path)
                raise
            # Rotate: current → .bak, then new → current  (two renames, both atomic)
            if os.path.exists(ENERGY_RECORDING_FILE):
                os.replace(ENERGY_RECORDING_FILE, ENERGY_RECORDING_BAK)
            os.replace(tmp_path, ENERGY_RECORDING_FILE)
        except Exception as e:
            print(f"[WARN] save_energy_recording failed: {e}")

def load_energy_recording():
    """
    Load recording state from JSON file.
    Falls back to .bak if the main file is corrupted.
    Handles active-recording-on-restart: keeps active=True so the phase tag
    continues to be applied to new InfluxDB writes, but logs a warning.
    Also restores _lamp_energy_kwh so the accumulator survives restarts.
    """
    global energy_phase, energy_recording, lamp_phase, lamp_recording
    global _lamp_energy_kwh, _lamp_energy_last_ts

    for candidate in (ENERGY_RECORDING_FILE, ENERGY_RECORDING_BAK):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, 'r') as f:
                data = json.load(f)
            # Validate top-level types
            if not isinstance(data, dict):
                raise ValueError("root is not a dict")

            energy_phase = data.get('phase', 'idle')
            if energy_phase not in ('before', 'after', 'idle'):
                energy_phase = 'idle'

            raw_rec = data.get('recording', {})
            if isinstance(raw_rec, dict):
                for ph in ('before', 'after'):
                    energy_recording[ph] = _validate_recording_entry(raw_rec.get(ph, {}))

            lamp_phase = data.get('lamp_phase', 'idle')
            if lamp_phase not in ('before', 'after', 'idle'):
                lamp_phase = 'idle'

            raw_lrec = data.get('lamp_recording', {})
            if isinstance(raw_lrec, dict):
                for ph in ('before', 'after'):
                    lamp_recording[ph] = _validate_recording_entry(raw_lrec.get(ph, {}))

            # Restore accumulated kWh so the counter doesn't reset to 0 on restart
            saved_kwh = data.get('lamp_energy_kwh', 0.0)
            if isinstance(saved_kwh, (int, float)) and saved_kwh >= 0:
                _lamp_energy_kwh = float(saved_kwh)
                _lamp_energy_last_ts = time.time()  # treat 'now' as last known point

            # Warn if any recording was still active when the server stopped
            for ph in ('before', 'after'):
                if energy_recording[ph]['active']:
                    print(f"[WARN] AC recording '{ph}' was active at last shutdown — resuming. "
                          f"Gap since {energy_recording[ph]['start']} will appear in data.")
                if lamp_recording[ph]['active']:
                    print(f"[WARN] Lamp recording '{ph}' was active at last shutdown — resuming.")

            src = "main" if candidate == ENERGY_RECORDING_FILE else "BACKUP"
            print(f"[OK] Loaded energy recording ({src}): "
                  f"ac_phase={energy_phase}, lamp_phase={lamp_phase}, "
                  f"lamp_kwh={_lamp_energy_kwh:.4f}")
            return  # success — stop trying candidates
        except Exception as e:
            print(f"[WARN] load_energy_recording failed for {candidate}: {e}")

    print("[INFO] No valid energy_recording.json found — starting fresh.")

load_energy_recording()

# ==================== GA/PSO OPTIMIZATION ENGINE (embedded) ====================
# Optimization bounds
OPT_TEMP_MIN, OPT_TEMP_MAX = 16.0, 30.0
OPT_FAN_MIN, OPT_FAN_MAX = 1, 4  # ESP32 AC supports fan speed 1-4
OPT_BRIGHTNESS_MIN, OPT_BRIGHTNESS_MAX = 0, 255  # PWM range 0-255 sesuai spesifikasi
OPT_RH_MIN, OPT_RH_MAX = 30, 80  # Set RH range 30-80%

# AC mode gene: 0=COOL, 1=DRY, 2=FAN, 3=AUTO
OPT_MODE_MIN, OPT_MODE_MAX = 0, 3
AC_MODE_NAMES = {0: 'COOL', 1: 'DRY', 2: 'FAN', 3: 'AUTO'}

# Live sensor data for fitness calculation
opt_sensor_data = {
    'temperature': 28.0, 'humidity': 55.0, 'person_detected': False,
    'lux': 0, 'lux1': 0, 'lux2': 0, 'lux3': 0,
    'temp1': 0.0, 'hum1': 0.0, 'temp2': 0.0, 'hum2': 0.0, 'temp3': 0.0, 'hum3': 0.0,
    'temp_trend': 0.0, 'temp_history': [], 'data_source': 'default',
    'actual_watt': 0.0, 'power_factor': 1.0,          # real power from MySQL energy meter (0 = not fetched yet)
    'curr_brightness1': 0, 'curr_brightness2': 0,      # current lamp brightness for delta lux model
    'person_count': 0,                                  # number of people detected (0 = empty)
}

# Auto optimization config — separate intervals for AC (slow) and Lamp (fast)
AUTO_OPT_INTERVAL_AC = 600   # 10 min for AC — aligned with AC_ADAPTIVE_DEBOUNCE so every GA run can apply
AUTO_OPT_INTERVAL_LAMP = 300 # 5 min for Lamp — aligned with LAMP_ADAPTIVE_DEBOUNCE so every run can apply
optimization_lock = threading.Lock()
optimization_run_count = 0
last_opt_results = {
    'ga': {'fitness': 0, 'temp': 0, 'fan': 0, 'stats': []},
    'pso': {'fitness': 0, 'pwm1': 0, 'pwm2': 0, 'brightness': 0, 'brightness1': 0, 'brightness2': 0, 'stats': []}
}

ga_params = {'population_size': 15, 'generations': 20, 'mutation_rate': 0.3, 'crossover_rate': 0.85, 'elitism_ratio': 0.2}
pso_params = {'swarm_size': 10, 'iterations': 20, 'w': 0.5, 'c1': 1.5, 'c2': 1.5}  # w=0.5 konstan (sesuai spesifikasi)

# Algorithm configuration: which algorithm to use for AC and Lamp
# Options: 'ga_pso' (default), 'pso_ga' (swap), 'ga_ga' (all GA), 'pso_pso' (all PSO)
# Format: '{ac_algo}_{lamp_algo}'
opt_algo_config = 'ga_pso'
OPT_ALGO_OPTIONS = {'ga_pso', 'pso_ga', 'ga_ga', 'pso_pso'}

def _get_ac_algo():
    """Return 'ga' or 'pso' for AC based on current config."""
    return 'pso' if opt_algo_config in ('pso_ga', 'pso_pso') else 'ga'

def _get_lamp_algo():
    """Return 'pso' or 'ga' for Lamp based on current config."""
    return 'ga' if opt_algo_config in ('pso_ga', 'ga_ga') else 'pso'

def _gaussian_score(value, target, sigma, max_score):
    return max_score * math.exp(-((value - target) ** 2) / (2 * sigma ** 2))

def _get_time_period():
    hour = datetime.now().hour
    if 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 17: return 'afternoon'
    elif 17 <= hour < 22: return 'evening'
    else: return 'night'

def _get_temp_uniformity():
    temps = [opt_sensor_data.get(k, 0) for k in ('temp1', 'temp2', 'temp3') if opt_sensor_data.get(k, 0) > 0]
    if len(temps) < 2: return 1.0
    return math.exp(-((max(temps) - min(temps)) ** 2) / 18.0)

def calculate_ac_fitness(temp_set, fan_speed, mode_idx=0, set_rh=50):
    """Fitness for GA. mode_idx: 0=COOL, 1=DRY, 2=FAN, 3=AUTO
    Genes: [temp, fan_speed, mode_idx, set_rh]
    Crowd-based tier:
      Tier 0 — 0 persons  : energy saving (27-29°C, fan 1)
      Tier 1 — 1-2 persons: standard temp by time (24-26°C, fan adaptive)
      Tier 2 — 3-5 persons: medium cooling (22°C, fan 3)
      Tier 3 — >5 persons : AC as cold as possible (16°C, fan 4 max)
    """
    temp_room = opt_sensor_data['temperature']
    humidity = opt_sensor_data['humidity']
    # Use 5-min window so brief camera misses don't cause GA to optimize for "no person" temps
    person_detected = opt_sensor_data['person_detected'] or _person_present_recently()
    person_count = max(0, int(opt_sensor_data.get('person_count', 1 if person_detected else 0)))
    temp_trend = opt_sensor_data.get('temp_trend', 0.0)
    time_period = _get_time_period()
    fitness = 0.0
    # ── Crowd-based temperature & fan tier ───────────────────────────────────────
    ideal_fan_crowd = None  # None = calculated from temp_gap (only for tier 1)
    if not person_detected:
        # Tier 0: no persons — energy saving mode
        target_map = {'morning': 28.0, 'afternoon': 27.0, 'evening': 28.0, 'night': 29.0}
        sigma_map   = {'morning': 3.0,  'afternoon': 2.5,  'evening': 3.0,  'night': 3.5}
        target_temp = target_map.get(time_period, 28.0)
        sigma = sigma_map.get(time_period, 3.0)
        ideal_fan_crowd = 1.0
    elif person_count > 5:
        # Tier 3: >5 persons — AC as cold as possible, max fan
        target_temp = OPT_TEMP_MIN   # 16°C (batas bawah GA)
        sigma = 0.8                  # very tight band — GA must choose minimum temperature
        ideal_fan_crowd = float(OPT_FAN_MAX)  # fan 4
    elif person_count >= 3:
        # Tier 2: 3-5 persons — medium cooling
        target_temp = 22.0
        sigma = 1.5
        ideal_fan_crowd = 3.0
    else:
        # Tier 1: 1-2 persons — standard temp by time
        std_target = {'morning': 25.0, 'afternoon': 24.0, 'evening': 25.0, 'night': 26.0}
        std_sigma  = {'morning': 2.0,  'afternoon': 1.5,  'evening': 2.0,  'night': 2.5}
        target_temp = std_target.get(time_period, 25.0)
        if person_count == 2:
            target_temp = max(target_temp - 0.5, 22.0)  # slightly cooler for 2 persons
        sigma = std_sigma.get(time_period, 2.0)
        # ideal_fan_crowd = None → will be calculated from temp_gap below
    # Trend offset: if room temp rises fast, lower target; if falls, raise target
    # For tier >5, trend offset still applied but target is already at minimum
    trend_offset = max(-2.0, min(2.0, -temp_trend * 2.0))
    adjusted_target = max(OPT_TEMP_MIN, target_temp + trend_offset) if person_count > 5 \
                      else target_temp + trend_offset
    fitness += _gaussian_score(temp_set, adjusted_target, sigma, 40.0)
    if humidity > 60:
        excess_factor = min(1.0, (humidity - 60) / 30.0)
        dehumid_target_temp = 24.0 - excess_factor * 2.0
        fitness += _gaussian_score(temp_set, dehumid_target_temp, 3.0, 10.0) + (fan_speed - 1) / 2.0 * 5.0 * excess_factor
    elif humidity < 40:
        dry_factor = min(1.0, (40 - humidity) / 20.0)
        fitness += (15.0 * (1 - dry_factor * 0.3)) if temp_set >= 25 else _gaussian_score(temp_set, 26.0, 3.0, 15.0) * (1 - dry_factor)
    else:
        fitness += 15.0 * _gaussian_score(humidity, 50.0, 10.0, 1.0)
    temp_gap = abs(temp_room - temp_set)
    # Ideal fan: tier 2 & 3 already set above; tier 1 calculated from temp_gap
    if ideal_fan_crowd is not None:
        ideal_fan = ideal_fan_crowd
    else:
        ideal_fan = min(3.0, max(1.0, 1.0 + (temp_gap - 1.0) / 2.0))
    fitness += _gaussian_score(abs(fan_speed - ideal_fan), 0.0, 1.0, 15.0)
    # ── Energy efficiency ─────────────────────────────────────────────────────────
    # Use real PZEM watt data when available; fall back to COP-based model.
    # COP model: efficiency drops exponentially as setpoint moves further below room temp.
    # Fan also consumes power — penalise high fan speed when gap is small.
    actual_watt = opt_sensor_data.get('actual_watt', 0.0)
    if actual_watt > 50.0:   # MySQL energy meter has live data and AC is drawing power
        MAX_WATT_REF = 1500.0
        energy_ratio = max(0.0, 1.0 - actual_watt / MAX_WATT_REF)
    else:
        temp_delta = max(0.0, temp_room - temp_set)   # how hard AC must work
        cop_efficiency = math.exp(-temp_delta * 0.22)  # exp(-0.44)≈0.64 at 2°C, exp(-1.1)≈0.33 at 5°C
        fan_efficiency = 1.0 - (fan_speed - 1) / 6.0  # fan1→1.0, fan4→0.5
        energy_ratio = cop_efficiency * 0.65 + fan_efficiency * 0.35
    # >5 people: comfort & cooling takes priority over energy efficiency
    if person_count > 5:
        energy_weight = 4.0
    elif person_detected:
        energy_weight = 12.0
    else:
        energy_weight = 20.0
    fitness += energy_ratio * energy_weight
    uniformity = _get_temp_uniformity()
    if uniformity < 0.7 and fan_speed >= 2:
        fitness += 8.0 * (1 - uniformity) * (fan_speed / 3.0)
    elif uniformity >= 0.7:
        fitness += 8.0 * uniformity
    if temp_trend > 0.3 and temp_set <= target_temp - 1:
        fitness += min(7.0, temp_trend * 5.0)
    elif temp_trend < -0.3 and temp_set >= target_temp:
        fitness += min(7.0, abs(temp_trend) * 5.0)
    else:
        fitness += _gaussian_score(temp_set, target_temp, 2.0, 4.0)
    if not person_detected and temp_set < 24:
        fitness -= (24.0 - temp_set) * 3.0
    # Temperature < 18 with mid fan: penalty only if not tier >5 (tier >5 wants this)
    if temp_set < 18 and fan_speed == 3 and person_count <= 5:
        fitness -= 10.0
    # Overcooling: penalty ONLY for tier 1-2 (tier >5 wants minimum temperature)
    if person_detected and person_count <= 5 and temp_set < target_temp - 1.5:
        fitness -= (target_temp - 1.5 - temp_set) * 5.0
    # Unnecessary high fan: penalty only for small crowd
    if fan_speed >= 3 and abs(temp_room - target_temp) <= 1.0 and person_count < 3:
        fitness -= (fan_speed - 2) * 4.0
    # Bonus: reward maximum fan when crowd >5 (maximum cooling comfort)
    if person_count > 5 and fan_speed == OPT_FAN_MAX:
        fitness += 12.0

    # ── Mode bonus / penalty (max ±10 pts) ────────────────────────────────────
    # COOL (0): best for cooling — reward when room is warm and cooling needed
    # DRY  (1): dehumidification — reward when humidity > 65%
    # FAN  (2): no cooling, just air circulation — penalise when hot, reward when mild
    # AUTO (3): neutral — small bonus for flexibility
    if mode_idx == 0:   # COOL
        if temp_room > target_temp:
            fitness += min(10.0, (temp_room - target_temp) * 2.0)
    elif mode_idx == 1:  # DRY
        if humidity > 65:
            fitness += min(10.0, (humidity - 65) / 3.5)
        elif humidity < 50:
            fitness -= 8.0   # DRY when already dry — bad
    elif mode_idx == 2:  # FAN
        if temp_room <= target_temp + 1:
            fitness += 5.0   # mild room — fan sufficient
        else:
            fitness -= (temp_room - (target_temp + 1)) * 3.0  # hot room — penalise FAN
    elif mode_idx == 3:  # AUTO
        fitness += 3.0  # small neutral bonus

    # ── Set RH fitness (max ±15 pts) ───────────────────────────────────────
    # Target RH for comfort: 45-55% (ideal 50%)
    # High humidity (>65%): reward low set_rh + DRY mode → dehumidify
    # Low humidity (<45%):  reward high set_rh → don't over-dry
    # Normal (45-55%):      reward set_rh near 50% (comfort zone)
    if humidity > 65:
        # High humidity — prefer low RH setpoint to dehumidify
        ideal_rh = max(OPT_RH_MIN, 30 + int((humidity - 65) / 5) * -2)  # lower as humidity rises
        ideal_rh = max(OPT_RH_MIN, min(45, 50 - int((humidity - 65) * 0.5)))
        fitness += _gaussian_score(set_rh, ideal_rh, 10.0, 10.0)
        # Strong bonus for DRY mode when humidity is high
        if mode_idx == 1:  # DRY
            fitness += min(8.0, (humidity - 65) * 0.5)
    elif humidity < 45:
        # Low humidity — prefer higher RH setpoint
        ideal_rh = min(OPT_RH_MAX, max(55, 50 + int((45 - humidity) * 0.8)))
        fitness += _gaussian_score(set_rh, ideal_rh, 10.0, 10.0)
        # Penalize DRY mode when humidity is already low
        if mode_idx == 1:
            fitness -= 5.0
    else:
        # Normal humidity — reward set_rh near 50%
        fitness += _gaussian_score(set_rh, 50, 8.0, 10.0)

    return max(0.0, round(fitness, 2))

def calculate_lamp_fitness_2d(pwm1, pwm2):
    """PSO fitness function — MINIMIZE (smaller = better).

    Goal:
    1. Lux_avg near 350 lux (main target)
    2. Balanced distribution between sensors — no sensor should be much
       brighter or darker than the others
    3. Each sensor at least 200 lux (no completely dark corners)

    Fitness formula:
        error_avg     = (Lux_avg - 350)^2              → pursue average target
        error_balance = variance(L1, L2, L3)            → penalize uneven distribution
        error_min     = penalty if any sensor < 200     → penalize dark corners
        Fitness = error_avg + w_bal * error_balance + w_min * error_min

    Estimasi lux per sensor menggunakan model proporsional berbasis gain aktual.
    """
    lux1 = float(opt_sensor_data.get('lux1', opt_sensor_data.get('lux', 0)))
    lux2 = float(opt_sensor_data.get('lux2', opt_sensor_data.get('lux', 0)))
    lux3 = float(opt_sensor_data.get('lux3', opt_sensor_data.get('lux', 0)))
    person_detected = opt_sensor_data['person_detected'] or _person_present_recently_lamp()
    TARGET_LUX  = 350.0 if person_detected else 0.0
    MIN_LUX     = 200.0  # minimum accepted lux per sensor
    W_BALANCE   = 0.5    # bobot penalty distribusi
    W_MIN       = 1.5    # bobot penalty sensor di bawah minimum (lebih ketat)

    # Current average PWM (0-255)
    b1_pct  = float(opt_sensor_data.get('curr_brightness1', 0))
    b2_pct  = float(opt_sensor_data.get('curr_brightness2', 0))
    pwm_now = ((b1_pct + b2_pct) / 2.0) * 255.0 / 100.0

    # Candidate average PWM
    pwm_cand = (float(pwm1) + float(pwm2)) / 2.0

    # Estimated lux per sensor for candidate PWM
    lux_now = (lux1 + lux2 + lux3) / 3.0
    if pwm_now > 5:
        ratio    = pwm_cand / pwm_now
        est_lux1 = max(0.0, lux1 * ratio)
        est_lux2 = max(0.0, lux2 * ratio)
        est_lux3 = max(0.0, lux3 * ratio)
    else:
        GAIN_DEFAULT = 1.57
        est_lux1 = pwm_cand * GAIN_DEFAULT
        est_lux2 = pwm_cand * GAIN_DEFAULT
        est_lux3 = pwm_cand * GAIN_DEFAULT

    lux_est_avg = (est_lux1 + est_lux2 + est_lux3) / 3.0

    # 1. Average error relative to target
    error_avg = (lux_est_avg - TARGET_LUX) ** 2

    # 2. Distribution penalty: variance between sensors (balanced = small variance)
    mean_e = lux_est_avg
    variance = ((est_lux1 - mean_e)**2 + (est_lux2 - mean_e)**2 + (est_lux3 - mean_e)**2) / 3.0
    error_balance = variance

    # 3. Penalty for sensor below minimum 200 lux
    error_min = 0.0
    for est in [est_lux1, est_lux2, est_lux3]:
        if est < MIN_LUX:
            error_min += (MIN_LUX - est) ** 2

    # Tolerance: fitness = 0 if avg 315-385 AND all sensors >= 200
    if (TARGET_LUX > 0 and 315.0 <= lux_est_avg <= 385.0
            and est_lux1 >= MIN_LUX and est_lux2 >= MIN_LUX and est_lux3 >= MIN_LUX):
        return 0.0

    fitness = error_avg + W_BALANCE * error_balance + W_MIN * error_min
    return round(fitness, 4)

def update_opt_sensor_data(**kwargs):
    for k, v in kwargs.items():
        if v is not None and k in opt_sensor_data:
            opt_sensor_data[k] = v
    if 'temperature' in kwargs and kwargs['temperature'] is not None:
        opt_sensor_data['data_source'] = 'mqtt'
        now = time.time()
        opt_sensor_data['temp_history'].append((now, kwargs['temperature']))
        if len(opt_sensor_data['temp_history']) > 10:
            opt_sensor_data['temp_history'] = opt_sensor_data['temp_history'][-10:]
        h = opt_sensor_data['temp_history']
        if len(h) >= 3:
            dt_min = (h[-1][0] - h[0][0]) / 60.0
            if dt_min > 0.1:
                opt_sensor_data['temp_trend'] = round((h[-1][1] - h[0][1]) / dt_min, 3)

# PHP proxy URL for MySQL energy meter
MYSQL_ENERGY_PHP_URL = 'https://iotlab-uns.com/api_energy.php?key=iotlab_smartroom_2024'
_last_mysql_power_fetch = 0.0
MYSQL_POWER_FETCH_INTERVAL = 60  # fetch at most once per minute

def _fetch_mysql_power():
    """Fetch latest AC active_power from MySQL via PHP proxy.
    Updates opt_sensor_data['actual_watt'] and ['power_factor'].
    Runs at most once per MYSQL_POWER_FETCH_INTERVAL seconds.
    """
    global _last_mysql_power_fetch
    now = time.time()
    if now - _last_mysql_power_fetch < MYSQL_POWER_FETCH_INTERVAL:
        return
    _last_mysql_power_fetch = now
    try:
        import urllib.request as _ureq
        req = _ureq.Request(MYSQL_ENERGY_PHP_URL,
                            headers={'User-Agent': 'SmartRoom-Optimizer/1.0'})
        with _ureq.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        ac_data        = data.get('ac') or {}
        active_power   = float(ac_data.get('active_power')   or ac_data.get('active_power') or 0)
        apparent_power = float(ac_data.get('apparent_power') or 0)
        voltage        = float(ac_data.get('voltage')  or 0)
        current        = float(ac_data.get('current')       or 0)
        frequency      = float(ac_data.get('frequency')  or 0)
        energy_wh      = float(ac_data.get('total_energy') or 0)   # Wh from PHP
        pf = round(active_power / apparent_power, 3) if apparent_power > 0.001 else 1.0
        opt_sensor_data['actual_watt']   = active_power
        opt_sensor_data['power_factor']  = pf
        print(f"[OPT] MySQL power: {active_power:.1f}W {voltage:.0f}V {current:.2f}A PF={pf:.2f}")
        # Write all fields to InfluxDB — for complete energy CSV export
        try:
            write_to_influxdb('energy_monitor', {
                'voltage':      voltage,
                'current':      current,
                'power':        active_power,
                'energy_kwh':   round(energy_wh / 1000.0, 4),  # Wh → kWh
                'frequency':    frequency,
                'power_factor': pf,
            }, tags={'device': 'esp32_ac', 'source': 'mysql'})
        except Exception as _ie:
            print(f"[OPT] InfluxDB write after MySQL fetch failed: {_ie}")
    except Exception as e:
        print(f"[OPT] MySQL power fetch failed: {e}")

def fetch_sensor_data_from_db(time_range_minutes=30):
    # Fetch real power from MySQL energy meter first
    _fetch_mysql_power()

    # --- Priority 1: use real-time MQTT data if available (freshest) ---
    # lux di mqtt_data['lamp'] is already filtered by _filter_lux di MQTT callback.
    # Only falls back to InfluxDB if ESP32 not connected (MQTT timeout > 2 minutes).
    mqtt_has_lux = (time.time() - _last_lamp_mqtt_ts) < 120  # fresh if < 2 minutes
    if mqtt_has_lux:
        # MQTT data is fresh — value in mqtt_data['lamp'] already passed _filter_lux
        # in MQTT callback, so sync directly to opt_sensor_data without refiltering.
        # opt_sensor_data lux is already updated in MQTT callback via update_opt_sensor_data,
        # but we sync again here to ensure consistency when PSO reads it.
        l1 = float(mqtt_data['lamp'].get('lux1', 0))
        l2 = float(mqtt_data['lamp'].get('lux2', 0))
        l3 = float(mqtt_data['lamp'].get('lux3', 0))
        opt_sensor_data['lux1'] = round(l1, 1)
        opt_sensor_data['lux2'] = round(l2, 1)
        opt_sensor_data['lux3'] = round(l3, 1)
        opt_sensor_data['lux']  = round((l1 + l2 + l3) / 3.0, 1)
        opt_sensor_data['curr_brightness1'] = mqtt_data['lamp'].get('brightness1', opt_sensor_data['curr_brightness1'])
        opt_sensor_data['curr_brightness2'] = mqtt_data['lamp'].get('brightness2', opt_sensor_data['curr_brightness2'])
        print(f"[OPT] Lux from MQTT (filtered): L1={l1} L2={l2} L3={l3} "
              f"B1={opt_sensor_data['curr_brightness1']}% B2={opt_sensor_data['curr_brightness2']}%")

    try:
        _, _, query_api = _get_influx_client()
        # AC: use mean() — temperature changes slowly, average is still relevant for GA
        ac_query = (f'from(bucket: "{INFLUX_BUCKET}")'
                    f' |> range(start: -{time_range_minutes}m)'
                    f' |> filter(fn: (r) => r._measurement == "ac_sensor")'
                    f' |> filter(fn: (r) => r._field == "temperature" or r._field == "humidity")'
                    f' |> mean()')
        for table in query_api.query(ac_query):
            for rec in table.records:
                if rec.get_field() == 'temperature' and rec.get_value() is not None:
                    opt_sensor_data['temperature'] = round(float(rec.get_value()), 1)
                elif rec.get_field() == 'humidity' and rec.get_value() is not None:
                    opt_sensor_data['humidity'] = round(float(rec.get_value()), 1)

        # Lux: use last() instead of mean() — light changes fast, latest value is relevant
        # Only query InfluxDB if MQTT has no data yet (ESP32 not connected)
        # Stale limit: if InfluxDB record older than 10 minutes, ignore to avoid
        # overwriting opt_sensor_data with old values (e.g., 350 from previous session).
        INFLUX_LAX_STALE_LIMIT_S = 600  # 10 minutes
        if not mqtt_has_lux:
            lamp_query = (f'from(bucket: "{INFLUX_BUCKET}")'
                          f' |> range(start: -{time_range_minutes}m)'
                          f' |> filter(fn: (r) => r._measurement == "lamp_sensor")'
                          f' |> filter(fn: (r) => r._field == "lux1" or r._field == "lux2"'
                          f'            or r._field == "lux3" or r._field == "lux_avg"'
                          f'            or r._field == "brightness1" or r._field == "brightness2")'
                          f' |> last()')
            influx_lux_used = False
            for table in query_api.query(lamp_query):
                for rec in table.records:
                    field = rec.get_field()
                    val = rec.get_value()
                    if val is None:
                        continue
                    # Check record age — reject if too old (stale data)
                    rec_time = rec.get_time()
                    if rec_time is not None:
                        try:
                            from datetime import timezone
                            now_utc = datetime.now(timezone.utc)
                            age_s = (now_utc - rec_time).total_seconds()
                            if age_s > INFLUX_LAX_STALE_LIMIT_S:
                                print(f"[OPT] InfluxDB {field} diabaikan — usia {age_s:.0f}s > {INFLUX_LAX_STALE_LIMIT_S}s (stale)")
                                continue
                        except Exception:
                            pass  # if time parse fails, still use data
                    if field == 'lux_avg':
                        opt_sensor_data['lux'] = round(float(val), 1)
                        influx_lux_used = True
                    elif field in ('lux1', 'lux2', 'lux3'):
                        # Skip _filter_lux so stale values from InfluxDB are also filtered
                        # brightness=0 because this fallback only runs when MQTT is absent
                        sensor_idx = {'lux1': 0, 'lux2': 1, 'lux3': 2}[field]
                        filtered = _filter_lux(float(val), sensor_idx,
                                               brightness1=opt_sensor_data.get('curr_brightness1', 0),
                                               brightness2=opt_sensor_data.get('curr_brightness2', 0))
                        opt_sensor_data[field] = filtered
                        influx_lux_used = True
                    elif field == 'brightness1':
                        opt_sensor_data['curr_brightness1'] = round(float(val), 1)
                    elif field == 'brightness2':
                        opt_sensor_data['curr_brightness2'] = round(float(val), 1)
            if influx_lux_used:
                print(f"[OPT] Lux from InfluxDB (fresh): L1={opt_sensor_data.get('lux1')} L2={opt_sensor_data.get('lux2')} L3={opt_sensor_data.get('lux3')}")
            else:
                print(f"[OPT] InfluxDB lux stale/empty — using current opt_sensor_data: L1={opt_sensor_data.get('lux1')} L2={opt_sensor_data.get('lux2')} L3={opt_sensor_data.get('lux3')}")

        opt_sensor_data['data_source'] = 'mqtt' if mqtt_has_lux else 'influxdb_last'
    except Exception as e:
        print(f"[OPT] InfluxDB fetch failed: {e}, using MQTT data")
        opt_sensor_data['data_source'] = 'mqtt_fallback'

def run_ga_optimization(verbose=False):
    pop_size = ga_params['population_size']
    generations = ga_params['generations']
    mutation_rate = ga_params['mutation_rate']
    crossover_rate = ga_params['crossover_rate']
    elitism_ratio = ga_params.get('elitism_ratio', 0.2)
    elite_count = max(2, int(pop_size * elitism_ratio))
    seed_solutions = []
    if last_opt_results['ga']['temp'] > 0 and last_opt_results['ga']['fan'] > 0:
        seed_mode = last_opt_results['ga'].get('mode_idx', 0)
        seed_rh = last_opt_results['ga'].get('set_rh', 50)
        seed_solutions.append([last_opt_results['ga']['temp'], last_opt_results['ga']['fan'], seed_mode, seed_rh])

    def create_ind():
        # ind = [temp, fan_speed, mode_idx, set_rh]
        return [round(random.uniform(OPT_TEMP_MIN, OPT_TEMP_MAX), 1),
                random.randint(OPT_FAN_MIN, OPT_FAN_MAX),
                random.randint(OPT_MODE_MIN, OPT_MODE_MAX),
                random.randint(OPT_RH_MIN, OPT_RH_MAX)]

    population = []
    for s in seed_solutions[:max(1, int(pop_size * 0.3))]:
        population.append([float(s[0]), int(s[1]), int(s[2]), int(s[3])])
        if len(population) < pop_size:
            population.append([round(max(OPT_TEMP_MIN, min(OPT_TEMP_MAX, s[0] + random.uniform(-1.5, 1.5))), 1),
                               max(OPT_FAN_MIN, min(OPT_FAN_MAX, s[1] + random.choice([-1, 0, 0, 1]))),
                               int(s[2]),
                               max(OPT_RH_MIN, min(OPT_RH_MAX, int(s[3]) + random.choice([-5, 0, 0, 5])))])
    while len(population) < pop_size:
        population.append(create_ind())
    population = population[:pop_size]

    best_solution, best_fitness = None, 0
    fitness_history = []
    stagnation_counter, stagnation_limit = 0, max(5, generations // 4)
    prev_best = 0

    for gen in range(generations):
        scores = [calculate_ac_fitness(int(round(ind[0])), ind[1], ind[2], ind[3]) for ind in population]
        paired = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        population = [p[0] for p in paired]
        scores = [p[1] for p in paired]
        if scores[0] > best_fitness:
            best_fitness = scores[0]
            best_solution = population[0][:]
        fitness_history.append(best_fitness)
        improvement = best_fitness - prev_best
        stagnation_counter = stagnation_counter + 1 if improvement < 0.01 else 0
        prev_best = best_fitness
        boost = False
        if stagnation_counter >= stagnation_limit:
            for i in range(max(2, pop_size // 4)):
                population[-(i + 1)] = create_ind()
            scores = [calculate_ac_fitness(int(round(ind[0])), ind[1], ind[2], ind[3]) for ind in population]
            boost = True
            stagnation_counter = 0
        if stagnation_counter >= stagnation_limit * 2 and gen > generations // 2:
            break
        next_pop = [ind[:] for ind in population[:elite_count]]
        # tournament selection
        selected = []
        for _ in range(len(population)):
            contestants = random.sample(range(len(population)), min(3, len(population)))
            best_idx = max(contestants, key=lambda i: scores[i])
            selected.append(population[best_idx][:])
        while len(next_pop) < pop_size:
            p1, p2 = random.sample(selected, 2)
            # BLX-alpha crossover for temp and RH; uniform for fan & mode
            if random.random() < crossover_rate:
                alpha = 0.3
                lo, hi = min(p1[0], p2[0]), max(p1[0], p2[0])
                span = hi - lo
                c1t = round(max(OPT_TEMP_MIN, min(OPT_TEMP_MAX, random.uniform(lo - alpha * span, hi + alpha * span))), 1)
                c2t = round(max(OPT_TEMP_MIN, min(OPT_TEMP_MAX, random.uniform(lo - alpha * span, hi + alpha * span))), 1)
                c1f = p1[1] if random.random() < 0.5 else p2[1]
                c2f = p2[1] if random.random() < 0.5 else p1[1]
                c1m = p1[2] if random.random() < 0.5 else p2[2]
                c2m = p2[2] if random.random() < 0.5 else p1[2]
                # BLX-alpha crossover for set_rh
                rh_lo, rh_hi = min(p1[3], p2[3]), max(p1[3], p2[3])
                rh_span = rh_hi - rh_lo
                c1r = int(max(OPT_RH_MIN, min(OPT_RH_MAX, random.uniform(rh_lo - alpha * rh_span, rh_hi + alpha * rh_span))))
                c2r = int(max(OPT_RH_MIN, min(OPT_RH_MAX, random.uniform(rh_lo - alpha * rh_span, rh_hi + alpha * rh_span))))
                child1, child2 = [c1t, c1f, c1m, c1r], [c2t, c2f, c2m, c2r]
            else:
                child1, child2 = p1[:], p2[:]
            # mutate
            progress = gen / max(1, generations)
            adaptive_rate = mutation_rate * (1.0 - 0.7 * progress)
            if boost: adaptive_rate = min(0.9, adaptive_rate * 2.5)
            for child in [child1, child2]:
                if random.random() < adaptive_rate:
                    step = (2.0 * (1 - progress) + 0.5) * (2.0 if boost else 1.0)
                    child[0] = round(max(OPT_TEMP_MIN, min(OPT_TEMP_MAX, child[0] + random.gauss(0, step))), 1)
                if random.random() < adaptive_rate:
                    child[1] = random.randint(OPT_FAN_MIN, OPT_FAN_MAX)
                if random.random() < adaptive_rate * 0.5:  # mode mutates less often
                    child[2] = random.randint(OPT_MODE_MIN, OPT_MODE_MAX)
                if random.random() < adaptive_rate:
                    rh_step = int((10.0 * (1 - progress) + 2.0) * (2.0 if boost else 1.0))
                    child[3] = max(OPT_RH_MIN, min(OPT_RH_MAX, child[3] + random.randint(-rh_step, rh_step)))
            next_pop.append(child1)
            if len(next_pop) < pop_size:
                next_pop.append(child2)
        population = next_pop[:pop_size]

    # brute-force validation (temp × fan × mode × rh_samples search)
    bf_best_fit, bf_best_sol = -1, None
    rh_samples = [30, 40, 45, 50, 55, 60, 70, 80]
    for t in range(int(OPT_TEMP_MIN), int(OPT_TEMP_MAX) + 1):
        for f in range(OPT_FAN_MIN, OPT_FAN_MAX + 1):
            for m in range(OPT_MODE_MIN, OPT_MODE_MAX + 1):
                for rh in rh_samples:
                    fit = calculate_ac_fitness(t, f, m, rh)
                    if fit > bf_best_fit:
                        bf_best_fit, bf_best_sol = fit, [t, f, m, rh]
    if bf_best_fit > best_fitness:
        best_solution = [float(bf_best_sol[0]), bf_best_sol[1], bf_best_sol[2], bf_best_sol[3]]
        best_fitness = bf_best_fit
    final = [int(round(best_solution[0])), best_solution[1], best_solution[2], best_solution[3]]
    return final, best_fitness, fitness_history, {'solution': bf_best_sol, 'fitness': bf_best_fit}

def run_pso_for_ac(verbose=False):
    """PSO optimizer for AC settings — same search space as GA.
    Particles: [temp (float), fan_speed (int), mode_idx (int), set_rh (int)]
    Fitness: maximize calculate_ac_fitness() (same as GA).
    Returns same format as run_ga_optimization() for drop-in replacement.
    """
    swarm_size = pso_params.get('swarm_size', 10)
    iterations = pso_params.get('iterations', 20)
    w  = pso_params.get('w', 0.5)
    c1 = pso_params.get('c1', 1.5)
    c2 = pso_params.get('c2', 1.5)
    DIM = 4  # [temp, fan, mode, rh]

    # Bounds for each dimension
    lo = [OPT_TEMP_MIN, OPT_FAN_MIN, OPT_MODE_MIN, OPT_RH_MIN]
    hi = [OPT_TEMP_MAX, OPT_FAN_MAX, OPT_MODE_MAX, OPT_RH_MAX]
    max_vel = [(hi[d] - lo[d]) * 0.3 for d in range(DIM)]

    # Initialize swarm
    positions = []
    velocities = []
    for _ in range(swarm_size):
        pos = [
            round(random.uniform(OPT_TEMP_MIN, OPT_TEMP_MAX), 1),
            random.randint(OPT_FAN_MIN, OPT_FAN_MAX),
            random.randint(OPT_MODE_MIN, OPT_MODE_MAX),
            random.randint(OPT_RH_MIN, OPT_RH_MAX),
        ]
        vel = [random.uniform(-max_vel[d], max_vel[d]) for d in range(DIM)]
        positions.append(pos)
        velocities.append(vel)

    # Seed with previous best (if available)
    if last_opt_results['ga']['temp'] > 0 and last_opt_results['ga']['fan'] > 0:
        seed = [
            float(last_opt_results['ga']['temp']),
            int(last_opt_results['ga']['fan']),
            int(last_opt_results['ga'].get('mode_idx', 0)),
            int(last_opt_results['ga'].get('set_rh', 50)),
        ]
        positions[0] = seed[:]
        velocities[0] = [0.0] * DIM

    # Evaluate initial fitness (maximize)
    def eval_particle(p):
        return calculate_ac_fitness(int(round(p[0])), int(round(p[1])),
                                    int(round(p[2])), int(round(p[3])))

    pb_pos = [p[:] for p in positions]
    pb_fit = [eval_particle(p) for p in positions]
    g_idx  = pb_fit.index(max(pb_fit))
    g_pos  = pb_pos[g_idx][:]
    g_fit  = pb_fit[g_idx]

    fitness_history = []

    for it in range(iterations):
        # Adaptive inertia
        current_w = w - (w - 0.3) * (it / max(1, iterations))

        for i in range(swarm_size):
            for d in range(DIM):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (current_w * velocities[i][d]
                    + c1 * r1 * (pb_pos[i][d] - positions[i][d])
                    + c2 * r2 * (g_pos[d] - positions[i][d]))
                velocities[i][d] = max(-max_vel[d], min(max_vel[d], velocities[i][d]))
                positions[i][d] += velocities[i][d]

            # Clip to bounds and discretize integer dimensions
            positions[i][0] = round(max(lo[0], min(hi[0], positions[i][0])), 1)  # temp (float)
            positions[i][1] = int(round(max(lo[1], min(hi[1], positions[i][1]))))  # fan (int)
            positions[i][2] = int(round(max(lo[2], min(hi[2], positions[i][2]))))  # mode (int)
            positions[i][3] = int(round(max(lo[3], min(hi[3], positions[i][3]))))  # rh (int)

            fit = eval_particle(positions[i])

            # Update personal best (maximize)
            if fit > pb_fit[i]:
                pb_fit[i] = fit
                pb_pos[i] = positions[i][:]

            # Update global best
            if fit > g_fit:
                g_fit = fit
                g_pos = positions[i][:]

        fitness_history.append(g_fit)

    # Brute-force validation (same as GA)
    bf_best_fit, bf_best_sol = -1, None
    rh_samples = [30, 40, 45, 50, 55, 60, 70, 80]
    for t in range(int(OPT_TEMP_MIN), int(OPT_TEMP_MAX) + 1):
        for f in range(OPT_FAN_MIN, OPT_FAN_MAX + 1):
            for m in range(OPT_MODE_MIN, OPT_MODE_MAX + 1):
                for rh in rh_samples:
                    fit = calculate_ac_fitness(t, f, m, rh)
                    if fit > bf_best_fit:
                        bf_best_fit, bf_best_sol = fit, [t, f, m, rh]
    if bf_best_fit > g_fit:
        g_pos = [float(bf_best_sol[0]), bf_best_sol[1], bf_best_sol[2], bf_best_sol[3]]
        g_fit = bf_best_fit

    final = [int(round(g_pos[0])), int(round(g_pos[1])), int(round(g_pos[2])), int(round(g_pos[3]))]
    print(f"[PSO-AC] Done: {final[0]}°C Fan={final[1]} Mode={AC_MODE_NAMES.get(final[2],'COOL')} "
          f"RH={final[3]}% fitness={g_fit:.2f}")
    return final, g_fit, fitness_history, {'solution': bf_best_sol, 'fitness': bf_best_fit}

def run_pso_optimization(verbose=False):
    """2D PSO with real lux read per iteration.

    Alur per iterasi:
        1. Update kecepatan & posisi semua partikel (di memori)
        2. Tentukan Gbest terbaik dari seluruh swarm
        3. Send Gbest ke lampu via MQTT
        4. Wait for BH1750 sensor to stabilize (~2.5 seconds)
        5. Baca lux nyata dari sensor
        6. Hitung fitness berdasarkan lux nyata
        7. Cek stop early: jika lux 315-385 → berhenti
        8. Check timeout: if total > 60 seconds → use last Gbest

    Lamp hanya berubah SEKALI per iterasi (bukan per partikel).

    Return tambahan: iteration_log — list of dict per iterasi berisi
        {iter, pwm1, pwm2, b1, b2, lux_avg, fitness}
        to display on the dashboard chart.
    """
    swarm_size = pso_params['swarm_size']
    iterations = min(pso_params['iterations'], 10)
    w  = pso_params['w']
    c1 = pso_params['c1']
    c2 = pso_params['c2']
    DIM = 2
    max_vel = 60
    SENSOR_SETTLE_S = 5.0   # 5 seconds per iteration: PWM changes → lamp stabilizes → sensor reads
    TIMEOUT_S = 60.0

    # ── Phase 3: Inisialisasi swarm ─────────────────────────────────────────
    positions  = [
        [int(OPT_BRIGHTNESS_MIN + random.random() * (OPT_BRIGHTNESS_MAX - OPT_BRIGHTNESS_MIN))
         for _ in range(DIM)]
        for _ in range(swarm_size)
    ]
    velocities = [[random.uniform(-max_vel, max_vel) for _ in range(DIM)]
                  for _ in range(swarm_size)]

    # ── Phase 4: Fitness awal ───────────────────────────────────────────────
    pb_pos = [p[:] for p in positions]
    pb_fit = [calculate_lamp_fitness_2d(p[0], p[1]) for p in positions]
    g_idx  = pb_fit.index(min(pb_fit))
    g_pos  = pb_pos[g_idx][:]
    g_fit  = pb_fit[g_idx]

    fitness_history = []   # error per iteration (for convergence chart)
    iteration_log   = []   # {iter, pwm1, pwm2, b1, b2, lux_avg, fitness} for detail chart

    start_time = time.time()

    # ── Phase 5-7: Iteration loop with real lux reading ───────────────────────
    for it in range(iterations):
        # Timeout check
        elapsed = time.time() - start_time
        if elapsed >= TIMEOUT_S:
            print(f"[PSO] Timeout {TIMEOUT_S:.0f}s di iterasi {it} — pakai Gbest terakhir")
            break

        # Update kecepatan dan posisi semua partikel (di memori)
        for i in range(swarm_size):
            for d in range(DIM):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (w * velocities[i][d]
                    + c1 * r1 * (pb_pos[i][d] - positions[i][d])
                    + c2 * r2 * (g_pos[d]      - positions[i][d]))
                velocities[i][d] = max(-max_vel, min(max_vel, velocities[i][d]))
                positions[i][d]  = max(OPT_BRIGHTNESS_MIN, min(OPT_BRIGHTNESS_MAX,
                                       int(round(positions[i][d] + velocities[i][d]))))

        # Determine Gbest candidate from new position (estimation)
        cand_fits      = [calculate_lamp_fitness_2d(positions[i][0], positions[i][1])
                          for i in range(swarm_size)]
        best_cand_idx  = cand_fits.index(min(cand_fits))
        best_cand_pos  = positions[best_cand_idx][:]

        # Send only Gbest candidate to lamp — once per iteration
        b1_send = round(best_cand_pos[0] * 100.0 / 255.0, 1)
        b2_send = round(best_cand_pos[1] * 100.0 / 255.0, 1)
        b1_send, b2_send = _safe_lamp_brightness(b1_send, b2_send)
        mqtt_client.publish(
            'smartroom/lamp/control',
            json.dumps({'brightness1': b1_send, 'brightness2': b2_send, 'source': 'pso_iter'})
        )

        # Emit iteration progress to dashboard in real-time
        socketio.emit('pso_iter_progress', {
            'iter': it + 1,
            'pwm1': best_cand_pos[0], 'pwm2': best_cand_pos[1],
            'b1': b1_send, 'b2': b2_send,
            'status': 'waiting'
        })

        print(f"[PSO] Iteration {it+1}/{iterations} — B1={b1_send}% B2={b2_send}% "
              f"(PWM1={best_cand_pos[0]} PWM2={best_cand_pos[1]}), tunggu sensor...")

        # Tunggu sensor stabil
        time.sleep(SENSOR_SETTLE_S)

        # Baca lux nyata
        lux1_r = float(opt_sensor_data.get('lux1', opt_sensor_data.get('lux', 0)))
        lux2_r = float(opt_sensor_data.get('lux2', opt_sensor_data.get('lux', 0)))
        lux3_r = float(opt_sensor_data.get('lux3', opt_sensor_data.get('lux', 0)))
        lux_real = round((lux1_r + lux2_r + lux3_r) / 3.0, 1)

        # Hitung fitness nyata menggunakan fungsi objektif yang sama dengan estimasi
        person_now = opt_sensor_data.get('person_detected', False) or _person_present_recently_lamp()
        TARGET_LUX = 350.0 if person_now else 0.0
        
        err_avg = (lux_real - TARGET_LUX) ** 2
        var_real = ((lux1_r - lux_real)**2 + (lux2_r - lux_real)**2 + (lux3_r - lux_real)**2) / 3.0
        err_min = 0.0
        MIN_LUX = 200.0
        for lux_val in [lux1_r, lux2_r, lux3_r]:
            if lux_val < MIN_LUX:
                err_min += (MIN_LUX - lux_val) ** 2
                
        real_fit = err_avg + 0.5 * var_real + 1.5 * err_min
        
        if TARGET_LUX > 0 and 315.0 <= lux_real <= 385.0 and lux1_r >= MIN_LUX and lux2_r >= MIN_LUX and lux3_r >= MIN_LUX:
            real_fit = 0.0
            
        real_fit = round(real_fit, 4)

        print(f"[PSO] Lux nyata={lux_real} | fitness={real_fit} | target={TARGET_LUX:.0f}")

        # Record iteration log for dashboard chart
        log_entry = {
            'iter':    it + 1,
            'pwm1':    best_cand_pos[0],
            'pwm2':    best_cand_pos[1],
            'b1':      b1_send,
            'b2':      b2_send,
            'lux1':    round(lux1_r, 1),
            'lux2':    round(lux2_r, 1),
            'lux3':    round(lux3_r, 1),
            'lux_avg': lux_real,
            'fitness': real_fit,
        }
        iteration_log.append(log_entry)
        fitness_history.append(real_fit)

        # Emit iteration result to dashboard (update live chart)
        socketio.emit('pso_iter_progress', {**log_entry, 'status': 'done'})

        # Update Pbest
        if real_fit < pb_fit[best_cand_idx]:
            pb_fit[best_cand_idx] = real_fit
            pb_pos[best_cand_idx] = best_cand_pos[:]

        # Update Gbest
        if real_fit < g_fit:
            g_fit = real_fit
            g_pos = best_cand_pos[:]
        elif abs(real_fit - g_fit) < 1.0:
            if (best_cand_pos[0] + best_cand_pos[1]) < (g_pos[0] + g_pos[1]):
                g_pos = best_cand_pos[:]

        # Stop early: lux avg dalam toleransi DAN semua sensor >= 200 lux
        if TARGET_LUX > 0 and 315.0 <= lux_real <= 385.0:
            if lux1_r >= 200.0 and lux2_r >= 200.0 and lux3_r >= 200.0:
                print(f"[PSO] Stop early iterasi {it+1} — lux {lux_real} dalam 315-385, "
                      f"L1={lux1_r} L2={lux2_r} L3={lux3_r} semua ≥200")
                break
            else:
                print(f"[PSO] Lux avg={lux_real} in target but some sensor < 200 "
                      f"(L1={lux1_r} L2={lux2_r} L3={lux3_r}) — lanjut iterasi")

    elapsed_total = time.time() - start_time
    print(f"[PSO] Selesai {len(fitness_history)} iterasi dalam {elapsed_total:.1f}s | "
          f"Gbest: PWM1={g_pos[0]} PWM2={g_pos[1]} | fitness={g_fit:.2f}")

    initial_error = fitness_history[0] if fitness_history else g_fit
    return list(g_pos), g_fit, fitness_history, initial_error, iteration_log

def run_ga_for_lamp(verbose=False):
    """GA optimizer for Lamp brightness — same search space as PSO.
    Chromosome: [PWM1 (0-255), PWM2 (0-255)]
    Fitness: minimize calculate_lamp_fitness_2d() (same as PSO, lower = better).
    Real sensor feedback per generation: send best → wait → read lux.
    Returns same format as run_pso_optimization() for drop-in replacement.
    """
    pop_size = ga_params.get('population_size', 15)
    generations = min(ga_params.get('generations', 10), 10)  # cap at 10 (5s per gen)
    mutation_rate = ga_params.get('mutation_rate', 0.3)
    crossover_rate = ga_params.get('crossover_rate', 0.85)
    elitism_ratio = ga_params.get('elitism_ratio', 0.2)
    elite_count = max(2, int(pop_size * elitism_ratio))
    SENSOR_SETTLE_S = 5.0
    TIMEOUT_S = 60.0

    # Initialize population
    def create_ind():
        return [random.randint(OPT_BRIGHTNESS_MIN, OPT_BRIGHTNESS_MAX),
                random.randint(OPT_BRIGHTNESS_MIN, OPT_BRIGHTNESS_MAX)]

    population = [create_ind() for _ in range(pop_size)]

    # Seed with current brightness if available
    b1_pct = opt_sensor_data.get('curr_brightness1', 0)
    b2_pct = opt_sensor_data.get('curr_brightness2', 0)
    if b1_pct > 0 or b2_pct > 0:
        seed_pwm1 = int(round(b1_pct * 255.0 / 100.0))
        seed_pwm2 = int(round(b2_pct * 255.0 / 100.0))
        population[0] = [seed_pwm1, seed_pwm2]

    fitness_history = []
    iteration_log = []
    start_time = time.time()
    best_solution = population[0][:]
    best_fitness = float('inf')

    for gen in range(generations):
        elapsed = time.time() - start_time
        if elapsed >= TIMEOUT_S:
            print(f"[GA-LAMP] Timeout {TIMEOUT_S:.0f}s at generation {gen}")
            break

        # Evaluate fitness using estimation for all individuals
        scores = [calculate_lamp_fitness_2d(ind[0], ind[1]) for ind in population]

        # Sort by fitness (ascending — minimize)
        paired = sorted(zip(population, scores), key=lambda x: x[1])
        population = [p[0] for p in paired]
        scores = [p[1] for p in paired]

        # Best individual of this generation (estimated)
        gen_best = population[0][:]

        # Send best individual to lamp for real sensor reading
        b1_send = round(gen_best[0] * 100.0 / 255.0, 1)
        b2_send = round(gen_best[1] * 100.0 / 255.0, 1)
        b1_send, b2_send = _safe_lamp_brightness(b1_send, b2_send)
        mqtt_client.publish(
            'smartroom/lamp/control',
            json.dumps({'brightness1': b1_send, 'brightness2': b2_send, 'source': 'ga_lamp_gen'})
        )

        # Emit progress
        socketio.emit('pso_iter_progress', {
            'iter': gen + 1, 'pwm1': gen_best[0], 'pwm2': gen_best[1],
            'b1': b1_send, 'b2': b2_send, 'status': 'waiting'
        })

        print(f"[GA-LAMP] Gen {gen+1}/{generations} — B1={b1_send}% B2={b2_send}% "
              f"(PWM1={gen_best[0]} PWM2={gen_best[1]}), waiting sensor...")

        # Wait for sensor to stabilize
        time.sleep(SENSOR_SETTLE_S)

        # Read real lux
        lux1_r = float(opt_sensor_data.get('lux1', opt_sensor_data.get('lux', 0)))
        lux2_r = float(opt_sensor_data.get('lux2', opt_sensor_data.get('lux', 0)))
        lux3_r = float(opt_sensor_data.get('lux3', opt_sensor_data.get('lux', 0)))
        lux_real = round((lux1_r + lux2_r + lux3_r) / 3.0, 1)

        # Compute real fitness
        person_now = opt_sensor_data.get('person_detected', False) or _person_present_recently_lamp()
        TARGET_LUX = 350.0 if person_now else 0.0
        MIN_LUX = 200.0

        err_avg = (lux_real - TARGET_LUX) ** 2
        var_real = ((lux1_r - lux_real)**2 + (lux2_r - lux_real)**2 + (lux3_r - lux_real)**2) / 3.0
        err_min = sum((MIN_LUX - lv) ** 2 for lv in [lux1_r, lux2_r, lux3_r] if lv < MIN_LUX)

        real_fit = err_avg + 0.5 * var_real + 1.5 * err_min
        if TARGET_LUX > 0 and 315.0 <= lux_real <= 385.0 and lux1_r >= MIN_LUX and lux2_r >= MIN_LUX and lux3_r >= MIN_LUX:
            real_fit = 0.0
        real_fit = round(real_fit, 4)

        print(f"[GA-LAMP] Real lux={lux_real} | fitness={real_fit} | target={TARGET_LUX:.0f}")

        # Record iteration log
        log_entry = {
            'iter': gen + 1, 'pwm1': gen_best[0], 'pwm2': gen_best[1],
            'b1': b1_send, 'b2': b2_send,
            'lux1': round(lux1_r, 1), 'lux2': round(lux2_r, 1), 'lux3': round(lux3_r, 1),
            'lux_avg': lux_real, 'fitness': real_fit,
        }
        iteration_log.append(log_entry)
        fitness_history.append(real_fit)

        socketio.emit('pso_iter_progress', {**log_entry, 'status': 'done'})

        # Track overall best
        if real_fit < best_fitness:
            best_fitness = real_fit
            best_solution = gen_best[:]

        # Stop early if converged
        if TARGET_LUX > 0 and 315.0 <= lux_real <= 385.0:
            if lux1_r >= 200.0 and lux2_r >= 200.0 and lux3_r >= 200.0:
                print(f"[GA-LAMP] Stop early gen {gen+1} — lux {lux_real} in 315-385")
                break

        # --- GA operators for next generation ---
        next_pop = [ind[:] for ind in population[:elite_count]]

        # Tournament selection
        selected = []
        for _ in range(len(population)):
            contestants = random.sample(range(len(population)), min(3, len(population)))
            best_idx = min(contestants, key=lambda i: scores[i])  # minimize
            selected.append(population[best_idx][:])

        # Crossover & mutation
        while len(next_pop) < pop_size:
            p1, p2 = random.sample(selected, 2)
            if random.random() < crossover_rate:
                # BLX-alpha crossover
                alpha = 0.3
                children = []
                for p in [p1, p2]:
                    child = []
                    for d in range(2):
                        lo_v, hi_v = min(p1[d], p2[d]), max(p1[d], p2[d])
                        span = hi_v - lo_v
                        val = random.uniform(lo_v - alpha * span, hi_v + alpha * span)
                        child.append(int(round(max(OPT_BRIGHTNESS_MIN, min(OPT_BRIGHTNESS_MAX, val)))))
                    children.append(child)
                child1, child2 = children
            else:
                child1, child2 = p1[:], p2[:]

            # Mutate
            progress = gen / max(1, generations)
            adaptive_rate = mutation_rate * (1.0 - 0.7 * progress)
            for child in [child1, child2]:
                for d in range(2):
                    if random.random() < adaptive_rate:
                        step = int((30 * (1 - progress) + 5))
                        child[d] = max(OPT_BRIGHTNESS_MIN, min(OPT_BRIGHTNESS_MAX,
                                       child[d] + random.randint(-step, step)))

            next_pop.append(child1)
            if len(next_pop) < pop_size:
                next_pop.append(child2)

        population = next_pop[:pop_size]

    elapsed_total = time.time() - start_time
    print(f"[GA-LAMP] Done {len(fitness_history)} generations in {elapsed_total:.1f}s | "
          f"Best: PWM1={best_solution[0]} PWM2={best_solution[1]} | fitness={best_fitness:.2f}")

    initial_error = fitness_history[0] if fitness_history else best_fitness
    return list(best_solution), best_fitness, fitness_history, initial_error, iteration_log

def run_optimization_cycle(algo='both'):
    global optimization_run_count
    if not optimization_lock.acquire(blocking=False):
        print("[OPT] Already running, skipping")
        return False
    try:
        ac_algo = _get_ac_algo()
        lamp_algo = _get_lamp_algo()
        socketio.emit('ml_status', {'status': 'running', 'algorithm': algo,
                                    'ac_algo': ac_algo, 'lamp_algo': lamp_algo,
                                    'algo_config': opt_algo_config})
        fetch_sensor_data_from_db(30)
        if algo in ('ga', 'both'):
            # Dynamic dispatch: GA or PSO for AC
            if ac_algo == 'pso':
                sol, fit, hist, bf = run_pso_for_ac()
                print(f"[OPT] AC using PSO (config: {opt_algo_config})")
            else:
                sol, fit, hist, bf = run_ga_optimization()
                print(f"[OPT] AC using GA (config: {opt_algo_config})")
            mode_idx = sol[2] if len(sol) > 2 else 0
            opt_set_rh = sol[3] if len(sol) > 3 else 50
            last_opt_results['ga'] = {
                'fitness': fit, 'temp': sol[0], 'fan': sol[1],
                'mode_idx': mode_idx, 'mode': AC_MODE_NAMES.get(mode_idx, 'COOL'),
                'set_rh': opt_set_rh,
                'stats': hist, 'brute_force': bf,
                'initial_fitness': round(hist[0], 2) if hist else 0,
                'final_fitness': round(fit, 2),
                'run_time': datetime.now().strftime('%d %b %Y %H:%M'),
                'params': {
                    'population_size': ga_params['population_size'],
                    'generations': ga_params['generations'],
                    'mutation_rate': ga_params['mutation_rate'],
                    'crossover_rate': ga_params['crossover_rate'],
                    'elitism_ratio': ga_params['elitism_ratio'],
                } if ac_algo == 'ga' else {
                    'swarm_size': pso_params['swarm_size'],
                    'iterations': pso_params['iterations'],
                    'w': pso_params['w'],
                    'c1': pso_params['c1'],
                    'c2': pso_params['c2'],
                },
                'sensor_snapshot': {
                    'temp_room': round(opt_sensor_data.get('temperature', 0), 1),
                    'humidity': round(opt_sensor_data.get('humidity', 0), 1),
                    'person_detected': opt_sensor_data.get('person_detected', False),
                    'person_count': opt_sensor_data.get('person_count', 0),
                    'actual_watt': round(opt_sensor_data.get('actual_watt', 0), 1),
                },
            }
            print(f"[{ac_algo.upper()}-AC] Done: {sol[0]}°C Fan={sol[1]} Mode={AC_MODE_NAMES.get(mode_idx,'COOL')} RH={opt_set_rh}% fitness={fit:.2f}")
            persist_opt_results('ga')
        if algo in ('pso', 'both'):
            # Dynamic dispatch: PSO or GA for Lamp
            if lamp_algo == 'ga':
                sol, fit, hist, initial_err, iter_log = run_ga_for_lamp()
                print(f"[OPT] Lamp using GA (config: {opt_algo_config})")
            else:
                sol, fit, hist, initial_err, iter_log = run_pso_optimization()
                print(f"[OPT] Lamp using PSO (config: {opt_algo_config})")
            # sol is PWM 0-255 directly from algorithm
            pwm1_val = int(round(min(255, max(0, sol[0]))))
            pwm2_val = int(round(min(255, max(0, sol[1]))))
            # Convert to brightness % for MQTT/ESP32
            b1 = round(pwm1_val * 100 / 255)
            b2 = round(pwm2_val * 100 / 255)
            # Estimate achieved lux from optimal brightness
            person_now = opt_sensor_data.get('person_detected', False) or _person_present_recently_lamp()
            target_lux = 350.0 if person_now else 0.0
            lux1 = opt_sensor_data.get('lux1', opt_sensor_data.get('lux', 0))
            lux2 = opt_sensor_data.get('lux2', opt_sensor_data.get('lux', 0))
            lux3 = opt_sensor_data.get('lux3', opt_sensor_data.get('lux', 0))
            # lux_achieved: average REAL SENSOR L1+L2+L3 (not model estimation)
            lux_achieved = round((lux1 + lux2 + lux3) / 3.0, 1)
            last_opt_results['pso'] = {
                'fitness': fit,
                'pwm1': pwm1_val, 'pwm2': pwm2_val,        # raw PWM value 0-255 from algorithm
                'brightness': int((b1 + b2) / 2),           # average brightness % for reference
                'brightness1': b1, 'brightness2': b2,        # brightness % 0-100 for MQTT/ESP32
                'stats': hist,
                'initial_error': round(initial_err, 2),
                'final_error': round(fit, 2),
                'lux_achieved': lux_achieved,
                'target_lux': target_lux,
                'run_time': datetime.now().strftime('%d %b %Y %H:%M'),
                'params': {
                    'swarm_size': pso_params['swarm_size'],
                    'iterations': pso_params['iterations'],
                    'w': pso_params['w'],
                    'c1': pso_params['c1'],
                    'c2': pso_params['c2'],
                } if lamp_algo == 'pso' else {
                    'population_size': ga_params['population_size'],
                    'generations': ga_params['generations'],
                    'mutation_rate': ga_params['mutation_rate'],
                    'crossover_rate': ga_params['crossover_rate'],
                    'elitism_ratio': ga_params['elitism_ratio'],
                },
            }
            print(f"[PSO] Done: PWM1={pwm1_val}/255 PWM2={pwm2_val}/255 (B1={b1}% B2={b2}%) lux_error={fit:.2f}")
            persist_opt_results('pso')
        optimization_run_count += 1
        # Update mqtt_data system
        mqtt_data['system'].update({
            'algo_config': opt_algo_config,
            'ac_algo': ac_algo, 'lamp_algo': lamp_algo,
            'ga_fitness': last_opt_results['ga']['fitness'],
            'pso_fitness': last_opt_results['pso']['fitness'],
            'optimization_runs': optimization_run_count,
            'ga_temp': last_opt_results['ga']['temp'],
            'ga_fan': last_opt_results['ga']['fan'],
            'ga_mode': last_opt_results['ga'].get('mode', 'COOL'),
            'ga_set_rh': last_opt_results['ga'].get('set_rh', 50),
            'pso_pwm1': last_opt_results['pso'].get('pwm1', 0),
            'pso_pwm2': last_opt_results['pso'].get('pwm2', 0),
            'pso_brightness': last_opt_results['pso']['brightness'],
            'pso_brightness1': last_opt_results['pso'].get('brightness1', 0),
            'pso_brightness2': last_opt_results['pso'].get('brightness2', 0),
            'ga_history': last_opt_results['ga'].get('stats', []),
            'pso_history': last_opt_results['pso'].get('stats', [])
        })
        socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
        # Write to InfluxDB
        write_to_influxdb('optimization_result', {
            'ga_fitness': float(last_opt_results['ga']['fitness']),
            'pso_fitness': float(last_opt_results['pso']['fitness']),
            'ga_temp': float(last_opt_results['ga']['temp']),
            'ga_fan': float(last_opt_results['ga']['fan']),
            'ga_mode_idx': float(last_opt_results['ga'].get('mode_idx', 0)),
            'ga_set_rh': float(last_opt_results['ga'].get('set_rh', 50)),
            'pso_pwm1': float(last_opt_results['pso'].get('pwm1', 0)),
            'pso_pwm2': float(last_opt_results['pso'].get('pwm2', 0)),
            'pso_brightness': float(last_opt_results['pso']['brightness']),
            'pso_brightness1': float(last_opt_results['pso'].get('brightness1', 0)),
            'pso_brightness2': float(last_opt_results['pso'].get('brightness2', 0)),
            'pso_error': float(last_opt_results['pso']['fitness']),   # PSO: error value (lower=better)
            # combined_fitness: GA maximize, PSO minimize — PSO fitness is already in % (0-100)
            'combined_fitness': float(last_opt_results['ga']['fitness']) * 0.5
                                + max(0.0, 100.0 - float(last_opt_results['pso']['fitness'])) * 0.5
        })
        # Auto-apply AC — only when GA produced fresh results this cycle
        global _last_adaptive_ac_apply, _last_sent_ac_temp, _last_sent_ac_fan
        if algo in ('ga', 'both') and mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE' and _person_present_recently():
            opt_temp = last_opt_results['ga']['temp']
            opt_fan  = last_opt_results['ga']['fan']
            opt_mode = last_opt_results['ga'].get('mode', 'COOL')
            opt_rh   = last_opt_results['ga'].get('set_rh', 50)
            if 16 <= opt_temp <= 30 and opt_fan >= 1:
                now = time.time()
                temp_changed = abs(int(opt_temp) - _last_sent_ac_temp) >= AC_CHANGE_THRESHOLD_TEMP
                fan_changed  = abs(int(opt_fan)  - _last_sent_ac_fan)  >= AC_CHANGE_THRESHOLD_FAN
                if (now - _last_adaptive_ac_apply >= AC_ADAPTIVE_DEBOUNCE) and (temp_changed or fan_changed):
                    _last_adaptive_ac_apply = now
                    _last_sent_ac_temp = int(opt_temp)
                    _last_sent_ac_fan  = int(opt_fan)
                    ac_cmd = {'command': 'SET', 'temperature': int(opt_temp), 'fan_speed': int(opt_fan),
                              'mode': opt_mode, 'set_rh': int(opt_rh), 'source': 'adaptive'}
                    mqtt_client.publish('smartroom/ac/control', json.dumps(ac_cmd))
                    # Update local state
                    mqtt_data['ac']['set_rh'] = int(opt_rh)
                    mqtt_data['ac']['ac_fan_mode'] = opt_mode
                    print(f"  [GA→AC] Applied: {int(opt_temp)}°C Fan={int(opt_fan)} Mode={opt_mode} RH={int(opt_rh)}%")
        # Note: Auto-apply lamp PSO is NOT done here.
        # Apply lamp PWM controlled fully by _pso_lamp_cycle()
        # agar urutan Baca→Hitung→Send→Tunggu→Baca tetap terjaga.
        # Emit status — only include solution fields for the algorithm that actually ran
        status_payload = {
            'status': 'completed', 'algorithm': algo,
            'algo_config': opt_algo_config,
            'ac_algo': ac_algo, 'lamp_algo': lamp_algo,
            'ga_fitness': last_opt_results['ga']['fitness'],
            'pso_fitness': last_opt_results['pso']['fitness'],
            'optimization_count': optimization_run_count,
            'ga_history': last_opt_results['ga'].get('stats', []),
            'pso_history': last_opt_results['pso'].get('stats', [])
        }
        if algo in ('ga', 'both'):
            status_payload['ga_solution'] = {
                'temperature': last_opt_results['ga']['temp'],
                'fan_speed': last_opt_results['ga']['fan'],
                'mode': last_opt_results['ga'].get('mode', 'COOL'),
                'mode_idx': last_opt_results['ga'].get('mode_idx', 0),
                'set_rh': last_opt_results['ga'].get('set_rh', 50)
            }
        if algo in ('pso', 'both'):
            status_payload['pso_solution'] = {'brightness': last_opt_results['pso']['brightness'], 'brightness1': last_opt_results['pso'].get('brightness1', 0), 'brightness2': last_opt_results['pso'].get('brightness2', 0), 'pwm1': last_opt_results['pso'].get('pwm1', 0), 'pwm2': last_opt_results['pso'].get('pwm2', 0)}
        socketio.emit('ml_status', status_payload)
        return True
    except Exception as e:
        print(f"[OPT] Error: {e}")
        import traceback; traceback.print_exc()
        socketio.emit('ml_status', {'status': 'error', 'message': str(e)})
        return False
    finally:
        optimization_lock.release()

# ==================== PERSIST & RESTORE OPT RESULTS ====================
OPT_PERSIST_MEASUREMENT = 'opt_results'  # InfluxDB measurement name
OPT_RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opt_results.json')

def save_opt_results_file():
    """Save last_opt_results (including stats/history arrays) to JSON file.
    This supplements InfluxDB persistence which cannot store arrays.
    Called after every GA or PSO completion."""
    try:
        ga = last_opt_results.get('ga', {})
        pso = last_opt_results.get('pso', {})
        payload = {
            'ga': {
                'fitness':         ga.get('fitness', 0),
                'temp':            ga.get('temp', 0),
                'fan':             ga.get('fan', 0),
                'mode_idx':        ga.get('mode_idx', 0),
                'mode':            ga.get('mode', 'COOL'),
                'initial_fitness': ga.get('initial_fitness', 0),
                'final_fitness':   ga.get('final_fitness', 0),
                'run_time':        ga.get('run_time', ''),
                'stats':           ga.get('stats', []),
                'params':          ga.get('params', {}),
                'sensor_snapshot': ga.get('sensor_snapshot', {}),
            },
            'pso': {
                'fitness':       pso.get('fitness', 0),
                'pwm1':          pso.get('pwm1', 0),
                'pwm2':          pso.get('pwm2', 0),
                'brightness':    pso.get('brightness', 0),
                'brightness1':   pso.get('brightness1', 0),
                'brightness2':   pso.get('brightness2', 0),
                'initial_error': pso.get('initial_error', 0),
                'final_error':   pso.get('final_error', 0),
                'lux_achieved':  pso.get('lux_achieved', 0),
                'target_lux':    pso.get('target_lux', 350.0),
                'run_time':      pso.get('run_time', ''),
                'stats':         pso.get('stats', []),
                'iteration_log': pso.get('iteration_log', []),
                'params':        pso.get('params', {}),
            },
            'saved_at': datetime.utcnow().isoformat() + 'Z',
        }
        dir_path = os.path.dirname(OPT_RESULTS_FILE)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            os.unlink(tmp_path)
            raise
        os.replace(tmp_path, OPT_RESULTS_FILE)
    except Exception as e:
        print(f"[PERSIST] save_opt_results_file failed: {e}")

def load_opt_results_file():
    """Load last_opt_results from JSON file on startup.
    Restores stats/history arrays that InfluxDB cannot store."""
    if not os.path.exists(OPT_RESULTS_FILE):
        return False
    try:
        with open(OPT_RESULTS_FILE, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        ga_data  = data.get('ga', {})
        pso_data = data.get('pso', {})
        if ga_data and ga_data.get('fitness', 0) > 0:
            last_opt_results['ga'].update(ga_data)
            mqtt_data['system']['ga_fitness']  = ga_data.get('fitness', 0)
            mqtt_data['system']['ga_temp']     = ga_data.get('temp', 0)
            mqtt_data['system']['ga_fan']      = ga_data.get('fan', 0)
            mqtt_data['system']['ga_mode']     = ga_data.get('mode', 'COOL')
            mqtt_data['system']['ga_set_rh']   = ga_data.get('set_rh', 50)
            mqtt_data['system']['ga_history']  = ga_data.get('stats', [])
            print(f"  [RESTORE-FILE] GA: {ga_data.get('temp')}°C Fan={ga_data.get('fan')} "
                  f"fitness={ga_data.get('fitness',0):.2f} stats={len(ga_data.get('stats',[]))} pts")
        if pso_data and (pso_data.get('brightness1', 0) > 0 or pso_data.get('fitness', 0) > 0):
            last_opt_results['pso'].update(pso_data)
            mqtt_data['system']['pso_fitness']     = pso_data.get('fitness', 0)
            mqtt_data['system']['pso_pwm1']        = pso_data.get('pwm1', 0)
            mqtt_data['system']['pso_pwm2']        = pso_data.get('pwm2', 0)
            mqtt_data['system']['pso_brightness']   = pso_data.get('brightness', 0)
            mqtt_data['system']['pso_brightness1']  = pso_data.get('brightness1', 0)
            mqtt_data['system']['pso_brightness2']  = pso_data.get('brightness2', 0)
            mqtt_data['system']['pso_history']      = pso_data.get('stats', [])
            print(f"  [RESTORE-FILE] PSO: PWM1={pso_data.get('pwm1')}/255 PWM2={pso_data.get('pwm2')}/255 "
                  f"stats={len(pso_data.get('stats',[]))} pts iter_log={len(pso_data.get('iteration_log',[]))} entries")
        return True
    except Exception as e:
        print(f"  [RESTORE-FILE] load_opt_results_file failed: {e}")
        return False

def persist_opt_results(algo):
    """Write current GA or PSO results to InfluxDB so they survive restarts.
    Also saves full results (including stats arrays) to JSON file."""
    try:
        if algo == 'ga':
            ga = last_opt_results['ga']
            if not ga or ga.get('fitness', 0) == 0:
                return
            fields = {
                'fitness':          float(ga['fitness']),
                'temp':             float(ga.get('temp', 0)),
                'fan':              float(ga.get('fan', 0)),
                'mode_idx':         float(ga.get('mode_idx', 0)),
                'set_rh':           float(ga.get('set_rh', 50)),
                'initial_fitness':  float(ga.get('initial_fitness', 0)),
                'final_fitness':    float(ga.get('final_fitness', ga['fitness'])),
                'run_time':         str(ga.get('run_time', '')),
                'ga_stats_json':    json.dumps(ga.get('stats', [])),
            }
            write_to_influxdb(OPT_PERSIST_MEASUREMENT, fields,
                              tags={'algo': 'ga', 'version': '1'})
        elif algo == 'pso':
            pso = last_opt_results['pso']
            if not pso or pso.get('fitness', 0) == 0 and pso.get('brightness1', 0) == 0:
                return
            fields = {
                'fitness':        float(pso['fitness']),
                'pwm1':           float(pso.get('pwm1', 0)),
                'pwm2':           float(pso.get('pwm2', 0)),
                'brightness':     float(pso.get('brightness', 0)),
                'brightness1':    float(pso.get('brightness1', 0)),
                'brightness2':    float(pso.get('brightness2', 0)),
                'initial_error':  float(pso.get('initial_error', 0)),
                'final_error':    float(pso.get('final_error', pso['fitness'])),
                'lux_achieved':   float(pso.get('lux_achieved', 0)),
                'target_lux':     float(pso.get('target_lux', 350.0)),
                'run_time':       str(pso.get('run_time', '')),
                'pso_stats_json':    json.dumps(pso.get('stats', [])),
                'pso_iterlog_json':  json.dumps(pso.get('iteration_log', [])),
            }
            write_to_influxdb(OPT_PERSIST_MEASUREMENT, fields,
                              tags={'algo': 'pso', 'version': '1'})
        # Also save full results (with stats arrays) to JSON file
        save_opt_results_file()
    except Exception as e:
        print(f"[PERSIST] Error saving {algo} results: {e}")

def restore_opt_results():
    """Read the latest GA and PSO results from InfluxDB on startup.
    NOTE: InfluxDB does not store stats/history arrays. If load_opt_results_file()
    already loaded stats from JSON, we preserve them here (don't overwrite with [])."""
    try:
        _, _, query_api = _get_influx_client()
        for algo in ('ga', 'pso'):
            q = (f'from(bucket: "{INFLUX_BUCKET}")'
                 f' |> range(start: -30d)'
                 f' |> filter(fn:(r) => r._measurement == "{OPT_PERSIST_MEASUREMENT}" and r.algo == "{algo}")'
                 f' |> last()')
            rec_map = {}
            for table in query_api.query(q):
                for rec in table.records:
                    rec_map[rec.get_field()] = rec.get_value()
            if not rec_map:
                continue
            if algo == 'ga' and rec_map.get('fitness', 0) > 0:
                # Try to restore stats from InfluxDB JSON field
                restored_stats = []
                if rec_map.get('ga_stats_json'):
                    try:
                        restored_stats = json.loads(rec_map['ga_stats_json'])
                    except Exception:
                        pass
                # Preserve stats already loaded from JSON file, or use InfluxDB stats
                existing_stats = last_opt_results['ga'].get('stats', [])
                last_opt_results['ga'].update({
                    'fitness':          float(rec_map.get('fitness', 0)),
                    'temp':             int(rec_map.get('temp', 0)),
                    'fan':              int(rec_map.get('fan', 0)),
                    'mode_idx':         int(rec_map.get('mode_idx', 0)),
                    'mode':             AC_MODE_NAMES.get(int(rec_map.get('mode_idx', 0)), 'COOL'),
                    'set_rh':           int(rec_map.get('set_rh', 50)),
                    'initial_fitness':  float(rec_map.get('initial_fitness', 0)),
                    'final_fitness':    float(rec_map.get('final_fitness', 0)),
                    'run_time':         str(rec_map.get('run_time', '')),
                })
                # Use existing stats from file, or restored from InfluxDB, or empty
                if existing_stats:
                    last_opt_results['ga']['stats'] = existing_stats
                elif restored_stats:
                    last_opt_results['ga']['stats'] = restored_stats
                else:
                    last_opt_results['ga']['stats'] = []
                mqtt_data['system']['ga_fitness']  = last_opt_results['ga']['fitness']
                mqtt_data['system']['ga_temp']     = last_opt_results['ga']['temp']
                mqtt_data['system']['ga_fan']      = last_opt_results['ga']['fan']
                mqtt_data['system']['ga_mode']     = last_opt_results['ga']['mode']
                mqtt_data['system']['ga_set_rh']   = last_opt_results['ga'].get('set_rh', 50)
                mqtt_data['system']['ga_history']  = last_opt_results['ga'].get('stats', [])
                print(f"  [RESTORE] GA: {last_opt_results['ga']['temp']}°C Fan={last_opt_results['ga']['fan']} Mode={last_opt_results['ga']['mode']} RH={last_opt_results['ga'].get('set_rh',50)}% fitness={last_opt_results['ga']['fitness']:.2f} stats={len(last_opt_results['ga']['stats'])} pts")
            elif algo == 'pso' and (rec_map.get('brightness1', 0) > 0 or rec_map.get('brightness', 0) > 0):
                b1_restored = int(rec_map.get('brightness1', 0))
                b2_restored = int(rec_map.get('brightness2', 0))
                # Try to restore stats from InfluxDB JSON fields
                restored_stats = []
                restored_iter_log = []
                if rec_map.get('pso_stats_json'):
                    try:
                        restored_stats = json.loads(rec_map['pso_stats_json'])
                    except Exception:
                        pass
                if rec_map.get('pso_iterlog_json'):
                    try:
                        restored_iter_log = json.loads(rec_map['pso_iterlog_json'])
                    except Exception:
                        pass
                existing_stats = last_opt_results['pso'].get('stats', [])
                existing_iter_log = last_opt_results['pso'].get('iteration_log', [])
                last_opt_results['pso'].update({
                    'fitness':       float(rec_map.get('fitness', 0)),
                    'pwm1':          int(rec_map.get('pwm1', round(b1_restored * 255 / 100))),
                    'pwm2':          int(rec_map.get('pwm2', round(b2_restored * 255 / 100))),
                    'brightness':    int(rec_map.get('brightness', 0)),
                    'brightness1':   b1_restored,
                    'brightness2':   b2_restored,
                    'initial_error': float(rec_map.get('initial_error', 0)),
                    'final_error':   float(rec_map.get('final_error', 0)),
                    'lux_achieved':  float(rec_map.get('lux_achieved', 0)),
                    'target_lux':    float(rec_map.get('target_lux', 350.0)),
                    'run_time':      str(rec_map.get('run_time', '')),
                })
                # Use existing stats from file, or restored from InfluxDB, or empty
                if existing_stats:
                    last_opt_results['pso']['stats'] = existing_stats
                elif restored_stats:
                    last_opt_results['pso']['stats'] = restored_stats
                else:
                    last_opt_results['pso']['stats'] = []
                if existing_iter_log:
                    last_opt_results['pso']['iteration_log'] = existing_iter_log
                elif restored_iter_log:
                    last_opt_results['pso']['iteration_log'] = restored_iter_log
                else:
                    last_opt_results['pso']['iteration_log'] = []
                mqtt_data['system']['pso_fitness']    = last_opt_results['pso']['fitness']
                mqtt_data['system']['pso_pwm1']       = last_opt_results['pso']['pwm1']
                mqtt_data['system']['pso_pwm2']       = last_opt_results['pso']['pwm2']
                mqtt_data['system']['pso_brightness']  = last_opt_results['pso']['brightness']
                mqtt_data['system']['pso_brightness1'] = last_opt_results['pso']['brightness1']
                mqtt_data['system']['pso_brightness2'] = last_opt_results['pso']['brightness2']
                mqtt_data['system']['pso_history']     = last_opt_results['pso'].get('stats', [])
                print(f"  [RESTORE] PSO: PWM1={last_opt_results['pso']['pwm1']}/255 PWM2={last_opt_results['pso']['pwm2']}/255 (B1={last_opt_results['pso']['brightness1']}% B2={last_opt_results['pso']['brightness2']}%) error={last_opt_results['pso']['fitness']:.2f} stats={len(last_opt_results['pso']['stats'])} pts")
    except Exception as e:
        print(f"  [RESTORE] Could not restore opt results: {e}")

# ==================== SENSOR FAULT DETECTION ====================
# Thresholds: how many seconds without a message = sensor is "stale"
SENSOR_STALE_WARN_S  = 120   # 2 min  → WARNING  (yellow)
SENSOR_STALE_FAULT_S = 300   # 5 min  → FAULT    (red)
# Track per-sensor last emit time so we don't spam the same fault
_fault_last_emit = {}

def sensor_fault_loop():
    """Background thread: check sensor staleness every 60 s and push alerts."""
    time.sleep(20)  # let devices connect first
    print("[FAULT] Sensor fault detection started")
    while True:
        try:
            now = datetime.now()
            checks = [
                ('esp32_ac',   'AC Sensor (ESP32)',  ['temperature', 'humidity']),
                ('esp32_lamp', 'Lamp Sensor (ESP32)', ['lux1', 'lux2', 'lux3']),
                ('camera',     'Camera (YOLO)',       []),
            ]
            health = {}
            for dev_id, dev_label, fields in checks:
                last = device_last_seen[dev_id]['last_seen']
                if last is None:
                    age = float('inf')
                else:
                    age = (now - last).total_seconds()

                if age == float('inf') or age > SENSOR_STALE_FAULT_S:
                    lvl = 'fault'
                elif age > SENSOR_STALE_WARN_S:
                    lvl = 'warn'
                else:
                    lvl = 'ok'

                age_str = 'never' if age == float('inf') else f"{int(age)}s ago"
                health[dev_id] = {'label': dev_label, 'status': lvl, 'age': age_str}

                # Only emit alert once per transition to avoid spam
                prev = _fault_last_emit.get(dev_id, 'ok')
                if lvl != 'ok' and prev == 'ok':
                    msg = f'[SENSOR FAULT] {dev_label} no data ({age_str})'
                    level = 'danger' if lvl == 'fault' else 'warning'
                    socketio.emit('sensor_fault', {'device': dev_id, 'label': dev_label, 'status': lvl, 'age': age_str})
                    socketio.emit('alert', {'type': 'sensor_fault', 'level': level, 'message': msg, 'time': now.strftime('%H:%M:%S')})
                    log_messages.append({'time': now.strftime('%H:%M:%S'), 'msg': msg, 'level': level})
                    print(msg)
                elif lvl == 'ok' and prev != 'ok':
                    # Recovery
                    msg = f'[SENSOR OK] {dev_label} back online'
                    socketio.emit('sensor_fault', {'device': dev_id, 'label': dev_label, 'status': 'ok', 'age': age_str})
                    socketio.emit('alert', {'type': 'sensor_recovered', 'level': 'success', 'message': msg, 'time': now.strftime('%H:%M:%S')})
                    log_messages.append({'time': now.strftime('%H:%M:%S'), 'msg': msg, 'level': 'success'})
                    print(msg)
                _fault_last_emit[dev_id] = lvl

            # Push health snapshot to frontend every cycle
            socketio.emit('sensor_health', health)
        except Exception as e:
            print(f"[FAULT] sensor_fault_loop error: {e}")
        time.sleep(60)

def _pso_lamp_cycle():
    """One full lamp PSO cycle with sequence:
        1. Baca lux sekarang
        2. PSO hitung PWM baru
        3. Send PWM ke lampu
        4. Wait 5 minutes (timer managed by optimization_auto_loop)
        5. Baca lux baru  → evaluasi → ulang dari langkah 2

    Return: True jika siklus berjalan normal, False jika dilewati.
    """
    global _pso_locked, _pso_lock_pwm

    # ── Step 1: Baca lux sekarang ─────────────────────────────────────────
    # Force refresh opt_sensor_data from MQTT/InfluxDB before reading lux value
    fetch_sensor_data_from_db()
    lux1 = float(opt_sensor_data.get('lux1', opt_sensor_data.get('lux', 0)))
    lux2 = float(opt_sensor_data.get('lux2', opt_sensor_data.get('lux', 0)))
    lux3 = float(opt_sensor_data.get('lux3', opt_sensor_data.get('lux', 0)))
    lux_avg = (lux1 + lux2 + lux3) / 3.0
    person_now = opt_sensor_data.get('person_detected', False) or _person_present_recently_lamp()
    mode_now = mqtt_data['lamp'].get('mode', 'MANUAL')

    print(f"[PSO] New cycle — Lux avg={lux_avg:.1f} | Person={'yes' if person_now else 'no'} | Mode={mode_now}")

    # Hanya jalan di mode ADAPTIVE
    if mode_now != 'ADAPTIVE':
        print(f"[PSO] Mode {mode_now} — siklus dilewati")
        return False

    # PSO only runs if there are persons in the room
    # If no persons: turn off lamps and stop
    if not person_now:
        b1_now = mqtt_data['lamp'].get('brightness1', 0)
        b2_now = mqtt_data['lamp'].get('brightness2', 0)
        if b1_now > 0 or b2_now > 0:
            mqtt_client.publish('smartroom/lamp/control',
                json.dumps({'brightness1': 0, 'brightness2': 0, 'source': 'adaptive_no_person'}))
            mqtt_data['lamp']['brightness1'] = 0
            mqtt_data['lamp']['brightness2'] = 0
            mqtt_data['lamp']['brightness_avg'] = 0
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
            _record_lamp_apply(0, 0)
            print(f"[PSO] No persons — lamps turned off, PSO stopped")
        else:
            print(f"[PSO] No persons — lamps already off, PSO stopped")
        return False

    # ── Evaluate current lux against target ─────────────────────────────────
    # Here person_now is definitely True (already checked above)
    # Converged if: lux avg 315-385 AND all sensors >= 200 lux
    lux_in_range = (315.0 <= lux_avg <= 385.0
                    and lux1 >= 200.0 and lux2 >= 200.0 and lux3 >= 200.0)

    if lux_in_range:
        if not _pso_locked:
            _pso_lock_pwm[0] = mqtt_data['lamp'].get('brightness1', 0)
            _pso_lock_pwm[1] = mqtt_data['lamp'].get('brightness2', 0)
            _pso_locked = True
        print(f"[PSO] Lux {lux_avg:.1f} dalam toleransi — brightness dikunci "
              f"(B1={_pso_lock_pwm[0]}% B2={_pso_lock_pwm[1]}%), no PWM change")
        return False  # no need to proceed to steps 2-3

    # Lux di luar target — buka kunci
    _pso_locked = False
    _pso_lock_pwm[0] = None
    _pso_lock_pwm[1] = None
    print(f"[PSO] Lux {lux_avg:.1f} outside 315-385 — proceeding to calculate PWM")

    # ── Step 2: PSO — calculate PWM, send per iteration, stop if converged ──
    g_pos, g_fit, fitness_history, initial_err, iteration_log = run_pso_optimization()
    pwm1_val = int(g_pos[0])
    pwm2_val = int(g_pos[1])
    b1 = round(pwm1_val * 100.0 / 255.0, 1)
    b2 = round(pwm2_val * 100.0 / 255.0, 1)
    b_avg = round((b1 + b2) / 2.0, 1)
    target_lux = 350.0 if person_now else 0.0

    # Save PSO results including iteration_log for dashboard chart
    last_opt_results['pso'].update({
        'brightness': b_avg, 'brightness1': b1, 'brightness2': b2,
        'pwm1': pwm1_val, 'pwm2': pwm2_val,
        'fitness': round(g_fit, 4),
        'stats': fitness_history[-20:] if len(fitness_history) >= 20 else fitness_history,
        'iteration_log': iteration_log,
        'initial_error': round(initial_err, 2),
        'final_error': round(g_fit, 2),
        'lux_achieved': lux_avg,
        'target_lux': target_lux,
        'run_time': datetime.now().strftime('%d %b %Y %H:%M'),
        'params': {
            'swarm_size': pso_params['swarm_size'], 'iterations': pso_params['iterations'],
            'w': pso_params['w'], 'c1': pso_params['c1'], 'c2': pso_params['c2'],
        },
    })

    # ── Step 3: Send final Gbest to lamp ────────────────────────────────
    # PSO already sends Gbest in each iteration. This step ensures
    # lamp is at the best Gbest position after all iterations complete.
    opt_b1, opt_b2 = _safe_lamp_brightness(b1, b2)
    mqtt_client.publish(
        'smartroom/lamp/control',
        json.dumps({'brightness1': opt_b1, 'brightness2': opt_b2, 'source': 'adaptive_final'})
    )
    # Sync mqtt_data['lamp'] directly — don't wait for MQTT back from ESP32
    # agar Lamp Dashboard brightness terupdate segera
    mqtt_data['lamp']['brightness1']    = opt_b1
    mqtt_data['lamp']['brightness2']    = opt_b2
    mqtt_data['lamp']['brightness_avg'] = round((opt_b1 + opt_b2) / 2.0, 1)
    _record_lamp_apply(opt_b1, opt_b2)
    # Emit to all clients so Lamp Dashboard and ML Optimization sync instantly
    socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
    print(f"[PSO] Gbest final dikirim — B1={opt_b1}% B2={opt_b2}% | fitness={g_fit:.2f}")

    # Fitness dalam persen: 100% = lux tepat 350, turun seiring error membesar
    # Skala: error=0 → 100%, error=350^2=122500 → 0%
    fitness_pct = round(max(0.0, 100.0 - (g_fit / 122500.0) * 100.0), 1)

    # Update dashboard & InfluxDB
    persist_opt_results('pso')
    mqtt_data['system'].update({
        'pso_fitness': fitness_pct,
        'pso_pwm1': pwm1_val, 'pso_pwm2': pwm2_val,
        'pso_brightness': b_avg, 'pso_brightness1': b1, 'pso_brightness2': b2,
        'pso_history': fitness_history,
    })
    socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
    socketio.emit('ml_status', {
        'status': 'completed', 'algorithm': 'pso',
        'pso_fitness': fitness_pct,
        'pso_history': fitness_history,
        'pso_iteration_log': iteration_log,
        'pso_solution': {
            'brightness': b_avg, 'brightness1': b1, 'brightness2': b2,
            'pwm1': pwm1_val, 'pwm2': pwm2_val,
        }
    })
    write_to_influxdb('optimization_result', {
        'pso_fitness': float(g_fit),
        'pso_pwm1': float(pwm1_val), 'pso_pwm2': float(pwm2_val),
        'pso_brightness': float(b_avg),
        'pso_brightness1': float(b1), 'pso_brightness2': float(b2),
        'pso_error': float(g_fit),
    })

    # ── Step 4 & 5: Wait 5 minutes then read new lux ────────────────────
    # The 5 minute wait is controlled by optimization_auto_loop via last_pso_time.
    # New lux reading automatically occurs when next cycle is called (Step 1).
    print(f"[PSO] Done — waiting 5 minutes, new lux will be read in next cycle")

    # Cek apakah PSO berhasil konvergen (lux akhir dalam toleransi)
    lux1_f = float(opt_sensor_data.get('lux1', 0))
    lux2_f = float(opt_sensor_data.get('lux2', 0))
    lux3_f = float(opt_sensor_data.get('lux3', 0))
    lux_final = (lux1_f + lux2_f + lux3_f) / 3.0
    # Converged if: persons present + lux avg 315-385 + all sensors >= 200
    converged = (315.0 <= lux_final <= 385.0
                 and lux1_f >= 200.0 and lux2_f >= 200.0 and lux3_f >= 200.0)
    print(f"[PSO] Lux akhir={lux_final:.1f} (L1={lux1_f} L2={lux2_f} L3={lux3_f}) "
          f"— {'CONVERGED' if converged else 'NOT converged, will retry'}")
    return converged


def optimization_auto_loop():
    """Background thread: GA (AC) dan PSO (Lamp) berjalan di siklus terpisah.
    PSO lamp:
      - If lux is within 315-385: wait 5 minutes then check again
      - If lux not at target: immediately retry PSO without waiting 5 minutes
    GA and PSO never run simultaneously."""
    time.sleep(10)
    print(f"[OPT] Auto-optimization started (AC every {AUTO_OPT_INTERVAL_AC}s, Lamp every {AUTO_OPT_INTERVAL_LAMP}s)")
    last_ga_time  = 0
    last_pso_time = 0
    while True:
        try:
            now     = time.time()
            run_ga  = (now - last_ga_time)  >= AUTO_OPT_INTERVAL_AC
            run_pso = (now - last_pso_time) >= AUTO_OPT_INTERVAL_LAMP

            if run_ga:
                run_optimization_cycle('ga')
                last_ga_time = time.time()
            elif run_pso:
                # Signal dashboard that a new PSO cycle has started
                socketio.emit('pso_iter_progress', {'status': 'new_cycle'})
                converged = _pso_lamp_cycle()
                now_after = time.time()
                if converged:
                    # Lux within target — wait normal 5 minutes
                    last_pso_time = now_after
                    print(f"[PSO] Konvergen — tunggu {AUTO_OPT_INTERVAL_LAMP}s")
                else:
                    # Lux not at target (or mode is not ADAPTIVE) —
                    # check if it's because already at target (_pso_locked)
                    # or because not converged
                    if _pso_locked:
                        # Already at target, lock active — wait normal
                        last_pso_time = now_after
                        print(f"[PSO] Lux dikunci dalam target — tunggu {AUTO_OPT_INTERVAL_LAMP}s")
                    else:
                        # Not converged — immediately retry PSO without waiting 5 minutes
                        # Give a short 10 second delay for sensor to stabilize
                        print(f"[PSO] Not converged — retrying PSO in 10 seconds")
                        last_pso_time = now_after - AUTO_OPT_INTERVAL_LAMP + 10

        except Exception as e:
            print(f"[OPT] Auto cycle error: {e}")
        time.sleep(15)  # check timer more often (15 seconds) for fast re-run

# InfluxDB Configuration
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "rfi_HvWdjwaG8jB3Rqx6g0y5kMWRfSfq_HmLLUvkom1yaHKvwonU9Qfj6nlZjTqb_I0leIREUnMhvQQXtgETfg=="
INFLUX_ORG = "IOTLAB"
INFLUX_BUCKET = "SENSORDATA"

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Camera Configuration
camera = None
camera_lock = threading.Lock()
camera_enabled = True
_actual_fps = 0  # Real measured FPS from detection loop

# YOLO Configuration
yolo_model = None
yolo_lock = threading.Lock()

# Smart Person-Based Auto ON/OFF
_person_consecutive_frames = 0
_no_person_start_time = None
PERSON_CONFIRM_FRAMES = 3        # 3 frames with person -> auto ON (~2 seconds)
PERSON_RECONFIRM_FRAMES = 15     # 15 frames (~10s) required to re-enable after auto-OFF
NO_PERSON_TIMEOUT_SECONDS = 300  # 5 min no person -> AC auto OFF
NO_PERSON_LAMP_TIMEOUT = 300     # 5 min no person -> Lamp auto OFF
AUTO_OFF_COOLDOWN = 120          # 2 min cooldown: block auto-ON after auto-OFF
_last_person_confirmed_time = 0.0  # Unix time when person last confirmed (0 = never since startup)
_auto_off_triggered = False        # Prevents repeated POWER_OFF from firing
_auto_off_time = 0.0               # Unix time when auto-OFF last fired (for cooldown)
_lamp_auto_off_triggered = False   # Prevents repeated lamp OFF
_no_person_lamp_start_time = None  # When no person started (for lamp timer)

# Background detection thread state
_latest_frame_bytes = None
_latest_frame_lock = threading.Lock()
_detection_thread_running = False

# Lamp apply: no EMA gain calibration (according to specification doc)
_pending_calibration = None  # not used, kept for reference compatibility

# ── Filter BH1750: validasi + EMA smoothing ───────────────────────────────────
# BH1750 sometimes produces high spikes (hundreds/thousands lux) in dark conditions
# due to I2C noise or reading before conversion completes. This filter prevents
# anomaly values from entering PSO and InfluxDB.
LUX_MAX_PHYSICAL   = 65535.0  # batas hardware BH1750 (16-bit)
LUX_SPIKE_RATIO    = 5.0      # bacaan > 5x EMA sebelumnya dianggap spike
LUX_DARK_THRESHOLD = 10.0     # if brightness1=brightness2=0, lux should be < this
LUX_EMA_ALPHA      = 0.3      # bobot data baru dalam smoothing (0=ignore, 1=raw)
# EMA state per sensor (None = not initialized, use raw for first time)
_lux_ema = [None, None, None]  # [L1_ema, L2_ema, L3_ema]

# ── PWM lock: don't change brightness if lux already at target ────────────────
_pso_locked = False   # True = lux in range, PWM locked, PSO does not apply
_pso_lock_pwm = [None, None]  # [b1%, b2%] locked

def _filter_lux(raw_val, sensor_idx, brightness1=0, brightness2=0):
    """Validate BH1750 reading received from ESP32.

    ESP32 sudah menjalankan: raw -> kalibrasi -> median filter (5 sampel).
    Raspi DOES NOT apply EMA again to prevent double-smoothing that
    menyebabkan nilai dashboard berbeda jauh dari Serial Monitor ESP32.

    Raspi hanya melakukan:
    1. Physical bounds: negative value or > 65535 -> use last valid value.
    2. Ghost lux: lampu OFF tapi sensor baca tinggi -> kembalikan 0.
       - Grace period 8 seconds after PSO sends brightness=0.
    3. Spike detection: nilai > LUX_SPIKE_RATIO x referensi -> buang.
       Referensi = nilai valid terakhir (bukan EMA).
    """
    global _lux_ema  # used as "last valid value", not EMA
    raw = float(raw_val)

    # 1. Batas fisik BH1750 (16-bit)
    if raw < 0 or raw > LUX_MAX_PHYSICAL:
        return round(_lux_ema[sensor_idx], 1) if _lux_ema[sensor_idx] is not None else 0.0

    lamp_is_off = (brightness1 <= 0 and brightness2 <= 0)

    # 2. Ghost lux when lamp is off
    # IMPORTANT: lamp_is_off DOES NOT mean room is dark — there could be natural light (sunlight, etc.).
    # We only suspect ghost lux if:
    # - Value is MUCH higher than previous reference (large spike when lamp just turned off)
    # - AND still within grace period (lamp just turned off)
    if lamp_is_off:
        time_since_apply = time.time() - _last_adaptive_lamp_apply
        in_grace = time_since_apply < 8.0

        if in_grace and raw > LUX_DARK_THRESHOLD:
            # In grace period: sensor might still read residual lamp light
            # Keep old reference until lamp is fully dark
            if _lux_ema[sensor_idx] is not None:
                return round(_lux_ema[sensor_idx], 1)
            return round(raw, 1)
        # After grace period: accept value as is — there might be natural light
        # Do not reset to 0 automatically, let spike detection handle it

    # 3. Spike detection: discard if far exceeds last reference
    if _lux_ema[sensor_idx] is not None and _lux_ema[sensor_idx] > 5.0:
        if raw > _lux_ema[sensor_idx] * LUX_SPIKE_RATIO:
            print(f"[LUX_FILTER] L{sensor_idx+1}: spike {raw:.1f} lx "
                  f"(ref={_lux_ema[sensor_idx]:.1f}) -> dibuang")
            return round(_lux_ema[sensor_idx], 1)

    # Passed all checks — save as last reference and return directly
    _lux_ema[sensor_idx] = raw
    return round(raw, 1)

# Adaptive apply debounce: prevent duplicate AC/Lamp commands within 5 seconds
_last_adaptive_ac_apply = 0
_last_adaptive_lamp_apply = 0
_last_sent_ac_temp = 0
_last_sent_ac_fan  = 0
_last_sent_lamp_b1 = -1
_last_sent_lamp_b2 = -1
AC_ADAPTIVE_DEBOUNCE = 600      # 10 minutes between adaptive AC SET commands
AC_CHANGE_THRESHOLD_TEMP = 1    # skip AC SET if temp recommendation unchanged by < 1 deg vs last sent
AC_CHANGE_THRESHOLD_FAN = 1     # skip AC SET if fan speed unchanged vs last sent
LAMP_ADAPTIVE_DEBOUNCE = 300    # 5 minutes — sesuai interval PSO dokumen

# Global data storage
mqtt_data = {
    'ac': {'temperature': 0, 'humidity': 0, 'heat_index': 0, 'ac_state': 'OFF', 'ac_temp': 24, 'fan_speed': 1, 'set_rh': 50, 'mode': 'ADAPTIVE', 'ac_fan_mode': 'COOL', 'rssi': 0, 'uptime': 0, 'temp1': 0, 'hum1': 0, 'temp2': 0, 'hum2': 0, 'temp3': 0, 'hum3': 0},
    'lamp': {'lux1': 0, 'lux2': 0, 'lux3': 0, 'lux_avg': 0, 'motion': False, 'brightness1': 0, 'brightness2': 0, 'brightness_avg': 0, 'mode': 'ADAPTIVE', 'rssi': 0, 'uptime': 0},
    'camera': {'person_detected': False, 'count': 0, 'confidence': 0, 'status': 'inactive'},
    'energy': {'voltage': 0, 'current': 0, 'power': 0, 'energy': 0, 'frequency': 0, 'pf': 0, 'connected': False, 'ac_state': 'OFF'},
    'system': {'ga_fitness': 0, 'pso_fitness': 0, 'optimization_runs': 0, 'ga_temp': 0, 'ga_fan': 0, 'ga_mode': 'COOL', 'pso_pwm1': 0, 'pso_pwm2': 0, 'pso_brightness': 0, 'pso_brightness1': 0, 'pso_brightness2': 0, 'ga_history': [], 'pso_history': []},
    'ir_codes': {},
    'ir_states': {},  # Track toggle states for power buttons
    'outlet': {'1': 'OFF', '2': 'OFF', '3': 'OFF', '4': 'OFF'}
}

log_messages = deque(maxlen=100)
ir_learning_mode = False
ir_learning_button = ""
ir_learning_device = ""  # Track device name

# MQTT connection status tracking
mqtt_status = {
    'connected': False,
    'last_connect_time': None,
    'last_message_time': None,
    'message_count': 0,
    'error': None,
    'broker': 'localhost:1883'
}

# Runtime energy history fallback (used when Influx query returns no points)
energy_runtime_history = deque(maxlen=5000)
# Runtime outlet energy history fallback (MySQL id_kwh=2)
outlet_runtime_history = deque(maxlen=5000)
# Runtime lamp energy history fallback (similar to AC — starts filling immediately on MQTT data)
lamp_runtime_history = deque(maxlen=5000)

# InfluxDB write throttle — save every 6 minutes so database does not fill up
INFLUX_WRITE_INTERVAL = 360  # seconds (6 minutes)
_last_sensor_influx_ts   = 0.0   # when last ac_sensor written to InfluxDB
_last_lamp_influx_ts     = 0.0   # when last lamp_sensor written to InfluxDB
_last_lamp_mqtt_ts       = 0.0   # kapan terakhir pesan MQTT lamp/sensors diterima (termasuk lux=0)
_last_ac_energy_influx_ts = 0.0  # when last AC energy written to InfluxDB

# ==================== INFLUXDB SINGLETON ====================
# One persistent client + write_api shared across all threads.
# Query-API is stateless and thread-safe; write_api (SYNCHRONOUS) is also safe.
# A RLock guards reinitialisation so two threads cannot race on _influx_write_api.
_influx_lock    = threading.RLock()
_influx_client  = None
_influx_write_api = None

def _get_influx_client():
    """Return (client, write_api, query_api) — creates or recreates if closed/None."""
    global _influx_client, _influx_write_api
    with _influx_lock:
        if _influx_client is None:
            _influx_client   = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            _influx_write_api = _influx_client.write_api(write_options=SYNCHRONOUS)
        return _influx_client, _influx_write_api, _influx_client.query_api()

def _reset_influx_client():
    """Force-close and nullify so the next call recreates it (used after unrecoverable errors)."""
    global _influx_client, _influx_write_api
    with _influx_lock:
        try:
            if _influx_write_api:
                _influx_write_api.close()
        except Exception:
            pass
        try:
            if _influx_client:
                _influx_client.close()
        except Exception:
            pass
        _influx_client    = None
        _influx_write_api = None

# Eager initialisation — fail fast at startup if InfluxDB is unreachable
try:
    _get_influx_client()
    print("[OK] InfluxDB singleton initialised")
except Exception as _e:
    print(f"[WARN] InfluxDB not reachable at startup: {_e} — will retry on first use")

# ==================== INFLUXDB WRITE / QUERY HELPERS ====================
def write_to_influxdb(measurement, fields, tags=None):
    """Write a single data point using the shared singleton client.
    Retries once after resetting the client on connection-level errors."""
    point = Point(measurement).time(datetime.utcnow(), WritePrecision.NS)
    if tags:
        for key, value in tags.items():
            point = point.tag(key, str(value))
    for key, value in fields.items():
        if isinstance(value, bool):
            point = point.field(key, value)
        elif isinstance(value, (int, float)):
            point = point.field(key, float(value))
        else:
            point = point.field(key, str(value))

    for attempt in range(2):
        try:
            _, write_api, _ = _get_influx_client()
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
            return True
        except Exception as e:
            print(f"[ERROR] InfluxDB write ({measurement}) attempt {attempt+1}: {e}")
            if attempt == 0:
                _reset_influx_client()   # recreate connection, then retry once
    return False

def save_sensor_data(temperature, humidity, heat_index,
                     temp1=0, hum1=0, temp2=0, hum2=0, temp3=0, hum3=0):
    global _last_sensor_influx_ts
    now = time.time()
    if now - _last_sensor_influx_ts < INFLUX_WRITE_INTERVAL:
        return  # throttle: not yet 6 minutes
    _last_sensor_influx_ts = now
    try:
        write_to_influxdb('ac_sensor', {
            'temperature': float(temperature), 'humidity': float(humidity), 'heat_index': float(heat_index),
            'temp1': float(temp1), 'hum1': float(hum1),
            'temp2': float(temp2), 'hum2': float(hum2),
            'temp3': float(temp3), 'hum3': float(hum3),
            'ac_temp': float(mqtt_data['ac'].get('ac_temp', 0)),
            'fan_speed': int(mqtt_data['ac'].get('fan_speed', 0)),
            'ac_fan_mode': str(mqtt_data['ac'].get('ac_fan_mode', '')),
            'mode': str(mqtt_data['ac'].get('mode', '')),
            'set_rh': int(mqtt_data['ac'].get('set_rh', 50)),
            'ac_state': str(mqtt_data['ac'].get('ac_state', 'OFF')),
        }, tags={'device': 'esp32_ac', 'location': 'room'})
    except Exception as e:
        print(f'[ERROR] save_sensor_data: {e}')

def save_lamp_data(lux1, lux2, lux3, brightness1, brightness2, motion):
    global _lamp_energy_kwh, _lamp_energy_last_ts, _last_lamp_influx_ts
    now_ts_throttle = time.time()
    do_influx_write = (now_ts_throttle - _last_lamp_influx_ts >= INFLUX_WRITE_INTERVAL)
    if do_influx_write:
        _last_lamp_influx_ts = now_ts_throttle
    try:
        lux_avg = (lux1 + lux2 + lux3) / 3.0
        bright_avg = (brightness1 + brightness2) / 2.0
        if do_influx_write:
            write_to_influxdb('lamp_sensor', {
                'lux1': float(lux1), 'lux2': float(lux2), 'lux3': float(lux3), 'lux_avg': float(lux_avg),
                'brightness1': float(brightness1), 'brightness2': float(brightness2), 'brightness_avg': float(bright_avg),
                'motion': bool(motion),
                'mode': str(mqtt_data['lamp'].get('mode', 'ADAPTIVE'))
            }, tags={'device': 'esp32_lamp', 'location': 'room'})
        # Estimate lamp energy from brightness (no PZEM on ESP32 Lamp)
        now_ts = time.time()
        lamp_power = round((bright_avg / 100.0) * LAMP_RATED_WATT, 2)
        lamp_current = round(lamp_power / LAMP_VOLTAGE, 3) if lamp_power > 0 else 0.0
        with _recording_lock:
            if _lamp_energy_last_ts > 0:
                dt_h = (now_ts - _lamp_energy_last_ts) / 3600.0
                _lamp_energy_kwh += lamp_power * dt_h / 1000.0
            _lamp_energy_last_ts = now_ts
            kwh_snapshot  = round(_lamp_energy_kwh, 4)
            phase_snapshot = lamp_phase
        if do_influx_write:
            write_to_influxdb('energy_monitor', {
                'power': lamp_power,
                'current': lamp_current,
                'voltage': LAMP_VOLTAGE if lamp_power > 0 else 0.0,
                'energy_kwh': kwh_snapshot
            }, tags={'device': 'esp32_lamp', 'phase': phase_snapshot})
        # Fill runtime buffer for immediate chart display (fallback when InfluxDB has no data yet)
        lamp_runtime_history.append({
            'ts': datetime.now(),
            'phase': phase_snapshot,
            'power': lamp_power,
            'current': lamp_current,
            'voltage': LAMP_VOLTAGE if lamp_power > 0 else 0.0,
            'energy_kwh': kwh_snapshot
        })
    except Exception as e:
        print(f'[ERROR] save_lamp_data: {e}')

def save_person_detection(person_count, confidence):
    try:
        write_to_influxdb('camera_detection', {
            'person_count': int(person_count), 'confidence': float(confidence), 'person_detected': bool(person_count > 0)
        }, tags={'device': 'camera_yolo', 'model': 'yolov8n'})
    except Exception as e:
        print(f'[ERROR] save_person_detection: {e}')

def save_ir_command(device, command, signal_length):
    try:
        write_to_influxdb('ir_remote', {
            'command': str(command), 'signal_length': int(signal_length), 'learned': True
        }, tags={'device': str(device), 'type': 'ir_code'})
    except Exception as e:
        print(f'[ERROR] save_ir_command: {e}')

def save_ac_control(ac_temp, fan_speed, ac_state):
    try:
        write_to_influxdb('ac_sensor', {
            'ac_temp': float(ac_temp), 'fan_speed': int(fan_speed), 'ac_state': str(ac_state)
        }, tags={'device': 'esp32_ac', 'type': 'control'})
    except Exception as e:
        print(f'[ERROR] save_ac_control: {e}')

# ==================== YOLO INITIALIZATION ====================
def load_yolo_model():
    global yolo_model
    try:
        yolo_dir = '/home/iotlab/smartroom/yolo'
        model_path = f'{yolo_dir}/yolov8n.pt'
        
        # Try local model first, fallback to auto-download
        if os.path.exists(model_path):
            print(f"[YOLO] Loading YOLOv8n from {model_path}...")
            yolo_model = YOLO(model_path)
        else:
            print("[YOLO] YOLOv8n not found locally, downloading...")
            os.makedirs(yolo_dir, exist_ok=True)
            yolo_model = YOLO('yolov8n.pt')
        
        # Warm up with dummy inference
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        yolo_model.predict(dummy, verbose=False)
        
        print("[OK] YOLOv8n model loaded successfully!")
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 
                           'msg': 'YOLOv8n loaded successfully', 'level': 'success'})
        return True
        
    except Exception as e:
        print(f"[ERROR] YOLO loading error: {str(e)}")
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 
                           'msg': f'YOLO error: {str(e)}', 'level': 'error'})
        return False

def detect_persons(frame):
    """Detect persons using YOLOv8 — returns frame, count, confidence, and bounding boxes list"""
    global yolo_model
    
    if yolo_model is None:
        return frame, 0, 0.0, []
    
    try:
        with yolo_lock:
            # YOLOv8 inference — classes=[0] filters for 'person' only
            results = yolo_model.predict(frame, conf=0.35, classes=[0], verbose=False, imgsz=640)
            
            person_count = 0
            max_confidence = 0.0
            boxes_list = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"Person {confidence*100:.0f}%"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    boxes_list.append((x1, y1, x2, y2, confidence))
                    person_count += 1
                    if confidence > max_confidence:
                        max_confidence = confidence
            
            return frame, person_count, max_confidence, boxes_list
            
    except Exception as e:
        print(f"[ERROR] YOLO detection error: {str(e)}")
        return frame, 0, 0.0, []

def _person_present_recently():
    """True if person confirmed within 5 min (NO_PERSON_TIMEOUT_SECONDS) — used for AC."""
    return _last_person_confirmed_time > 0 and \
           (time.time() - _last_person_confirmed_time) < NO_PERSON_TIMEOUT_SECONDS

def _person_present_recently_lamp():
    """True if person confirmed within 20 min (NO_PERSON_LAMP_TIMEOUT) — used for Lamp.
    Checks ALL sources: local YOLO thread, MQTT camera/detection, and opt_sensor_data."""
    # 1. Time-based confirmation from local YOLO camera thread
    time_confirmed = (_last_person_confirmed_time > 0 and
                      (time.time() - _last_person_confirmed_time) < NO_PERSON_LAMP_TIMEOUT)
    # 2. Current live state from ANY camera source (MQTT or local)
    currently_detected = (mqtt_data['camera'].get('person_detected', False) or
                          opt_sensor_data.get('person_detected', False))
    return time_confirmed or currently_detected

def _safe_lamp_brightness(b1, b2):
    """Ensure brightness value is not negative."""
    return max(0, int(b1)), max(0, int(b2))

def _should_apply_lamp(b1, b2):
    """Return True if 5-minute debounce has passed (per document interval)."""
    global _last_adaptive_lamp_apply
    now = time.time()
    if (now - _last_adaptive_lamp_apply) < LAMP_ADAPTIVE_DEBOUNCE:
        return False
    return True

def _record_lamp_apply(b1, b2):
    """Record last time brightness was applied to lamp.
    Jika brightness=0 (lampu dimatikan), schedule EMA reset setelah grace period
    agar filter langsung siap menerima lux=0 tanpa drift lambat.
    """
    global _last_sent_lamp_b1, _last_sent_lamp_b2, _last_adaptive_lamp_apply
    _last_sent_lamp_b1 = int(b1)
    _last_sent_lamp_b2 = int(b2)
    _last_adaptive_lamp_apply = time.time()
    # If lamp is turned off, reset EMA now to low value so
    # after 8 second grace period, ghost lux detected with exact threshold.
    if int(b1) == 0 and int(b2) == 0:
        for i in range(3):
            if _lux_ema[i] is not None and _lux_ema[i] > LUX_DARK_THRESHOLD:
                # Set to threshold value, not 0, so spike detection still works
                # in case real natural light enters the room
                _lux_ema[i] = LUX_DARK_THRESHOLD

# ==================== SMART PERSON-BASED CONTROL ====================
def handle_person_based_control(person_count):
    """Smart auto ON/OFF: turn ON AC when person confirmed, OFF after 5 min empty, Lamp OFF after 15 min empty"""
    global _person_consecutive_frames, _no_person_start_time, _auto_off_triggered, _last_person_confirmed_time, _auto_off_time
    global _lamp_auto_off_triggered, _no_person_lamp_start_time
    
    ac_adaptive = mqtt_data['ac'].get('mode') == 'ADAPTIVE'
    lamp_adaptive = mqtt_data['lamp'].get('mode') == 'ADAPTIVE'
    
    # Only act if at least one is ADAPTIVE
    if not ac_adaptive and not lamp_adaptive:
        _person_consecutive_frames = 0
        _no_person_start_time = None
        _no_person_lamp_start_time = None
        _auto_off_triggered = False
        _lamp_auto_off_triggered = False
        return
    
    # How many consecutive frames needed to confirm person?
    # After auto-OFF -> require 15 frames (~10s) to prevent false positives from turning AC on
    # Normal mode -> require 3 frames (~2s)
    in_cooldown = _auto_off_time > 0 and (time.time() - _auto_off_time) < AUTO_OFF_COOLDOWN
    required_frames = PERSON_RECONFIRM_FRAMES if _auto_off_triggered else PERSON_CONFIRM_FRAMES
    
    if person_count > 0:
        _person_consecutive_frames += 1
        
        # Only reset auto-off state when person is CONFIRMED (required frames)
        # AND cooldown has elapsed
        if _person_consecutive_frames >= required_frames and not in_cooldown:
            _last_person_confirmed_time = time.time()  # Record time person was confirmed
            _no_person_start_time = None
            _no_person_lamp_start_time = None
            _auto_off_triggered = False  # Allow auto-OFF to fire again if person leaves later
            # When lamp was auto-off and person returns: reset so lamp turns on instantly
            if _lamp_auto_off_triggered:
                _lamp_auto_off_triggered = False
                global _last_sent_lamp_b1, _last_sent_lamp_b2, _last_adaptive_lamp_apply
                _last_sent_lamp_b1 = -1
                _last_sent_lamp_b2 = -1
                _last_adaptive_lamp_apply = 0
            _auto_off_time = 0.0
        elif _person_consecutive_frames >= required_frames and in_cooldown:
            cooldown_left = AUTO_OFF_COOLDOWN - (time.time() - _auto_off_time)
        
        # Auto ON: person confirmed AND cooldown elapsed
        if _person_consecutive_frames >= required_frames and not in_cooldown and not _auto_off_triggered:
            if mqtt_data['ac'].get('ac_state') == 'OFF':
                try:
                    mqtt_client.publish("smartroom/ac/control", json.dumps({
                        "action": "POWER_ON",
                        "source": "camera_auto"
                    }))
                    mqtt_data['ac']['ac_state'] = 'ON'
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'),
                                       'msg': 'Auto ON: Person detected', 'level': 'success'})
                    socketio.emit('alert', {
                        'type': 'auto_on', 'level': 'success',
                        'message': 'Person detected — AC turned ON automatically',
                        'time': datetime.now().strftime('%H:%M:%S')
                    })
                except Exception as e:
                    print(f"[ERROR] Auto ON error: {e}")
            # Lamp auto ON: fires only when lamp was previously OFF (brightness=0)
            # CRITICAL: update mqtt_data['lamp'] locally right after publish so this block
            # does NOT fire again on the next YOLO frame while waiting for ESP32 confirmation
            if lamp_adaptive and (mqtt_data['lamp'].get('brightness1', 0) == 0 and mqtt_data['lamp'].get('brightness2', 0) == 0):
                # Use PSO result if available; fall back to 60% if PSO hasn't run yet
                # (PSO first runs ~3 min after startup — 60% is a safe default)
                b1 = mqtt_data['system'].get('pso_brightness1', 0) or mqtt_data['system'].get('pso_brightness', 0) or 60
                b2 = mqtt_data['system'].get('pso_brightness2', 0) or b1
                if b1 > 0 or b2 > 0:
                    try:
                        mqtt_client.publish("smartroom/lamp/control", json.dumps({
                            "brightness1": int(b1), "brightness2": int(b2),
                            "source": "camera_auto"
                        }))
                        # Update locally immediately — prevents re-firing on next YOLO frame
                        mqtt_data['lamp']['brightness1'] = int(b1)
                        mqtt_data['lamp']['brightness2'] = int(b2)
                        mqtt_data['lamp']['brightness_avg'] = round((b1 + b2) / 2.0, 1)
                        _record_lamp_apply(b1, b2)
                        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'),
                                           'msg': f'Auto ON Lamp: L1={b1}% L2={b2}%', 'level': 'success'})
                        socketio.emit('alert', {
                            'type': 'lamp_auto_on', 'level': 'success',
                            'message': 'Person detected — Lamps turned ON automatically',
                            'time': datetime.now().strftime('%H:%M:%S')
                        })
                    except Exception as e:
                        print(f"[ERROR] Auto ON Lamp error: {e}")
            _person_consecutive_frames = required_frames  # Cap to avoid overflow
    else:
        _person_consecutive_frames = 0
        
        # Start no-person timer
        if _no_person_start_time is None:
            _no_person_start_time = time.time()
        if _no_person_lamp_start_time is None:
            _no_person_lamp_start_time = time.time()
        
        elapsed = time.time() - _no_person_start_time
        elapsed_lamp = time.time() - _no_person_lamp_start_time
        
        # Auto OFF AC: no person for 5 minutes (ADAPTIVE only)
        if ac_adaptive and elapsed >= NO_PERSON_TIMEOUT_SECONDS and not _auto_off_triggered:
            _auto_off_triggered = True
            _auto_off_time = time.time()
            if mqtt_data['ac'].get('ac_state') != 'OFF':
                try:
                    mqtt_client.publish("smartroom/ac/control", json.dumps({
                        "action": "POWER_OFF",
                        "source": "camera_auto"
                    }))
                    mqtt_data['ac']['ac_state'] = 'OFF'
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'),
                                       'msg': f'Auto OFF AC: No person for {int(elapsed/60)} min', 'level': 'warning'})
                    socketio.emit('alert', {
                        'type': 'auto_off', 'level': 'warning',
                        'message': f'No person for {int(elapsed/60)} min — AC turned OFF automatically',
                        'time': datetime.now().strftime('%H:%M:%S')
                    })
                except Exception as e:
                    print(f"[ERROR] Auto OFF AC error: {e}")
        
        # Auto OFF Lamp: no person for 20 minutes (ADAPTIVE only)
        if lamp_adaptive and elapsed_lamp >= NO_PERSON_LAMP_TIMEOUT and not _lamp_auto_off_triggered:
            _lamp_auto_off_triggered = True
            current_b1 = mqtt_data['lamp'].get('brightness1', 0)
            current_b2 = mqtt_data['lamp'].get('brightness2', 0)
            if current_b1 > 0 or current_b2 > 0:
                try:
                    mqtt_client.publish("smartroom/lamp/control", json.dumps({
                        "brightness1": 0, "brightness2": 0,
                        "source": "camera_auto"
                    }))
                    mqtt_data['lamp']['brightness1'] = 0
                    mqtt_data['lamp']['brightness2'] = 0
                    mqtt_data['lamp']['brightness_avg'] = 0
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'),
                                       'msg': f'Auto OFF Lamp: No person for {int(elapsed_lamp/60)} min', 'level': 'warning'})
                    socketio.emit('alert', {
                        'type': 'lamp_auto_off', 'level': 'warning',
                        'message': f'No person for {int(elapsed_lamp/60)} min — Lamps turned OFF automatically',
                        'time': datetime.now().strftime('%H:%M:%S')
                    })
                except Exception as e:
                    print(f"[ERROR] Auto OFF Lamp error: {e}")

# ==================== CAMERA FUNCTIONS ====================
def get_camera():
    global camera
    if camera is None:
        # Auto-detect camera: try index 0-4, skip non-capture V4L2 devices
        for idx in range(5):
            dev_path = f'/dev/video{idx}'
            # On Linux, check if device exists and is a real capture device
            if os.path.exists('/dev'):
                if not os.path.exists(dev_path):
                    continue
                # Use v4l2-ctl to verify it's a capture device (skip IR/metadata devices)
                try:
                    import subprocess
                    result = subprocess.run(['v4l2-ctl', '-d', dev_path, '--all'],
                                          capture_output=True, text=True, timeout=3)
                    if 'Video Capture' not in result.stdout:
                        print(f"[CAM] Skipping {dev_path} (not a video capture device)")
                        continue
                except Exception:
                    pass  # v4l2-ctl not available, try anyway
            try:
                cam = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            except Exception:
                cam = cv2.VideoCapture(idx)
            if cam is not None and cam.isOpened():
                ret, test_frame = cam.read()
                if ret and test_frame is not None:
                    camera = cam
                    print(f"[CAM] Opened /dev/video{idx} successfully")
                    break
                else:
                    cam.release()
            else:
                if cam is not None:
                    cam.release()
        
        if camera is not None and camera.isOpened():
            # 720p for best FPS — YOLO only uses 640px input anyway
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            # Force MJPEG codec for USB cameras — much faster than default YUY2
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # Minimal buffer — always get latest frame, not stale buffered ones
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual values
            actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Fallback to 640x480 if 720p not supported
            if actual_w < 1280:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            actual_fps = int(camera.get(cv2.CAP_PROP_FPS))
            
            mqtt_data['camera']['status'] = 'active'
            log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 
                               'msg': f'Camera: {actual_w}x{actual_h} @ {actual_fps}fps', 
                               'level': 'success'})
        else:
            mqtt_data['camera']['status'] = 'error'
            log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 
                               'msg': 'Camera failed to open', 'level': 'error'})
    return camera

def camera_detection_loop():
    """Background thread: continuously detect persons & auto ON/OFF even when no one views the feed"""
    global _latest_frame_bytes, _detection_thread_running, camera, _actual_fps
    _detection_thread_running = True
    last_save_time = time.time()
    save_interval = 5
    retry_delay = 5
    camera_fail_count = 0
    MAX_CAMERA_RETRIES = 10
    frame_count = 0
    YOLO_EVERY_N = 4  # Run YOLO every 4th frame — other frames just capture+encode
    last_person_count = 0
    last_confidence = 0.0
    last_boxes = []  # Cache bounding boxes to draw on non-YOLO frames
    fps_counter = 0
    fps_timer = time.time()
    
    last_camera_status_publish = 0  # Track last MQTT publish of camera status to ESP32
    CAMERA_STATUS_INTERVAL = 10     # Publish every 10 seconds
    
    while _detection_thread_running:
        if not camera_enabled:
            time.sleep(0.5)
            continue
        
        try:
            frame = None
            with camera_lock:
                cam = get_camera()
                if cam is None or not cam.isOpened():
                    mqtt_data['camera']['status'] = 'error'
                else:
                    success, frame = cam.read()
                    if not success:
                        # Camera read failed — release and retry
                        if camera is not None:
                            camera.release()
                            camera = None
                        mqtt_data['camera']['status'] = 'reconnecting'
            
            # If no frame captured, wait and retry with increasing backoff
            if frame is None:
                camera_fail_count += 1
                if camera_fail_count >= MAX_CAMERA_RETRIES:
                    print(f"[CAM] No valid camera after {MAX_CAMERA_RETRIES} retries, pausing detection (retry in 60s)")
                    mqtt_data['camera']['status'] = 'no_camera'
                    time.sleep(60)
                    camera_fail_count = 0  # Reset to try again
                else:
                    time.sleep(retry_delay)
                continue
            
            mqtt_data['camera']['status'] = 'active'
            camera_fail_count = 0  # Reset on successful frame
            # Update device_last_seen so sensor fault detection shows 'Online'
            device_last_seen['camera']['last_seen'] = datetime.now()
            device_last_seen['camera']['status'] = 'online'
            frame_count += 1
            
            # Run YOLO only every Nth frame — other frames reuse last detection
            if frame_count % YOLO_EVERY_N == 0:
                frame, person_count, confidence, last_boxes = detect_persons(frame)
                last_person_count = person_count
                last_confidence = confidence
                
                # Update shared data
                mqtt_data['camera']['person_detected'] = person_count > 0
                mqtt_data['camera']['count'] = person_count
                mqtt_data['camera']['confidence'] = int(confidence * 100)
                # Sync to opt_sensor_data so PSO fitness function uses current local YOLO result
                update_opt_sensor_data(person_detected=person_count > 0, person_count=person_count)
                
                # * Smart auto ON/OFF -- only on YOLO frames
                handle_person_based_control(person_count)
                
                # Save to InfluxDB every 5 seconds
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    save_person_detection(person_count, confidence)
                    last_save_time = current_time
                
                # Publish person status to ESP32 OLED every CAMERA_STATUS_INTERVAL seconds
                if current_time - last_camera_status_publish >= CAMERA_STATUS_INTERVAL:
                    last_camera_status_publish = current_time
                    # Calculate seconds since last person confirmed
                    if _last_person_confirmed_time > 0:
                        last_seen_ago = int(current_time - _last_person_confirmed_time)
                    else:
                        last_seen_ago = -1  # Never detected
                    
                    # Calculate no-person timer if running
                    no_person_elapsed = 0
                    if _no_person_start_time is not None:
                        no_person_elapsed = int(current_time - _no_person_start_time)
                    
                    camera_status_data = {
                        'person_detected': person_count > 0,
                        'person_count': person_count,
                        'last_seen_ago': last_seen_ago,
                        'no_person_elapsed': no_person_elapsed,
                        'auto_off_in': max(0, NO_PERSON_TIMEOUT_SECONDS - no_person_elapsed) if no_person_elapsed > 0 else -1,
                        'auto_off_triggered': _auto_off_triggered,
                        'ac_mode': mqtt_data['ac'].get('mode', 'MANUAL')
                    }
                    try:
                        mqtt_client.publish('smartroom/camera/status', json.dumps(camera_status_data))
                    except Exception:
                        pass
                
                # Emit to WebSocket for live dashboard updates
                # Include last person detection time for dashboard display
                camera_ws_data = dict(mqtt_data['camera'])
                if _last_person_confirmed_time > 0:
                    camera_ws_data['last_seen_ago'] = int(time.time() - _last_person_confirmed_time)
                else:
                    camera_ws_data['last_seen_ago'] = -1
                if _no_person_start_time is not None:
                    camera_ws_data['no_person_elapsed'] = int(time.time() - _no_person_start_time)
                    camera_ws_data['auto_off_in'] = max(0, NO_PERSON_TIMEOUT_SECONDS - int(time.time() - _no_person_start_time))
                else:
                    camera_ws_data['no_person_elapsed'] = 0
                    camera_ws_data['auto_off_in'] = -1
                camera_ws_data['auto_off_triggered'] = _auto_off_triggered
                socketio.emit('mqtt_update', {'type': 'camera', 'data': camera_ws_data})
            else:
                # Non-YOLO frame: just draw cached bounding boxes
                for (x1, y1, x2, y2, conf) in last_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"Person {conf*100:.0f}%"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw overlays on every frame
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, 'Smart Room Camera - YOLOv8 Detection', (20, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if last_person_count > 0:
                cv2.putText(frame, f'Persons Detected: {last_person_count}', (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Encode frame and store for video_feed consumers
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                with _latest_frame_lock:
                    _latest_frame_bytes = buffer.tobytes()
            
            # Measure real FPS
            fps_counter += 1
            elapsed_fps = time.time() - fps_timer
            if elapsed_fps >= 1.0:
                _actual_fps = round(fps_counter / elapsed_fps)
                fps_counter = 0
                fps_timer = time.time()
            
        except Exception as e:
            print(f"[ERROR] Detection loop error: {e}")
            time.sleep(retry_delay)

def generate_frames():
    """Stream latest frames to /video_feed — reads from background thread"""
    # Send a blank frame first so the MJPEG stream starts immediately
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, 'Waiting for camera...', (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    _, buf = cv2.imencode('.jpg', blank)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    
    while True:
        with _latest_frame_lock:
            frame_bytes = _latest_frame_bytes
        
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~20 FPS to client

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        mqtt_data['camera']['status'] = 'inactive'

# ==================== MQTT ====================
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.max_message_size = 0  # NO LIMIT! Default could truncate large RAW IR codes

def on_connect(client, userdata, flags, reason_code, properties=None):
    global mqtt_status
    is_success = False
    if hasattr(reason_code, 'is_failure'):
        is_success = not reason_code.is_failure
    else:
        is_success = (int(reason_code) == 0)
    
    if is_success:
        mqtt_status['connected'] = True
        mqtt_status['last_connect_time'] = datetime.now().strftime('%d %b %Y %H:%M:%S')
        mqtt_status['error'] = None
        print("[OK] MQTT connected")
        client.subscribe("smartroom/#")
        client.subscribe("ir/#")
        client.subscribe("IR/#")
        client.subscribe("+/ir/#")
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': 'MQTT Connected!', 'level': 'success'})
    else:
        mqtt_status['connected'] = False
        mqtt_status['error'] = f'Connection failed: {reason_code}'
        print(f"[ERROR] MQTT CONNECTION FAILED! RC={reason_code}")

def on_message(client, userdata, msg):
    global ir_learning_mode, ir_learning_button, ir_learning_device, mqtt_status
    mqtt_status['last_message_time'] = datetime.now().strftime('%H:%M:%S')
    mqtt_status['message_count'] = mqtt_status.get('message_count', 0) + 1
    
    try:
        topic = msg.topic
        
        # Broadcast to frontend for debugging
        try:
            socketio.emit('mqtt_debug', {
                'topic': topic,
                'payload': msg.payload.decode()[:100],
                'time': datetime.now().strftime('%H:%M:%S')
            })
        except Exception:
            pass
        
        # Parse JSON payload
        try:
            full_payload_str = msg.payload.decode()
            payload = json.loads(full_payload_str)
        except Exception:
            payload = {'raw': msg.payload.decode()}
        
        if 'ac/sensors' in topic:
            mqtt_data['ac'].update({
                'temperature': payload.get('temperature', 0),
                'humidity': payload.get('humidity', 0),
                'heat_index': payload.get('heat_index', 0),
                'ac_state': payload.get('ac_state', 'OFF'),
                'ac_temp': payload.get('ac_temp', 24),
                'fan_speed': payload.get('fan_speed', 1),
                'set_rh': payload.get('set_rh', 50),
                'ac_mode': payload.get('ac_mode', 'ADAPTIVE'),
                'ac_fan_mode': payload.get('ac_fan_mode', 'COOL'),
                'rssi': payload.get('rssi', 0),
                'uptime': payload.get('uptime', 0),
                'temp1': payload.get('temp1', 0),
                'hum1': payload.get('hum1', 0),
                'temp2': payload.get('temp2', 0),
                'hum2': payload.get('hum2', 0),
                'temp3': payload.get('temp3', 0),
                'hum3': payload.get('hum3', 0),
                'swing': payload.get('swing', False),
                'turbo': payload.get('turbo', False),
                'econo': payload.get('econo', False),
            })
            save_sensor_data(
                mqtt_data['ac']['temperature'], mqtt_data['ac']['humidity'], mqtt_data['ac']['heat_index'],
                mqtt_data['ac']['temp1'], mqtt_data['ac']['hum1'],
                mqtt_data['ac']['temp2'], mqtt_data['ac']['hum2'],
                mqtt_data['ac']['temp3'], mqtt_data['ac']['hum3'],
            )
            save_ac_control(mqtt_data['ac']['ac_temp'], mqtt_data['ac']['fan_speed'], mqtt_data['ac']['ac_state'])
            # Feed sensor data to optimization engine
            update_opt_sensor_data(
                temperature=payload.get('temperature'), humidity=payload.get('humidity'),
                temp1=payload.get('temp1'), hum1=payload.get('hum1'),
                temp2=payload.get('temp2'), hum2=payload.get('hum2'),
                temp3=payload.get('temp3'), hum3=payload.get('hum3')
            )
            socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
            device_last_seen['esp32_ac']['last_seen'] = datetime.now()
            device_last_seen['esp32_ac']['status'] = 'online'
            check_alert_rules()
            
        elif 'lamp/sensors' in topic:
            global _last_lamp_mqtt_ts
            _last_lamp_mqtt_ts = time.time()
            b1 = payload.get('brightness1', payload.get('brightness', 0))
            b2 = payload.get('brightness2', b1)
            # Filter BH1750: validasi + ghost lux + spike detection
            l1 = _filter_lux(payload.get('lux1', payload.get('lux', 0)), 0, b1, b2)
            l2 = _filter_lux(payload.get('lux2', payload.get('lux', 0)), 1, b1, b2)
            l3 = _filter_lux(payload.get('lux3', payload.get('lux', 0)), 2, b1, b2)
            mqtt_data['lamp'].update({
                'lux1': l1, 'lux2': l2, 'lux3': l3,
                'lux_avg': round((l1 + l2 + l3) / 3.0, 1),
                'motion': payload.get('motion', False),
                'brightness1': b1, 'brightness2': b2,
                'brightness_avg': round((b1 + b2) / 2.0, 1),
                'mode': payload.get('mode', mqtt_data['lamp'].get('mode', 'ADAPTIVE')),
                'rssi': payload.get('rssi', 0),
                'uptime': payload.get('uptime', 0)
            })
            # Save lamp data to InfluxDB
            save_lamp_data(l1, l2, l3, b1, b2, mqtt_data['lamp']['motion'])
            update_opt_sensor_data(lux=round((l1 + l2 + l3) / 3.0, 1), lux1=l1, lux2=l2, lux3=l3,
                                    curr_brightness1=b1, curr_brightness2=b2)
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
            # Track device status
            device_last_seen['esp32_lamp']['last_seen'] = datetime.now()
            device_last_seen['esp32_lamp']['status'] = 'online'
            
        elif 'energy/data' in topic:
            mqtt_data['energy'].update({
                'voltage': payload.get('voltage', 0),
                'current': payload.get('current', 0),
                'power': payload.get('power', 0),
                'energy': float(payload.get('energy', 0)) / 1000.0,
                'frequency': payload.get('frequency', 0),
                'pf': payload.get('pf', 0),
                'connected': True
            })
            socketio.emit('mqtt_update', {'type': 'energy', 'data': mqtt_data['energy']})
            # Save to InfluxDB energy_monitor every 6 minutes (for export feature)
            global _last_ac_energy_influx_ts
            _now_e = time.time()
            if _now_e - _last_ac_energy_influx_ts >= INFLUX_WRITE_INTERVAL:
                _last_ac_energy_influx_ts = _now_e
                try:
                    write_to_influxdb('energy_monitor', {
                        'voltage':    float(mqtt_data['energy']['voltage']),
                        'current':    float(mqtt_data['energy']['current']),
                        'power':      float(mqtt_data['energy']['power']),
                        'energy_kwh': float(mqtt_data['energy']['energy']),
                    }, tags={'device': 'esp32_ac'})
                except Exception as _e:
                    print(f'[ERROR] energy_monitor write: {_e}')

        elif 'camera/detection' in topic:
            person_from_mqtt = payload.get('person_detected', False)
            mqtt_data['camera'].update({
                'person_detected': person_from_mqtt,
                'count': payload.get('count', 0),
                'confidence': payload.get('confidence', 0)
            })
            # Update _last_person_confirmed_time from MQTT camera so lamp protection
            # works even when local camera thread is not running
            if person_from_mqtt:
                global _last_person_confirmed_time
                _last_person_confirmed_time = time.time()
            socketio.emit('mqtt_update', {'type': 'camera', 'data': mqtt_data['camera']})
            update_opt_sensor_data(person_detected=person_from_mqtt,
                                   person_count=int(payload.get('count', 1 if person_from_mqtt else 0)))
            # Track device status
            device_last_seen['camera']['last_seen'] = datetime.now()
            device_last_seen['camera']['status'] = 'online'
            
        elif 'dashboard/state' in topic or 'optimization/stats' in topic or 'ml/result' in topic:
            # Update from optimization algorithm (GA->AC / PSO->Lamp)
            ga_sol = payload.get('ga_solution', {})
            pso_sol = payload.get('pso_solution', {})
            mqtt_data['system'].update({
                'ga_fitness': payload.get('ga_best_fitness', payload.get('ga_fitness', 0)),
                'pso_fitness': payload.get('pso_best_fitness', payload.get('pso_fitness', 0)),
                'optimization_runs': payload.get('optimization_count', payload.get('runs', mqtt_data['system'].get('optimization_runs', 0) + 1)),
                'ga_temp': ga_sol.get('temperature', payload.get('ga_temp', mqtt_data['system'].get('ga_temp', 0))),
                'ga_fan': ga_sol.get('fan_speed', payload.get('ga_fan', mqtt_data['system'].get('ga_fan', 0))),
                'pso_brightness': pso_sol.get('brightness', payload.get('pso_brightness', mqtt_data['system'].get('pso_brightness', 0))),
                'ga_history': payload.get('ga_history', mqtt_data['system'].get('ga_history', [])),
                'pso_history': payload.get('pso_history', mqtt_data['system'].get('pso_history', []))
            })
            socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
            # Write optimization history to InfluxDB
            write_to_influxdb('optimization_result', {
                'ga_fitness': float(mqtt_data['system']['ga_fitness']),
                'pso_fitness': float(mqtt_data['system']['pso_fitness']),
                'ga_temp': float(mqtt_data['system']['ga_temp']),
                'ga_fan': float(mqtt_data['system']['ga_fan']),
                'pso_brightness': float(mqtt_data['system']['pso_brightness']),
                # PSO is minimize (error), GA is maximize — normalize before combining
                'combined_fitness': float(mqtt_data['system']['ga_fitness']) * 0.5
                                    + max(0.0, 100.0 - float(mqtt_data['system']['pso_fitness'])) * 0.5
            })
            # AUTO-APPLY AC — only if ga_solution was present in this payload
            if ga_sol and mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE' and _person_present_recently():
                opt_temp = ga_sol.get('temperature', payload.get('ga_temp', 0))
                opt_fan  = ga_sol.get('fan_speed', payload.get('ga_fan', 0))
                opt_mode = ga_sol.get('mode', 'COOL')
                opt_rh   = ga_sol.get('set_rh', 50)
                if opt_temp >= 16 and opt_temp <= 30 and opt_fan >= 1:
                    global _last_adaptive_ac_apply
                    now = time.time()
                    if now - _last_adaptive_ac_apply >= AC_ADAPTIVE_DEBOUNCE:
                        _last_adaptive_ac_apply = now
                        ac_cmd = {'command': 'SET', 'temperature': int(opt_temp), 'fan_speed': int(opt_fan),
                                  'mode': opt_mode, 'set_rh': int(opt_rh), 'source': 'adaptive'}
                        client.publish('smartroom/ac/control', json.dumps(ac_cmd))
                        mqtt_data['ac']['set_rh'] = int(opt_rh)
            # AUTO-APPLY Lamp — only if pso_solution was present in this payload
            if pso_sol and mqtt_data['lamp'].get('mode', 'MANUAL') == 'ADAPTIVE' and _person_present_recently_lamp():
                opt_b1, opt_b2 = _safe_lamp_brightness(
                    pso_sol.get('brightness1', payload.get('pso_brightness1', pso_sol.get('brightness', 0))),
                    pso_sol.get('brightness2', payload.get('pso_brightness2', pso_sol.get('brightness', 0)))
                )
                if (opt_b1 > 0 or opt_b2 > 0) and _should_apply_lamp(opt_b1, opt_b2):
                    client.publish('smartroom/lamp/control', json.dumps({'brightness1': opt_b1, 'brightness2': opt_b2, 'source': 'adaptive'}))
                    _record_lamp_apply(opt_b1, opt_b2)
        
        elif 'ml/status' in topic:
            socketio.emit('ml_status', payload)
            log_messages.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'msg': f"ML {payload.get('algorithm', '')}: {payload.get('status', '')}",
                'level': 'info' if payload.get('status') == 'completed' else 'warning'
            })
            
            # If completed, also try to extract result data directly from status payload
            if payload.get('status') == 'completed':
                ga_sol = payload.get('ga_solution', {})
                pso_sol = payload.get('pso_solution', {})
                has_data = (payload.get('ga_fitness') is not None or payload.get('ga_best_fitness') is not None or payload.get('pso_fitness') is not None or payload.get('pso_best_fitness') is not None)
                if has_data:
                    mqtt_data['system'].update({
                        'ga_fitness': payload.get('ga_best_fitness', payload.get('ga_fitness', mqtt_data['system'].get('ga_fitness', 0))),
                        'pso_fitness': payload.get('pso_best_fitness', payload.get('pso_fitness', mqtt_data['system'].get('pso_fitness', 0))),
                        'optimization_runs': mqtt_data['system'].get('optimization_runs', 0) + 1,
                        'ga_temp': ga_sol.get('temperature', payload.get('ga_temp', mqtt_data['system'].get('ga_temp', 0))),
                        'ga_fan': ga_sol.get('fan_speed', payload.get('ga_fan', mqtt_data['system'].get('ga_fan', 0))),
                        'pso_pwm1': pso_sol.get('pwm1', payload.get('pso_pwm1', mqtt_data['system'].get('pso_pwm1', 0))),
                        'pso_pwm2': pso_sol.get('pwm2', payload.get('pso_pwm2', mqtt_data['system'].get('pso_pwm2', 0))),
                        'pso_brightness': pso_sol.get('brightness', payload.get('pso_brightness', mqtt_data['system'].get('pso_brightness', 0))),
                        'pso_brightness1': pso_sol.get('brightness1', payload.get('pso_brightness1', mqtt_data['system'].get('pso_brightness1', 0))),
                        'pso_brightness2': pso_sol.get('brightness2', payload.get('pso_brightness2', mqtt_data['system'].get('pso_brightness2', 0))),
                        'ga_history': payload.get('ga_history', mqtt_data['system'].get('ga_history', [])),
                        'pso_history': payload.get('pso_history', mqtt_data['system'].get('pso_history', []))
                    })
                    
        elif 'ac/mode' in topic:
            mqtt_data['ac']['mode'] = payload.get('mode', 'ADAPTIVE')
            socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
            
        elif 'lamp/mode' in topic:
            mqtt_data['lamp']['mode'] = payload.get('mode', 'ADAPTIVE')
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
        
        elif 'ir/learned' in topic or 'IR/learned' in topic.lower():
            esp32_status = payload.get('status', '') if isinstance(payload, dict) else ''
            
            ir_code = ''
            device = ir_learning_device or 'remote'
            button_name = ir_learning_button
            
            if isinstance(payload, dict):
                if not button_name:
                    button_name = payload.get('button', '')
                ir_code = payload.get('code', payload.get('ir_code', payload.get('raw', '')))
                if not ir_learning_device:
                    device = payload.get('device', device)
            elif isinstance(payload, str):
                ir_code = payload
            else:
                ir_code = str(payload)
            
            if not ir_learning_mode:
                pass  # Not in learning mode, ignore
            elif button_name and ir_code and len(ir_code) > 0:
                is_power_toggle = False
                if 'power' in button_name.lower():
                    existing_power_codes = {k: v for k, v in mqtt_data['ir_codes'].items() if 'power' in k.lower() and device in k}
                    if existing_power_codes and ir_code in existing_power_codes.values():
                        is_power_toggle = True
                        button_name = f"{device}_power_toggle"
                        mqtt_data['ir_states'][button_name] = 'OFF'
                
                mqtt_data['ir_codes'][button_name] = ir_code
                save_ir_command(device, button_name, len(ir_code))
                
                try:
                    ir_file = os.path.join(os.path.dirname(__file__), 'ir_codes.json')
                    with open(ir_file, 'w') as f:
                        json.dump(mqtt_data['ir_codes'], f, indent=2)
                except Exception as e:
                    print(f"[ERROR] Error saving IR codes to file: {e}")
                
                log_messages.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'msg': f'IR Code learned: {button_name}{" (TOGGLE)" if is_power_toggle else ""}',
                    'level': 'success'
                })
                
                socketio.emit('ir_learned', {
                    'button': button_name,
                    'code': ir_code[:50] + '...' if len(ir_code) > 50 else ir_code,
                    'device': device,
                    'is_toggle': is_power_toggle,
                    'status': 'success'
                })
                
                ir_learning_mode = False
                ir_learning_button = ""
                ir_learning_device = ""
            else:
                socketio.emit('ir_learned', {'status': 'error', 'message': 'Invalid IR data received'})
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] MQTT Message Handler Error: {str(e)}\n{tb}")
        # Only log to dashboard if it's a real unexpected error, not a known transient one
        err_str = str(e)
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'MQTT Error: {err_str}', 'level': 'error'})

def on_disconnect(client, userdata, flags, reason_code, properties=None):
    global mqtt_status
    mqtt_status['connected'] = False
    print(f"[WARN] MQTT Disconnected (RC: {reason_code})")
    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'MQTT Disconnected! RC: {reason_code}', 'level': 'warning'})

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

# Enable auto-reconnect with 5 second delay
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

def start_mqtt():
    global mqtt_status
    mqtt_status['broker'] = f'{MQTT_BROKER}:{MQTT_PORT}'
    retries = 0
    while retries < 5:
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
            time.sleep(2)
            return
        except Exception as e:
            retries += 1
            mqtt_status['error'] = str(e)
            if retries < 5:
                time.sleep(3)
    print("[ERROR] MQTT: All connection attempts failed.")

mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()

# ==================== INFLUXDB ====================
def get_influx_data(measurement, field, hours=1, device_tag=None):
    try:
        _, _, query_api = _get_influx_client()
        # Use 10-minute windows for occupancy (camera_detection) to reduce noise,
        # 5-minute windows for all other measurements.
        window = '10m' if measurement == 'camera_detection' else '5m'
        device_filter = f'|> filter(fn: (r) => r["device"] == "{device_tag}")\n          ' if device_tag else ''
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["_field"] == "{field}")
          {device_filter}|> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        '''
        result = query_api.query(query=query)
        data_points = []
        for table in result:
            for record in table.records:
                data_points.append({
                    'time': record.get_time().astimezone().strftime('%H:%M'),
                    'value': round(float(record.get_value()), 2)
                })
        return data_points
    except Exception as e:
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'InfluxDB Query Error: {str(e)}', 'level': 'error'})
        return []

# ==================== AUTHENTICATION ====================
from functools import wraps

def admin_required(f):
    """Decorator: restrict route to admin role only."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        if session.get('role') != 'admin':
            return jsonify({'status': 'error', 'message': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated

@app.before_request
def require_login():
    public_paths = ['/login', '/api/optimization/update']
    if request.path in public_paths:
        return None
    if request.path.startswith('/socket.io'):
        return None
    if session.get('logged_in'):
        return None
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = USERS.get(username)
        if user and user['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            session['role'] = user['role']
            return redirect('/')
        return render_template('login.html', error='Invalid username or password')
    return render_template('login.html', error=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/api/auth/role')
def get_auth_role():
    """Return current session role for frontend."""
    return jsonify({
        'role': session.get('role', 'user'),
        'username': session.get('username', ''),
    })

# ==================== API ROUTES ====================
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/status')
def camera_status():
    try:
        cam = get_camera()
        if cam and cam.isOpened():
            w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = _actual_fps if _actual_fps > 0 else int(cam.get(cv2.CAP_PROP_FPS))
            return jsonify({'status': 'active', 'width': w, 'height': h, 'fps': fps})
        else:
            return jsonify({'status': 'inactive'})
    except:
        return jsonify({'status': 'error'})

@app.route('/api/camera/restart', methods=['POST'])
@admin_required
def restart_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        time.sleep(1)
        cam = get_camera()
        if cam and cam.isOpened():
            return jsonify({'status': 'success', 'message': 'Camera restarted'})
        else:
            return jsonify({'status': 'error', 'message': 'Camera failed to restart'}), 500

@app.route('/api/camera/toggle', methods=['POST'])
@admin_required
def toggle_camera():
    global camera_enabled, camera
    camera_enabled = not camera_enabled
    if not camera_enabled:
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
            mqtt_data['camera']['status'] = 'inactive'
            mqtt_data['camera']['person_detected'] = False
            mqtt_data['camera']['count'] = 0
            mqtt_data['camera']['confidence'] = 0
        return jsonify({'status': 'success', 'enabled': False, 'message': 'Camera OFF'})
    else:
        with camera_lock:
            cam = get_camera()
            if cam and cam.isOpened():
                return jsonify({'status': 'success', 'enabled': True, 'message': 'Camera ON'})
            else:
                return jsonify({'status': 'error', 'enabled': True, 'message': 'Camera failed to start'}), 500

@app.route('/api/data')
def get_data():
    response = jsonify(mqtt_data)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/system/tz-status')
def get_tz_status():
    return jsonify({'warning': _system_tz_warn, 'ok': _system_tz_warn == ''})

# ==================== OUTDOOR WEATHER (Open-Meteo — UNS Surakarta) ====================
# Lokasi: Universitas Sebelas Maret, Surakarta, Jawa Tengah
# Koordinat: -7.5561, 110.8316
# API: Open-Meteo (https://open-meteo.com) — 100% gratis, tidak butuh API key

WEATHER_LAT  = -7.5561
WEATHER_LON  = 110.8316
WEATHER_FETCH_INTERVAL = 600  # Update setiap 10 menit
_last_weather_fetch    = 0.0

outdoor_weather_data = {
    'temperature':       None,
    'apparent_temp':     None,
    'humidity':          None,
    'wind_speed':        None,
    'precipitation':     None,
    'cloud_cover':       None,
    'uv_index':          None,
    'weather_code':      None,
    'weather_desc':      'Memuat data...',
    'weather_icon':      '🌤️',
    'is_day':            True,
    'last_updated':      None,
    'fetch_ok':          False,
    'error':             None,
}

def _wmo_to_desc_icon(code, is_day=True):
    """Convert WMO weather interpretation code to Indonesian description + emoji icon."""
    if code == 0:
        return ('Cerah', '☀️') if is_day else ('Langit Cerah', '🌙')
    elif code in (1, 2, 3):
        labels = {1: 'Sebagian Cerah', 2: 'Berawan Sebagian', 3: 'Mendung'}
        icons  = {1: '🌤️', 2: '⛅', 3: '☁️'}
        return (labels[code], icons[code])
    elif code in (45, 48):
        return ('Berkabut', '🌫️')
    elif code in (51, 53, 55):
        return ('Gerimis', '🌦️')
    elif code in (61, 63, 65):
        return ('Hujan', '🌧️')
    elif code in (71, 73, 75, 77):
        return ('Salju/Es', '🌨️')
    elif code in (80, 81, 82):
        return ('Hujan Lebat', '🌧️')
    elif code in (85, 86):
        return ('Hujan Salju', '🌨️')
    elif code in (95,):
        return ('Hujan + Petir', '⛈️')
    elif code in (96, 99):
        return ('Badai Petir', '⛈️')
    else:
        return ('Tidak Diketahui', '❓')

def fetch_outdoor_weather():
    """Fetch current weather from Open-Meteo API for UNS Surakarta.
    Updates outdoor_weather_data global dict.
    Called by the background thread every WEATHER_FETCH_INTERVAL seconds.
    """
    global _last_weather_fetch
    now = time.time()
    if now - _last_weather_fetch < WEATHER_FETCH_INTERVAL:
        return
    _last_weather_fetch = now

    url = (
        f'https://api.open-meteo.com/v1/forecast'
        f'?latitude={WEATHER_LAT}&longitude={WEATHER_LON}'
        f'&current=temperature_2m,relative_humidity_2m,apparent_temperature,'
        f'weather_code,wind_speed_10m,precipitation,cloud_cover,uv_index,is_day'
        f'&timezone=Asia%2FJakarta'
        f'&forecast_days=1'
    )
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SmartRoom-Weather/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode('utf-8'))

        cur = raw.get('current', {})
        code   = int(cur.get('weather_code', 0))
        is_day = bool(cur.get('is_day', 1))
        desc, icon = _wmo_to_desc_icon(code, is_day)

        outdoor_weather_data.update({
            'temperature':   round(float(cur.get('temperature_2m')    or 0), 1),
            'apparent_temp': round(float(cur.get('apparent_temperature') or 0), 1),
            'humidity':      round(float(cur.get('relative_humidity_2m') or 0), 1),
            'wind_speed':    round(float(cur.get('wind_speed_10m')    or 0), 1),
            'precipitation': round(float(cur.get('precipitation')     or 0), 1),
            'cloud_cover':   round(float(cur.get('cloud_cover')       or 0), 1),
            'uv_index':      round(float(cur.get('uv_index')          or 0), 1),
            'weather_code':  code,
            'weather_desc':  desc,
            'weather_icon':  icon,
            'is_day':        is_day,
            'last_updated':  datetime.now().strftime('%H:%M:%S'),
            'fetch_ok':      True,
            'error':         None,
        })
        print(f"[WEATHER] {desc} {icon} | Luar: {outdoor_weather_data['temperature']}°C "
              f"RH={outdoor_weather_data['humidity']}% Angin={outdoor_weather_data['wind_speed']}km/h "
              f"UV={outdoor_weather_data['uv_index']}")
    except Exception as e:
        outdoor_weather_data['fetch_ok']  = False
        outdoor_weather_data['error']     = str(e)
        outdoor_weather_data['last_updated'] = datetime.now().strftime('%H:%M:%S')
        print(f"[WEATHER] Fetch gagal: {e}")

def weather_poll_loop():
    """Background thread: fetch outdoor weather setiap WEATHER_FETCH_INTERVAL detik."""
    # Fetch segera saat startup (bypass interval check pertama kali)
    global _last_weather_fetch
    _last_weather_fetch = 0.0
    while True:
        try:
            fetch_outdoor_weather()
        except Exception as e:
            print(f"[WEATHER] Thread error: {e}")
        time.sleep(60)   # check setiap 1 menit, fungsi sendiri yang throttle ke 10 menit

@app.route('/api/outdoor-weather')
def get_outdoor_weather():
    """Return current outdoor weather data for UNS Surakarta."""
    # Also include indoor data for comparison
    indoor = {
        'temperature': mqtt_data['ac'].get('temperature', 0),
        'humidity':    mqtt_data['ac'].get('humidity', 0),
        'heat_index':  mqtt_data['ac'].get('heat_index', 0),
    }
    resp = jsonify({
        'outdoor': outdoor_weather_data,
        'indoor':  indoor,
        'location': 'UNS Surakarta, Jawa Tengah',
        'coords':  {'lat': WEATHER_LAT, 'lon': WEATHER_LON},
    })
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp

@app.route('/api/mqtt/status')
def get_mqtt_status():
    return jsonify({
        'connected': mqtt_client.is_connected(),
        'broker': f'{MQTT_BROKER}:{MQTT_PORT}',
        'last_connect': mqtt_status.get('last_connect_time'),
        'last_message': mqtt_status.get('last_message_time'),
        'message_count': mqtt_status.get('message_count', 0),
        'error': mqtt_status.get('error'),
        'subscriptions': ['smartroom/#', 'ir/#', 'IR/#', '+/ir/#']
    })

@app.route('/api/mqtt/reconnect', methods=['POST'])
def mqtt_reconnect():
    try:
        mqtt_client.reconnect()
        return jsonify({'status': 'success', 'message': 'Reconnect initiated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/mqtt/config', methods=['POST'])
def mqtt_config():
    """Change MQTT broker IP at runtime and reconnect"""
    global MQTT_BROKER, MQTT_PORT
    data = request.json or {}
    new_broker = data.get('broker', '').strip()
    new_port = int(data.get('port', 1883))
    if not new_broker:
        return jsonify({'status': 'error', 'message': 'Broker IP cannot be empty'}), 400
    old_broker = MQTT_BROKER
    MQTT_BROKER = new_broker
    MQTT_PORT = new_port
    mqtt_status['broker'] = f'{MQTT_BROKER}:{MQTT_PORT}'
    try:
        mqtt_client.disconnect()
    except Exception:
        pass
    import threading
    def reconnect_thread():
        import time
        time.sleep(1)
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
        except Exception as e:
            mqtt_status['error'] = str(e)
    threading.Thread(target=reconnect_thread, daemon=True).start()
    return jsonify({'status': 'ok', 'message': f'Trying to connect to {MQTT_BROKER}:{MQTT_PORT}', 'broker': MQTT_BROKER, 'port': MQTT_PORT})

@app.route('/api/simulate', methods=['POST'])
def simulate_data():
    """Inject dummy sensor data directly into mqtt_data (bypasses MQTT) for frontend testing"""
    import random
    mqtt_data['ac'].update({
        'temperature': round(27.5 + random.uniform(-2, 2), 1),
        'humidity': round(65.0 + random.uniform(-5, 5), 1),
        'heat_index': round(29.5 + random.uniform(-2, 2), 1),
        'ac_state': 'ON', 'ac_temp': 24, 'fan_speed': 2,
        'temp1': round(27.0 + random.uniform(-1, 1), 1),
        'temp2': round(27.3 + random.uniform(-1, 1), 1),
        'temp3': round(27.8 + random.uniform(-1, 1), 1),
        'hum1': round(64.0 + random.uniform(-3, 3), 1),
        'hum2': round(65.0 + random.uniform(-3, 3), 1),
        'hum3': round(66.0 + random.uniform(-3, 3), 1),
        'rssi': -55, 'uptime': 3600
    })
    mqtt_data['lamp'].update({
        'lux1': round(350 + random.uniform(-50, 50)),
        'lux2': round(380 + random.uniform(-50, 50)),
        'lux3': round(340 + random.uniform(-50, 50)),
        'lux_avg': round(357 + random.uniform(-30, 30)),
        'brightness1': 80, 'brightness2': 75, 'brightness3': 70, 'brightness_avg': 75,
        'motion': True, 'rssi': -60, 'uptime': 3600
    })
    mqtt_data['energy'].update({
        'voltage': round(220.0 + random.uniform(-5, 5), 1),
        'current': round(1.5 + random.uniform(-0.2, 0.2), 2),
        'power': round(330.0 + random.uniform(-10, 10), 1),
        'energy': round(1.25 + random.uniform(0, 0.1), 3),
        'frequency': 50.0, 'pf': 0.95, 'connected': True, 'ac_state': 'ON'
    })
    socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
    socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
    return jsonify({'status': 'ok', 'message': 'Dummy data injected', 'ac_temp': mqtt_data['ac']['temperature'], 'lamp_lux': mqtt_data['lamp']['lux1']})

@app.route('/api/mqtt/selftest', methods=['POST'])
def mqtt_selftest():
    """Publish a test message to MQTT broker to verify broker connection works"""
    if not mqtt_client.is_connected():
        return jsonify({'status': 'error', 'message': f'MQTT client not connected to {MQTT_BROKER}:{MQTT_PORT}. Is Mosquitto running?'}), 503
    try:
        test_payload = json.dumps({'temperature': 28.5, 'humidity': 65.0, 'heat_index': 30.0, 'ac_state': 'ON', 'ac_temp': 24, 'fan_speed': 2, 'rssi': -55, 'uptime': 100, 'temp1': 28.0, 'temp2': 28.5, 'temp3': 29.0, 'hum1': 64.0, 'hum2': 65.0, 'hum3': 66.0})
        result = mqtt_client.publish('smartroom/ac/sensors', test_payload, qos=1)
        result.wait_for_publish(timeout=3)
        return jsonify({'status': 'ok', 'message': 'Test message published to smartroom/ac/sensors. Check if data appears in dashboard.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/occupancy/feedback', methods=['POST'])
def occupancy_feedback_submit():
    try:
        data = request.json or {}
        rating = int(data.get('rating', 0))
        comment = (data.get('comment') or '').strip()
        occupancy_count = int(data.get('occupancy_count', 0))
        google_form_url = (data.get('google_form_url') or GOOGLE_FORM_URL).strip()

        if rating < 1 or rating > 5:
            return jsonify({'status': 'error', 'message': 'Rating must be 1-5'}), 400

        row = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rating': rating,
            'comment': comment,
            'occupancy_count': occupancy_count,
            'google_form_url': google_form_url
        }
        occupancy_feedback.appendleft(row)
        log_messages.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'msg': f'Occupancy Feedback: rating={rating}, occupancy={occupancy_count}, comment={comment[:40]}',
            'level': 'info'
        })

        return jsonify({'status': 'success', 'message': 'Feedback saved', 'google_form_url': google_form_url})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/occupancy/feedback/list')
def occupancy_feedback_list():
    return jsonify({'feedback': list(occupancy_feedback)[:20]})

# ── Server-side recording state (shared across all devices) ──
_rec_state = {'energy': False, 'temp': False, 'lux': False}

@app.route('/api/rec/state', methods=['GET', 'POST'])
def rec_state_api():
    global _rec_state
    if request.method == 'POST':
        data = request.get_json(force=True) or {}
        for k in ('energy', 'temp', 'lux'):
            if k in data:
                _rec_state[k] = bool(data[k])
        return jsonify({'ok': True, 'state': _rec_state})
    return jsonify(_rec_state)

# ── Server-side recorded data (shared across all devices, in-memory) ──
_rec_data = {'energy': [], 'temp': [], 'lux': []}

@app.route('/api/rec/data', methods=['GET', 'POST', 'DELETE'])
def rec_data_api():
    global _rec_data
    dtype = request.args.get('type', 'energy')
    if dtype not in ('energy', 'temp', 'lux'):
        return jsonify({'error': 'invalid type'}), 400
    if request.method == 'DELETE':
        _rec_data[dtype] = []
        return jsonify({'ok': True})
    if request.method == 'POST':
        body = request.get_json(force=True) or {}
        rows = body.get('rows', [])
        if isinstance(rows, list) and len(rows) <= 50000:
            _rec_data[dtype] = rows   # replace (not append) to stay idempotent
        return jsonify({'ok': True, 'count': len(_rec_data[dtype])})
    return jsonify({'rows': _rec_data[dtype], 'count': len(_rec_data[dtype])})


# ==================== EXPORT CSV ====================
_PHP_ENERGY_HISTORY_URL = 'https://iotlab-uns.com/api_energy.php'
_PHP_ENERGY_KEY         = 'iotlab_smartroom_2024'

def _fetch_energy_history_from_mysql(id_kwh, from_dt, to_dt, limit=5000):
    """Fetch historical energy data from MySQL via PHP proxy.
    Returns list of dicts or raises Exception on failure.
    id_kwh: 1=AC, 3=Lamp
    """
    import urllib.request as _ureq, urllib.parse as _uparse
    params = _uparse.urlencode({
        'key':    _PHP_ENERGY_KEY,
        'action': 'history',
        'id_kwh': id_kwh,
        'from':   from_dt,
        'to':     to_dt,
        'limit':  limit,
    })
    url = f'{_PHP_ENERGY_HISTORY_URL}?{params}'
    req = _ureq.Request(url, headers={'User-Agent': 'SmartRoom-Export/1.0'})
    with _ureq.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    if 'error' in data:
        raise ValueError(data['error'])
    return data.get('rows', [])


@app.route('/api/export/csv')
@admin_required
def export_csv_from_db():
    """Export CSV from MySQL (energy) or InfluxDB (temp/lux/occupancy).
    Params: type=energy_ac|energy_lamp|temp|lux|occupancy
            from=YYYY-MM-DD  to=YYYY-MM-DD  (WIB timezone)
    """
    import io, csv as _csv
    from datetime import timezone, timedelta as _td

    dtype   = request.args.get('type', 'temp')
    from_dt = request.args.get('from', '')
    to_dt   = request.args.get('to', '')

    if not from_dt or not to_dt:
        return jsonify({'error': 'from and to are required (YYYY-MM-DD)'}), 400

    try:
        datetime.strptime(from_dt, '%Y-%m-%d')
        datetime.strptime(to_dt,   '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

    valid_types = ('energy_ac', 'energy_lamp', 'energy_outlet', 'energy_total', 'temp', 'lux', 'occupancy')
    if dtype not in valid_types:
        return jsonify({'error': f'Invalid type. Options: {", ".join(valid_types)}'}), 400

    # ── Energy Total: fetch AC + Outlet + Lamp, merge into one CSV ────────────
    if dtype == 'energy_total':
        all_rows = []
        for id_kwh, dev_label in [(1, 'AC'), (2, 'Outlet'), (3, 'Lamp')]:
            try:
                rows_dev = _fetch_energy_history_from_mysql(id_kwh, from_dt, to_dt)
                for r in (rows_dev or []):
                    r['_device'] = dev_label
                    all_rows.append(r)
            except Exception:
                pass
        if not all_rows:
            return jsonify({'error': f'No data for any device in MySQL for range {from_dt} to {to_dt}'}), 404
        # Sort by timestamp then device
        all_rows.sort(key=lambda r: (r.get('timestamp', ''), r.get('_device', '')))
        output = io.StringIO()
        output.write("sep=,\n")
        writer = _csv.writer(output)
        writer.writerow(['Timestamp', 'Device', 'Voltage (V)', 'Current (A)', 'Active Power (W)',
                         'Reactive Power (VAR)', 'Apparent Power (VA)', 'Power Factor',
                         'Frequency (Hz)', 'Energy (kWh)'])
        for r in all_rows:
            ap = float(r.get('apparent_power') or 0)
            p  = float(r.get('active_power') or 0)
            pf = round(p / ap, 3) if ap > 0.001 else 0.0
            writer.writerow([
                r.get('timestamp', ''),
                r.get('_device', ''),
                r.get('voltage', ''),
                r.get('current', ''),
                r.get('active_power', ''),
                r.get('reactive_power', ''),
                r.get('apparent_power', ''),
                pf,
                r.get('frequency', ''),
                r.get('energy_kwh', ''),
            ])
        csv_bytes = output.getvalue().encode('utf-8-sig')
        fname = f"{dtype}_{from_dt}_sd_{to_dt}.csv"
        return Response(
            csv_bytes,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{fname}"'}
        )

    # ── Energy: fetch from MySQL via PHP (complete data source) ──────────────
    if dtype in ('energy_ac', 'energy_lamp', 'energy_outlet'):
        id_kwh_map = {'energy_ac': 1, 'energy_outlet': 2, 'energy_lamp': 3}
        dev_name_map = {'energy_ac': 'AC', 'energy_outlet': 'Outlet', 'energy_lamp': 'Lamp'}
        id_kwh   = id_kwh_map[dtype]
        dev_name = dev_name_map[dtype]
        try:
            mysql_rows = _fetch_energy_history_from_mysql(id_kwh, from_dt, to_dt)
        except Exception as ex:
            # Fallback to InfluxDB if PHP is unreachable
            mysql_rows = None
            influx_err = str(ex)

        if mysql_rows:
            output = io.StringIO()
            output.write("sep=,\n")
            fields  = ['timestamp','voltage','current','active_power','reactive_power',
                       'apparent_power','power_factor','frequency','energy_kwh']
            headers = ['Timestamp','Voltage (V)','Current (A)','Power Active (W)',
                       'Reactive Power (VAR)','Apparent Power (VA)','Power Factor','Frequency (Hz)',
                       'Konsumsi Energy (kWh)']
            writer = _csv.writer(output)
            writer.writerow(headers)
            for r in mysql_rows:
                writer.writerow([
                    r.get('timestamp', ''),
                    r.get('voltage',       ''),
                    r.get('current',           ''),
                    r.get('active_power',   ''),
                    r.get('reactive_power', ''),
                    r.get('apparent_power', ''),
                    r.get('power_factor',   ''),
                    r.get('frequency',      ''),
                    r.get('energy_kwh',     ''),
                ])
            csv_bytes = output.getvalue().encode('utf-8-sig')  # BOM so Excel reads correctly
            fname = f"{dtype}_{from_dt}_sd_{to_dt}.csv"
            return Response(
                csv_bytes,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename="{fname}"'}
            )
        else:
            # Fallback InfluxDB — pakai kedua device tag (esp32_ac/mysql_ac)
            print(f'[EXPORT] MySQL failed ({influx_err}), falling back to InfluxDB')
            tag_vals = ('esp32_ac', 'mysql_ac') if dtype == 'energy_ac' else ('esp32_lamp', 'mysql_lamp')
            influx_fields = ['voltage', 'current', 'power', 'energy_kwh', 'frequency', 'power_factor']
            wib = timezone(_td(hours=7))
            start_utc = datetime.strptime(from_dt, '%Y-%m-%d').replace(tzinfo=wib).astimezone(timezone.utc)
            end_utc   = datetime.strptime(to_dt, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=wib).astimezone(timezone.utc)
            start_iso = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_iso   = end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
            try:
                _, _, query_api = _get_influx_client()
                rows_by_ts = {}
                tag_filter = ' or '.join([f'r["device"] == "{v}"' for v in tag_vals])
                for fld in influx_fields:
                    q = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start_iso}, stop: {end_iso})
  |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
  |> filter(fn: (r) => r["_field"] == "{fld}")
  |> filter(fn: (r) => {tag_filter})
  |> yield(name: "raw")
'''
                    for table in query_api.query(query=q):
                        for rec in table.records:
                            t_wib  = rec.get_time().astimezone(wib)
                            ts_str = t_wib.strftime('%Y-%m-%d %H:%M:%S')
                            rows_by_ts.setdefault(ts_str, {})
                            v = rec.get_value()
                            if v is not None:
                                rows_by_ts[ts_str][fld] = v
                if not rows_by_ts:
                    return jsonify({'error': f'No data ditemukan. MySQL error: {influx_err}'}), 404
                sorted_ts = sorted(rows_by_ts.keys())
                # Make energy_kwh relative (subtract first value)
                first_kwh = next((float(rows_by_ts[t]['energy_kwh'])
                                  for t in sorted_ts if rows_by_ts[t].get('energy_kwh') is not None), None)
                output = io.StringIO()
                output.write("sep=,\n")
                writer = _csv.writer(output)
                writer.writerow(['Timestamp','Voltage (V)','Current (A)','Power Active (W)',
                                 'Frequency (Hz)','Power Factor','Konsumsi Energy (kWh)'])
                for ts in sorted_ts:
                    r = rows_by_ts[ts]
                    kwh_raw = r.get('energy_kwh')
                    kwh = round(float(kwh_raw) - (first_kwh or 0), 4) if kwh_raw is not None else ''
                    writer.writerow([ts,
                        round(r.get('voltage',  0) or 0, 2),
                        round(r.get('current',  0) or 0, 3),
                        round(r.get('power',    0) or 0, 1),
                        round(r.get('frequency',0) or 0, 2),
                        round(r.get('power_factor', 0) or 0, 3),
                        kwh,
                    ])
                csv_bytes = output.getvalue().encode('utf-8-sig')
                fname = f"{dtype}_{from_dt}_sd_{to_dt}.csv"
                return Response(csv_bytes, mimetype='text/csv',
                                headers={'Content-Disposition': f'attachment; filename="{fname}"'})
            except Exception as e2:
                return jsonify({'error': f'MySQL: {influx_err} | InfluxDB: {e2}'}), 500

    # ── Temp / Lux / Occupancy: remain from InfluxDB ──────────────────────────
    wib = timezone(_td(hours=7))
    try:
        start_utc = datetime.strptime(from_dt, '%Y-%m-%d').replace(tzinfo=wib).astimezone(timezone.utc)
        end_utc   = datetime.strptime(to_dt, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=wib).astimezone(timezone.utc)
    except ValueError:
        return jsonify({'error': 'Invalid date'}), 400
    start_iso = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_iso   = end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Human-readable column header labels per field
    _FIELD_LABELS = {
        # Sensor readings
        'temperature':  'Temp Ruangan (°C)',
        'humidity':     'Kelembapan Ruangan (%)',
        'heat_index':   'Heat Index (°C)',
        # Multi-sensor
        'temp1': 'Sensor 1 – Suhu (°C)',  'hum1': 'Sensor 1 – RH (%)',
        'temp2': 'Sensor 2 – Suhu (°C)',  'hum2': 'Sensor 2 – RH (%)',
        'temp3': 'Sensor 3 – Suhu (°C)',  'hum3': 'Sensor 3 – RH (%)',
        # AC status
        'ac_temp':     'Set Temperature AC (°C)',
        'fan_speed':   'Fan Speed (1=Low 2=Med 3=High 4=Turbo)',
        'ac_fan_mode': 'Mode AC (COOL/DRY/FAN/AUTO)',
        'set_rh':      'Set RH (%)',
        'ac_state':    'Status Kontrol AC (ON/OFF)',
        'mode':        'Mode Kontrol (ADAPTIVE/MANUAL)',
        # Lamp
        'lux1': 'Lux Sensor 1', 'lux2': 'Lux Sensor 2', 'lux3': 'Lux Sensor 3',
        'lux_avg': 'Lux Average', 'brightness1': 'Brightness 1 (%)', 'brightness2': 'Brightness 2 (%)',
        # Occupancy
        'person_count': 'Person Count', 'confidence': 'Confidence',
    }

    TYPE_MAP = {
        'temp':      ('ac_sensor',        [
                         'temperature', 'humidity', 'heat_index',
                         'temp1', 'hum1', 'temp2', 'hum2', 'temp3', 'hum3',
                         'ac_temp', 'fan_speed', 'ac_fan_mode', 'set_rh', 'ac_state', 'mode',
                     ], None, None),
        'lux':       ('lamp_sensor',      ['lux1','lux2','lux3','lux_avg','brightness1','brightness2','mode'], None, None),
        'occupancy': ('camera_detection', ['person_count','confidence'],                               None, None),
    }
    measurement, fields, tag_key, tag_val = TYPE_MAP[dtype]

    try:
        _, _, query_api = _get_influx_client()
        rows_by_ts = {}
        for field in fields:
            tag_filter = f'|> filter(fn: (r) => r["{tag_key}"] == "{tag_val}")\n' if tag_key else ''
            if dtype == 'occupancy':
                agg_fn = 'max' if field == 'person_count' else 'mean'
                q = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start_iso}, stop: {end_iso})
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => r["_field"] == "{field}")
  {tag_filter}  |> aggregateWindow(every: 1h, fn: {agg_fn}, createEmpty: false)
  |> yield(name: "agg")
'''
            else:
                q = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start_iso}, stop: {end_iso})
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => r["_field"] == "{field}")
  {tag_filter}  |> aggregateWindow(every: 1m, fn: last, createEmpty: false)
  |> yield(name: "raw")
'''
            for table in query_api.query(query=q):
                for record in table.records:
                    t_wib  = record.get_time().astimezone(wib)
                    ts_str = t_wib.strftime('%Y-%m-%d %H:%M:%S')
                    rows_by_ts.setdefault(ts_str, {'ts': ts_str})
                    rows_by_ts[ts_str][field] = record.get_value()

        if not rows_by_ts:
            return jsonify({'error': 'No data found for the selected date range'}), 404

        sorted_ts = sorted(rows_by_ts.keys())
        output    = io.StringIO()
        output.write("sep=,\n")
        writer    = _csv.writer(output)
        # Use human-readable headers
        header_row = ['Timestamp'] + [_FIELD_LABELS.get(f, f) for f in fields]
        writer.writerow(header_row)
        for ts_str in sorted_ts:
            row_d = rows_by_ts[ts_str]
            
            # Default missing 'mode' for backward compatibility
            if 'mode' in fields and 'mode' not in row_d:
                row_d['mode'] = 'ADAPTIVE'
                
            row = [ts_str] + [
                round(row_d.get(f, ''), 4) if isinstance(row_d.get(f), float) else row_d.get(f, '')
                for f in fields
            ]
            
            # Hanya sertakan baris yang datanya lengkap (tidak ada kolom kosong)
            if '' in row:
                continue
                
            writer.writerow(row)

        csv_bytes = output.getvalue().encode('utf-8-sig')
        fname = f"{dtype}_{from_dt}_sd_{to_dt}.csv"
        return Response(
            csv_bytes,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{fname}"'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== GENERIC CHART API ====================
# Allowed lists prevent InfluxDB injection
_CHART_ALLOWED_MEAS = {'ac_sensor', 'lamp_sensor', 'camera_detection', 'energy_monitor'}
_CHART_ALLOWED_FIELDS = {
    'temperature', 'humidity', 'heat_index', 'ac_temp', 'fan_speed', 'set_rh',
    'lux', 'lux_avg', 'lux1', 'lux2', 'lux3',
    'brightness', 'brightness_avg', 'brightness1', 'brightness2',
    'person_count', 'confidence',
    'power', 'voltage', 'current', 'energy_kwh',
    'temp1', 'hum1', 'temp2', 'hum2', 'temp3', 'hum3',
}

@app.route('/api/chart/<measurement>/<field>/<int:hours>')
def chart_data_route(measurement, field, hours):
    if measurement not in _CHART_ALLOWED_MEAS:
        return jsonify({'error': 'Invalid measurement'}), 400
    if field not in _CHART_ALLOWED_FIELDS:
        return jsonify({'error': 'Invalid field'}), 400
    if hours not in (1, 6, 24, 48, 72, 168):
        return jsonify({'error': 'Invalid hours'}), 400
    return jsonify(get_influx_data(measurement, field, hours))

@app.route('/api/chart/ac_power/<int:hours>')
def chart_ac_power(hours):
    """AC power from energy_monitor filtered by device=esp32_ac"""
    if hours not in (1, 6, 24, 48, 72, 168):
        return jsonify({'error': 'Invalid hours'}), 400
    return jsonify(get_influx_data('energy_monitor', 'power', hours, device_tag='esp32_ac'))

@app.route('/api/chart/lamp_power/<int:hours>')
def chart_lamp_power(hours):
    """Lamp power from energy_monitor filtered by device=esp32_lamp"""
    if hours not in (1, 6, 24, 48, 72, 168):
        return jsonify({'error': 'Invalid hours'}), 400
    return jsonify(get_influx_data('energy_monitor', 'power', hours, device_tag='esp32_lamp'))

@app.route('/api/energy/phase', methods=['GET', 'POST'])
def energy_phase_api():
    """Get or set energy monitoring phase (before/after adaptive AC)"""
    global energy_phase
    if request.method == 'POST':
        new_phase = request.json.get('phase', '').lower()
        if new_phase not in ('before', 'after', 'idle'):
            return jsonify({'error': 'Phase must be before, after, or idle'}), 400
        energy_phase = new_phase
        return jsonify({'phase': energy_phase})
    return jsonify({'phase': energy_phase})

@app.route('/api/energy/record', methods=['GET', 'POST'])
def energy_record_api():
    """Start/stop recording for before/after energy comparison phases"""
    global energy_phase, energy_recording
    if request.method == 'POST':
        body = request.json or {}
        phase = body.get('phase', '').lower()
        action = body.get('action', '').lower()
        if phase not in ('before', 'after'):
            return jsonify({'error': 'Phase must be before or after'}), 400
        if action not in ('start', 'stop'):
            return jsonify({'error': 'Action must be start or stop'}), 400

        with _recording_lock:
            if action == 'start':
                other = 'after' if phase == 'before' else 'before'
                if energy_recording[other]['active']:
                    energy_recording[other]['active'] = False
                    energy_recording[other]['end'] = datetime.utcnow().isoformat()
                energy_recording[phase]['active'] = True
                energy_recording[phase]['start'] = datetime.utcnow().isoformat()
                energy_recording[phase]['end'] = None
                energy_phase = phase
            else:
                energy_recording[phase]['active'] = False
                energy_recording[phase]['end'] = datetime.utcnow().isoformat()
                energy_phase = 'idle'
            snapshot_rec   = {k: dict(v) for k, v in energy_recording.items()}
            snapshot_phase = energy_phase

        save_energy_recording()
        socketio.emit('energy_recording', {'recording': snapshot_rec, 'phase': snapshot_phase})
        return jsonify({'recording': snapshot_rec, 'phase': snapshot_phase})

    with _recording_lock:
        return jsonify({'recording': {k: dict(v) for k, v in energy_recording.items()}, 'phase': energy_phase})

@app.route('/api/energy/compare')
def energy_compare():
    """Compare energy data between recorded before and after adaptive AC periods"""
    field = request.args.get('field', 'power')
    range_param = request.args.get('range', 'all')  # all, 7d, 30d

    allowed_fields = ['voltage', 'current', 'power', 'energy_kwh', 'frequency', 'power_factor']
    if field not in allowed_fields:
        return jsonify({'error': f'Invalid field. Use: {", ".join(allowed_fields)}'}), 400

    try:
        _, _, query_api = _get_influx_client()

        results = {}
        for phase in ['before', 'after']:
            rec = energy_recording[phase]
            if not rec['start']:
                results[phase] = []
                continue

            start_dt = datetime.fromisoformat(rec['start'])
            if rec['end']:
                end_dt = datetime.fromisoformat(rec['end'])
            elif rec['active']:
                end_dt = datetime.utcnow()
            else:
                results[phase] = []
                continue

            # Apply range filter: clip start_dt so only last N days of data shown
            if range_param == '7d':
                clipped = end_dt - timedelta(days=7)
                if clipped > start_dt:
                    start_dt = clipped
            elif range_param == '30d':
                clipped = end_dt - timedelta(days=30)
                if clipped > start_dt:
                    start_dt = clipped

            dur_h = (end_dt - start_dt).total_seconds() / 3600
            if dur_h <= 1:
                window = '30s'
            elif dur_h <= 24:
                window = '10m'
            elif dur_h <= 168:
                window = '1h'
            else:
                window = '6h'

            start_iso = start_dt.isoformat() + 'Z'
            end_iso = end_dt.isoformat() + 'Z'

            query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: {start_iso}, stop: {end_iso})
              |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> filter(fn: (r) => r["phase"] == "{phase}")
              |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
              |> yield(name: "mean")
            '''

            result = query_api.query(query=query)
            data_points = []
            for table in result:
                for record in table.records:
                    rec_time = record.get_time().replace(tzinfo=None)
                    offset_h = (rec_time - start_dt).total_seconds() / 3600
                    if dur_h <= 1:
                        label = f"{int(offset_h * 60)}m"
                    elif dur_h <= 24:
                        label = f"{int(offset_h)}:{int((offset_h % 1) * 60):02d}"
                    elif dur_h <= 168:
                        day = int(offset_h / 24) + 1
                        hr = int(offset_h % 24)
                        label = f"Day {day} {hr:02d}:00"
                    else:
                        day = int(offset_h / 24) + 1
                        label = f"Day {day}"
                    data_points.append({
                        'offset': round(offset_h, 2),
                        'label': label,
                        'value': round(float(record.get_value()), 2)
                    })
            results[phase] = data_points

        # Fallback to runtime buffer when InfluxDB has no data yet
        if not results['before'] and not results['after']:
            for phase in ['before', 'after']:
                phase_data = [r for r in energy_runtime_history if r.get('phase') == phase]
                if phase_data:
                    end_ts = phase_data[-1]['ts']
                    start_ts = phase_data[0]['ts']
                    # Apply range filter to runtime buffer
                    if range_param == '7d':
                        cutoff = end_ts - timedelta(days=7)
                        phase_data = [r for r in phase_data if r['ts'] >= cutoff]
                    elif range_param == '30d':
                        cutoff = end_ts - timedelta(days=30)
                        phase_data = [r for r in phase_data if r['ts'] >= cutoff]
                    if not phase_data:
                        results[phase] = []
                        continue
                    start_ts = phase_data[0]['ts']
                    dur_h = max(0.01, (phase_data[-1]['ts'] - start_ts).total_seconds() / 3600)
                    step = max(1, len(phase_data) // 120)
                    sampled = phase_data[::step]
                    pts = []
                    for r in sampled:
                        off = (r['ts'] - start_ts).total_seconds() / 3600
                        if dur_h <= 1:
                            label = f"{int(off * 60)}m"
                        elif dur_h <= 24:
                            label = f"{int(off)}:{int((off % 1) * 60):02d}"
                        else:
                            day = int(off / 24) + 1
                            hr = int(off % 24)
                            label = f"Day {day} {hr:02d}:00"
                        pts.append({
                            'offset': round(off, 2),
                            'label': label,
                            'value': round(float(r.get(field, 0)), 2)
                        })
                    results[phase] = pts

        # Calculate averages for summary
        before_vals = [d['value'] for d in results['before']] if results['before'] else []
        after_vals = [d['value'] for d in results['after']] if results['after'] else []
        avg_before = round(sum(before_vals) / len(before_vals), 2) if before_vals else 0
        avg_after = round(sum(after_vals) / len(after_vals), 2) if after_vals else 0
        savings_pct = round((1 - avg_after / avg_before) * 100, 1) if avg_before > 0 else 0

        return jsonify({
            'field': field,
            'before': results['before'],
            'after': results['after'],
            'recording': energy_recording,
            'summary': {
                'avg_before': avg_before,
                'avg_after': avg_after,
                'savings_percent': savings_pct
            }
        })

    except Exception as e:
        print(f"[ERROR] Energy compare error: {e}")
        return jsonify({'error': str(e), 'before': [], 'after': [], 'summary': {}}), 500

@app.route('/api/energy/export-csv')
@admin_required
def energy_export_csv():
    """Export energy data as CSV — from realtime ring buffer or PHP proxy.
    ?device=ac|lamp|all   (default: all)
    ?source=realtime|php  (default: php — fetch from PHP proxy)
    """
    import io, csv as csv_mod

    device = request.args.get('device', 'all')   # ac, lamp, all
    source = request.args.get('source', 'php')   # php or realtime

    rows = []

    if source == 'realtime':
        # Fetch from _mysqlBuf ring buffer in JS — cannot directly.
        # Use energy_runtime_history in Python.
        with _recording_lock:
            hist = list(energy_runtime_history)
        for rec in hist:
            rows.append({
                'timestamp': rec.get('ts', '').strftime('%Y-%m-%d %H:%M:%S') if hasattr(rec.get('ts',''), 'strftime') else str(rec.get('ts','')),
                'device':    'AC',
                'voltage':  rec.get('voltage', 0),
                'current':      rec.get('current', 0),
                'active_power':rec.get('power', 0),
                'energy_kwh':rec.get('energy_kwh', 0),
                'frequency': rec.get('frequency', 0),
                'pf':        rec.get('power_factor', 0),
                'reactive_power':   '',
                'semu':      '',
            })
    else:
        # Fetch directly from PHP proxy (fresh data)
        import urllib.request as _ureq, json as _json
        PHP_URL = 'https://iotlab-uns.com/api_energy.php?key=iotlab_smartroom_2024'
        try:
            req = _ureq.Request(PHP_URL, headers={'User-Agent': 'SmartRoom-Export/1.0'})
            with _ureq.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read().decode('utf-8'))
        except Exception as ex:
            return jsonify({'error': f'Failed to fetch PHP data: {ex}'}), 502

        if 'error' in data:
            return jsonify({'error': data['error']}), 500

        def _row(d, label):
            ap = float(d.get('apparent_power') or 0)
            p  = float(d.get('active_power') or 0)
            pf = round(p / ap, 3) if ap > 0.001 else 0.0
            return {
                'timestamp':  str(d.get('created_at', '')),
                'device':     label,
                'voltage':   float(d.get('voltage')   or 0),
                'current':       float(d.get('current')       or 0),
                'active_power': p,
                'energy_kwh': float(d.get('total_energy') or 0) / 1000.0,
                'frequency':  float(d.get('frequency')  or 0),
                'pf':         pf,
                'reactive_power':    float(d.get('reactive_power') or 0),
                'semu':       ap,
            }

        if device in ('ac', 'all') and data.get('ac'):
            rows.append(_row(data['ac'], 'AC'))
        if device in ('lamp', 'all') and data.get('lamp'):
            rows.append(_row(data['lamp'], 'Lamp'))

    # Buat CSV — human-readable headers, 1 data = 1 kolom, BOM for Excel
    output = io.StringIO()
    output.write("sep=,\n")
    fields = ['timestamp','device','voltage','current','active_power','reactive_power','semu','pf','frequency','energy_kwh']
    header_labels = {
        'timestamp': 'Timestamp',
        'device': 'Device',
        'voltage': 'Voltage (V)',
        'current': 'Current (A)',
        'active_power': 'Active Power (W)',
        'reactive_power': 'Reactive Power (VAR)',
        'semu': 'Apparent Power (VA)',
        'pf': 'Power Factor',
        'frequency': 'Frequency (Hz)',
        'energy_kwh': 'Energy (kWh)',
    }
    writer = csv_mod.writer(output)
    writer.writerow([header_labels.get(f, f) for f in fields])
    for r in rows:
        writer.writerow([r.get(f, '') for f in fields])
    csv_bytes = output.getvalue().encode('utf-8-sig')  # BOM so Excel reads correctly

    from flask import Response
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Response(
        csv_bytes,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=energy_{device}_{now_str}.csv'}
    )

@app.route('/api/energy/history')
def energy_history():
    """Get energy history from InfluxDB — supports 1h to 30 days.
    device=ac  -> MySQL AC data (tag device='mysql_ac')
    device=lamp -> MySQL Lamp data (tag device='mysql_lamp')
    device omitted -> AC data (backward compat)
    """
    period = request.args.get('period', '24h')
    field = request.args.get('field')
    device = request.args.get('device', 'ac')  # 'ac' or 'lamp'

    period_map = {
        '1h':   {'range': '-1h',   'window': '1m'},
        '6h':   {'range': '-6h',   'window': '5m'},
        '24h':  {'range': '-24h',  'window': '1h'},    # 24 titik (per jam)
        '7d':   {'range': '-7d',   'window': '1d'},    # 7 titik (per hari)
        '30d':  {'range': '-30d',  'window': '1d'},    # 30 titik (per hari)
        '12mo': {'range': '-12mo', 'window': '1mo'},   # 12 titik (per bulan)
        '5y':   {'range': '-5y',   'window': '1y'},    # yearly: last 5 years
    }

    if period not in period_map:
        return jsonify({'error': 'Invalid period. Use: 1h, 6h, 24h, 7d, 30d, 12mo, 5y'}), 400

    allowed_fields = ['voltage', 'current', 'power', 'energy_kwh', 'frequency', 'power_factor']
    if field and field not in allowed_fields:
        return jsonify({'error': f'Invalid field. Use: {", ".join(allowed_fields)}'}), 400

    # Map device to InfluxDB tag value
    device_tag = {'ac': 'mysql_ac', 'lamp': 'mysql_lamp', 'outlet': 'mysql_outlet'}.get(device, 'mysql_ac')
    p = period_map[period]

    try:
        _, _, query_api = _get_influx_client()

        if period in ('1h', '6h'):
            time_format = '%H:%M'
        elif period == '24h':
            time_format = '%H:00'   # jam bulat: 00:00, 01:00, ..., 23:00
        elif period == '7d':
            time_format = '%a %d/%m'   # Mon 02/06, Tue 03/06, ...
        elif period == '30d':
            time_format = '%d/%m'      # 05/05, 06/05, ..., 03/06
        elif period == '12mo':
            time_format = '%b %Y'   # Jan 2026, Feb 2026, ...
        else:  # 5y
            time_format = '%Y'      # 2024, 2025, 2026, ...

        def _add_months(dt, months):
            month_index = dt.month - 1 + months
            year = dt.year + month_index // 12
            month = month_index % 12 + 1
            return dt.replace(year=year, month=month, day=1)

        def _to_flux_time(dt):
            return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

        def _fixed_bucket_spec(extra_baseline=False):
            """Return fixed local bucket starts so charts always show the requested count."""
            now = datetime.now()
            if period == '24h':
                end = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                starts = [end - timedelta(hours=i) for i in range(24, 0, -1)]
                prev_start = starts[0] - timedelta(hours=1)
                window = '1h'
            elif period == '7d':
                end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                starts = [end - timedelta(days=i) for i in range(7, 0, -1)]
                prev_start = starts[0] - timedelta(days=1)
                window = '1d'
            elif period == '30d':
                end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                starts = [end - timedelta(days=i) for i in range(30, 0, -1)]
                prev_start = starts[0] - timedelta(days=1)
                window = '1d'
            elif period == '12mo':
                this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                end = _add_months(this_month, 1)
                starts = [_add_months(end, -i) for i in range(12, 0, -1)]
                prev_start = _add_months(starts[0], -1)
                window = '1mo'
            else:
                return None

            if extra_baseline:
                starts = [prev_start] + starts
            return {'starts': starts, 'start': starts[0], 'end': end, 'window': window}

        def _bucket_key(dt):
            if period == '24h':
                return dt.replace(minute=0, second=0, microsecond=0)
            if period in ('7d', '30d'):
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
            if period == '12mo':
                return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return dt

        def _runtime_bucket_values(field_name, spec):
            if device == 'lamp':
                buf = lamp_runtime_history
            elif device == 'outlet':
                buf = outlet_runtime_history
            else:
                buf = energy_runtime_history
            rows = [r for r in buf if r.get('ts') and spec['start'] <= r['ts'] < spec['end']]
            if not rows:
                return {}

            grouped = {}
            for row in rows:
                key = _bucket_key(row['ts'])
                grouped.setdefault(key, []).append(row)

            values = {}
            for key, items in grouped.items():
                if field_name == 'energy_kwh':
                    latest = max(items, key=lambda r: r.get('ts'))
                    values[key] = float(latest.get(field_name, 0) or 0)
                else:
                    vals = [float(r.get(field_name, 0) or 0) for r in items]
                    values[key] = sum(vals) / len(vals) if vals else 0.0
            return values

        def _points_from_fixed_buckets(field_name, values_by_bucket, spec):
            dec = 5 if field_name == 'energy_kwh' else 2
            starts = spec['starts']

            if field_name == 'energy_kwh':
                first_available = next((values_by_bucket[s] for s in starts if s in values_by_bucket), 0.0)
                last_value = first_available
                points = []
                for start in starts:
                    if start in values_by_bucket:
                        last_value = values_by_bucket[start]
                    val = float(last_value)
                    val = val / 1000.0
                    points.append({'time': start.strftime(time_format), 'value': round(val, dec)})
                return points

            return [
                {'time': start.strftime(time_format), 'value': round(float(values_by_bucket.get(start, 0.0)), dec)}
                for start in starts
            ]

        def query_field_points(field_name):
            # energy_kwh is cumulative — use last() per window so final value
            # of each interval is taken (not average), so JS can diff between points
            agg_fn = 'last' if field_name == 'energy_kwh' else 'mean'
            fixed_spec = _fixed_bucket_spec(extra_baseline=(field_name == 'energy_kwh'))
            if fixed_spec:
                query = f'''
                from(bucket: "{INFLUX_BUCKET}")
                  |> range(start: time(v: "{_to_flux_time(fixed_spec['start'])}"), stop: time(v: "{_to_flux_time(fixed_spec['end'])}"))
                  |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
                  |> filter(fn: (r) => r["_field"] == "{field_name}")
                  |> filter(fn: (r) => r["device"] == "{device_tag}")
                  |> aggregateWindow(every: {fixed_spec['window']}, fn: {agg_fn}, createEmpty: false, timeSrc: "_start")
                  |> yield(name: "{agg_fn}")
                '''
                result = query_api.query(query=query)
                values_by_bucket = {}
                expected_starts = set(fixed_spec['starts'])
                for table in result:
                    for record in table.records:
                        rec_time = record.get_time().astimezone().replace(tzinfo=None)
                        key = _bucket_key(rec_time)
                        if key in expected_starts:
                            values_by_bucket[key] = float(record.get_value())

                if not values_by_bucket:
                    values_by_bucket = _runtime_bucket_values(field_name, fixed_spec)

                return _points_from_fixed_buckets(field_name, values_by_bucket, fixed_spec)
            query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: {p['range']})
              |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
              |> filter(fn: (r) => r["_field"] == "{field_name}")
              |> filter(fn: (r) => r["device"] == "{device_tag}")
              |> aggregateWindow(every: {p['window']}, fn: {agg_fn}, createEmpty: false)
              |> yield(name: "{agg_fn}")
            '''
            result = query_api.query(query=query)
            points = []
            for table in result:
                for record in table.records:
                    dec = 5 if field_name == 'energy_kwh' else 2
                    val = float(record.get_value())
                    if field_name == 'energy_kwh':
                        val = val / 1000.0
                    points.append({
                        'time': record.get_time().astimezone().strftime(time_format),
                        'value': round(val, dec)
                    })
            return points

        if field:
            data_points = query_field_points(field)
            # Fallback to runtime buffer when InfluxDB has no data yet
            if not data_points:
                now = datetime.now()
                lookback = {'1h': timedelta(hours=1), '6h': timedelta(hours=6), '24h': timedelta(hours=24), '7d': timedelta(days=7), '30d': timedelta(days=30), '12mo': timedelta(days=365), '5y': timedelta(days=1825)}.get(period, timedelta(days=1))
                cutoff = now - lookback
                if device == 'lamp':
                    buf = lamp_runtime_history
                elif device == 'outlet':
                    buf = outlet_runtime_history
                else:
                    buf = energy_runtime_history
                runtime = [r for r in buf if r.get('ts') and r['ts'] >= cutoff]
                if runtime:
                    step = max(1, len(runtime) // 120)
                    data_points = [{'time': r['ts'].strftime(time_format), 'value': round(float(r.get(field, 0)), 2)} for r in runtime[::step]]
            return jsonify({'period': period, 'field': field, 'device': device, 'data': data_points})

        power_points = query_field_points('power')
        voltage_points = query_field_points('voltage')
        kwh_points = query_field_points('energy_kwh')

        if not power_points and not voltage_points and not kwh_points:
            now = datetime.now()
            lookback = {'1h': timedelta(hours=1), '6h': timedelta(hours=6), '24h': timedelta(hours=24), '7d': timedelta(days=7), '30d': timedelta(days=30), '12mo': timedelta(days=365), '5y': timedelta(days=1825)}.get(period, timedelta(days=1))
            cutoff = now - lookback
            if device == 'lamp':
                buf = lamp_runtime_history
            elif device == 'outlet':
                buf = outlet_runtime_history
            else:
                buf = energy_runtime_history
            runtime = [r for r in buf if r.get('ts') and r['ts'] >= cutoff]
            if runtime:
                step = max(1, len(runtime) // 120)
                sampled = runtime[::step]
                power_points   = [{'time': r['ts'].strftime(time_format), 'value': round(float(r.get('power', 0)), 2)} for r in sampled]
                voltage_points = [{'time': r['ts'].strftime(time_format), 'value': round(float(r.get('voltage', 0)), 2)} for r in sampled]
                kwh_points     = [{'time': r['ts'].strftime(time_format), 'value': round(float(r.get('energy_kwh', 0)), 3)} for r in sampled]

        return jsonify({'period': period, 'device': device, 'power': power_points, 'voltage': voltage_points, 'energy_kwh': kwh_points})

    except Exception as e:
        print(f"[ERROR] Energy history query error: {e}")
        if field:
            return jsonify({'error': str(e), 'period': period, 'field': field, 'device': device, 'data': []}), 500
        return jsonify({'error': str(e), 'period': period, 'device': device, 'power': [], 'voltage': [], 'energy_kwh': []}), 500

@app.route('/api/ml/status')
def ml_status():
    """Return current ML optimization state for the ML page"""
    return jsonify({
        'ga_fitness':       mqtt_data['system'].get('ga_fitness', 0),
        'pso_fitness':      mqtt_data['system'].get('pso_fitness', 0),
        'ga_temp':          mqtt_data['system'].get('ga_temp', 0),
        'ga_fan':           mqtt_data['system'].get('ga_fan', 0),
        'ga_mode':          mqtt_data['system'].get('ga_mode', 'COOL'),
        'pso_brightness':   mqtt_data['system'].get('pso_brightness', 0),
        'pso_brightness1':  mqtt_data['system'].get('pso_brightness1', 0),
        'pso_brightness2':  mqtt_data['system'].get('pso_brightness2', 0),
        'pso_pwm1':         mqtt_data['system'].get('pso_pwm1', 0),
        'pso_pwm2':         mqtt_data['system'].get('pso_pwm2', 0),
        'optimization_runs': mqtt_data['system'].get('optimization_runs', 0),
        'ga_history':       mqtt_data['system'].get('ga_history', []),
        'pso_history':      mqtt_data['system'].get('pso_history', []),
        # iteration_log for detailed PSO chart (PWM1, PWM2, Lux per iteration)
        'pso_iteration_log': last_opt_results['pso'].get('iteration_log', []),
        'algo_config':      mqtt_data['system'].get('algo_config', 'ga_pso'),
        'ac_algo':          mqtt_data['system'].get('ac_algo', 'ga'),
        'lamp_algo':        mqtt_data['system'].get('lamp_algo', 'pso'),
    })

@app.route('/api/ga/export-csv')
@admin_required
def ga_export_csv():
    """Export GA results report (AC Optimization) in full CSV format."""
    import io, csv as csv_mod
    ga = last_opt_results.get('ga', {})
    if not ga or ga.get('fitness', 0) == 0:
        return jsonify({'error': 'No GA results yet. Run GA first.'}), 404

    params    = ga.get('params', ga_params)
    run_time  = ga.get('run_time', datetime.now().strftime('%d %b %Y %H:%M'))
    temp_opt  = ga.get('temp', 0)
    fan_opt   = ga.get('fan', 0)
    mode_opt  = ga.get('mode', 'COOL')
    init_fit  = ga.get('initial_fitness', '-')
    final_fit = ga.get('final_fitness', round(ga.get('fitness', 0), 2))
    bf        = ga.get('brute_force', {})
    snap      = ga.get('sensor_snapshot', {})
    history   = ga.get('stats', [])

    # Fan label
    fan_label_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Turbo'}
    fan_label = fan_label_map.get(int(fan_opt), str(fan_opt))

    # Improvement %
    if isinstance(init_fit, (int, float)) and init_fit > 0:
        improvement = round((final_fit - init_fit) / init_fit * 100, 1)
    else:
        improvement = '-'

    buf = io.StringIO()
    buf.write("sep=,\n")
    wr  = csv_mod.writer(buf)

    # ── Header ──────────────────────────────────────────────────────────────
    ac_algo_name = 'Genetic Algorithm' if mqtt_data['system'].get('ac_algo', 'ga') == 'ga' else 'Particle Swarm Optimization'
    wr.writerow([f'=== AC OPTIMIZATION RESULT ({ac_algo_name.upper()}) - SMART ROOM ==='])
    wr.writerow([])
    wr.writerow(['Run Date', run_time])
    wr.writerow(['Algorithm Used', ac_algo_name])
    wr.writerow([])

    # ── Parameter ────────────────────────────────────────────────────────
    wr.writerow([f'--- PARAMETER {ac_algo_name.upper()} ---'])
    if mqtt_data['system'].get('ac_algo', 'ga') == 'ga':
        wr.writerow(['Ukuran Populasi',   params.get('population_size', '-')])
        wr.writerow(['Generation Count',  params.get('generations', '-')])
        wr.writerow(['Mutation Rate',     params.get('mutation_rate', '-')])
        wr.writerow(['Crossover Rate',    params.get('crossover_rate', '-')])
        wr.writerow(['Elitism Ratio',     params.get('elitism_ratio', '-')])
    else:
        wr.writerow(['Swarm Size',        params.get('swarm_size', '-')])
        wr.writerow(['Iterations',        params.get('iterations', '-')])
        wr.writerow(['Inertia (w)',       params.get('w', '-')])
        wr.writerow(['Cognitive (c1)',    params.get('c1', '-')])
        wr.writerow(['Social (c2)',       params.get('c2', '-')])
    wr.writerow([])

    # ── Room Condition on Run ─────────────────────────────────────────────
    wr.writerow(['--- ROOM CONDITIONS DURING OPTIMIZATION ---'])
    wr.writerow(['Temperature Room',   f"{snap.get('temp_room', '-')} °C"])
    wr.writerow(['Kelembapan',     f"{snap.get('humidity', '-')} %"])
    wr.writerow(['Person Detected', 'Yes' if snap.get('person_detected') else 'No'])
    wr.writerow(['Person Count',  snap.get('person_count', 0)])
    wr.writerow(['Power AC (real)', f"{snap.get('actual_watt', '-')} W"])
    wr.writerow([])

    # ── Hasil Optimal ────────────────────────────────────────────────────────
    wr.writerow(['--- HASIL OPTIMASI AC ---'])
    wr.writerow(['Temperature AC Optimal',        f"{temp_opt} °C"])
    wr.writerow(['Speed Fan Optimal',   f"{fan_opt} ({fan_label})"])
    wr.writerow(['Mode AC Optimal',         mode_opt])
    wr.writerow([])

    # ── Performa GA ─────────────────────────────────────────────────────────
    wr.writerow(['--- PERFORMA ALGORITMA ---'])
    wr.writerow(['Fitness Awal (Gen 1)',     init_fit])
    wr.writerow(['Final Fitness (Best)',  final_fit])
    wr.writerow(['Peningkatan Fitness',      f"{improvement}%" if improvement != '-' else '-'])
    if bf:
        wr.writerow(['Brute-Force Validation',
                     f"Temp={bf.get('solution', ['-','-','-'])[0]}°C "
                     f"Fan={bf.get('solution', ['-','-','-'])[1]} "
                     f"Mode={AC_MODE_NAMES.get(bf.get('solution', [None,None,0])[2], 'COOL')} "
                     f"Fitness={round(bf.get('fitness', 0), 2)}"])
    wr.writerow([])

    # ── History Konvergensi ──────────────────────────────────────────────────
    if history:
        wr.writerow(['=== FITNESS CONVERGENCE HISTORY ==='])
        wr.writerow(['Generation', 'Fitness Best', 'Delta'])
        prev = None
        for idx, val in enumerate(history, 1):
            v = round(val, 4)
            delta = round(v - prev, 4) if prev is not None else '-'
            wr.writerow([idx, v, delta])
            prev = v

    output = buf.getvalue().encode('utf-8-sig')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=ga_report_{ts}.csv'}
    )

@app.route('/api/pso/export-csv')
@admin_required
def pso_export_csv():
    """Export PSO results per iteration as CSV table."""
    import io, csv as csv_mod
    pso = last_opt_results.get('pso', {})
    iteration_log = pso.get('iteration_log', [])

    if not pso or not iteration_log:
        return jsonify({'error': 'No PSO iteration data yet. Run PSO first.'}), 404

    buf = io.StringIO()
    buf.write("sep=,\n")
    wr  = csv_mod.writer(buf)

    # Header
    lamp_algo_name = 'Particle Swarm Optimization' if mqtt_data['system'].get('lamp_algo', 'pso') == 'pso' else 'Genetic Algorithm'
    wr.writerow([f'=== LAMP OPTIMIZATION LOG ({lamp_algo_name.upper()}) ==='])
    wr.writerow(['Algorithm Used:', lamp_algo_name])
    wr.writerow([])

    # Parameters
    params = pso.get('params', {})
    wr.writerow([f'--- PARAMETER {lamp_algo_name.upper()} ---'])
    if mqtt_data['system'].get('lamp_algo', 'pso') == 'pso':
        wr.writerow(['Swarm Size',        params.get('swarm_size', '-')])
        wr.writerow(['Iterations',        params.get('iterations', '-')])
        wr.writerow(['Inertia (w)',       params.get('w', '-')])
        wr.writerow(['Cognitive (c1)',    params.get('c1', '-')])
        wr.writerow(['Social (c2)',       params.get('c2', '-')])
    else:
        wr.writerow(['Ukuran Populasi',   params.get('population_size', '-')])
        wr.writerow(['Generation Count',  params.get('generations', '-')])
        wr.writerow(['Mutation Rate',     params.get('mutation_rate', '-')])
        wr.writerow(['Crossover Rate',    params.get('crossover_rate', '-')])
        wr.writerow(['Elitism Ratio',     params.get('elitism_ratio', '-')])
    wr.writerow([])
    wr.writerow(['Iteration', 'PWM 1', 'PWM 2', 'Lux 1', 'Lux 2', 'Lux 3',
                 'Average Lux', 'Fitness'])

    # Baris per iterasi
    for entry in iteration_log:
        it       = entry.get('iter',    '--')
        pwm1     = entry.get('pwm1',    '--')
        pwm2     = entry.get('pwm2',    '--')
        lux1     = entry.get('lux1',    '--')
        lux2     = entry.get('lux2',    '--')
        lux3     = entry.get('lux3',    '--')
        lux_avg  = entry.get('lux_avg', '--')
        fitness  = entry.get('fitness', '--')
        wr.writerow([it, pwm1, pwm2, lux1, lux2, lux3, lux_avg, fitness])

    output = buf.getvalue().encode('utf-8-sig')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=pso_iterations_{ts}.csv'}
    )

@app.route('/api/ml/algo', methods=['GET', 'POST'])
def ml_algo_config_api():
    """Get or set the active ML algorithm configuration."""
    global opt_algo_config
    if request.method == 'GET':
        return jsonify({
            'status': 'success',
            'config': opt_algo_config,
            'ac_algo': _get_ac_algo(),
            'lamp_algo': _get_lamp_algo(),
            'options': list(OPT_ALGO_OPTIONS)
        })
    elif request.method == 'POST':
        if session.get('role') != 'admin':
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
        data = request.json or {}
        new_config = data.get('config')
        if new_config in OPT_ALGO_OPTIONS:
            opt_algo_config = new_config
            log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Algorithm config changed to {new_config}', 'level': 'info'})
            # Broadcast the change
            socketio.emit('mqtt_update', {
                'type': 'system',
                'data': {
                    'algo_config': opt_algo_config,
                    'ac_algo': _get_ac_algo(),
                    'lamp_algo': _get_lamp_algo()
                }
            })
            return jsonify({
                'status': 'success',
                'message': f'Algorithm configuration updated to {new_config}',
                'config': opt_algo_config,
                'ac_algo': _get_ac_algo(),
                'lamp_algo': _get_lamp_algo()
            })
        else:
            return jsonify({'status': 'error', 'message': f'Invalid config. Must be one of {OPT_ALGO_OPTIONS}'}), 400

@app.route('/api/ml/run', methods=['POST'])
@admin_required
def ml_run():
    """Trigger optimization run (GA, PSO, or both) — runs embedded engine"""
    try:
        data = request.json
        algo = data.get('algorithm', 'both')
        params = data.get('params', {})
        
        # Update params if provided
        if algo == 'ga' and params:
            ga_params.update({k: v for k, v in params.items() if k in ga_params})
        elif algo == 'pso' and params:
            pso_params.update({k: v for k, v in params.items() if k in pso_params})
        elif algo == 'both' and params:
            if 'ga' in params: ga_params.update({k: v for k, v in params['ga'].items() if k in ga_params})
            if 'pso' in params: pso_params.update({k: v for k, v in params['pso'].items() if k in pso_params})
        
        # PSO: gunakan _pso_lamp_cycle() agar urutan Baca→Hitung→Send konsisten
        # with automatic cycle. GA still uses run_optimization_cycle().
        if algo == 'pso':
            t = threading.Thread(target=_pso_lamp_cycle, daemon=True)
        elif algo == 'ga':
            t = threading.Thread(target=run_optimization_cycle, args=('ga',), daemon=True)
        else:
            t = threading.Thread(target=run_optimization_cycle, args=(algo,), daemon=True)
        t.start()
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'ML Run triggered: {algo}', 'level': 'info'})
        return jsonify({'status': 'success', 'message': f'{algo} optimization triggered'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/lamp/energy/record', methods=['GET', 'POST'])
def lamp_energy_record_api():
    """Start/stop recording for lamp before/after comparison"""
    global lamp_phase, lamp_recording
    if request.method == 'POST':
        body = request.json or {}
        phase = body.get('phase', '').lower()
        action = body.get('action', '').lower()
        if phase not in ('before', 'after'):
            return jsonify({'error': 'Phase must be before or after'}), 400
        if action not in ('start', 'stop'):
            return jsonify({'error': 'Action must be start or stop'}), 400
        with _recording_lock:
            if action == 'start':
                other = 'after' if phase == 'before' else 'before'
                if lamp_recording[other]['active']:
                    lamp_recording[other]['active'] = False
                    lamp_recording[other]['end'] = datetime.utcnow().isoformat()
                lamp_recording[phase]['active'] = True
                lamp_recording[phase]['start'] = datetime.utcnow().isoformat()
                lamp_recording[phase]['end'] = None
                lamp_phase = phase
            else:
                lamp_recording[phase]['active'] = False
                lamp_recording[phase]['end'] = datetime.utcnow().isoformat()
                lamp_phase = 'idle'
            snapshot_rec   = {k: dict(v) for k, v in lamp_recording.items()}
            snapshot_phase = lamp_phase
        save_energy_recording()
        socketio.emit('lamp_recording', {'recording': snapshot_rec, 'phase': snapshot_phase})
        return jsonify({'recording': snapshot_rec, 'phase': snapshot_phase})
    with _recording_lock:
        return jsonify({'recording': {k: dict(v) for k, v in lamp_recording.items()}, 'phase': lamp_phase})

@app.route('/api/lamp/energy/compare')
def lamp_energy_compare():
    """Compare lamp estimated energy between before and after adaptive lamp periods"""
    field = request.args.get('field', 'power')
    range_param = request.args.get('range', 'all')

    allowed_fields = ['power', 'current', 'energy_kwh']
    if field not in allowed_fields:
        return jsonify({'error': f'Invalid field. Use: {", ".join(allowed_fields)}'}), 400

    try:
        _, _, query_api = _get_influx_client()

        results = {}
        for phase in ['before', 'after']:
            rec = lamp_recording[phase]
            if not rec['start']:
                results[phase] = []
                continue
            start_dt = datetime.fromisoformat(rec['start'])
            if rec['end']:
                end_dt = datetime.fromisoformat(rec['end'])
            elif rec['active']:
                end_dt = datetime.utcnow()
            else:
                results[phase] = []
                continue

            if range_param == '7d':
                clipped = end_dt - timedelta(days=7)
                if clipped > start_dt: start_dt = clipped
            elif range_param == '30d':
                clipped = end_dt - timedelta(days=30)
                if clipped > start_dt: start_dt = clipped

            dur_h = (end_dt - start_dt).total_seconds() / 3600
            window = '30s' if dur_h <= 1 else ('10m' if dur_h <= 24 else ('1h' if dur_h <= 168 else '6h'))

            query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: {start_dt.isoformat()}Z, stop: {end_dt.isoformat()}Z)
              |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> filter(fn: (r) => r["device"] == "esp32_lamp")
              |> filter(fn: (r) => r["phase"] == "{phase}")
              |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
              |> yield(name: "mean")
            '''
            result = query_api.query(query=query)
            data_points = []
            for table in result:
                for record in table.records:
                    rec_time = record.get_time().replace(tzinfo=None)
                    offset_h = (rec_time - start_dt).total_seconds() / 3600
                    if dur_h <= 1:       label = f"{int(offset_h * 60)}m"
                    elif dur_h <= 24:    label = f"{int(offset_h)}:{int((offset_h % 1) * 60):02d}"
                    elif dur_h <= 168:   label = f"Day {int(offset_h/24)+1} {int(offset_h%24):02d}:00"
                    else:                label = f"Day {int(offset_h/24)+1}"
                    data_points.append({'offset': round(offset_h, 2), 'label': label, 'value': round(float(record.get_value()), 2)})
            results[phase] = data_points

        before_vals = [d['value'] for d in results.get('before', [])]
        after_vals  = [d['value'] for d in results.get('after', [])]
        avg_before  = round(sum(before_vals) / len(before_vals), 2) if before_vals else 0
        avg_after   = round(sum(after_vals) / len(after_vals), 2)   if after_vals  else 0
        savings_pct = round((1 - avg_after / avg_before) * 100, 1)  if avg_before > 0 else 0

        return jsonify({
            'field': field,
            'before': results.get('before', []),
            'after':  results.get('after', []),
            'recording': lamp_recording,
            'summary': {'avg_before': avg_before, 'avg_after': avg_after, 'savings_percent': savings_pct}
        })
    except Exception as e:
        print(f"[ERROR] Lamp energy compare error: {e}")
        return jsonify({'error': str(e), 'before': [], 'after': [], 'summary': {}}), 500

@app.route('/api/energy/daily-summary')
def energy_daily_summary():
    """Return today's energy summary: kWh, peak power, avg power, estimated cost, runtime hours."""
    try:
        _, _, query_api = _get_influx_client()
        now_local = datetime.now()
        midnight_utc = (now_local.replace(hour=0, minute=0, second=0, microsecond=0)).strftime('%Y-%m-%dT%H:%M:%SZ')

        def _query_today(device_tag, field, agg='mean'):
            q = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: {midnight_utc})
              |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
              |> filter(fn: (r) => r["device"] == "{device_tag}")
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> {agg}()
            '''
            try:
                result = query_api.query(q)
                for table in result:
                    for rec in table.records:
                        v = rec.get_value()
                        if v is not None:
                            val = float(v)
                            if field == 'energy_kwh':
                                val = val / 1000.0
                            return round(val, 3)
            except Exception:
                pass
            return 0.0

        def _query_today_max(device_tag, field):
            q = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: {midnight_utc})
              |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
              |> filter(fn: (r) => r["device"] == "{device_tag}")
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> max()
            '''
            try:
                result = query_api.query(q)
                for table in result:
                    for rec in table.records:
                        v = rec.get_value()
                        if v is not None:
                            val = float(v)
                            if field == 'energy_kwh':
                                val = val / 1000.0
                            return round(val, 3)
            except Exception:
                pass
            return 0.0

        def _runtime_hours_from_buffer(buf, threshold_power=10.0):
            """Count hours today where power > threshold (device was 'on')."""
            today = now_local.date()
            on_count = sum(1 for r in buf if r.get('ts') and r['ts'].date() == today and r.get('power', 0) > threshold_power)
            total_today = sum(1 for r in buf if r.get('ts') and r['ts'].date() == today)
            if total_today == 0:
                return 0.0
            # estimate hours based on fraction of day elapsed so far
            elapsed_h = now_local.hour + now_local.minute / 60.0
            return round(elapsed_h * (on_count / total_today), 2)

        # AC metrics (mysql_ac)
        ac_kwh     = _query_today('mysql_ac', 'energy_kwh', 'max')  # cumulative, so max is total
        ac_power_avg = _query_today('mysql_ac', 'power', 'mean')
        ac_power_peak = _query_today_max('mysql_ac', 'power')

        # Lamp metrics (mysql_lamp)
        lamp_kwh     = _query_today('mysql_lamp', 'energy_kwh', 'max')
        lamp_power_avg = _query_today('mysql_lamp', 'power', 'mean')
        lamp_power_peak = _query_today_max('mysql_lamp', 'power')

        # Outlet metrics (mysql_outlet)
        outlet_kwh     = _query_today('mysql_outlet', 'energy_kwh', 'max')
        outlet_power_avg = _query_today('mysql_outlet', 'power', 'mean')
        outlet_power_peak = _query_today_max('mysql_outlet', 'power')

        # Runtime from ring buffers (fallback when InfluxDB has no data yet today)
        ac_runtime_h   = _runtime_hours_from_buffer(energy_runtime_history, threshold_power=50.0)
        lamp_runtime_h = _runtime_hours_from_buffer(lamp_runtime_history, threshold_power=5.0)
        outlet_runtime_h = _runtime_hours_from_buffer(outlet_runtime_history, threshold_power=5.0)

        # If InfluxDB returned 0 kWh, estimate from ring buffer
        if ac_kwh == 0 and ac_power_avg == 0:
            today = now_local.date()
            ac_vals = [r.get('power', 0) for r in energy_runtime_history if r.get('ts') and r['ts'].date() == today]
            if ac_vals:
                ac_power_avg = round(sum(ac_vals) / len(ac_vals), 1)
                ac_power_peak = round(max(ac_vals), 1)
                ac_kwh = round(ac_power_avg * ac_runtime_h / 1000.0, 4)
        if lamp_kwh == 0 and lamp_power_avg == 0:
            today = now_local.date()
            lamp_vals = [r.get('power', 0) for r in lamp_runtime_history if r.get('ts') and r['ts'].date() == today]
            if lamp_vals:
                lamp_power_avg = round(sum(lamp_vals) / len(lamp_vals), 1)
                lamp_power_peak = round(max(lamp_vals), 1)
                lamp_kwh = round(lamp_power_avg * lamp_runtime_h / 1000.0, 4)
        if outlet_kwh == 0 and outlet_power_avg == 0:
            today = now_local.date()
            outlet_vals = [r.get('power', 0) for r in outlet_runtime_history if r.get('ts') and r['ts'].date() == today]
            if outlet_vals:
                outlet_power_avg = round(sum(outlet_vals) / len(outlet_vals), 1)
                outlet_power_peak = round(max(outlet_vals), 1)
                outlet_kwh = round(outlet_power_avg * outlet_runtime_h / 1000.0, 4)

        total_kwh = round(ac_kwh + lamp_kwh + outlet_kwh, 4)
        cost_rp   = round(total_kwh * 1500)

        return jsonify({
            'date': now_local.strftime('%Y-%m-%d'),
            'ac': {
                'kwh': ac_kwh,
                'power_avg_w': ac_power_avg,
                'power_peak_w': ac_power_peak,
                'runtime_h': ac_runtime_h
            },
            'lamp': {
                'kwh': lamp_kwh,
                'power_avg_w': lamp_power_avg,
                'power_peak_w': lamp_power_peak,
                'runtime_h': lamp_runtime_h
            },
            'outlet': {
                'kwh': outlet_kwh,
                'power_avg_w': outlet_power_avg,
                'power_peak_w': outlet_power_peak,
                'runtime_h': outlet_runtime_h
            },
            'total_kwh': total_kwh,
            'cost_rp': cost_rp
        })
    except Exception as e:
        print(f"[ERROR] daily-summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ac/control', methods=['POST'])
def control_ac():
    try:
        data = request.json or {}
        command = str(data.get('command', data.get('action', '')) or '').strip().upper()
        current_temp = int(mqtt_data['ac'].get('ac_temp', 24) or 24)
        current_fan = int(mqtt_data['ac'].get('fan_speed', 1) or 1)
        current_rh = int(mqtt_data['ac'].get('set_rh', 50) or 50)
        current_mode = str(mqtt_data['ac'].get('ac_fan_mode', 'COOL') or 'COOL').upper()
        current_state = str(mqtt_data['ac'].get('ac_state', 'OFF') or 'OFF').upper()
        payload = dict(data)
        payload['command'] = command
        payload['action'] = command
        payload['temperature'] = int(payload.get('temperature', current_temp) or current_temp)
        payload['fan_speed'] = int(payload.get('fan_speed', current_fan) or current_fan)
        payload['set_rh'] = int(payload.get('set_rh', current_rh) or current_rh)
        payload['mode'] = str(payload.get('mode', current_mode) or current_mode).upper()
        payload['ac_state'] = str(payload.get('ac_state', current_state) or current_state).upper()
        payload['power'] = payload['ac_state']

        # Mitsubishi Heavy frames are stateful, so always publish a complete state snapshot.
        mode_map = {'MODE_COOL': 'COOL', 'MODE_HEAT': 'HEAT', 'MODE_DRY': 'DRY', 'MODE_FAN': 'FAN', 'MODE_AUTO': 'AUTO'}
        if command == 'POWER_ON':
            payload['ac_state'] = 'ON'
            payload['power'] = 'ON'
        elif command == 'POWER_OFF':
            payload['ac_state'] = 'OFF'
            payload['power'] = 'OFF'
        elif command == 'TEMP_UP':
            payload['temperature'] = min(30, current_temp + 1)
            payload['ac_state'] = 'ON'
            payload['power'] = 'ON'
        elif command == 'TEMP_DOWN':
            payload['temperature'] = max(16, current_temp - 1)
            payload['ac_state'] = 'ON'
            payload['power'] = 'ON'
        elif command in mode_map:
            payload['mode'] = mode_map[command]
            payload['ac_state'] = 'ON'
            payload['power'] = 'ON'
        elif command == 'SET':
            payload['temperature'] = max(16, min(30, int(payload.get('temperature', current_temp) or current_temp)))
            payload['fan_speed'] = max(1, min(4, int(payload.get('fan_speed', current_fan) or current_fan)))
            payload['mode'] = str(payload.get('mode', current_mode) or current_mode).upper()
            payload['ac_state'] = str(payload.get('ac_state', 'ON') or 'ON').upper()
            payload['power'] = payload['ac_state']

        mqtt_client.publish('smartroom/ac/control', json.dumps(payload))

        mqtt_data['ac']['ac_temp'] = payload['temperature']
        mqtt_data['ac']['fan_speed'] = payload['fan_speed']
        mqtt_data['ac']['set_rh'] = payload['set_rh']
        mqtt_data['ac']['ac_state'] = payload['ac_state']
        mqtt_data['ac']['ac_fan_mode'] = payload['mode']
        socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})

        # Force immediate InfluxDB write with updated set_rh so the chart reflects the change
        try:
            write_to_influxdb('ac_sensor', {
                'ac_temp': float(payload['temperature']),
                'fan_speed': int(payload['fan_speed']),
                'set_rh': int(payload['set_rh']),
                'ac_state': str(payload['ac_state']),
                'ac_fan_mode': str(payload['mode']),
                'temperature': float(mqtt_data['ac'].get('temperature', 0)),
                'humidity': float(mqtt_data['ac'].get('humidity', 0)),
                'heat_index': float(mqtt_data['ac'].get('heat_index', 0)),
            }, tags={'device': 'esp32_ac', 'type': 'control', 'location': 'room'})
        except Exception as e:
            print(f'[WARN] control_ac influx write: {e}')

        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'AC Control: {command}', 'level': 'info'})
        return jsonify({'status': 'success', 'message': f'AC command sent: {command}', 'ac': mqtt_data['ac'], 'payload': payload, 'ir_sent': True})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/lamp/control', methods=['POST'])
def control_lamp():
    try:
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({'status': 'error', 'message': 'Invalid JSON payload'}), 400
        # Validate brightness values if present
        for key in ('brightness1', 'brightness2'):
            if key in data:
                data[key] = max(0, min(100, int(data[key])))
        mqtt_client.publish('smartroom/lamp/control', json.dumps(data))
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Lamp Control: {data}', 'level': 'info'})
        return jsonify({'status': 'success', 'message': 'Lamp command sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/outlet/control', methods=['POST'])
def control_outlet():
    try:
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({'status': 'error', 'message': 'Invalid JSON payload'}), 400
        
        outlet_num = data.get('id', data.get('outlet', data.get('outlet_num', 1)))
        if 'status' in data:
            state = 'ON' if data['status'] == 1 else 'OFF'
        else:
            state = data.get('state', 'OFF')
            
        # Gunakan "Shotgun Approach": Karena kita tidak bisa melihat kode ESP32,
        # kita kirimkan semua variasi yang mungkin (status angka vs teks, ID 8 vs 1)
        raw_id = 1 if outlet_num == 8 else outlet_num
        variations = [
            # Variasi 1: ID dari web (8), status angka (1/0) -> seperti format asli database
            {"id": outlet_num, "nama": "Outlet", "group_key": "master-room", "status": 1 if state == 'ON' else 0, "is_master": False},
            # Variasi 2: ID dari web (8), status teks ("ON"/"OFF") -> seperti format AC
            {"id": outlet_num, "nama": "Outlet", "group_key": "master-room", "status": state, "is_master": False},
            # Variasi 3: ID mentah (1), status angka (1/0)
            {"id": raw_id, "nama": "Outlet", "group_key": "master-room", "status": 1 if state == 'ON' else 0, "is_master": False},
            # Variasi 4: ID mentah (1), status teks ("ON"/"OFF")
            {"id": raw_id, "nama": "Outlet", "group_key": "master-room", "status": state, "is_master": False},
        ]
        
        for payload_var in variations:
            mqtt_client.publish('smartroom/outlet/control', json.dumps(payload_var))
        
        # Track outlet state (map 8 back to 1 for internal tracking)
        internal_outlet_num = 1 if outlet_num == 8 else outlet_num
        mqtt_data['outlet'][str(internal_outlet_num)] = state
        
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Outlet {internal_outlet_num} Control: {state}', 'level': 'info'})
        return jsonify({'status': 'ok', 'message': f'Outlet {internal_outlet_num} {state} command sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/outlet/status', methods=['GET'])
def get_outlet_status():
    try:
        outlets = []
        for i in range(1, 5):
            outlets.append({
                'id': i,
                'state': mqtt_data['outlet'].get(str(i), 'OFF')
            })
            
        return jsonify({
            'status': 'ok',
            'outlets': outlets
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/outlet/history', methods=['GET'])
def api_outlet_history():
    try:
        range_str = request.args.get('range', '1h')
        hours = 1
        if 'h' in range_str:
            try: hours = int(range_str.replace('h', ''))
            except: hours = 1
        elif 'd' in range_str:
            try: hours = int(range_str.replace('d', '')) * 24
            except: hours = 24
                
        res = get_influx_data('energy_monitor', 'power', hours, device_tag='mysql_outlet')
        return jsonify({
            'labels': res.get('time', []),
            'outlet1': res.get('value', []),
            'outlet2': [], 'outlet3': [], 'outlet4': []
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/outlet/energy', methods=['GET'])
def api_outlet_energy():
    try:
        range_str = request.args.get('range', '24h')
        hours = 24
        if 'h' in range_str:
            try: hours = int(range_str.replace('h', ''))
            except: hours = 24
        elif 'd' in range_str:
            try: hours = int(range_str.replace('d', '')) * 24
            except: hours = 24
                
        res = get_influx_data('energy_monitor', 'energy_kwh', hours, device_tag='mysql_outlet')
        return jsonify({
            'labels': res.get('time', []),
            'outlet1': res.get('value', []),
            'outlet2': [], 'outlet3': [], 'outlet4': []
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ==================== SBMS API PROXY ENDPOINTS ====================
def _sbms_request(method, path, json_data=None):
    """Helper: make HTTP request to SBMS server. Returns (status_code, response_dict)."""
    url = sbms_config['server_url'].rstrip('/') + path
    headers = {'Content-Type': 'application/json'}
    if sbms_config['token']:
        headers['Authorization'] = f"Bearer {sbms_config['token']}"
    body = json.dumps(json_data).encode('utf-8') if json_data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_data = json.loads(resp.read().decode('utf-8'))
            return resp.status, resp_data
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode('utf-8'))
        except Exception:
            err_body = {'message': str(e)}
        return e.code, err_body
    except Exception as e:
        return 0, {'message': str(e)}

@app.route('/api/sbms/config', methods=['GET', 'POST'])
def sbms_api_config():
    """Get or set SBMS server URL."""
    if request.method == 'POST':
        data = request.json or {}
        url = data.get('server_url', '').strip().rstrip('/')
        if not url:
            return jsonify({'status': 'error', 'message': 'server_url is required'}), 400
        sbms_config['server_url'] = url
        sbms_config['token'] = None
        sbms_config['user'] = None
        sbms_config['connected'] = False
        return jsonify({'status': 'ok', 'server_url': url})
    return jsonify({
        'server_url': sbms_config['server_url'],
        'connected': sbms_config['connected'],
        'user': sbms_config['user'],
    })

@app.route('/api/sbms/login', methods=['POST'])
def sbms_api_login():
    """Login to SBMS server and store token."""
    data = request.json or {}
    email = data.get('email', '')
    password = data.get('password', '')
    if not sbms_config['server_url']:
        return jsonify({'status': 'error', 'message': 'SBMS server URL not configured'}), 400
    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Email and password are required'}), 400
    code, resp = _sbms_request('POST', '/api/auth/login', {'email': email, 'password': password})
    if code == 200 and resp.get('token'):
        sbms_config['token'] = resp['token']
        sbms_config['user'] = resp.get('user', {}).get('email', email)
        sbms_config['connected'] = True
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'SBMS Login OK: {email}', 'level': 'info'})
        return jsonify({'status': 'ok', 'user': sbms_config['user']})
    msg = resp.get('message', 'Login failed')
    return jsonify({'status': 'error', 'message': msg}), 401

@app.route('/api/sbms/logout', methods=['POST'])
def sbms_api_logout():
    """Logout from SBMS server."""
    if sbms_config['token']:
        _sbms_request('POST', '/api/auth/logout')
    sbms_config['token'] = None
    sbms_config['user'] = None
    sbms_config['connected'] = False
    return jsonify({'status': 'ok', 'message': 'Disconnected from SBMS'})

@app.route('/api/sbms/devices', methods=['GET'])
def sbms_api_devices():
    """Get status of all SBMS devices (no auth needed)."""
    if not sbms_config['server_url']:
        return jsonify({'status': 'error', 'message': 'SBMS server URL not configured'}), 400
    code, resp = _sbms_request('GET', '/api/device-status')
    if code == 200:
        return jsonify({'status': 'ok', 'devices': resp if isinstance(resp, list) else resp.get('devices', resp)})
    return jsonify({'status': 'error', 'message': resp.get('message', 'Failed to fetch')}), code or 500

@app.route('/api/sbms/control/<int:device_id>', methods=['POST'])
def sbms_api_control(device_id):
    """Control a SBMS device: set ON/OFF or toggle."""
    if not sbms_config['token']:
        return jsonify({'status': 'error', 'message': 'Not logged in to SBMS'}), 401
    data = request.json or {}
    action = data.get('action', 'set')  # 'set' or 'toggle'
    if action == 'toggle':
        code, resp = _sbms_request('POST', f'/api/control-device/{device_id}/toggle')
    else:
        status_val = data.get('status', 0)  # 1=ON, 0=OFF
        code, resp = _sbms_request('POST', f'/api/control-device/{device_id}', {'status': status_val})
    if code == 200:
        device_name = {5: 'Master', 6: 'AC', 7: 'Lampu', 8: 'Outlet'}.get(device_id, f'Device {device_id}')
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'SBMS {device_name}: {action} (status={resp.get("status", "?")})', 'level': 'info'})
        return jsonify({'status': 'ok', 'device': resp})
    
    # DEBUG: Log the 404 or other errors to the console to help debugging
    target_url = sbms_config['server_url'].rstrip('/') + f'/api/control-device/{device_id}'
    print(f"[SBMS ERROR] Control failed for {target_url}. Code: {code}, Response: {resp}")
    
    msg = resp.get('message', 'Control failed')
    if code == 401:
        sbms_config['connected'] = False
        sbms_config['token'] = None
    return jsonify({'status': 'error', 'message': f"{msg} (HTTP {code} from SBMS)"}), code or 500

@app.route('/api/ac/mode', methods=['POST'])
def set_ac_mode():
    try:
        data = request.json
        mode = data.get('mode', 'ADAPTIVE')
        mqtt_client.publish('smartroom/ac/mode', json.dumps({'mode': mode}))
        mqtt_data['ac']['mode'] = mode
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'AC Mode: {mode}', 'level': 'info'})
        return jsonify({'status': 'success', 'mode': mode})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/lamp/mode', methods=['POST'])
def set_lamp_mode():
    try:
        data = request.json
        mode = data.get('mode', 'ADAPTIVE')
        mqtt_client.publish('smartroom/lamp/mode', json.dumps({'mode': mode}))
        mqtt_data['lamp']['mode'] = mode
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Lamp Mode: {mode}', 'level': 'info'})
        return jsonify({'status': 'success', 'mode': mode})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/optimization/update', methods=['POST'])
def update_optimization():
    """Receive optimization data from main.py — GA->AC / PSO->Lamp"""
    try:
        data = request.json
        ga_sol = data.get('ga_solution', {})
        pso_sol = data.get('pso_solution', {})
        
        # Update system data with solutions
        mqtt_data['system'].update({
            'ga_fitness': data.get('ga_fitness', 0),
            'pso_fitness': data.get('pso_fitness', 0),
            'optimization_runs': data.get('optimization_count', data.get('runs', 0)),
            'ga_temp': ga_sol.get('temperature', mqtt_data['system'].get('ga_temp', 0)),
            'ga_fan': ga_sol.get('fan_speed', mqtt_data['system'].get('ga_fan', 0)),
            'pso_brightness': pso_sol.get('brightness', mqtt_data['system'].get('pso_brightness', 0)),
            'pso_brightness1': pso_sol.get('brightness1', mqtt_data['system'].get('pso_brightness1', 0)),
            'pso_brightness2': pso_sol.get('brightness2', mqtt_data['system'].get('pso_brightness2', 0)),
            'ga_history': data.get('ga_history', mqtt_data['system'].get('ga_history', [])),
            'pso_history': data.get('pso_history', mqtt_data['system'].get('pso_history', []))
        })
        
        # Broadcast to all connected clients
        socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
        
        # AUTO-APPLY AC — only if ga_solution was provided in this request
        if ga_sol and mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE' and _person_present_recently():
            opt_temp = ga_sol.get('temperature', mqtt_data['system'].get('ga_temp', 0))
            opt_fan  = ga_sol.get('fan_speed', mqtt_data['system'].get('ga_fan', 0))
            opt_mode = ga_sol.get('mode', 'COOL')
            opt_rh   = ga_sol.get('set_rh', 50)
            if opt_temp >= 16 and opt_temp <= 30 and opt_fan >= 1:
                global _last_adaptive_ac_apply
                now = time.time()
                if now - _last_adaptive_ac_apply >= AC_ADAPTIVE_DEBOUNCE:
                    _last_adaptive_ac_apply = now
                    ac_cmd = {'command': 'SET', 'temperature': int(opt_temp), 'fan_speed': int(opt_fan),
                              'mode': opt_mode, 'set_rh': int(opt_rh), 'source': 'adaptive'}
                    mqtt_client.publish('smartroom/ac/control', json.dumps(ac_cmd))
                    mqtt_data['ac']['set_rh'] = int(opt_rh)
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Adaptive AC: {opt_temp}°C Fan:{opt_fan} Mode:{opt_mode} RH:{opt_rh}%', 'level': 'success'})
        
        # AUTO-APPLY Lamp — only if pso_solution was provided in this request
        if pso_sol and mqtt_data['lamp'].get('mode', 'MANUAL') == 'ADAPTIVE' and _person_present_recently_lamp():
            opt_b1, opt_b2 = _safe_lamp_brightness(
                pso_sol.get('brightness1', mqtt_data['system'].get('pso_brightness1', pso_sol.get('brightness', 0))),
                pso_sol.get('brightness2', mqtt_data['system'].get('pso_brightness2', pso_sol.get('brightness', 0)))
            )
            if (opt_b1 > 0 or opt_b2 > 0) and _should_apply_lamp(opt_b1, opt_b2):
                lamp_cmd = {'brightness1': opt_b1, 'brightness2': opt_b2, 'source': 'adaptive'}
                mqtt_client.publish('smartroom/lamp/control', json.dumps(lamp_cmd))
                _record_lamp_apply(opt_b1, opt_b2)
                log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Adaptive Lamp: L1={opt_b1}% L2={opt_b2}%', 'level': 'success'})
        
        return jsonify({'status': 'success', 'message': 'Optimization data updated'})
    except Exception as e:
        print(f"[ERROR] Optimization update error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ir/learn', methods=['POST'])
def learn_ir():
    global ir_learning_mode, ir_learning_button, ir_learning_device
    try:
        data = request.json
        button_name = data.get('button', '')
        device_name = data.get('device', 'remote')  # Get device name (AC, TV, etc)
        
        if not button_name:
            return jsonify({'status': 'error', 'message': 'Button name required'}), 400
        
        ir_learning_mode = True
        ir_learning_button = button_name
        ir_learning_device = device_name
        
        # MQTT PAYLOAD
        mqtt_payload = {
            'button': button_name, 
            'device': device_name,
            'action': 'start'
        }
        mqtt_payload_str = json.dumps(mqtt_payload)
        
        # DEBUG LOGGING - SEE WHAT WE'RE SENDING!
        result = mqtt_client.publish('smartroom/ir/learn', mqtt_payload_str)
        
        log_messages.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'msg': f'IR Learning started for: {device_name} - {button_name}',
            'level': 'info'
        })
        
        return jsonify({
            'status': 'success', 
            'message': f'Learning mode activated for {device_name} - {button_name}',
            'mqtt_published': result.rc == 0
        })
    except Exception as e:
        print(f"[ERROR] learn_ir: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ir/send', methods=['POST'])
def send_ir():
    try:
        data = request.json
        button_name = data.get('button', '')
        
        if button_name not in mqtt_data['ir_codes']:
            return jsonify({'status': 'error', 'message': 'IR code not learned yet'}), 400
        
        ir_code = mqtt_data['ir_codes'][button_name]
        
        # Handle toggle buttons (power ON/OFF with same code)
        action_suffix = ''
        if 'toggle' in button_name.lower() or button_name in mqtt_data['ir_states']:
            current_state = mqtt_data['ir_states'].get(button_name, 'OFF')
            new_state = 'ON' if current_state == 'OFF' else 'OFF'
            mqtt_data['ir_states'][button_name] = new_state
            action_suffix = f' ({new_state})'
        
        mqtt_payload = json.dumps({'button': button_name, 'code': ir_code})
        result = mqtt_client.publish('smartroom/ir/send', mqtt_payload)
        
        log_messages.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'msg': f'IR Code sent: {button_name}{action_suffix}',
            'level': 'success' if result.rc == 0 else 'error'
        })
        
        return jsonify({
            'status': 'success' if result.rc == 0 else 'error', 
            'message': f'IR command sent: {button_name}{action_suffix}',
            'state': mqtt_data['ir_states'].get(button_name, None),
            'mqtt_published': result.rc == 0
        })
    except Exception as e:
        print(f"[ERROR] send_ir: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ir/codes')
def get_ir_codes():
    return jsonify({
        'codes': mqtt_data['ir_codes'],
        'states': mqtt_data['ir_states']
    })

@app.route('/api/ir/delete', methods=['POST'])
def delete_ir_code():
    try:
        data = request.json
        button_name = data.get('button', '')
        
        if button_name in mqtt_data['ir_codes']:
            del mqtt_data['ir_codes'][button_name]
            
            # Also delete state if exists
            if button_name in mqtt_data['ir_states']:
                del mqtt_data['ir_states'][button_name]
            
            # Update file
            try:
                ir_file = os.path.join(os.path.dirname(__file__), 'ir_codes.json')
                with open(ir_file, 'w') as f:
                    json.dump(mqtt_data['ir_codes'], f, indent=2)
            except Exception as e:
                print(f"[ERROR] Error updating IR codes file: {e}")
            
            return jsonify({'status': 'success', 'message': f'IR code {button_name} deleted'})
        else:
            return jsonify({'status': 'error', 'message': 'IR code not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ir/status')
def get_ir_status():
    """Get current IR learning status"""
    return jsonify({
        'learning_mode': ir_learning_mode,
        'learning_button': ir_learning_button,
        'learning_device': ir_learning_device,
        'total_codes': len(mqtt_data['ir_codes']),
        'codes': list(mqtt_data['ir_codes'].keys())
    })

@app.route('/api/ir/manual_save', methods=['POST'])
def manual_save_ir():
    """Fallback: Manually save IR code if MQTT fails"""
    try:
        data = request.json
        button_name = data.get('button', '')
        ir_code = data.get('code', '')
        device = data.get('device', 'remote')
        
        if not button_name or not ir_code:
            return jsonify({'status': 'error', 'message': 'Button name and code required'}), 400
        
        # Save to memory
        mqtt_data['ir_codes'][button_name] = ir_code
        
        # Save to InfluxDB
        save_ir_command(device, button_name, len(ir_code))
        
        # Save to file
        try:
            ir_file = os.path.join(os.path.dirname(__file__), 'ir_codes.json')
            with open(ir_file, 'w') as f:
                json.dump(mqtt_data['ir_codes'], f, indent=2)
        except Exception as e:
            print(f"[ERROR] Error saving to file: {e}")
        
        # Notify frontend
        socketio.emit('ir_learned', {
            'button': button_name,
            'code': ir_code[:50] + '...',
            'device': device,
            'status': 'success',
            'is_toggle': False
        })
        
        return jsonify({'status': 'success', 'message': f'IR code manually saved: {button_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/logs')
def get_logs():
    return jsonify(list(log_messages))

@app.route('/api/device/status')
def get_device_status():
    now = datetime.now()
    for dev_id, dev in device_last_seen.items():
        if dev['last_seen'] is None:
            dev['status'] = 'offline'
        elif (now - dev['last_seen']).total_seconds() > 30:
            dev['status'] = 'offline'
        else:
            dev['status'] = 'online'
    return jsonify({k: {'status': v['status'], 'last_seen': v['last_seen'].strftime('%d %b %H:%M:%S') if v['last_seen'] else 'Never'} for k, v in device_last_seen.items()})

@app.route('/api/sensor/health')
def sensor_health_api():
    """Return real-time sensor health: status (ok/warn/fault) and data age."""
    now = datetime.now()
    health = {}
    labels = {'esp32_ac': 'AC Sensor', 'esp32_lamp': 'Lamp Sensor', 'camera': 'Camera'}
    for dev_id, dev in device_last_seen.items():
        last = dev['last_seen']
        age = float('inf') if last is None else (now - last).total_seconds()
        if age == float('inf') or age > SENSOR_STALE_FAULT_S:
            status = 'fault'
        elif age > SENSOR_STALE_WARN_S:
            status = 'warn'
        else:
            status = 'ok'
        health[dev_id] = {
            'label':   labels.get(dev_id, dev_id),
            'status':  status,
            'age':     'never' if age == float('inf') else f"{int(age)}s ago",
            'last_seen': last.strftime('%d %b %H:%M:%S') if last else None,
        }
    return jsonify(health)

@app.route('/api/alerts')
def get_alerts():
    return jsonify({'rules': alert_rules, 'active': list(active_alerts)})

@app.route('/api/alerts/config', methods=['POST'])
def config_alerts():
    try:
        data = request.json
        for rule_name, config in data.items():
            if rule_name in alert_rules:
                alert_rules[rule_name].update(config)
        return jsonify({'status': 'success', 'rules': alert_rules})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def check_alert_rules():
    """Check sensor data against alert rules and emit alerts"""
    try:
        now = datetime.now()
        temp = mqtt_data['ac'].get('temperature', 0)
        humidity = mqtt_data['ac'].get('humidity', 0)
        person = mqtt_data['camera'].get('person_detected', False)
        
        # High temperature alert
        rule = alert_rules.get('high_temp', {})
        if rule.get('enabled') and temp > rule.get('threshold', 35):
            if not rule.get('triggered'):
                rule['triggered'] = True
                alert = {'type': 'high_temp', 'message': f'Temperature {temp:.1f}°C exceeds {rule.get("threshold", 35)}°C!', 'level': 'danger', 'time': now.strftime('%H:%M:%S')}
                active_alerts.append(alert)
                socketio.emit('alert', alert)
        else:
            rule['triggered'] = False
        
        # High humidity alert
        rule = alert_rules.get('high_humidity', {})
        if rule.get('enabled') and humidity > rule.get('threshold', 80):
            if not rule.get('triggered'):
                rule['triggered'] = True
                alert = {'type': 'high_humidity', 'message': f'Humidity {humidity:.1f}% exceeds {rule.get("threshold", 80)}%!', 'level': 'warning', 'time': now.strftime('%H:%M:%S')}
                active_alerts.append(alert)
                socketio.emit('alert', alert)
        else:
            rule['triggered'] = False
        
        # No person timeout -> suggest turning off AC
        rule = alert_rules.get('no_person_timeout', {})
        if rule.get('enabled'):
            if person:
                rule['last_person_seen'] = now
            elif rule.get('last_person_seen') is not None:
                elapsed = (now - rule['last_person_seen']).total_seconds() / 60
                ac_timeout = rule.get('ac_timeout_minutes', rule.get('timeout_minutes', 5))
                if elapsed > ac_timeout and mqtt_data['ac'].get('ac_state') != 'OFF':
                    alert = {'type': 'no_person', 'message': f'No person detected for {int(elapsed)} min. Consider turning off AC.', 'level': 'warning', 'time': now.strftime('%H:%M:%S')}
                    active_alerts.append(alert)
                    socketio.emit('alert', alert)
                    rule['last_person_seen'] = now  # Reset to avoid spamming
    except Exception as e:
        import traceback
        print(f'[WARN] check_alert_rules error: {e}\n{traceback.format_exc()}')

# ==================== LOGIN TEMPLATE ====================
# LOGIN_TEMPLATE moved to templates/login.html

# ==================== HTML TEMPLATE ====================
# HTML_TEMPLATE (CSS + HTML + JS) moved to:
#   static/css/dashboard.css
#   static/js/dashboard.js
#   templates/ (dashboard.html, pages/, partials/)

if __name__ == '__main__':
    print("Smart Room Dashboard starting...")

    # Check system timezone (RPi must be Asia/Jakarta for correct timestamps)
    check_timezone()

    # Load saved IR codes from file
    try:
        ir_file = os.path.join(os.path.dirname(__file__), 'ir_codes.json')
        if os.path.exists(ir_file):
            with open(ir_file, 'r') as f:
                mqtt_data['ir_codes'] = json.load(f)
            print(f"  [OK] Loaded {len(mqtt_data['ir_codes'])} IR codes from file")
        else:
            print("  [INFO] No saved IR codes found")
    except Exception as e:
        print(f"  [WARN] Error loading IR codes: {e}")
    
    # Load YOLO SYNCHRONOUSLY
    yolo_loaded = load_yolo_model()
    print(f"  [YOLO] {'Ready' if yolo_loaded else 'Failed to load'}")
    
    # Start background camera detection thread
    detection_thread = threading.Thread(target=camera_detection_loop, daemon=True)
    detection_thread.start()
    
    # Start GA/PSO auto-optimization background thread
    opt_thread = threading.Thread(target=optimization_auto_loop, daemon=True)
    opt_thread.start()
    print(f"  [OPT] GA auto-opt every {AUTO_OPT_INTERVAL_AC}s, PSO every {AUTO_OPT_INTERVAL_LAMP}s")

    # Restore last GA/PSO results — first from JSON file (has stats/history arrays),
    # then supplement with InfluxDB (has latest scalar values)
    print("  [RESTORE] Loading last optimization results...")
    load_opt_results_file()
    restore_opt_results()

    # Start sensor fault detection thread
    fault_thread = threading.Thread(target=sensor_fault_loop, daemon=True)
    fault_thread.start()
    print("  [FAULT] Sensor fault detection thread started")

    # Start outdoor weather polling thread (Open-Meteo, UNS Surakarta)
    weather_thread = threading.Thread(target=weather_poll_loop, daemon=True)
    weather_thread.start()
    print(f"  [WEATHER] Outdoor weather polling started (UNS Surakarta, update every {WEATHER_FETCH_INTERVAL}s)")

    print("  [URL] Dashboard: http://172.20.0.65:5000")


    # Start MySQL energy polling (Jagoan Hosting)
    try:
        import mysql_energy
        mysql_energy.start_polling({
            'mqtt_data':              mqtt_data,
            'energy_runtime_history': energy_runtime_history,
            'outlet_runtime_history': outlet_runtime_history,
            'get_energy_phase':       lambda: energy_phase,
            'write_influx':           write_to_influxdb,
            'socketio':               socketio,
        })
        print("  [MySQL] Energy polling thread started")
    except Exception as _me:
        print(f"  [MySQL] WARNING: mysql_energy could not be loaded: {_me}")

    def auto_connect_sbms():
        time.sleep(2)
        print("  [SBMS] Auto-connecting to SBMS Server...")
        sbms_config['server_url'] = 'https://iotlab-uns.com/neo-sbms'
        try:
            code, resp = _sbms_request('POST', '/api/auth/login', {'email': 'admin@example.com', 'password': '123456'})
            if code == 200 and resp.get('token'):
                sbms_config['token'] = resp['token']
                sbms_config['user'] = resp.get('user', {}).get('email', 'admin@example.com')
                sbms_config['connected'] = True
                log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': 'SBMS Auto-Login OK', 'level': 'info'})
                print(f"  [SBMS] Successfully auto-connected as {sbms_config['user']}")
            else:
                print(f"  [SBMS] Auto-connect failed: HTTP {code} - {resp}")
        except Exception as e:
            print(f"  [SBMS] Auto-connect exception: {e}")

    threading.Thread(target=auto_connect_sbms, daemon=True).start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
