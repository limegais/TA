from flask import Flask, render_template_string, jsonify, request, Response, session, redirect, url_for
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

        # Hitung fitness nyata
        person_now = opt_sensor_data.get('person_detected', False) or _person_present_recently_lamp()
        TARGET_LUX = 350.0 if person_now else 0.0
        real_fit   = round((lux_real - TARGET_LUX) ** 2, 2)

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

def run_optimization_cycle(algo='both'):
    global optimization_run_count
    if not optimization_lock.acquire(blocking=False):
        print("[OPT] Already running, skipping")
        return False
    try:
        socketio.emit('ml_status', {'status': 'running', 'algorithm': algo})
        fetch_sensor_data_from_db(30)
        if algo in ('ga', 'both'):
            sol, fit, hist, bf = run_ga_optimization()
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
                },
                'sensor_snapshot': {
                    'temp_room': round(opt_sensor_data.get('temperature', 0), 1),
                    'humidity': round(opt_sensor_data.get('humidity', 0), 1),
                    'person_detected': opt_sensor_data.get('person_detected', False),
                    'person_count': opt_sensor_data.get('person_count', 0),
                    'actual_watt': round(opt_sensor_data.get('actual_watt', 0), 1),
                },
            }
            print(f"[GA] Done: {sol[0]}°C Fan={sol[1]} Mode={AC_MODE_NAMES.get(mode_idx,'COOL')} RH={opt_set_rh}% fitness={fit:.2f}")
            persist_opt_results('ga')
        if algo in ('pso', 'both'):
            sol, fit, hist, initial_err, iter_log = run_pso_optimization()
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
                },
            }
            print(f"[PSO] Done: PWM1={pwm1_val}/255 PWM2={pwm2_val}/255 (B1={b1}% B2={b2}%) lux_error={fit:.2f}")
            persist_opt_results('pso')
        optimization_run_count += 1
        # Update mqtt_data system
        mqtt_data['system'].update({
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
    'ir_states': {}  # Track toggle states for power buttons
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
                'motion': bool(motion)
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
        return render_template_string(LOGIN_TEMPLATE, error='Invalid username or password')
    return render_template_string(LOGIN_TEMPLATE, error=None)

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
    return render_template_string(HTML_TEMPLATE)

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

        if mysql_rows is not None:
            if not mysql_rows:
                return jsonify({'error': f'No data for {dev_name} in MySQL for range {from_dt} to {to_dt}'}), 404
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
        'lux':       ('lamp_sensor',      ['lux1','lux2','lux3','lux_avg','brightness1','brightness2'], None, None),
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
  {tag_filter}  |> yield(name: "raw")
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
            row = [ts_str] + [
                round(row_d.get(f, ''), 4) if isinstance(row_d.get(f), float) else row_d.get(f, '')
                for f in fields
            ]
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
                    if val > 10000:
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
                    if field_name == 'energy_kwh' and val > 10000:
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
    wr.writerow(['=== GA RESULT REPORT (GENETIC ALGORITHM) - SMART ROOM ==='])
    wr.writerow([])
    wr.writerow(['Run Date', run_time])
    wr.writerow([])

    # ── Parameter GA ────────────────────────────────────────────────────────
    wr.writerow(['--- PARAMETER GA ---'])
    wr.writerow(['Ukuran Populasi',   params.get('population_size', '-')])
    wr.writerow(['Generation Count',   params.get('generations', '-')])
    wr.writerow(['Mutation Rate',     params.get('mutation_rate', '-')])
    wr.writerow(['Crossover Rate',    params.get('crossover_rate', '-')])
    wr.writerow(['Elitism Ratio',     params.get('elitism_ratio', '-')])
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
                            if field == 'energy_kwh' and val > 10000:
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
                            if field == 'energy_kwh' and val > 10000:
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
        
        outlet_num = data.get('outlet', data.get('outlet_num'))
        state = data.get('state')
        
        mqtt_client.publish('smartroom/outlet/control', json.dumps({
            'outlet_num': outlet_num,
            'state': state
        }))
        
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Outlet {outlet_num} Control: {state}', 'level': 'info'})
        return jsonify({'status': 'success', 'message': f'Outlet {outlet_num} {state} command sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/outlet/status', methods=['GET'])
def get_outlet_status():
    try:
        # Assuming only 1 outlet as per user request
        return jsonify({
            'status': 'ok',
            'outlets': [
                {'id': 1, 'state': 'OFF', 'power': 0, 'energy': 0}
            ],
            'total_power': 0,
            'today_kwh': 0,
            'peak_power': 0
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Room - Login</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .login-box { background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 40px; width: 380px; box-shadow: 0 20px 60px rgba(0,0,0,0.5); }
        .login-logo { text-align: center; margin-bottom: 30px; }
        .login-logo i { font-size: 48px; color: #2563eb; }
        .login-logo h1 { color: #eff6ff; font-size: 24px; margin-top: 10px; }
        .login-logo p { color: #94a3b8; font-size: 14px; margin-top: 5px; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; color: #94a3b8; font-size: 13px; margin-bottom: 6px; font-weight: 500; }
        .form-group input { width: 100%; padding: 12px 16px; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #eff6ff; font-size: 14px; transition: border-color 0.3s; }
        .form-group input:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.2); }
        .login-btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s; }
        .login-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(37,99,235,0.4); }
        .error-msg { background: rgba(30,64,175,0.15); border: 1px solid #1e40af; color: #1e40af; padding: 10px; border-radius: 8px; margin-bottom: 20px; font-size: 13px; text-align: center; }
    </style>
</head>
<body>
    <div class="login-box">
        <div class="login-logo">
            <h1>Smart Room IoT</h1>
            <p>Login to access dashboard</p>
        </div>
        {% if error %}
        <div class="error-msg">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required autocomplete="username" placeholder="Enter username">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required autocomplete="current-password" placeholder="Enter password">
            </div>
            <button type="submit" class="login-btn">Login</button>
        </form>
    </div>
</body>
</html>
'''

# ==================== HTML TEMPLATE ====================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>Smart Room IoT Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0; padding: 0; box-sizing: border-box;
        }
        :root {
            --primary: #2563eb; --primary-dark: #1d4ed8; --success: #3b82f6; --warning: #0ea5e9; --danger: #1e40af; --bg-dark: #eff6ff;
            --bg-card: #ffffff; --bg-card-hover: #dbeafe; --text-primary: #1e293b; --text-secondary: #3b6cb7; --border: #bfdbfe; --shadow: rgba(37, 99, 235, 0.08);
            --input-bg: #f0f7ff;
        }
        [data-theme="dark"] {
            --bg-dark: #03111e; --bg-card: #0c1a35; --bg-card-hover: #1a3261; --text-primary: #e8f0fe;
            --text-secondary: #93c5fd; --border: #1e3a5f; --shadow: rgba(0, 0, 0, 0.3); --input-bg: #03111e;
        }
        body {
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg-dark);
            color: var(--text-primary); overflow-x: hidden; -webkit-tap-highlight-color: transparent;
        }
        .sidebar {
            position: fixed; left: 0; top: 0; width: 260px; height: 100vh;
            background: var(--bg-card); border-right: 1px solid var(--border); padding: 20px; overflow-y: auto; z-index: 1000;
        }
        .logo {
            font-size: 24px; font-weight: bold; color: var(--primary); margin-bottom: 30px; display: flex; align-items: center; gap: 10px;
        }
        .nav-item {
            padding: 12px 16px; margin: 8px 0; border-radius: 8px; cursor: pointer;
            transition: all 0.3s; display: flex; align-items: center; gap: 12px;
            color: var(--text-secondary); touch-action: manipulation;
        }
        .nav-item:hover {
            background: var(--bg-card-hover); color: var(--text-primary);
        }
        .nav-item.active {
            background: var(--primary); color: white;
        }
        .main-content {
            margin-left: 260px; padding: 22px; min-height: 100vh;
        }
        .page {
            display: none;
        }
        .page.active {
            display: block; animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .header {
            background: var(--bg-card); padding: 20px 30px; border-radius: 12px; margin-bottom: 16px; border: 1px solid var(--border);
        }
        .header h1 {
            font-size: 28px; margin-bottom: 5px;
        }
        .header p {
            color: var(--text-secondary); font-size: 14px;
        }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; margin-bottom: 16px;
        }
        .dashboard-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: stretch;
        }
        .dashboard-grid .stat-card {
            min-height: 185px;
        }
        .feedback-grid {
            display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px;
        }
        .occupancy-top {
            display: grid; grid-template-columns: 280px 1fr; gap: 20px; margin-bottom: 20px;
        }
        .occupancy-kpi {
            background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
        }
        .occupancy-kpi .kpi-label {
            font-size: 12px; color: var(--text-secondary); margin-bottom: 8px;
        }
        .occupancy-kpi .kpi-value {
            font-size: 36px; font-weight: 700; color: #0ea5e9; line-height: 1;
        }
        .occupancy-mini-note {
            margin-top: 8px; font-size: 12px; color: var(--text-secondary);
        }
        .occupancy-chart-card {
            background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 14px;
        }
        .occupancy-chart-card canvas {
            max-height: 160px;
        }
        .rating-row {
            display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;
        }
        .rating-btn {
            border: 1px solid var(--border); background: var(--bg-card); color: var(--text-primary); border-radius: 14px; min-width: 74px; min-height: 74px; padding: 10px 12px;
            cursor: pointer; transition: all 0.2s; font-size: 22px; font-weight: 700; display: inline-flex; align-items: center; justify-content: center;
        }
        .rating-btn.active {
            background: var(--primary); border-color: var(--primary);
            color: #fff; transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(37, 99, 235, 0.35);
        }
        .feedback-input {
            width: 100%; background: var(--input-bg); border: 1px solid var(--border);
            border-radius: 10px; color: var(--text-primary); padding: 10px 12px;
            margin-top: 10px;
        }
        .feedback-history-item {
            padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; margin-bottom: 10px; background: var(--bg-card-hover);
        }
        .stat-card {
            background: var(--bg-card); padding: 18px; border-radius: 12px; border: 1px solid var(--border); transition: all 0.3s;
        }
        .stat-card:hover {
            transform: translateY(-5px); box-shadow: 0 10px 30px var(--shadow);
        }
        .stat-header {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;
        }
        .stat-title {
            color: var(--text-secondary); font-size: 14px; font-weight: 500;
        }
        .stat-icon {
            width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;
        }
        .stat-value {
            font-size: 32px; font-weight: bold; margin-bottom: 8px;
        }
        .stat-change {
            font-size: 13px; display: flex; align-items: center; gap: 5px;
        }
            .stat-change.up { color: var(--success); }
            .stat-change.down { color: var(--danger); }
        .chart-container {
            background: var(--bg-card); padding: 24px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 30px;
        }
        .chart-header {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
        }
        .chart-title {
            font-size: 18px; font-weight: 600;
        }
        .chart-options {
            display: flex; gap: 10px;
        }
        .chart-option-btn {
            padding: 8px 16px; border: 1px solid var(--border); background: transparent;
            color: var(--text-secondary); border-radius: 6px; cursor: pointer;
            transition: all 0.3s; touch-action: manipulation;
        }
        .chart-option-btn:hover {
            background: var(--bg-card-hover); color: var(--text-primary);
        }
        .chart-option-btn.active {
            background: var(--primary); color: white; border-color: var(--primary);
        }
        /* Energy page — uses same chart styling as other pages for consistency */
        /* ML Optimization Page Styles */
        .ml-table {
            width: 100%; border-collapse: collapse; font-size: 13px;
        }
        .ml-table th, .ml-table td {
            padding: 10px 14px; text-align: center; border-bottom: 1px solid var(--border);
        }
        .ml-table th {
            background: rgba(37, 99, 235, 0.1); color: var(--primary); font-weight: 600;
            text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;
        }
        .ml-table tr:hover {
            background: var(--bg-card-hover);
        }
        .ml-param-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
        }
        .ml-param-item label {
            display: block; font-size: 12px; color: var(--text-secondary); margin-bottom: 6px;
        }
        .ml-input {
            width: 100%; padding: 8px 12px; background: var(--input-bg); border: 1px solid var(--border);
            border-radius: 8px; color: var(--text-primary); font-size: 14px; transition: border-color 0.3s;
        }
        .ml-input:focus {
            outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
        }
        .ml-action-btn {
            padding: 8px 18px !important; font-weight: 600 !important; cursor: pointer; transition: all 0.3s !important;
        }
        .ml-action-btn:hover {
            transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .ml-badge {
            display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;
        }
            .ml-badge.good { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
            .ml-badge.mid { background: rgba(14, 165, 233, 0.2); color: #0ea5e9; }
            .ml-badge.low { background: rgba(30, 64, 175, 0.2); color: #1e40af; }
        @media (max-width: 768px) {
            .ml-param-grid { grid-template-columns: 1fr; }
        }
        .mode-badge {
            display: inline-block; padding: 6px 14px; border-radius: 20px;
            font-size: 12px; font-weight: 600; cursor: pointer;
            transition: all 0.3s;
        }
        .mode-badge.adaptive {
            background: linear-gradient(135deg, #3b82f6, #2563eb); color: white;
        }
        .mode-badge.manual {
            background: linear-gradient(135deg, #0ea5e9, #0284c7); color: white;
        }
        .mode-badge:hover {
            transform: scale(1.05); box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4);
        }
        .control-panel {
            background: var(--bg-card); padding: 24px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 20px;
        }
        .control-title {
            font-size: 18px; font-weight: 600; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;
        }
        .control-group {
            margin-bottom: 20px;
        }
        .control-label {
            display: block; color: var(--text-secondary); margin-bottom: 10px; font-size: 14px;
        }
        .slider {
            width: 100%; height: 8px; border-radius: 5px; background: var(--border); outline: none; -webkit-appearance: none;
        }
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none; appearance: none; width: 20px;
            height: 20px; border-radius: 50%; background: var(--primary);
            cursor: pointer;
        }
        .btn {
            padding: 12px 24px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600;
            cursor: pointer; transition: all 0.3s; display: inline-flex; align-items: center; gap: 8px;
            touch-action: manipulation;
        }
        .btn-primary {
            background: var(--primary); color: white;
        }
        .btn-primary:hover {
            background: var(--primary-dark); transform: translateY(-2px); box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4);
        }
        .btn-success {
            background: var(--success); color: white;
        }
        .btn-danger {
            background: var(--danger); color: white;
        }
        .btn-warning {
            background: var(--warning); color: white;
        }
        .btn:disabled {
            opacity: 0.5; cursor: not-allowed;
        }
        .btn-sm {
            padding: 6px 12px; font-size: 12px;
        }
        .ir-button-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-top: 20px;
        }
        .ir-button {
            padding: 15px; background: var(--input-bg); border: 2px solid var(--border); border-radius: 8px;
            text-align: center; cursor: pointer; transition: all 0.3s; position: relative;
        }
        .ir-button:hover {
            border-color: var(--primary); transform: translateY(-2px);
        }
        .ir-button.learned {
            border-color: var(--success);
        }
        .ir-button.learning {
            border-color: var(--warning); animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.2; }
        }
        .ir-button-name {
            font-size: 12px; color: var(--text-secondary); margin-bottom: 5px;
        }
        .ir-button-icon {
            font-size: 24px; margin: 10px 0;
        }
        .ir-status {
            font-size: 10px; margin-top: 5px;
        }
        .log-container {
            background: var(--bg-card); padding: 20px; border-radius: 12px;
            border: 1px solid var(--border); max-height: 500px; overflow-y: auto;
        }
        .log-entry {
            padding: 10px; margin: 5px 0; border-left: 3px solid var(--border); font-family: 'Courier New', monospace; font-size: 13px;
        }
            .log-entry.success { border-left-color: var(--success); }
            .log-entry.error { border-left-color: var(--danger); }
            .log-entry.info { border-left-color: var(--primary); }
        .toast {
            position: fixed; bottom: 30px; right: 30px; background: var(--bg-card); padding: 16px 20px;
            border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 10px 30px var(--shadow); display: none; z-index: 9999;
            animation: slideIn 0.3s;
        }
        @keyframes slideIn {
            from { transform: translateX(400px); }
            to { transform: translateX(0); }
        }
            .toast.show { display: block; }
        /* Energy data bubble notification */
        .energy-bubble {
            position: fixed; top: 80px; right: 30px; background: linear-gradient(135deg, rgba(59, 130, 246, 0.95), rgba(37, 99, 235, 0.95)); color: #fff; padding: 10px 18px; border-radius: 20px; font-size: 13px; font-weight: 600;
            z-index: 9998; pointer-events: none; opacity: 0; transform: translateY(-10px) scale(0.9); transition: opacity 0.3s, transform 0.3s; box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4); display: flex; align-items: center; gap: 8px;
        }
        .energy-bubble.show {
            opacity: 1; transform: translateY(0) scale(1);
        }
        .energy-bubble .bubble-dot {
            width: 8px; height: 8px; background: #fff; border-radius: 50%; animation: bubblePulse 1s infinite;
        }
        @keyframes bubblePulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .power-card {
            background: var(--bg-card); padding: 20px; border-radius: 12px; border: 1px solid var(--border); text-align: center;
        }
        .power-value {
            font-size: 36px; font-weight: bold; color: var(--primary); margin: 10px 0;
        }
        /* ========== CAMERA STYLES ========== */
        .camera-view {
            background: var(--bg-card); padding: 20px; border-radius: 12px;
            border: 1px solid var(--border); text-align: center; margin-bottom: 20px;
        }
        .camera-feed-container {
            position: relative; background: #000; border-radius: 8px; overflow: hidden; max-width: 100%;
        }
        .camera-feed-container img {
            width: 100%; max-height: 800px; object-fit: contain; display: block; margin: 0 auto;
        }
        .camera-overlay-bar {
            position: absolute; top: 0; left: 0; right: 0;
            background: linear-gradient(180deg, rgba(0,0,0,0.7) 0%, transparent 100%); padding: 10px 15px; display: flex; justify-content: space-between;
            align-items: center;
        }
        .camera-rec-badge {
            background: var(--danger); color: white; padding: 3px 10px; border-radius: 4px; font-size: 12px;
            font-weight: bold; animation: pulse 1.5s infinite; display: flex; align-items: center; gap: 5px;
        }
        .camera-time-badge {
            color: white; font-size: 12px; background: rgba(0,0,0,0.5); padding: 3px 10px; border-radius: 4px; font-family: monospace;
        }
        .camera-error {
            padding: 80px 20px; color: var(--text-secondary);
        }
        .camera-error i {
            font-size: 64px; margin-bottom: 20px; color: var(--border);
        }
        .camera-info-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-top: 20px;
        }
        .camera-info-card {
            background: var(--input-bg); padding: 15px; border-radius: 8px; border: 1px solid var(--border);
        }
        .camera-info-label {
            color: var(--text-secondary); font-size: 12px; margin-bottom: 5px;
        }
        .camera-info-value {
            font-size: 18px; font-weight: bold; color: var(--text-primary);
        }
        .detection-alert {
            position: fixed; top: 100px; right: 30px; background: linear-gradient(135deg, #1e40af, #1d4ed8); color: white; padding: 20px 25px;
            border-radius: 12px; box-shadow: 0 10px 40px rgba(30, 64, 175, 0.5); display: none; z-index: 8888; animation: slideInRight 0.5s, pulse 2s infinite; max-width: 300px;
        }
        .detection-alert.show {
            display: block;
        }
        @keyframes slideInRight {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .detection-alert-header {
            display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 18px; font-weight: bold;
        }
        .detection-alert-icon {
            font-size: 24px; animation: bounce 1s infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        .detection-alert-body {
            font-size: 14px; opacity: 0.9;
        }
        .detection-close {
            position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.2); border: none; color: white;
            width: 25px; height: 25px; border-radius: 50%; cursor: pointer; display: flex; align-items: center;
            justify-content: center;
        }
        .person-badge {
            display: inline-block; padding: 4px 12px; border-radius: 12px;
            font-size: 11px; font-weight: 600; margin: 2px;
            animation: fadeIn 0.3s;
        }
        .person-badge.detected {
            background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        }
        .person-badge.not-detected {
            background: linear-gradient(135deg, #64748b, #475569); color: rgba(255,255,255,0.7);
        }
        /* ===== MOBILE (≤ 768px) ===== */
        @media (max-width: 768px) {
            /* Sidebar slide-in */
            .sidebar {
                transform: translateX(-100%); transition: transform 0.3s ease; z-index: 2000;
                width: 240px;
            }
            .sidebar.open { transform: translateX(0); }
            .hamburger-btn { display: flex !important; }

            /* Main content area */
            .main-content {
                margin-left: 0; padding: 10px; padding-top: 62px;
            }

            /* Header card */
            .header {
                padding: 14px 14px 12px; border-radius: 10px;
            }
            .header h1 { font-size: 20px; margin-bottom: 4px; }
            .header p  { font-size: 12px; }
            #device-status-bar { gap: 6px; margin-top: 8px; }
            .device-status-item { padding: 4px 9px; font-size: 11px; }

            /* Sensor health bar */
            #sensor-health-bar {
                padding: 5px 10px; gap: 8px; font-size: 11px;
            }
            .sensor-health-badge { padding: 2px 8px; font-size: 11px; }

            /* Grids → single column */
            .stats-grid       { grid-template-columns: 1fr; gap: 10px; }
            .dashboard-grid   { grid-template-columns: 1fr; gap: 10px; }
            .feedback-grid    { grid-template-columns: 1fr; }
            .occupancy-top    { grid-template-columns: 1fr; gap: 12px; }
            .ml-param-grid    { grid-template-columns: 1fr; }

            /* Stat cards */
            .stat-card { padding: 14px; border-radius: 10px; }
            .stat-value { font-size: 26px; }
            .stat-icon  { width: 34px; height: 34px; font-size: 16px; }

            /* Chart containers */
            .chart-container { padding: 14px 12px; border-radius: 10px; margin-bottom: 14px; }
            .chart-header {
                flex-direction: column; align-items: flex-start; gap: 8px; margin-bottom: 12px;
            }
            .chart-title { font-size: 15px; }
            .chart-options { display: flex; flex-wrap: wrap; gap: 6px; width: 100%; }
            .chart-option-btn { padding: 6px 10px; font-size: 11px; }

            /* Energy page */
            #energy .chart-container { padding: 12px 10px; }
            #energy canvas { height: 200px !important; max-height: 200px; }
            #energy .chart-header { flex-direction: column; align-items: flex-start; gap: 6px; }
            #energy .chart-options { flex-wrap: wrap; gap: 5px; }
            #energy .chart-option-btn { padding: 5px 9px; font-size: 11px; }

            /* Control panels */
            .control-panel { padding: 14px; border-radius: 10px; }
            .control-title { font-size: 15px; }

            /* Buttons */
            .btn { padding: 10px 18px; font-size: 13px; }

            /* Alert banner — full width on mobile */
            .alert-banner {
                top: auto; bottom: 12px;
                left: 10px; right: 10px;
                max-width: none; width: auto;
                border-radius: 10px;
            }

            /* Tables — horizontal scroll */
            .ml-table-wrap, .table-wrap {
                overflow-x: auto; -webkit-overflow-scrolling: touch;
            }
            .ml-table { min-width: 520px; }

            /* Logo */
            .logo { font-size: 20px; margin-bottom: 20px; }

            /* ML page stat cards */
            #ml-optimization .stats-grid { gap: 8px; }

            /* Occupancy KPI card */
            .occupancy-kpi .kpi-value { font-size: 28px; }
        }

        /* ===== EXTRA SMALL (≤ 480px) — e.g. narrow phones ===== */
        @media (max-width: 480px) {
            .main-content { padding: 8px; padding-top: 60px; }
            .header h1 { font-size: 17px; }
            .stat-value { font-size: 24px; }
            .chart-option-btn { padding: 5px 8px; font-size: 10px; }
            .btn { padding: 9px 14px; font-size: 12px; }
            .hamburger-btn { width: 40px; height: 40px; font-size: 17px; top: 10px; left: 10px; }
            .device-status-item { font-size: 10px; padding: 3px 7px; }
            #sensor-health-bar { font-size: 10px; padding: 4px 8px; }
        }

        @media (min-width: 769px) and (max-width: 1200px) {
            .dashboard-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .occupancy-top { grid-template-columns: 1fr; }
        }
        @media (min-width: 1201px) {
            .dashboard-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        .hamburger-btn {
            display: none; position: fixed; top: 12px; left: 12px; z-index: 1998; width: 44px; height: 44px; border-radius: 10px;
            background: var(--primary); color: white; border: none; align-items: center; justify-content: center; font-size: 20px; cursor: pointer; box-shadow: 0 4px 15px rgba(37,99,235,0.4);
            touch-action: manipulation;
        }
        .sidebar-overlay {
            display: none; position: fixed; top: 0; left: 0;
            width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1999;
            pointer-events: none;
        }
        .sidebar-overlay.active {
            display: block; pointer-events: auto;
        }
        /* Device Status Indicators */
        .device-status-item {
            display: flex; align-items: center; gap: 6px; padding: 5px 12px;
            background: var(--bg-card-hover); border-radius: 20px; font-size: 12px; font-weight: 500;
            color: var(--text-secondary);
        }
        .sensor-health-badge {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 3px 10px; border-radius: 14px;
            background: var(--bg-secondary); font-size: 12px; font-weight: 600;
            border: 1px solid var(--border);
        }
        .device-dot {
            width: 8px; height: 8px; border-radius: 50%; display: inline-block;
        }
        .device-dot.online {
            background: #3b82f6; box-shadow: 0 0 6px #3b82f6; animation: pulse-green 2s infinite;
        }
        .device-dot.offline {
            background: #1e40af;
        }
        .device-time {
            font-size: 10px; color: var(--text-secondary); opacity: 0.7;
        }
        @keyframes pulse-green {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        /* Alert Notification Banner */
        .alert-banner {
            position: fixed; top: 15px; right: 15px; z-index: 3000; max-width: 380px; padding: 14px 20px; border-radius: 12px;
            color: white; font-size: 13px; font-weight: 500; box-shadow: 0 10px 30px rgba(0,0,0,0.3); animation: slideInRight 0.4s ease; display: flex; align-items: center;
            gap: 10px;
        }
            .alert-banner.danger { background: linear-gradient(135deg, #1e40af, #1d4ed8); }
            .alert-banner.warning { background: linear-gradient(135deg, #0ea5e9, #0284c7); }
            .alert-banner .alert-close { background: none; border: none; color: white; cursor: pointer; font-size: 16px; margin-left: auto; opacity: 0.8; }
            .alert-banner .alert-close:hover { opacity: 1; }
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        /* Theme Toggle */
        .theme-toggle {
            display: flex; align-items: center; gap: 10px; padding: 12px 16px; margin: 8px 0; border-radius: 8px;
            cursor: pointer; transition: all 0.3s; color: var(--text-secondary); border: 1px solid var(--border); background: transparent; width: 100%;
            font-size: 14px;
        }
        .theme-toggle:hover {
            background: var(--bg-card-hover); color: var(--text-primary);
        }
        .theme-toggle i {
            font-size: 16px; width: 20px; text-align: center;
        }
        .theme-divider {
            border: none; border-top: 1px solid var(--border); margin: 15px 0;
        }
        /* Role-based access: hide admin-only nav items for user role */
        body.role-user .admin-only { display: none !important; }
        /* Light theme adjustments */
        .slider {
            background: var(--border);
        }
        select.slider {
            background: var(--input-bg); color: var(--text-primary);
        }
    </style>
</head>
<body>
    <!-- Hamburger Menu Button (mobile) -->
    <button class="hamburger-btn" id="hamburger-btn" onclick="toggleSidebar()">
        <i class="fas fa-bars" id="hamburger-icon"></i>
    </button>
    <div class="sidebar-overlay" id="sidebar-overlay" onclick="toggleSidebar()"></div>

    <!-- Sidebar -->
    <div class="sidebar" id="sidebar">
        <div class="logo">
            Smart Room
        </div>
        <!-- Role badge -->
        <div id="role-badge" style="display:flex;align-items:center;gap:8px;margin-bottom:14px;padding:8px 10px;border-radius:8px;background:rgba(37,99,235,0.12);border:1px solid rgba(37,99,235,0.25);">
            <i class="fas fa-user-circle" style="color:#2563eb;font-size:16px;"></i>
            <span id="role-username" style="color:var(--text-primary);font-size:13px;font-weight:600;">-</span>
            <span id="role-label" style="margin-left:auto;padding:2px 8px;border-radius:12px;background:#2563eb;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;">-</span>
        </div>
        <div class="nav-item active" onclick="showPage('dashboard-ac')">
            <i class="fas fa-snowflake"></i>
            <span>AC Dashboard</span>
        </div>
        <div class="nav-item" onclick="showPage('dashboard-lamp')">
            <span>Lamp Dashboard</span>
        </div>
        <div class="nav-item admin-only" onclick="showPage('ac-analytics')">
            <span>AC Analytics</span>
        </div>
        <div class="nav-item admin-only" onclick="showPage('lamp-analytics')">
            <span>Lamp Analytics</span>
        </div>
        <div class="nav-item admin-only" onclick="showPage('camera')">
            <span>Camera</span>
        </div>
        <div class="nav-item admin-only" onclick="showPage('energy')">
            <span>Energy Usage</span>
        </div>
        <div class="nav-item" onclick="showPage('control-ac')">
            <span>AC Control</span>
        </div>
        <div class="nav-item" onclick="showPage('control-lamp')">
            <span>Lamp Control</span>
        </div>
        <div class="nav-item" onclick="showPage('control-outlet')">
            <span>Outlet Control</span>
        </div>
        <div class="nav-item admin-only" onclick="showPage('outlet-analysis')">
            <span>Outlet Analysis</span>
        </div>
        <div class="nav-item admin-only" onclick="showPage('ml-optimization')">
            <span>ML Optimization</span>
        </div>
        <div class="nav-item admin-only" onclick="showPage('logs')">
            <span>System Logs</span>
        </div>
        <div class="nav-item admin-only" onclick="showPage('occupancy-feedback')">
            <span>Occupancy Trend & Feedback</span>
        </div>
        <hr class="theme-divider">
        <button class="theme-toggle" onclick="toggleTheme()" id="theme-toggle-btn">
            <i class="fas fa-moon" id="theme-icon"></i>
            <span id="theme-label">Dark Mode</span>
        </button>
        <a href="/logout" class="theme-toggle" style="text-decoration: none; color: var(--danger); border-color: var(--danger);">
            <span>Logout</span>
        </a>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <!-- Sensor Health Bar (global, shown on all pages) -->
        <div id="sensor-health-bar" style="
            display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
            padding: 7px 18px; margin-bottom: 4px;
            background: var(--card-bg); border-radius: 10px;
            border: 1px solid var(--border);
            font-size: 12.5px; font-weight: 500; min-height: 34px;
        ">
            <span style="color:var(--text-secondary); font-size:11px; font-weight:600; letter-spacing:.5px;">SENSOR:</span>
        </div>
        <!-- /Sensor Health Bar -->
        <!-- AC Dashboard Page -->
        <div id="dashboard-ac" class="page active">
            <div class="header">
                <h1>AC Dashboard</h1>
                <p>Air Conditioning monitoring & status <button onclick="document.getElementById('diag-panel').style.display='block'" style="margin-left: 10px; padding: 3px 10px; font-size: 11px; background: #0ea5e9; border: none; color: white; border-radius: 6px; cursor: pointer;">Diagnostics</button></p>
                <div id="device-status-bar" style="display: flex; gap: 15px; margin-top: 12px; flex-wrap: wrap;">
                    <div class="device-status-item" id="ds-mqtt-broker" style="cursor: pointer;" onclick="checkMqttStatus()">
                        <span class="device-dot offline" id="mqtt-dot"></span>
                        <span>MQTT</span>
                        <span class="device-time" id="mqtt-status-text">Checking...</span>
                    </div>
                    <div class="device-status-item" id="ds-esp32-ac">
                        <span class="device-dot offline"></span>
                        <span>ESP32-AC</span>
                        <span class="device-time" id="ds-ac-time">Never</span>
                    </div>
                    <div class="device-status-item" id="ds-camera">
                        <span class="device-dot offline"></span>
                        <span>Camera</span>
                        <span class="device-time" id="ds-cam-time">Never</span>
                    </div>
                </div>
            </div>

            <!-- DIAGNOSTIC PANEL -->
            <div id="diag-panel" style="background: var(--card-bg); border: 2px solid #0ea5e9; border-radius: 12px; padding: 16px; margin-bottom: 16px; display: none;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <div style="font-size: 18px; color: #0ea5e9; font-weight: 800;">D</div>
                    <strong style="color: #0ea5e9;">Diagnostic Mode</strong>
                    <button onclick="document.getElementById('diag-panel').style.display='none'" style="margin-left: auto; background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 18px;">&times;</button>
                </div>
                <div id="diag-result" style="font-family: monospace; font-size: 12px; background: var(--bg-secondary); padding: 10px; border-radius: 8px; margin-bottom: 12px; min-height: 60px; white-space: pre-wrap; color: var(--text-primary);">Click the button below to diagnose...</div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button onclick="runSimulate()" style="padding: 8px 16px; font-size: 13px; cursor: pointer; background: #3b82f6; border: none; color: white; border-radius: 8px; border-radius: 8px;">
                        Test Frontend (Inject Dummy Data)
                    </button>
                    <button onclick="runMqttSelftest()" style="padding: 8px 16px; font-size: 13px; cursor: pointer; background: #3b82f6; border: none; color: white; border-radius: 8px;">
                        Test MQTT Broker (Self-Test)
                    </button>
                    <button onclick="runMqttReconnect()" style="padding: 8px 16px; font-size: 13px; cursor: pointer; background: #3b82f6; border: none; color: white; border-radius: 8px;">
                        Reconnect MQTT
                    </button>
                    <button onclick="checkMqttStatus(true)" style="padding: 8px 16px; font-size: 13px; cursor: pointer; background: #6b7280; border: none; color: white; border-radius: 8px;">
                        MQTT Status Detail
                    </button>
                </div>
            </div>

            <div class="stats-grid dashboard-grid">
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Room Temperature</span>
                        <div class="stat-icon" style="background: rgba(30, 64, 175, 0.2); color: #1e40af;">
                            T
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-temp">0</span>°C</div>
                    <div class="stat-change up">
                        <span>Avg 3×DHT22 — Real-time</span>
                    </div>
                    <div style="display: flex; gap: 6px; margin-top: 8px; font-size: 11px; color: var(--text-secondary);">
                        <span style="flex:1; text-align:center; background: rgba(30,64,175,0.08); border-radius: 6px; padding: 3px 0;">S1 <small>(Near)</small>: <strong id="dash-temp1">0</strong>°C</span>
                        <span style="flex:1; text-align:center; background: rgba(30,64,175,0.08); border-radius: 6px; padding: 3px 0;">S2 <small>(Mid)</small>: <strong id="dash-temp2">0</strong>°C</span>
                        <span style="flex:1; text-align:center; background: rgba(30,64,175,0.08); border-radius: 6px; padding: 3px 0;">S3 <small>(Far)</small>: <strong id="dash-temp3">0</strong>°C</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Humidity</span>
                        <div class="stat-icon" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">
                            H
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-hum">0</span>%</div>
                    <div class="stat-change">
                        <span>Avg 3×DHT22 — Real-time</span>
                    </div>
                    <div style="display: flex; gap: 6px; margin-top: 8px; font-size: 11px; color: var(--text-secondary);">
                        <span style="flex:1; text-align:center; background: rgba(59,130,246,0.08); border-radius: 6px; padding: 3px 0;">S1 <small>(Near)</small>: <strong id="dash-hum1">0</strong>%</span>
                        <span style="flex:1; text-align:center; background: rgba(59,130,246,0.08); border-radius: 6px; padding: 3px 0;">S2 <small>(Mid)</small>: <strong id="dash-hum2">0</strong>%</span>
                        <span style="flex:1; text-align:center; background: rgba(59,130,246,0.08); border-radius: 6px; padding: 3px 0;">S3 <small>(Far)</small>: <strong id="dash-hum3">0</strong>%</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Heat Index</span>
                        <div class="stat-icon" style="background: rgba(14, 165, 233, 0.2); color: #0ea5e9;">
                            HI
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-heat-index">0</span>°C</div>
                    <div class="stat-change">
                        <span>Feels like — Combination of temp &amp; humidity</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">ESP32 Signal</span>
                        <div class="stat-icon" style="background: rgba(37, 99, 235, 0.2); color: #2563eb;">
                            dB
                        </div>
                    </div>
                    <div class="stat-value" style="font-size: 24px;"><span id="dash-rssi">0</span> dBm</div>
                    <div class="stat-change">
                        <span>Uptime: <span id="dash-uptime" style="font-weight: bold;">0</span>s</span>
                    </div>
                </div>

                <!-- AC Status - Full Width Detailed Panel -->
                <div class="stat-card" id="ac-status-panel" style="grid-column: 1 / -1; background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 0; overflow: hidden;">
                    <!-- Header Bar -->
                    <div id="ac-panel-header" style="padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(79, 70, 229, 0.08)); border-bottom: 1px solid var(--border);">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(37, 99, 235, 0.2); display: flex; align-items: center; justify-content: center; font-weight: 800; color: #2563eb; font-size: 14px;">
                                AC
                            </div>
                            <div>
                                <div style="font-size: 15px; font-weight: 700; color: var(--text);">AC Status</div>
                                <div style="font-size: 11px; color: var(--text-secondary);">Mitsubishi Heavy Industries — Real-time</div>
                            </div>
                        </div>
                        <div id="ac-panel-power" style="display: flex; align-items: center; gap: 8px;">
                            <div id="ac-panel-dot" style="width: 12px; height: 12px; border-radius: 50%; background: #1e40af; box-shadow: 0 0 8px rgba(30, 64, 175, 0.5);"></div>
                            <span id="dash-ac-state" style="font-size: 16px; font-weight: 800; color: #1e40af;">OFF</span>
                        </div>
                    </div>
                    <!-- Body Grid -->
                    <div style="padding: 16px 20px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                        <!-- Set Temperature -->
                        <div style="text-align: center; padding: 14px 8px; background: rgba(59, 130, 246, 0.06); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                Set Temperature
                            </div>
                            <div style="font-size: 28px; font-weight: 800; color: #3b82f6;" id="dash-ac-temp">24</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">°C</div>
                        </div>
                        <!-- Fan Speed -->
                        <div style="text-align: center; padding: 14px 8px; background: rgba(59, 130, 246, 0.06); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                Fan Speed
                            </div>
                            <div style="font-size: 28px; font-weight: 800; color: #3b82f6;" id="dash-ac-fan">1</div>
                            <div style="font-size: 12px; color: var(--text-secondary);" id="dash-ac-fan-label">Low</div>
                        </div>
                        <!-- AC Mode (COOL/HEAT/DRY/FAN/AUTO) -->
                        <div style="text-align: center; padding: 14px 8px; background: rgba(14, 165, 233, 0.06); border-radius: 12px; border: 1px solid rgba(14, 165, 233, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                Mode AC
                            </div>
                            <div style="font-size: 22px; font-weight: 800; color: #0ea5e9;" id="dash-ac-mode-icon">❄</div>
                            <div style="font-size: 14px; font-weight: 700; color: #0ea5e9; margin-top: 2px;" id="dash-ac-mode">COOL</div>
                        </div>
                        <!-- Operating Mode (ADAPTIVE/MANUAL) -->
                        <div style="text-align: center; padding: 14px 8px; background: rgba(59, 130, 246, 0.06); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                Control
                            </div>
                            <div style="font-size: 22px; font-weight: 800; color: #3b82f6;" id="dash-ac-ctrl-icon">A</div>
                            <div style="font-size: 14px; font-weight: 700; color: #3b82f6; margin-top: 2px;" id="dash-ac-ctrl-mode">ADAPTIVE</div>
                        </div>
                    </div>
                    <!-- Footer: Room Environment + extra info -->
                    <div style="padding: 10px 20px 14px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border); font-size: 12px; color: var(--text-secondary);">
                        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                            <span>Avg: <strong id="dash-ac-room-temp" style="color: var(--text);">0</strong>°C</span>
                            <span style="font-size: 11px; color: var(--text-secondary);">S1(Near): <strong id="dash-ac-temp1" style="color: var(--text);">0</strong>°C</span>
                            <span style="font-size: 11px; color: var(--text-secondary);">S2(Mid): <strong id="dash-ac-temp2" style="color: var(--text);">0</strong>°C</span>
                            <span style="font-size: 11px; color: var(--text-secondary);">S3(Far): <strong id="dash-ac-temp3" style="color: var(--text);">0</strong>°C</span>
                            <span>Hum: <strong id="dash-ac-room-hum" style="color: var(--text);">0</strong>%</span>
                        </div>
                        <div id="dash-ac-source" style="padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; background: rgba(59, 130, 246, 0.12); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.25);">
                            AI Controlled
                        </div>
                    </div>
                </div>

                <!-- Fan & Mode Decision Range Table -->
                <div class="stat-card" style="grid-column: 1 / -1; padding: 0; overflow: hidden;">
                    <div style="padding: 14px 20px; display: flex; align-items: center; gap: 10px; background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(234, 179, 8, 0.08)); border-bottom: 1px solid var(--border);">
                        <div style="width: 36px; height: 36px; border-radius: 10px; background: rgba(14, 165, 233, 0.2); display: flex; align-items: center; justify-content: center; font-weight: 800; color: #0ea5e9; font-size: 13px;">
                            &#9881;
                        </div>
                        <div>
                            <div style="font-size: 15px; font-weight: 700; color: var(--text);">Fan &amp; Mode — Definisi Range</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Temperature &amp; humidity limits that determine Fan Speed and AC Mode output (from GA)</div>
                        </div>
                    </div>
                    <div style="padding: 16px 20px; overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 12px; color: var(--text);">
                            <thead>
                                <tr style="background: rgba(14, 165, 233, 0.06); border-bottom: 2px solid var(--border);">
                                    <th style="padding: 10px 12px; text-align: left; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px;">Kondisi</th>
                                    <th style="padding: 10px 12px; text-align: center; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px;">Set Temp</th>
                                    <th style="padding: 10px 12px; text-align: center; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px;">Fan Speed</th>
                                    <th style="padding: 10px 12px; text-align: center; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px;">Mode AC</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style="border-bottom: 1px solid var(--border);">
                                    <td style="padding: 10px 12px;"><span style="color: #6b7280; font-weight: 600;">Tier 0</span> — No person</td>
                                    <td style="padding: 10px 12px; text-align: center; font-weight: 600; color: #3b82f6;">27–29°C</td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(59,130,246,0.12); color: #3b82f6; font-weight: 600;">1 (Low)</span></td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(14,165,233,0.12); color: #0ea5e9; font-weight: 600;">COOL / FAN</span></td>
                                </tr>
                                <tr style="border-bottom: 1px solid var(--border);">
                                    <td style="padding: 10px 12px;"><span style="color: #3b82f6; font-weight: 600;">Tier 1</span> — 1–2 persons, RH ≤ 65%</td>
                                    <td style="padding: 10px 12px; text-align: center; font-weight: 600; color: #3b82f6;">24–26°C</td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(14,165,233,0.12); color: #0ea5e9; font-weight: 600;">1–2 (Adaptif)</span></td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(14,165,233,0.12); color: #0ea5e9; font-weight: 600;">COOL</span></td>
                                </tr>
                                <tr style="border-bottom: 1px solid var(--border);">
                                    <td style="padding: 10px 12px;"><span style="color: #3b82f6; font-weight: 600;">Tier 1</span> — 1–2 persons, RH &gt; 65%</td>
                                    <td style="padding: 10px 12px; text-align: center; font-weight: 600; color: #3b82f6;">22–24°C</td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(14,165,233,0.12); color: #0ea5e9; font-weight: 600;">2–3</span></td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(37,99,235,0.12); color: #2563eb; font-weight: 600;">DRY</span></td>
                                </tr>
                                <tr style="border-bottom: 1px solid var(--border);">
                                    <td style="padding: 10px 12px;"><span style="color: #0ea5e9; font-weight: 600;">Tier 2</span> — 3–5 persons</td>
                                    <td style="padding: 10px 12px; text-align: center; font-weight: 600; color: #3b82f6;">22°C</td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(30,64,175,0.12); color: #1e40af; font-weight: 600;">3 (High)</span></td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(14,165,233,0.12); color: #0ea5e9; font-weight: 600;">COOL</span></td>
                                </tr>
                                <tr style="border-bottom: 1px solid var(--border);">
                                    <td style="padding: 10px 12px;"><span style="color: #1e40af; font-weight: 600;">Tier 3</span> — &gt;5 persons</td>
                                    <td style="padding: 10px 12px; text-align: center; font-weight: 600; color: #3b82f6;">16°C (Min)</td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(30,64,175,0.12); color: #1e40af; font-weight: 600;">4 (Max)</span></td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(14,165,233,0.12); color: #0ea5e9; font-weight: 600;">COOL</span></td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px 12px;"><span style="color: #2563eb; font-weight: 600;">Override</span> — RH &gt; 65% (all tiers)</td>
                                    <td style="padding: 10px 12px; text-align: center; font-weight: 600; color: #6b7280;">−2°C offset</td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(37,99,235,0.12); color: #2563eb; font-weight: 600;">+1 boost</span></td>
                                    <td style="padding: 10px 12px; text-align: center;"><span style="padding: 2px 10px; border-radius: 12px; background: rgba(37,99,235,0.12); color: #2563eb; font-weight: 600;">DRY</span></td>
                                </tr>
                            </tbody>
                        </table>
                        <div style="margin-top: 12px; padding: 8px 14px; border-radius: 8px; background: rgba(14, 165, 233, 0.06); border: 1px solid rgba(14, 165, 233, 0.15); font-size: 11px; color: var(--text-secondary);">
                            <strong style="color: #0ea5e9;">&#9432;</strong> GA Optimization automatically determines the optimal combination based on room conditions, number of people, and temperature trends. The table above is a general guide.
                        </div>
                    </div>
                </div>

                <!-- Energy Usage, Person Detection, ML Optimization removed from AC Dashboard -->

                <!-- ===== PANEL EXPORT TEMPERATURE & HUMIDITY FROM DATABASE ===== -->
                <div style="grid-column:1/-1;margin-top:6px;padding:22px;border-radius:18px;border:1px solid rgba(59,130,246,0.25);background:linear-gradient(160deg,rgba(59,130,246,0.06),rgba(241,245,249,0.96));box-shadow:0 6px 24px rgba(0,0,0,0.06);">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:10px;">
                        <div>
                            <h2 style="font-size:16px;font-weight:700;color:var(--text);margin:0 0 4px;"><i class="fas fa-database" style="color:#3b82f6;margin-right:8px;"></i>Export Temperature &amp; Humidity</h2>
                            <p style="font-size:12px;color:var(--text-secondary);margin:0;">Fetch temperature &amp; humidity data from database — select date range, export to CSV</p>
                        </div>
                        <div style="padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(59,130,246,0.12);color:#3b82f6;border:1px solid rgba(59,130,246,0.3);">
                            <i class="fas fa-database" style="font-size:9px;vertical-align:middle;margin-right:5px;"></i> InfluxDB
                        </div>
                    </div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px;">
                        <div>
                            <label style="display:block;font-size:11px;color:var(--text-secondary);margin-bottom:4px;">From Date</label>
                            <input type="date" id="db-temp-from" style="padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:13px;">
                        </div>
                        <div>
                            <label style="display:block;font-size:11px;color:var(--text-secondary);margin-bottom:4px;">To Date</label>
                            <input type="date" id="db-temp-to" style="padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:13px;">
                        </div>
                        <button onclick="dbExportCSV('temp')"
                            style="padding:10px 22px;border-radius:10px;border:none;background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;font-size:13px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:7px;">
                            <i class="fas fa-download"></i> Export CSV
                        </button>
                        <button onclick="dbExportSetToday('temp')"
                            style="padding:10px 14px;border-radius:10px;border:1px solid rgba(59,130,246,0.35);background:rgba(59,130,246,0.08);color:#3b82f6;font-size:12px;font-weight:600;cursor:pointer;">
                            Today
                        </button>
                        <button onclick="dbExportSet7d('temp')"
                            style="padding:10px 14px;border-radius:10px;border:1px solid rgba(59,130,246,0.35);background:rgba(59,130,246,0.08);color:#3b82f6;font-size:12px;font-weight:600;cursor:pointer;">
                            7 Days
                        </button>
                        <button onclick="dbExportSet30d('temp')"
                            style="padding:10px 14px;border-radius:10px;border:1px solid rgba(59,130,246,0.35);background:rgba(59,130,246,0.08);color:#3b82f6;font-size:12px;font-weight:600;cursor:pointer;">
                            30 Days
                        </button>
                    </div>
                    <div style="font-size:12px;color:var(--text-secondary);padding:10px 14px;background:rgba(59,130,246,0.05);border-radius:8px;border:1px solid rgba(59,130,246,0.12);">
                        <i class="fas fa-info-circle" style="color:#3b82f6;margin-right:6px;"></i>
                        Data fetched directly from InfluxDB — columns: <strong>Timestamp, Temp Ruangan (°C), Kelembapan (%), Heat Index (°C), Sensor 1–3 Suhu/RH, Set Temperature AC (°C), Fan Speed, Mode AC (COOL/DRY/FAN/AUTO), Set RH (%), Status Kontrol AC (ON/OFF), Mode Kontrol (ADAPTIVE/MANUAL)</strong>
                    </div>
                </div>

            </div>
        </div>

        <!-- Lamp Dashboard Page -->
        <div id="dashboard-lamp" class="page">
            <div class="header">
                <h1>Lamp Dashboard</h1>
                <p>Lighting monitoring — 3 Sensors, 2 Lamps (GPIO 25 & 26)</p>
                <div style="display: flex; gap: 15px; margin-top: 12px; flex-wrap: wrap;">
                    <div class="device-status-item" id="ds-esp32-lamp">
                        <span class="device-dot offline"></span>
                        <span>ESP32-Lamp</span>
                        <span class="device-time" id="ds-lamp-time">Never</span>
                    </div>
                </div>
            </div>

            <div class="stats-grid dashboard-grid">
                <!-- Sensor 1 -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Sensor 1 — Lux</span>
                        <div class="stat-icon" style="background: rgba(14, 165, 233, 0.2); color: #0ea5e9;">
                            S1
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux1">0</span> lx</div>
                    <div class="stat-change">
                        <span>BH1750 Channel 1</span>
                    </div>
                </div>

                <!-- Sensor 2 -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Sensor 2 — Lux</span>
                        <div class="stat-icon" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">
                            S2
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux2">0</span> lx</div>
                    <div class="stat-change">
                        <span>BH1750 Channel 2</span>
                    </div>
                </div>

                <!-- Sensor 3 -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Sensor 3 — Lux</span>
                        <div class="stat-icon" style="background: rgba(37, 99, 235, 0.2); color: #2563eb;">
                            S3
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux3">0</span> lx</div>
                    <div class="stat-change">
                        <span>BH1750 Channel 3</span>
                    </div>
                </div>

                <!-- Average Lux -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Average Lux (3 Sensors)</span>
                        <div class="stat-icon" style="background: rgba(14, 165, 233, 0.2); color: #0ea5e9;">
                            AVG
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux-avg">0</span> lx</div>
                    <div class="stat-change">
                        <span>Target: 400 lx</span>
                    </div>
                </div>

                <!-- Lamp 1 Brightness -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Lamp 1 — Brightness</span>
                        <div class="stat-icon" style="background: rgba(251, 191, 36, 0.2); color: #fbbf24;">
                            L1
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-bright1">0</span>%</div>
                    <div class="stat-change">
                        <span>GPIO 25 (DAC)</span>
                    </div>
                </div>

                <!-- Lamp 2 Brightness -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Lamp 2 — Brightness</span>
                        <div class="stat-icon" style="background: rgba(251, 191, 36, 0.2); color: #fbbf24;">
                            L2
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-bright2">0</span>%</div>
                    <div class="stat-change">
                        <span>GPIO 26 (DAC)</span>
                    </div>
                </div>

                <!-- Motion Detection -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Motion Detection</span>
                        <div class="stat-icon" style="background: rgba(37, 99, 235, 0.2); color: #2563eb;">
                            PIR
                        </div>
                    </div>
                    <div class="stat-value" style="font-size: 24px;"><span id="dash-motion">NO MOTION</span></div>
                    <div class="stat-change">
                        <span>PIR Sensor</span>
                    </div>
                </div>

                <!-- PSO Control -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">PSO -> Lamp Control</span>
                        <div class="stat-icon" style="background: rgba(14, 165, 233, 0.2); color: #0ea5e9;">
                            PSO
                        </div>
                    </div>
                    <div class="stat-value" style="font-size: 24px;"><span>0.00</span></div>
                    <div class="stat-change" style="display: flex; flex-direction: column; gap: 4px;">
                        <span>Fitness Score</span>
                        <span style="font-size: 11px; color: #94a3b8;">Brightness: <span style="color: #0ea5e9; font-weight: bold;">--</span>%</span>
                    </div>
                </div>
                <!-- ===== PANEL EXPORT LUX & BRIGHTNESS FROM DATABASE ===== -->
                <div style="grid-column:1/-1;margin-top:6px;padding:22px;border-radius:18px;border:1px solid rgba(234,179,8,0.25);background:linear-gradient(160deg,rgba(234,179,8,0.06),rgba(241,245,249,0.96));box-shadow:0 6px 24px rgba(0,0,0,0.06);">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:10px;">
                        <div>
                            <h2 style="font-size:16px;font-weight:700;color:var(--text);margin:0 0 4px;"><i class="fas fa-database" style="color:#eab308;margin-right:8px;"></i>Export Lux &amp; Lamp Brightness</h2>
                            <p style="font-size:12px;color:var(--text-secondary);margin:0;">Fetch lux &amp; brightness data from database — select date range, export to CSV</p>
                        </div>
                        <div style="padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(234,179,8,0.12);color:#ca8a04;border:1px solid rgba(234,179,8,0.3);">
                            <i class="fas fa-database" style="font-size:9px;vertical-align:middle;margin-right:5px;"></i> InfluxDB
                        </div>
                    </div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px;">
                        <div>
                            <label style="display:block;font-size:11px;color:var(--text-secondary);margin-bottom:4px;">From Date</label>
                            <input type="date" id="db-lux-from" style="padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:13px;">
                        </div>
                        <div>
                            <label style="display:block;font-size:11px;color:var(--text-secondary);margin-bottom:4px;">To Date</label>
                            <input type="date" id="db-lux-to" style="padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:13px;">
                        </div>
                        <button onclick="dbExportCSV('lux')"
                            style="padding:10px 22px;border-radius:10px;border:none;background:linear-gradient(135deg,#eab308,#ca8a04);color:#fff;font-size:13px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:7px;">
                            <i class="fas fa-download"></i> Export CSV
                        </button>
                        <button onclick="dbExportSetToday('lux')"
                            style="padding:10px 14px;border-radius:10px;border:1px solid rgba(234,179,8,0.35);background:rgba(234,179,8,0.08);color:#ca8a04;font-size:12px;font-weight:600;cursor:pointer;">
                            Today
                        </button>
                        <button onclick="dbExportSet7d('lux')"
                            style="padding:10px 14px;border-radius:10px;border:1px solid rgba(234,179,8,0.35);background:rgba(234,179,8,0.08);color:#ca8a04;font-size:12px;font-weight:600;cursor:pointer;">
                            7 Days
                        </button>
                        <button onclick="dbExportSet30d('lux')"
                            style="padding:10px 14px;border-radius:10px;border:1px solid rgba(234,179,8,0.35);background:rgba(234,179,8,0.08);color:#ca8a04;font-size:12px;font-weight:600;cursor:pointer;">
                            30 Days
                        </button>
                    </div>
                    <div style="font-size:12px;color:var(--text-secondary);padding:10px 14px;background:rgba(234,179,8,0.05);border-radius:8px;border:1px solid rgba(234,179,8,0.12);">
                        <i class="fas fa-info-circle" style="color:#eab308;margin-right:6px;"></i>
                        Data fetched directly from InfluxDB — columns: <strong>Time, Lux1 (lx), Lux2 (lx), Lux3 (lx), Avg Lux (lx), Brightness1 (%), Brightness2 (%)</strong>
                    </div>
                </div>

            </div>
        </div>

        <!-- AC Analytics Page -->
        <div id="ac-analytics" class="page">
            <div class="header">
                <h1>AC Analytics</h1>
                <p>Air Conditioning performance and optimization</p>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Set Temperature</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('acTemp', 'Set Temperature (C)')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="exportChartRange('acTemp', 'Set Temperature (C)')" title="Export date range">&#128197;</button>
                        <button class="chart-option-btn active" onclick="changeChartRange('acTemp', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('acTemp', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('acTemp', 24)">24h</button>
                    </div>
                </div>
                <canvas id="acTempChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Set RH</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('acHum', 'Set RH (%)')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="exportChartRange('acHum', 'Set RH (%)')" title="Export date range">&#128197;</button>
                        <button class="chart-option-btn active" onclick="changeChartRange('acHum', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('acHum', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('acHum', 24)">24h</button>
                    </div>
                </div>
                <canvas id="acHumChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Actual Temperature</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('temp', 'Actual Temperature (C)')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="exportChartRange('temp', 'Actual Temperature (C)')" title="Export date range">&#128197;</button>
                        <button class="chart-option-btn active" onclick="changeChartRange('temp', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('temp', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('temp', 24)">24h</button>
                    </div>
                </div>
                <canvas id="tempChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Actual RH</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('hum', 'Actual RH (%)')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="exportChartRange('hum', 'Actual RH (%)')" title="Export date range">&#128197;</button>
                        <button class="chart-option-btn active" onclick="changeChartRange('hum', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('hum', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('hum', 24)">24h</button>
                    </div>
                </div>
                <canvas id="humChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">AC Power (kW)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="changeChartRange('acPower', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('acPower', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('acPower', 24)">24h</button>
                    </div>
                </div>
                <canvas id="acPowerChart" height="80"></canvas>
            </div>

            <!-- AC Energy Usage Section -->
            <div style="margin-top:24px;padding:22px;border-radius:18px;border:1px solid rgba(59,130,246,0.25);background:linear-gradient(160deg,rgba(59,130,246,0.06),var(--bg-card));box-shadow:0 8px 28px rgba(15,23,42,0.08);">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
                    <div style="width:36px;height:36px;border-radius:10px;background:rgba(59,130,246,0.18);display:flex;align-items:center;justify-content:center;font-size:17px;">&#9889;</div>
                    <div>
                        <div style="font-size:16px;font-weight:700;color:var(--text);">AC Energy Consumption</div>
                        <div style="font-size:11px;color:var(--text-secondary);">Energy consumption per interval from PZEM sensor</div>
                    </div>
                </div>
                <div class="chart-container" style="border:none;box-shadow:none;padding:0;">
                    <div class="chart-header">
                        <div class="chart-title">AC Energy Consumption (kWh)</div>
                        <div class="chart-options">
                            <button class="chart-option-btn active" onclick="loadAnalyticsEnergy('ac','24h',this)">24h</button>
                            <button class="chart-option-btn" onclick="loadAnalyticsEnergy('ac','7d',this)">7d</button>
                            <button class="chart-option-btn" onclick="loadAnalyticsEnergy('ac','30d',this)">30d</button>
                        </div>
                    </div>
                    <canvas id="acEnergyKwhChart" height="90"></canvas>
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px;">
                    <div style="padding:12px 16px;border-radius:12px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.18);text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Last Interval</div>
                        <div id="ac-analytics-kwh-last" style="font-size:18px;font-weight:700;color:#3b82f6;">--</div>
                    </div>
                    <div style="padding:12px 16px;border-radius:12px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.18);text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Total Period</div>
                        <div id="ac-analytics-kwh-total" style="font-size:18px;font-weight:700;color:#3b82f6;">--</div>
                    </div>
                    <div style="padding:12px 16px;border-radius:12px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.18);text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Avg per Interval</div>
                        <div id="ac-analytics-kwh-avg" style="font-size:18px;font-weight:700;color:#3b82f6;">--</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Lamp Analytics Page -->
        <div id="lamp-analytics" class="page">
            <div class="header">
                <h1>Lamp Analytics</h1>
                <p>Lighting system performance and optimization</p>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Average Light Intensity (Lux) — 3 Sensors</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('lampLux', 'Lux')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="exportChartRange('lampLux', 'Lux')" title="Export date range">&#128197;</button>
                        <button class="chart-option-btn active" onclick="changeChartRange('lampLux', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampLux', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampLux', 24)">24h</button>
                    </div>
                </div>
                <canvas id="lampLuxChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Average Brightness Level — 2 Lamps</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('lampBright', 'Brightness (%)')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="exportChartRange('lampBright', 'Brightness (%)')" title="Export date range">&#128197;</button>
                        <button class="chart-option-btn active" onclick="changeChartRange('lampBright', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampBright', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampBright', 24)">24h</button>
                    </div>
                </div>
                <canvas id="lampBrightChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Lamp Power Consumption (W)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="changeChartRange('lampPower', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampPower', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampPower', 24)">24h</button>
                    </div>
                </div>
                <canvas id="lampPowerChart" height="80"></canvas>
            </div>

            <!-- Lamp Energy Usage Section -->
            <div style="margin-top:24px;padding:22px;border-radius:18px;border:1px solid rgba(37,99,235,0.25);background:linear-gradient(160deg,rgba(37,99,235,0.06),var(--bg-card));box-shadow:0 8px 28px rgba(15,23,42,0.08);">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
                    <div style="width:36px;height:36px;border-radius:10px;background:rgba(37,99,235,0.18);display:flex;align-items:center;justify-content:center;font-size:17px;">&#128161;</div>
                    <div>
                        <div style="font-size:16px;font-weight:700;color:var(--text);">Lamp Energy Usage</div>
                        <div style="font-size:11px;color:var(--text-secondary);">Energy consumption per interval from PZEM sensor</div>
                    </div>
                </div>
                <div class="chart-container" style="border:none;box-shadow:none;padding:0;">
                    <div class="chart-header">
                        <div class="chart-title">Lamp Energy per Interval (kWh)</div>
                        <div class="chart-options">
                            <button class="chart-option-btn active" onclick="loadAnalyticsEnergy('lamp','24h',this)">24h</button>
                            <button class="chart-option-btn" onclick="loadAnalyticsEnergy('lamp','7d',this)">7d</button>
                            <button class="chart-option-btn" onclick="loadAnalyticsEnergy('lamp','30d',this)">30d</button>
                        </div>
                    </div>
                    <canvas id="lampEnergyKwhChart" height="90"></canvas>
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px;">
                    <div style="padding:12px 16px;border-radius:12px;background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.18);text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Last Interval</div>
                        <div id="lamp-analytics-kwh-last" style="font-size:18px;font-weight:700;color:#2563eb;">--</div>
                    </div>
                    <div style="padding:12px 16px;border-radius:12px;background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.18);text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Total Period</div>
                        <div id="lamp-analytics-kwh-total" style="font-size:18px;font-weight:700;color:#2563eb;">--</div>
                    </div>
                    <div style="padding:12px 16px;border-radius:12px;background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.18);text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Avg per Interval</div>
                        <div id="lamp-analytics-kwh-avg" style="font-size:18px;font-weight:700;color:#2563eb;">--</div>
                    </div>
                </div>
            </div>
        </div>
        <div id="camera" class="page">
            <div class="header">
                <h1>Live Camera Feed - YOLOv8 Detection</h1>
                <p>Real-time person detection using YOLOv8n</p>
            </div>

            <div class="camera-view">
                <div class="camera-feed-container" id="camera-feed-container">
                    <div class="camera-overlay-bar">
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span class="camera-rec-badge"><i class="fas fa-circle"></i> LIVE</span>
                            <span class="person-badge not-detected" id="overlay-person-badge">
                                No Person
                            </span>
                        </div>
                        <span class="camera-time-badge" id="camera-time">00:00:00</span>
                    </div>
                    <img id="camera-img" src="/video_feed" alt="Camera Feed"
                         onerror="this.style.display='none'; document.getElementById('camera-error').style.display='flex';">
                    <div id="camera-error" class="camera-error" style="display:none; flex-direction:column; align-items:center;">
                        <h3>Camera Not Available</h3>
                        <p style="margin-top:10px; font-size:14px; color:var(--text-secondary);">
                            Make sure USB camera is properly connected
                        </p>
                        <button class="btn btn-primary" style="margin-top:15px;" onclick="retryCamera()">
                            Retry Connection
                        </button>
                    </div>
                </div>

                <div class="camera-info-grid">
                    <div class="camera-info-card">
                        <div class="camera-info-label">Status</div>
                        <div class="camera-info-value" id="cam-status">Connecting...</div>
                    </div>
                    <div class="camera-info-card">
                        <div class="camera-info-label">Resolution</div>
                        <div class="camera-info-value" id="cam-resolution">Loading...</div>
                    </div>
                    <div class="camera-info-card">
                        <div class="camera-info-label">Frame Rate</div>
                        <div class="camera-info-value" id="cam-fps">Loading...</div>
                    </div>
                    <div class="camera-info-card" id="person-detected-card" style="transition: all 0.3s;">
                        <div class="camera-info-label">Person Detected</div>
                        <div class="camera-info-value" id="cam-person" style="color: var(--danger);">No</div>
                    </div>
                    <div class="camera-info-card" style="transition: all 0.3s;">
                        <div class="camera-info-label">Person Count</div>
                        <div class="camera-info-value" id="cam-count-display">0</div>
                    </div>
                    <div class="camera-info-card" style="transition: all 0.3s;">
                        <div class="camera-info-label">Confidence</div>
                        <div class="camera-info-value" id="cam-confidence-display">0%</div>
                    </div>
                </div>
                
                <div style="margin-top: 20px; text-align: center; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
                    <button class="btn" id="camera-toggle-btn" onclick="toggleCamera()" style="padding: 12px 24px; border-radius: 12px; font-weight: 600; background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; border: none; transition: all 0.3s;">
                        <span id="camera-toggle-text">Camera ON</span>
                    </button>
                    <button class="btn" id="sound-toggle-btn" onclick="toggleDetectionSound()" style="padding: 12px 24px; border-radius: 12px; font-weight: 600; background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; border: none; transition: all 0.3s;">
                        <span id="sound-toggle-text">Sound ON</span>
                    </button>
                    <button class="btn btn-success" onclick="retryCamera()" style="padding: 12px 24px; border-radius: 12px; font-weight: 600;">
                        Refresh Feed
                    </button>
                </div>
            </div>

            <!-- ===== PANEL OCCUPANCY HARIAN ===== -->
            <div style="margin-top:24px;padding:22px;border-radius:18px;border:1px solid rgba(37,99,235,0.25);background:linear-gradient(160deg,rgba(37,99,235,0.06),rgba(241,245,249,0.97));box-shadow:0 6px 24px rgba(0,0,0,0.06);">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:10px;">
                    <div>
                        <h2 style="font-size:16px;font-weight:700;color:var(--text);margin:0 0 4px;"><i class="fas fa-users" style="color:#2563eb;margin-right:8px;"></i>Daily Occupancy Chart</h2>
                        <p style="font-size:12px;color:var(--text-secondary);margin:0;">Persons detected per hour (auto-recorded every hour) &mdash; max 24 data/day</p>
                    </div>
                    <div style="display:flex;gap:8px;align-items:center;">
                        <div id="occ-rec-badge" style="padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(59,130,246,0.15);color:#3b82f6;border:1px solid rgba(59,130,246,0.3);"><i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;"></i> Auto-Recording</div>
                        <button onclick="occExportCSV()" style="padding:8px 18px;border-radius:10px;border:none;background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;font-size:12px;font-weight:700;cursor:pointer;"><i class="fas fa-download" style="margin-right:5px;"></i>Export CSV</button>
                        <button onclick="dbExportCSV('occupancy')" title="Export from InfluxDB" style="padding:8px 14px;border-radius:10px;border:none;background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;font-size:12px;font-weight:700;cursor:pointer;"><i class="fas fa-database" style="margin-right:4px;"></i>DB Export</button>
                        <button onclick="occExportRange()" title="Export date range" style="padding:8px 13px;border-radius:10px;border:1px solid rgba(37,99,235,0.35);background:rgba(37,99,235,0.10);color:#2563eb;font-size:12px;font-weight:700;cursor:pointer;">&#128197; Rentang</button>
                        <button onclick="occClear()" style="padding:8px 14px;border-radius:10px;border:1px solid rgba(107,114,128,0.3);background:rgba(107,114,128,0.08);color:var(--text-secondary);font-size:12px;font-weight:600;cursor:pointer;"><i class="fas fa-trash-alt"></i></button>
                    </div>
                </div>
                <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;font-size:12px;">
                    <span style="color:var(--text-secondary);">Data shown: <strong id="occ-count" style="color:#2563eb;">0</strong> hrs</span>
                    <span style="color:var(--text-secondary);">Next: <strong id="occ-next-in" style="color:var(--text);">--</strong></span>
                    <span style="color:var(--text-secondary);">Total: <strong id="occ-total-today" style="color:#3b82f6;">0</strong> persons</span>
                </div>
                <!-- Range filter occupancy -->
                <div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;">
                    <button id="occ-tab-today" onclick="occSetRange('today')" style="padding:5px 14px;border-radius:8px;border:none;font-size:11px;font-weight:700;cursor:pointer;background:#2563eb;color:#fff;box-shadow:0 1px 4px rgba(37,99,235,0.3);">Today</button>
                    <button id="occ-tab-7d" onclick="occSetRange('7d')" style="padding:5px 14px;border-radius:8px;border:1px solid rgba(37,99,235,0.25);font-size:11px;font-weight:600;cursor:pointer;background:rgba(37,99,235,0.12);color:#2563eb;">7 Days</button>
                    <button id="occ-tab-30d" onclick="occSetRange('30d')" style="padding:5px 14px;border-radius:8px;border:1px solid rgba(37,99,235,0.25);font-size:11px;font-weight:600;cursor:pointer;background:rgba(37,99,235,0.12);color:#2563eb;">30 Days</button>
                    <button id="occ-tab-all" onclick="occSetRange('all')" style="padding:5px 14px;border-radius:8px;border:1px solid rgba(37,99,235,0.25);font-size:11px;font-weight:600;cursor:pointer;background:rgba(37,99,235,0.12);color:#2563eb;">All</button>
                </div>
                <!-- DB Export date range for occupancy -->
                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px;padding:10px 12px;border-radius:10px;background:rgba(37,99,235,0.04);border:1px solid rgba(37,99,235,0.12);">
                    <span style="font-size:11px;color:var(--text-secondary);font-weight:600;"><i class="fas fa-database" style="color:#2563eb;margin-right:4px;"></i>DB Export:</span>
                    <input type="date" id="db-occupancy-from" style="padding:6px 10px;border-radius:7px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:12px;">
                    <span style="font-size:11px;color:var(--text-secondary);">to</span>
                    <input type="date" id="db-occupancy-to" style="padding:6px 10px;border-radius:7px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:12px;">
                    <button onclick="dbExportSetToday('occupancy')" style="padding:6px 11px;border-radius:7px;border:1px solid rgba(37,99,235,0.25);background:rgba(37,99,235,0.08);color:#2563eb;font-size:11px;font-weight:600;cursor:pointer;">Today</button>
                    <button onclick="dbExportSet7d('occupancy')" style="padding:6px 11px;border-radius:7px;border:1px solid rgba(37,99,235,0.25);background:rgba(37,99,235,0.08);color:#2563eb;font-size:11px;font-weight:600;cursor:pointer;">7 Days</button>
                    <button onclick="dbExportSet30d('occupancy')" style="padding:6px 11px;border-radius:7px;border:1px solid rgba(37,99,235,0.25);background:rgba(37,99,235,0.08);color:#2563eb;font-size:11px;font-weight:600;cursor:pointer;">30 Days</button>
                </div>
                <div style="position:relative;height:220px;background:rgba(0,0,0,0.02);border-radius:12px;padding:12px;border:1px solid var(--border);">
                    <canvas id="occChart"></canvas>
                </div>
                <div style="overflow-x:auto;overflow-y:auto;max-height:280px;border-radius:10px;border:1px solid var(--border);margin-top:14px;">
                    <table style="width:100%;border-collapse:collapse;font-size:11px;">
                        <thead>
                            <tr style="background:rgba(37,99,235,0.08);">
                                <th style="padding:8px 12px;text-align:left;color:var(--text-secondary);font-weight:600;">Time</th>
                                <th style="padding:8px 12px;text-align:center;color:var(--text-secondary);font-weight:600;">Hour</th>
                                <th style="padding:8px 12px;text-align:right;color:#2563eb;font-weight:600;">Person Count</th>
                                <th style="padding:8px 12px;text-align:right;color:#0ea5e9;font-weight:600;">Confidence (%)</th>
                            </tr>
                        </thead>
                        <tbody id="occ-preview-body">
                            <tr><td colspan="4" style="text-align:center;padding:14px;color:var(--text-secondary);">Waiting for first hour data&hellip;</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Energy Usage Page -->
        <div id="energy" class="page">
            <div class="header">
                <h1>Energy Usage</h1>
                <p>Real-time &amp; historical energy monitoring &mdash; MySQL Database (iotlabun_sbms)</p>
            </div>

            <!-- ===== LIVE ENERGY MONITOR ===== -->
            <div style="background:var(--bg-card);border-radius:18px;border:1px solid rgba(37,99,235,0.2);overflow:hidden;margin-bottom:22px;box-shadow:0 4px 24px rgba(0,0,0,0.07);">
                <!-- Header -->
                <div style="padding:14px 20px;display:flex;justify-content:space-between;align-items:center;background:linear-gradient(135deg,rgba(37,99,235,0.10),rgba(79,70,229,0.06));border-bottom:1px solid rgba(37,99,235,0.15);">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div style="width:38px;height:38px;border-radius:10px;background:rgba(37,99,235,0.2);display:flex;align-items:center;justify-content:center;font-size:18px;">&#9889;</div>
                        <div>
                            <div style="font-size:15px;font-weight:700;color:var(--text);">Live Energy Monitor</div>
                            <div style="font-size:11px;color:var(--text-secondary);">MySQL iotlabun_sbms &mdash; updates every 5 seconds</div>
                        </div>
                    </div>
                    <div style="display:flex;align-items:center;gap:12px;">
                        <div style="font-size:11px;color:var(--text-secondary);">Update: <span id="energy-last-update" style="color:var(--text);font-weight:600;">--</span></div>
                        <div id="mysql-live-badge" style="padding:4px 14px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(107,114,128,0.15);color:#6b7280;border:1px solid rgba(107,114,128,0.3);transition:all 0.4s ease;">
                            <i class="fas fa-circle" style="font-size:6px;vertical-align:middle;margin-right:4px;"></i> Waiting...
                        </div>
                    </div>
                </div>

                <!-- AC + Outlet + Lamp columns -->
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;">
                    <!-- ── AC ── -->
                    <div style="padding:18px 20px;border-right:1px solid rgba(30,64,175,0.15);">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                            <div style="font-size:13px;font-weight:700;color:#1e40af;"><i class="fas fa-snowflake" style="margin-right:6px;"></i>AC (id_kwh=1)</div>
                            <div style="font-size:10px;padding:2px 10px;border-radius:20px;background:rgba(30,64,175,0.1);color:#1e40af;border:1px solid rgba(30,64,175,0.2);">Live</div>
                        </div>
                        <!-- Big power -->
                        <div style="text-align:center;margin-bottom:12px;">
                            <div style="font-size:48px;font-weight:800;color:#1e40af;line-height:1;transition:color 0.3s;"><span id="ac-power">--</span></div>
                            <div style="font-size:12px;color:var(--text-secondary);margin-top:3px;">kW &mdash; Power Active</div>
                            <div style="margin-top:8px;height:7px;border-radius:4px;background:rgba(30,64,175,0.12);overflow:hidden;">
                                <div id="eu-ac-pbar" style="height:100%;border-radius:4px;background:linear-gradient(90deg,#1e40af,#93c5fd);width:0%;transition:width 0.6s ease;"></div>
                            </div>
                            <div style="font-size:10px;color:var(--text-secondary);margin-top:3px;">of 3.0 kW max</div>
                        </div>
                        <!-- Metrics 3-col -->
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:7px;">
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(30,64,175,0.05);border:1px solid rgba(30,64,175,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Voltage</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-ac-voltage">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">V</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(30,64,175,0.05);border:1px solid rgba(30,64,175,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Current</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-ac-current">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">A</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.15);">
                                <div style="font-size:10px;color:var(--text-secondary);">Energy</div>
                                <div style="font-size:17px;font-weight:700;color:#3b82f6;transition:all 0.3s;"><span id="eu-ac-kwh">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">kWh</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(30,64,175,0.05);border:1px solid rgba(30,64,175,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Reaktif</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-ac-reactive">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">VAR</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(30,64,175,0.05);border:1px solid rgba(30,64,175,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Semu</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-ac-apparent">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">VA</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(30,64,175,0.05);border:1px solid rgba(30,64,175,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Frequency</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-ac-freq">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">Hz</div>
                            </div>
                        </div>
                        <!-- PF -->
                        <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-radius:10px;background:rgba(30,64,175,0.04);border:1px solid rgba(30,64,175,0.1);">
                            <span style="font-size:11px;color:var(--text-secondary);">Power Factor</span>
                            <div style="display:flex;align-items:center;gap:8px;">
                                <span style="font-size:17px;font-weight:700;color:var(--text);" id="eu-ac-pf">--</span>
                                <span id="eu-ac-pf-quality" style="font-size:10px;padding:2px 8px;border-radius:8px;background:rgba(107,114,128,0.15);color:#6b7280;transition:all 0.4s;">--</span>
                            </div>
                        </div>
                        <div style="margin-top:6px;font-size:10px;color:var(--text-secondary);">Updated: <span id="eu-ac-ts">--</span></div>
                    </div>

                    <!-- ── Outlet ── -->
                    <div style="padding:18px 20px;border-right:1px solid rgba(5,150,105,0.15);">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                            <div style="font-size:13px;font-weight:700;color:#059669;"><i class="fas fa-plug" style="margin-right:6px;"></i>Outlet (id_kwh=2)</div>
                            <div style="font-size:10px;padding:2px 10px;border-radius:20px;background:rgba(5,150,105,0.1);color:#059669;border:1px solid rgba(5,150,105,0.2);">Live</div>
                        </div>
                        <!-- Big power -->
                        <div style="text-align:center;margin-bottom:12px;">
                            <div style="font-size:48px;font-weight:800;color:#059669;line-height:1;transition:color 0.3s;"><span id="outlet-power">--</span></div>
                            <div style="font-size:12px;color:var(--text-secondary);margin-top:3px;">kW &mdash; Power Active</div>
                            <div style="margin-top:8px;height:7px;border-radius:4px;background:rgba(5,150,105,0.12);overflow:hidden;">
                                <div id="eu-outlet-pbar" style="height:100%;border-radius:4px;background:linear-gradient(90deg,#059669,#34d399);width:0%;transition:width 0.6s ease;"></div>
                            </div>
                            <div style="font-size:10px;color:var(--text-secondary);margin-top:3px;">of 1.0 kW max</div>
                        </div>
                        <!-- Metrics 3-col -->
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:7px;">
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Voltage</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-outlet-voltage">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">V</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Current</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-outlet-current">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">A</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.15);">
                                <div style="font-size:10px;color:var(--text-secondary);">Energy</div>
                                <div style="font-size:17px;font-weight:700;color:#3b82f6;transition:all 0.3s;"><span id="eu-outlet-kwh">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">kWh</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Reaktif</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-outlet-reactive">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">VAR</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Semu</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-outlet-apparent">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">VA</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Frequency</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-outlet-freq">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">Hz</div>
                            </div>
                        </div>
                        <!-- PF -->
                        <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-radius:10px;background:rgba(5,150,105,0.04);border:1px solid rgba(5,150,105,0.1);">
                            <span style="font-size:11px;color:var(--text-secondary);">Power Factor</span>
                            <div style="display:flex;align-items:center;gap:8px;">
                                <span style="font-size:17px;font-weight:700;color:var(--text);" id="eu-outlet-pf">--</span>
                                <span id="eu-outlet-pf-quality" style="font-size:10px;padding:2px 8px;border-radius:8px;background:rgba(107,114,128,0.15);color:#6b7280;transition:all 0.4s;">--</span>
                            </div>
                        </div>
                        <div style="margin-top:6px;font-size:10px;color:var(--text-secondary);">Updated: <span id="eu-outlet-ts">--</span></div>
                    </div>

                    <!-- ── Lamp ── -->
                    <div style="padding:18px 20px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                            <div style="font-size:13px;font-weight:700;color:#eab308;"><i class="fas fa-lightbulb" style="margin-right:6px;"></i>Lamp (id_kwh=3)</div>
                            <div style="font-size:10px;padding:2px 10px;border-radius:20px;background:rgba(234,179,8,0.1);color:#eab308;border:1px solid rgba(234,179,8,0.2);">Live</div>
                        </div>
                        <!-- Big power -->
                        <div style="text-align:center;margin-bottom:12px;">
                            <div style="font-size:48px;font-weight:800;color:#eab308;line-height:1;transition:color 0.3s;"><span id="lamp-power">--</span></div>
                            <div style="font-size:12px;color:var(--text-secondary);margin-top:3px;">kW &mdash; Power Active</div>
                            <div style="margin-top:8px;height:7px;border-radius:4px;background:rgba(234,179,8,0.12);overflow:hidden;">
                                <div id="eu-lamp-pbar" style="height:100%;border-radius:4px;background:linear-gradient(90deg,#eab308,#fde047);width:0%;transition:width 0.6s ease;"></div>
                            </div>
                            <div style="font-size:10px;color:var(--text-secondary);margin-top:3px;">of 0.5 kW max</div>
                        </div>
                        <!-- Metrics 3-col -->
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:7px;">
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(234,179,8,0.05);border:1px solid rgba(234,179,8,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Voltage</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-lamp-voltage">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">V</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(234,179,8,0.05);border:1px solid rgba(234,179,8,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Current</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-lamp-current">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">A</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.15);">
                                <div style="font-size:10px;color:var(--text-secondary);">Energy</div>
                                <div style="font-size:17px;font-weight:700;color:#3b82f6;transition:all 0.3s;"><span id="eu-lamp-kwh">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">kWh</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(234,179,8,0.05);border:1px solid rgba(234,179,8,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Reaktif</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-lamp-reactive">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">VAR</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(234,179,8,0.05);border:1px solid rgba(234,179,8,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Semu</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-lamp-apparent">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">VA</div>
                            </div>
                            <div style="text-align:center;padding:8px 5px;border-radius:10px;background:rgba(234,179,8,0.05);border:1px solid rgba(234,179,8,0.1);">
                                <div style="font-size:10px;color:var(--text-secondary);">Frequency</div>
                                <div style="font-size:17px;font-weight:700;color:var(--text);transition:all 0.3s;"><span id="eu-lamp-freq">--</span></div>
                                <div style="font-size:10px;color:var(--text-secondary);">Hz</div>
                            </div>
                        </div>
                        <!-- PF -->
                        <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-radius:10px;background:rgba(234,179,8,0.04);border:1px solid rgba(234,179,8,0.1);">
                            <span style="font-size:11px;color:var(--text-secondary);">Power Factor</span>
                            <div style="display:flex;align-items:center;gap:8px;">
                                <span style="font-size:17px;font-weight:700;color:var(--text);" id="eu-lamp-pf">--</span>
                                <span id="eu-lamp-pf-quality" style="font-size:10px;padding:2px 8px;border-radius:8px;background:rgba(107,114,128,0.15);color:#6b7280;transition:all 0.4s;">--</span>
                            </div>
                        </div>
                        <div style="margin-top:6px;font-size:10px;color:var(--text-secondary);">Updated: <span id="eu-lamp-ts">--</span></div>
                    </div>
                </div>

                <!-- Total footer row -->
                <div style="padding:14px 20px;background:linear-gradient(135deg,rgba(37,99,235,0.07),rgba(79,70,229,0.04));border-top:1px solid rgba(37,99,235,0.12);display:grid;grid-template-columns:repeat(5,1fr);gap:10px;">
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);margin-bottom:3px;">Total Power</div>
                        <div style="font-size:20px;font-weight:800;color:#2563eb;"><span id="total-power">--</span><span style="font-size:11px;"> kW</span></div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);margin-bottom:3px;">Total Current</div>
                        <div style="font-size:20px;font-weight:800;color:#2563eb;"><span id="total-current">--</span><span style="font-size:11px;"> A</span></div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);margin-bottom:3px;">Total Energy</div>
                        <div style="font-size:20px;font-weight:800;color:#3b82f6;"><span id="total-energy-kwh">--</span><span style="font-size:11px;"> kWh</span></div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);margin-bottom:3px;">Frequency</div>
                        <div style="font-size:20px;font-weight:800;color:var(--text);"><span id="total-freq-card">--</span><span style="font-size:11px;"> Hz</span></div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:var(--text-secondary);margin-bottom:3px;">Est. Cost</div>
                        <div style="font-size:17px;font-weight:800;color:#0ea5e9;">Rp <span id="daily-cost">--</span></div>
                        <div style="font-size:9px;color:var(--text-secondary);">@ Rp 1.500/kWh</div>
                    </div>
                </div>
            </div>

            <!-- ===== REAL-TIME CHARTS (last 30 points) ===== -->
            <div style="background:var(--bg-card);border-radius:16px;border:1px solid rgba(37,99,235,0.18);padding:20px;margin-bottom:22px;box-shadow:0 4px 16px rgba(0,0,0,0.05);">
                <h2 style="font-size:15px;font-weight:700;color:var(--text);margin:0 0 16px;"><i class="fas fa-chart-line" style="margin-right:8px;color:#2563eb;"></i>Real-time Charts &mdash; Last 30 Points</h2>
                <div class="chart-container" style="margin-bottom:16px;">
                    <div class="chart-header">
                        <div class="chart-title">Frequency <span style="color:#0ea5e9;">&#9632;</span> (Hz)</div>
                    </div>
                    <canvas id="mysqlVoltFreqChart" height="80"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Current &mdash; AC <span style="color:#1e40af;">&#9632;</span> &amp; Outlet <span style="color:#059669;">&#9632;</span> &amp; Lamp <span style="color:#eab308;">&#9632;</span> (A)</div>
                    </div>
                    <canvas id="mysqlCurrentChart" height="80"></canvas>
                </div>
            </div>

            <!-- ===== GRAFIK HISTORIS (InfluxDB) ===== -->
            <div style="background:var(--bg-card);border-radius:16px;border:1px solid var(--border);padding:20px;margin-bottom:22px;box-shadow:0 4px 16px rgba(0,0,0,0.05);">
                <h2 style="font-size:15px;font-weight:700;color:var(--text);margin:0 0 16px;"><i class="fas fa-history" style="margin-right:8px;color:#3b82f6;"></i>Historical Data (InfluxDB)</h2>

                <!-- Power -->
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Active Power &mdash; AC <span style="color:#1e40af;">&#9632;</span> &amp; Outlet <span style="color:#059669;">&#9632;</span> &amp; Lamp <span style="color:#eab308;">&#9632;</span> &amp; Total <span style="color:#7c3aed;">&#9632;</span> (kW)</div>
                        <div class="chart-options">
                            <button class="chart-option-btn active" onclick="loadEnergyHistory('power', '24h', this)">Daily</button>
                            <button class="chart-option-btn" onclick="loadEnergyHistory('power', '7d', this)">Weekly</button>
                            <button class="chart-option-btn" onclick="loadEnergyHistory('power', '30d', this)">Monthly</button>
                            <button class="chart-option-btn" onclick="loadEnergyHistory('power', '12mo', this)">Annual</button>
                        </div>
                    </div>
                    <div class="chart-visibility-toggles" style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;padding:6px 10px;background:rgba(148,163,184,0.06);border-radius:8px;font-size:12px;">
                        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;color:var(--text-secondary);"><input type="checkbox" checked onchange="toggleDataset('energyPower',0,this)" style="accent-color:#1e40af;"> <span style="color:#1e40af;font-weight:600;">AC</span></label>
                        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;color:var(--text-secondary);"><input type="checkbox" checked onchange="toggleDataset('energyPower',1,this)" style="accent-color:#059669;"> <span style="color:#059669;font-weight:600;">Outlet</span></label>
                        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;color:var(--text-secondary);"><input type="checkbox" checked onchange="toggleDataset('energyPower',2,this)" style="accent-color:#eab308;"> <span style="color:#eab308;font-weight:600;">Lamp</span></label>
                        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;color:var(--text-secondary);"><input type="checkbox" checked onchange="toggleDataset('energyPower',3,this)" style="accent-color:#7c3aed;"> <span style="color:#7c3aed;font-weight:600;">Total</span></label>
                    </div>
                    <canvas id="energyPowerChart" height="80"></canvas>
                    <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;font-size:12px;">
                        <span style="color:#1e40af;font-weight:600;">AC</span>
                        <span style="color:var(--text-secondary);">Latest: <strong id="energy-power-latest" style="color:var(--text);">--</strong></span>
                        <span style="color:var(--text-secondary);">Min: <strong id="energy-power-min" style="color:var(--text);">--</strong></span>
                        <span style="color:var(--text-secondary);">Max: <strong id="energy-power-max" style="color:var(--text);">--</strong></span>
                        <span style="color:var(--text-secondary);">Avg: <strong id="energy-power-avg" style="color:var(--text);">--</strong></span>
                        <span style="color:#059669;font-weight:600;margin-left:10px;">Outlet</span>
                        <span style="color:var(--text-secondary);">Latest: <strong id="outlet-power-latest" style="color:var(--text);">--</strong></span>
                        <span style="color:var(--text-secondary);">Avg: <strong id="outlet-power-avg" style="color:var(--text);">--</strong></span>
                        <span style="color:#eab308;font-weight:600;margin-left:10px;">Lamp</span>
                        <span style="color:var(--text-secondary);">Latest: <strong id="lamp-power-latest" style="color:var(--text);">--</strong></span>
                        <span style="color:var(--text-secondary);">Avg: <strong id="lamp-power-avg" style="color:var(--text);">--</strong></span>
                    </div>
                </div>

                <!-- Voltage + Current side by side -->
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;">
                    <div class="chart-container">
                        <div class="chart-header">
                            <div class="chart-title">Voltage &mdash; AC <span style="color:#1e40af;">&#9632;</span> &amp; Outlet <span style="color:#059669;">&#9632;</span> &amp; Lamp <span style="color:#eab308;">&#9632;</span> (V)</div>
                            <div class="chart-options">
                                <button class="chart-option-btn active" onclick="loadEnergyHistory('voltage', '24h', this)">Daily</button>
                                <button class="chart-option-btn" onclick="loadEnergyHistory('voltage', '7d', this)">Weekly</button>
                                <button class="chart-option-btn" onclick="loadEnergyHistory('voltage', '30d', this)">Monthly</button>
                                <button class="chart-option-btn" onclick="loadEnergyHistory('voltage', '12mo', this)">Annual</button>
                            </div>
                        </div>
                        <div class="chart-visibility-toggles" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:6px;padding:5px 8px;background:rgba(148,163,184,0.06);border-radius:8px;font-size:11px;">
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyVoltage',0,this)" style="accent-color:#1e40af;"> <span style="color:#1e40af;font-weight:600;">AC</span></label>
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyVoltage',1,this)" style="accent-color:#059669;"> <span style="color:#059669;font-weight:600;">Outlet</span></label>
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyVoltage',2,this)" style="accent-color:#eab308;"> <span style="color:#eab308;font-weight:600;">Lamp</span></label>
                        </div>
                        <canvas id="energyVoltageChart" height="100"></canvas>
                        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:11px;color:var(--text-secondary);">
                            <span>Latest: <strong id="energy-voltage-latest" style="color:var(--text);">--</strong></span>
                            <span>Min: <strong id="energy-voltage-min" style="color:var(--text);">--</strong></span>
                            <span>Max: <strong id="energy-voltage-max" style="color:var(--text);">--</strong></span>
                            <span>Avg: <strong id="energy-voltage-avg" style="color:var(--text);">--</strong></span>
                        </div>
                    </div>
                    <div class="chart-container">
                        <div class="chart-header">
                            <div class="chart-title">Current &mdash; AC <span style="color:#1e40af;">&#9632;</span> &amp; Outlet <span style="color:#059669;">&#9632;</span> &amp; Lamp <span style="color:#eab308;">&#9632;</span> &amp; Total <span style="color:#7c3aed;">&#9632;</span> (A)</div>
                            <div class="chart-options">
                                <button class="chart-option-btn active" onclick="loadEnergyHistory('current', '24h', this)">Daily</button>
                                <button class="chart-option-btn" onclick="loadEnergyHistory('current', '7d', this)">Weekly</button>
                                <button class="chart-option-btn" onclick="loadEnergyHistory('current', '30d', this)">Monthly</button>
                                <button class="chart-option-btn" onclick="loadEnergyHistory('current', '12mo', this)">Annual</button>
                            </div>
                        </div>
                        <div class="chart-visibility-toggles" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:6px;padding:5px 8px;background:rgba(148,163,184,0.06);border-radius:8px;font-size:11px;">
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyCurrent',0,this)" style="accent-color:#1e40af;"> <span style="color:#1e40af;font-weight:600;">AC</span></label>
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyCurrent',1,this)" style="accent-color:#059669;"> <span style="color:#059669;font-weight:600;">Outlet</span></label>
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyCurrent',2,this)" style="accent-color:#eab308;"> <span style="color:#eab308;font-weight:600;">Lamp</span></label>
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyCurrent',3,this)" style="accent-color:#7c3aed;"> <span style="color:#7c3aed;font-weight:600;">Total</span></label>
                        </div>
                        <canvas id="energyCurrentChart" height="100"></canvas>
                        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:11px;">
                            <span style="color:#1e40af;font-weight:600;">AC</span>
                            <span style="color:var(--text-secondary);">Latest: <strong id="energy-current-latest" style="color:var(--text);">--</strong></span>
                            <span style="color:var(--text-secondary);">Avg: <strong id="energy-current-avg" style="color:var(--text);">--</strong></span>
                            <span style="color:#059669;font-weight:600;margin-left:8px;">Outlet</span>
                            <span style="color:var(--text-secondary);">Latest: <strong id="outlet-current-latest" style="color:var(--text);">--</strong></span>
                            <span style="color:var(--text-secondary);">Avg: <strong id="outlet-current-avg" style="color:var(--text);">--</strong></span>
                            <span style="color:#eab308;font-weight:600;margin-left:8px;">Lamp</span>
                            <span style="color:var(--text-secondary);">Latest: <strong id="lamp-current-latest" style="color:var(--text);">--</strong></span>
                            <span style="color:var(--text-secondary);">Avg: <strong id="lamp-current-avg" style="color:var(--text);">--</strong></span>
                        </div>
                    </div>
                </div>

                <!-- kWh -->
                <div class="chart-container" style="margin-top:16px;">
                    <div class="chart-header">
                        <div class="chart-title">Energy per Interval &mdash; AC <span style="color:#1e40af;">&#9632;</span> &amp; Outlet <span style="color:#059669;">&#9632;</span> &amp; Lamp <span style="color:#eab308;">&#9632;</span> &amp; Total <span style="color:#7c3aed;">&#9632;</span> (kWh)</div>
                        <div class="chart-options">
                            <button class="chart-option-btn active" onclick="loadEnergyHistory('energy_kwh', '24h', this)">Daily</button>
                            <button class="chart-option-btn" onclick="loadEnergyHistory('energy_kwh', '7d', this)">Weekly</button>
                            <button class="chart-option-btn" onclick="loadEnergyHistory('energy_kwh', '30d', this)">Monthly</button>
                            <button class="chart-option-btn" onclick="loadEnergyHistory('energy_kwh', '12mo', this)">Annual</button>
                        </div>
                    </div>
                    <div class="chart-visibility-toggles" style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;padding:6px 10px;background:rgba(148,163,184,0.06);border-radius:8px;font-size:12px;">
                        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyKwh',0,this)" style="accent-color:#1e40af;"> <span style="color:#1e40af;font-weight:600;">AC</span></label>
                        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyKwh',1,this)" style="accent-color:#059669;"> <span style="color:#059669;font-weight:600;">Outlet</span></label>
                        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyKwh',2,this)" style="accent-color:#eab308;"> <span style="color:#eab308;font-weight:600;">Lamp</span></label>
                        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;"><input type="checkbox" checked onchange="toggleDataset('energyKwh',3,this)" style="accent-color:#7c3aed;"> <span style="color:#7c3aed;font-weight:600;">Total</span></label>
                    </div>
                    <canvas id="energyKwhChart" height="80"></canvas>
                    <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;font-size:12px;">
                        <span style="color:#1e40af;font-weight:600;">AC</span>
                        <span style="color:var(--text-secondary);">Last Interval: <strong id="energy-kwh-latest" style="color:var(--text);">--</strong></span>
                        <span style="color:var(--text-secondary);">Total: <strong id="energy-kwh-avg" style="color:var(--text);">--</strong></span>
                        <span style="color:#059669;font-weight:600;margin-left:10px;">Outlet</span>
                        <span style="color:var(--text-secondary);">Last Interval: <strong id="outlet-kwh-latest" style="color:var(--text);">--</strong></span>
                        <span style="color:var(--text-secondary);">Total: <strong id="outlet-kwh-avg" style="color:var(--text);">--</strong></span>
                        <span style="color:#eab308;font-weight:600;margin-left:10px;">Lamp</span>
                        <span style="color:var(--text-secondary);">Last Interval: <strong id="lamp-kwh-latest" style="color:var(--text);">--</strong></span>
                        <span style="color:var(--text-secondary);">Total: <strong id="lamp-kwh-avg" style="color:var(--text);">--</strong></span>
                    </div>
                </div>

            </div>



            <!-- ===== DAILY SUMMARY ===== -->
            <div id="daily-summary-section" style="margin-top: 28px; padding: 22px; border-radius: 18px; border: 1px solid rgba(59,130,246,0.2); background: linear-gradient(160deg, rgba(59,130,246,0.06), rgba(241,245,249,0.94)); box-shadow: 0 8px 28px rgba(15,23,42,0.08);">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; flex-wrap:wrap; gap:10px;">
                    <div>
                        <h2 style="font-size:18px; font-weight:700; color:var(--text-primary); margin:0 0 4px 0;">Daily Summary</h2>
                        <p style="color:var(--text-secondary); font-size:12px; margin:0;">Today's energy statistics — <span id="daily-summary-date">--</span></p>
                    </div>
                    <button onclick="loadDailySummary()" style="padding:7px 16px; border-radius:8px; border:1px solid rgba(59,130,246,0.4); background:rgba(59,130,246,0.1); color:#3b82f6; font-size:12px; cursor:pointer;">↻ Refresh</button>
                </div>
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(140px,1fr)); gap:14px;">
                    <!-- AC kWh -->
                    <div style="padding:14px; border-radius:12px; background:rgba(30,64,175,0.07); border:1px solid rgba(30,64,175,0.18); text-align:center;">
                        <div style="font-size:11px; color:var(--text-secondary); margin-bottom:6px;">AC Energy</div>
                        <div style="font-size:22px; font-weight:700; color:#1e40af;"><span id="ds-ac-kwh">--</span></div>
                        <div style="font-size:11px; color:var(--text-secondary);">kWh</div>
                    </div>
                    <!-- Lamp kWh -->
                    <div style="padding:14px; border-radius:12px; background:rgba(59,130,246,0.07); border:1px solid rgba(59,130,246,0.18); text-align:center;">
                        <div style="font-size:11px; color:var(--text-secondary); margin-bottom:6px;">Lamp Energy</div>
                        <div style="font-size:22px; font-weight:700; color:#3b82f6;"><span id="ds-lamp-kwh">--</span></div>
                        <div style="font-size:11px; color:var(--text-secondary);">kWh</div>
                    </div>
                    <!-- Total kWh -->
                    <div style="padding:14px; border-radius:12px; background:rgba(59,130,246,0.07); border:1px solid rgba(59,130,246,0.18); text-align:center;">
                        <div style="font-size:11px; color:var(--text-secondary); margin-bottom:6px;">Total Energy</div>
                        <div style="font-size:22px; font-weight:700; color:#3b82f6;"><span id="ds-total-kwh">--</span></div>
                        <div style="font-size:11px; color:var(--text-secondary);">kWh</div>
                    </div>
                    <!-- Estimated Cost -->
                    <div style="padding:14px; border-radius:12px; background:rgba(14,165,233,0.07); border:1px solid rgba(14,165,233,0.18); text-align:center;">
                        <div style="font-size:11px; color:var(--text-secondary); margin-bottom:6px;">Est. Cost</div>
                        <div style="font-size:18px; font-weight:700; color:#0ea5e9;">Rp<span id="ds-cost">--</span></div>
                        <div style="font-size:11px; color:var(--text-secondary);">@ Rp 1,500/kWh</div>
                    </div>
                    <!-- AC Peak Power -->
                    <div style="padding:14px; border-radius:12px; background:rgba(30,64,175,0.05); border:1px solid rgba(30,64,175,0.12); text-align:center;">
                        <div style="font-size:11px; color:var(--text-secondary); margin-bottom:6px;">AC Peak Power</div>
                        <div style="font-size:20px; font-weight:700; color:#1e40af;"><span id="ds-ac-peak">--</span></div>
                        <div style="font-size:11px; color:var(--text-secondary);">W</div>
                    </div>
                    <!-- Lamp Peak Power -->
                    <div style="padding:14px; border-radius:12px; background:rgba(59,130,246,0.05); border:1px solid rgba(59,130,246,0.12); text-align:center;">
                        <div style="font-size:11px; color:var(--text-secondary); margin-bottom:6px;">Lamp Peak Power</div>
                        <div style="font-size:20px; font-weight:700; color:#3b82f6;"><span id="ds-lamp-peak">--</span></div>
                        <div style="font-size:11px; color:var(--text-secondary);">W</div>
                    </div>
                    <!-- AC Runtime -->
                    <div style="padding:14px; border-radius:12px; background:rgba(30,64,175,0.05); border:1px solid rgba(30,64,175,0.12); text-align:center;">
                        <div style="font-size:11px; color:var(--text-secondary); margin-bottom:6px;">AC Runtime</div>
                        <div style="font-size:20px; font-weight:700; color:#1e40af;"><span id="ds-ac-runtime">--</span></div>
                        <div style="font-size:11px; color:var(--text-secondary);">hours</div>
                    </div>
                    <!-- Lamp Runtime -->
                    <div style="padding:14px; border-radius:12px; background:rgba(59,130,246,0.05); border:1px solid rgba(59,130,246,0.12); text-align:center;">
                        <div style="font-size:11px; color:var(--text-secondary); margin-bottom:6px;">Lamp Runtime</div>
                        <div style="font-size:20px; font-weight:700; color:#3b82f6;"><span id="ds-lamp-runtime">--</span></div>
                        <div style="font-size:11px; color:var(--text-secondary);">hours</div>
                    </div>
                </div>
            </div>

            <!-- ===== PANEL EXPORT ENERGY FROM DATABASE ===== -->
            <div style="margin-top:22px;padding:22px;border-radius:18px;border:1px solid rgba(59,130,246,0.25);background:linear-gradient(160deg,rgba(59,130,246,0.06),rgba(241,245,249,0.96));box-shadow:0 6px 24px rgba(0,0,0,0.06);">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:10px;">
                    <div>
                        <h2 style="font-size:16px;font-weight:700;color:var(--text);margin:0 0 4px;"><i class="fas fa-database" style="color:#3b82f6;margin-right:8px;"></i>Export Energy Data from Database</h2>
                        <p style="font-size:12px;color:var(--text-secondary);margin:0;">Fetch energy data directly from InfluxDB — select date range &amp; device, export to CSV</p>
                    </div>
                    <div style="padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(59,130,246,0.12);color:#3b82f6;border:1px solid rgba(59,130,246,0.3);">
                        <i class="fas fa-database" style="font-size:9px;vertical-align:middle;margin-right:5px;"></i> InfluxDB
                    </div>
                </div>

                <!-- Row 1: Date + AC -->
                <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-bottom:12px;">
                    <div>
                        <label style="display:block;font-size:11px;color:var(--text-secondary);margin-bottom:4px;">From Date</label>
                        <input type="date" id="db-energy-from" style="padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:13px;">
                    </div>
                    <div>
                        <label style="display:block;font-size:11px;color:var(--text-secondary);margin-bottom:4px;">To Date</label>
                        <input type="date" id="db-energy-to" style="padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:13px;">
                    </div>
                    <button onclick="dbExportCSV('energy_ac')"
                        style="padding:10px 20px;border-radius:10px;border:none;background:linear-gradient(135deg,#1e40af,#1d4ed8);color:#fff;font-size:13px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:7px;">
                        <i class="fas fa-download"></i> Export AC
                    </button>
                    <button onclick="dbExportCSV('energy_outlet')"
                        style="padding:10px 20px;border-radius:10px;border:none;background:linear-gradient(135deg,#059669,#047857);color:#fff;font-size:13px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:7px;">
                        <i class="fas fa-download"></i> Export Outlet
                    </button>
                    <button onclick="dbExportCSV('energy_lamp')"
                        style="padding:10px 20px;border-radius:10px;border:none;background:linear-gradient(135deg,#eab308,#ca8a04);color:#fff;font-size:13px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:7px;">
                        <i class="fas fa-download"></i> Export Lamp
                    </button>
                    <button onclick="dbExportCSV('energy_total')"
                        style="padding:10px 20px;border-radius:10px;border:none;background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#fff;font-size:13px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:7px;">
                        <i class="fas fa-download"></i> Export Total
                    </button>

                    <button onclick="dbExportSetToday('energy')"
                        style="padding:10px 14px;border-radius:10px;border:1px solid rgba(59,130,246,0.35);background:rgba(59,130,246,0.08);color:#3b82f6;font-size:12px;font-weight:600;cursor:pointer;">
                        Today
                    </button>
                    <button onclick="dbExportSet7d('energy')"
                        style="padding:10px 14px;border-radius:10px;border:1px solid rgba(59,130,246,0.35);background:rgba(59,130,246,0.08);color:#3b82f6;font-size:12px;font-weight:600;cursor:pointer;">
                        7 Days
                    </button>
                    <button onclick="dbExportSet30d('energy')"
                        style="padding:10px 14px;border-radius:10px;border:1px solid rgba(59,130,246,0.35);background:rgba(59,130,246,0.08);color:#3b82f6;font-size:12px;font-weight:600;cursor:pointer;">
                        30 Days
                    </button>
                </div>
                <div style="font-size:12px;color:var(--text-secondary);padding:10px 14px;background:rgba(59,130,246,0.05);border-radius:8px;border:1px solid rgba(59,130,246,0.12);">
                    <i class="fas fa-info-circle" style="color:#3b82f6;margin-right:6px;"></i>
                    Data fetched directly from InfluxDB — columns: <strong>Time, Voltage (V), Current (A), Power (W), Energy (kWh)</strong>
                </div>
            </div>
        </div>

        <!-- Control Panel Page -->
        <!-- AC Control Page -->
        <div id="control-ac" class="page">
            <div class="header">
                <h1>AC Control Panel</h1>
                <p>Mitsubishi Heavy — IRMitsubishiHeavy library</p>
            </div>

            <!-- ========== MODE SELECTOR (PROMINENT) ========== -->
            <div id="ac-mode-selector" style="margin-bottom: 20px; padding: 20px; border-radius: 16px; border: 2px solid var(--border); background: var(--bg-card);">
                <div style="text-align: center; margin-bottom: 14px; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary);">
                    Control Mode
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <button id="btn-mode-adaptive" onclick="setACMode('ADAPTIVE')" style="padding: 18px 16px; border-radius: 14px; border: 3px solid #3b82f6; background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 6px;">
                        <span style="font-size: 24px; font-weight: 800;">A</span>
                        <span>ADAPTIVE</span>
                        <span style="font-size: 11px; font-weight: 400; opacity: 0.9;">AI controls AC automatically</span>
                    </button>
                    <button id="btn-mode-manual" onclick="setACMode('MANUAL')" style="padding: 18px 16px; border-radius: 14px; border: 3px solid var(--border); background: var(--bg-card); color: var(--text-secondary); font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 6px; opacity: 0.6;">
                        <span style="font-size: 24px; font-weight: 800;">M</span>
                        <span>MANUAL</span>
                        <span style="font-size: 11px; font-weight: 400; opacity: 0.9;">Control AC manually</span>
                    </button>
                </div>
                <!-- Current mode indicator -->
                <div id="ac-mode-indicator" style="margin-top: 14px; padding: 10px 16px; border-radius: 10px; text-align: center; font-size: 13px; font-weight: 600; background: rgba(59, 130, 246, 0.1); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3);">
                    Current mode: <strong>ADAPTIVE</strong> — AC controlled automatically by GA optimization
                </div>
            </div>

            <!-- ========== ADAPTIVE INFO BANNER (shown when ADAPTIVE) ========== -->
            <div id="adaptive-info-banner" style="margin-bottom: 20px; padding: 20px; border-radius: 14px; background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(37, 99, 235, 0.12)); border: 2px solid rgba(59, 130, 246, 0.3);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #3b82f6, #2563eb); display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; font-weight: 800;">
                        GA
                    </div>
                    <div>
                        <div style="font-size: 16px; font-weight: 700; color: #3b82f6;">Adaptive Mode Active</div>
                        <div style="font-size: 12px; color: var(--text-secondary);">GA Optimization controls temperature & fan speed</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 13px;">
                    <div style="padding: 10px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; text-align: center;">
                        <div style="color: var(--text-secondary); font-size: 11px;">Temp</div>
                        <div style="font-weight: 700; color: #3b82f6; font-size: 18px;" id="adaptive-ga-temp">--</div>
                    </div>
                    <div style="padding: 10px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; text-align: center;">
                        <div style="color: var(--text-secondary); font-size: 11px;">Fan</div>
                        <div style="font-weight: 700; color: #3b82f6; font-size: 18px;" id="adaptive-ga-fan">--</div>
                    </div>
                    <div style="padding: 10px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; text-align: center;">
                        <div style="color: var(--text-secondary); font-size: 11px;">Fitness</div>
                        <div style="font-weight: 700; color: #3b82f6; font-size: 18px;" id="adaptive-ga-fitness">--</div>
                    </div>
                </div>
                <div style="margin-top: 12px; padding: 10px; background: rgba(14, 165, 233, 0.1); border-radius: 8px; border: 1px solid rgba(14, 165, 233, 0.3); font-size: 12px; color: #0ea5e9; text-align: center;">
                    Manual control disabled in Adaptive mode. Switch to Manual to control AC.
                </div>
            </div>

            <div class="control-panel" style="position: relative;">
                <!-- OVERLAY: blocks manual controls when ADAPTIVE -->
                <div id="ac-manual-overlay" style="display: block; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.45); backdrop-filter: blur(3px); border-radius: 12px; z-index: 10; display: flex; align-items: center; justify-content: center; cursor: not-allowed;">
                    <div style="text-align: center; color: white;">
                        <div style="font-size: 28px; margin-bottom: 10px; opacity: 0.8;">🔒</div>
                        <div style="font-size: 14px; font-weight: 600;">Adaptive Mode Active</div>
                        <div style="font-size: 12px; opacity: 0.7;">Switch to Manual to use these controls</div>
                    </div>
                </div>

                <div class="control-title">
                    <span>Air Conditioning Control</span>
                    <div class="mode-badge manual" id="ac-mode-badge" style="display:none;">MANUAL MODE</div>
                </div>
                
                <!-- AC Realtime Status Bar -->
                <div id="ac-live-status" style="margin-top: 12px; padding: 12px 16px; background: rgba(37, 99, 235, 0.08); border: 1px solid var(--border); border-radius: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 13px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div id="ac-live-dot" style="width: 10px; height: 10px; border-radius: 50%; background: #1e40af;"></div>
                        <span style="font-weight: 600; color: var(--text);">AC <span id="ac-live-state">OFF</span></span>
                    </div>
                    <div style="display: flex; gap: 16px; color: var(--text-secondary);">
                        <span><span id="ac-live-temp">24</span>°C</span>
                        <span>Fan <span id="ac-live-fan">1</span></span>
                        <span><span id="ac-live-mode">COOL</span></span>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <button class="btn btn-success" style="padding: 20px; font-size: 16px; border-radius: 12px;" onclick="sendACCommand('POWER_ON')">
                        AC ON
                    </button>
                    <button class="btn btn-danger" style="padding: 20px; font-size: 16px; border-radius: 12px;" onclick="sendACCommand('POWER_OFF')">
                        AC OFF
                    </button>
                    <button class="btn btn-primary" style="padding: 20px; font-size: 16px; border-radius: 12px;" onclick="sendACCommand('TEMP_UP')">
                        TEMP +
                    </button>
                    <button class="btn btn-primary" style="padding: 20px; font-size: 16px; border-radius: 12px;" onclick="sendACCommand('TEMP_DOWN')">
                        TEMP −
                    </button>
                </div>

                <!-- AC Mode Buttons -->
                <div style="margin-top: 20px;">
                    <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 10px; font-weight: 600;">
                        AC Mode
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
                        <button class="btn ac-mode-btn" id="mode-btn-auto" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: var(--primary); color: white;" onclick="sendACMode('MODE_AUTO', this)">
                            Auto
                        </button>
                        <button class="btn ac-mode-btn" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: #0ea5e9; color: white;" onclick="sendACMode('MODE_COOL', this)">
                            Cool
                        </button>
                        <button class="btn ac-mode-btn" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: #1e40af; color: white;" onclick="sendACMode('MODE_HEAT', this)">
                            Heat
                        </button>
                        <button class="btn ac-mode-btn" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: #3b82f6; color: white;" onclick="sendACMode('MODE_FAN', this)">
                            Fan
                        </button>
                        <button class="btn ac-mode-btn" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: #0ea5e9; color: white;" onclick="sendACMode('MODE_DRY', this)">
                            Dry
                        </button>
                    </div>
                </div>

                <!-- Swing / Turbo / Econo Toggle Buttons -->
                <div style="margin-top: 20px;">
                    <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 10px; font-weight: 600;">
                        Extra Features
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                        <button class="btn" id="btn-swing" onclick="sendACCommand('SWING_TOGGLE')" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: var(--bg-card); color: var(--text-secondary); border: 2px solid var(--border); cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                            <span style="font-size: 20px;">🔄</span>
                            <span>Swing</span>
                            <span id="swing-status" style="font-size: 11px; opacity: 0.7;">OFF</span>
                        </button>
                        <button class="btn" id="btn-turbo" onclick="sendACCommand('TURBO')" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: var(--bg-card); color: var(--text-secondary); border: 2px solid var(--border); cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                            <span style="font-size: 20px;">⚡</span>
                            <span>Turbo</span>
                            <span id="turbo-status" style="font-size: 11px; opacity: 0.7;">OFF</span>
                        </button>
                        <button class="btn" id="btn-econo" onclick="sendACCommand('ECONO')" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: var(--bg-card); color: var(--text-secondary); border: 2px solid var(--border); cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                            <span style="font-size: 20px;">🌿</span>
                            <span>Econo</span>
                            <span id="econo-status" style="font-size: 11px; opacity: 0.7;">OFF</span>
                        </button>
                    </div>
                </div>

                <!-- AC Temperature & Fan Speed Sliders -->
                <div style="margin-top: 20px; padding: 20px; background: rgba(37, 99, 235, 0.05); border-radius: 12px; border: 1px solid var(--border);">
                    <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 15px; font-weight: 600;">
                        AC Settings (Direct MQTT)
                    </div>
                    <div class="control-group" style="margin-bottom: 15px;">
                        <label class="control-label">Temperature: <span id="ac-temp-display" style="color: var(--primary); font-weight: bold;">24</span>°C</label>
                        <input type="range" min="16" max="30" value="24" class="slider" id="ac-temp-slider" oninput="updateACTemp(this.value)" style="width: 100%;">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
                            <span>16°C</span><span>20°C</span><span>24°C</span><span>28°C</span><span>30°C</span>
                        </div>
                    </div>
                    <div class="control-group" style="margin-bottom: 15px;">
                        <label class="control-label">Set RH: <span id="ac-rh-display" style="color: var(--primary); font-weight: bold;">50</span>%</label>
                        <input type="range" min="30" max="80" value="50" class="slider" id="ac-rh-slider" oninput="updateACRH(this.value)" style="width: 100%;">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
                            <span>30%</span><span>40%</span><span>50%</span><span>60%</span><span>70%</span><span>80%</span>
                        </div>
                    </div>
                    <div class="control-group" style="margin-bottom: 15px;">
                        <label class="control-label">Fan Speed: Level <span id="fan-speed-display" style="color: var(--primary); font-weight: bold;">1</span></label>
                        <input type="range" min="1" max="3" value="1" class="slider" id="fan-speed-slider" oninput="updateFanSpeed(this.value)" style="width: 100%;">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
                            <span>Low</span><span>Medium</span><span>High</span>
                        </div>
                    </div>
                    <div class="control-group" style="margin-bottom: 15px;">
                        <label class="control-label">AC Mode:</label>
                        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 8px;">
                            <button class="btn ac-set-mode-btn" data-mode="AUTO" onclick="selectACSetMode('AUTO', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--card-bg); border: 2px solid var(--border); color: var(--text); cursor: pointer;">
                                Auto
                            </button>
                            <button class="btn ac-set-mode-btn active" data-mode="COOL" onclick="selectACSetMode('COOL', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--primary); border: 2px solid var(--primary); color: white; cursor: pointer;">
                                Cool
                            </button>
                            <button class="btn ac-set-mode-btn" data-mode="HEAT" onclick="selectACSetMode('HEAT', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--card-bg); border: 2px solid var(--border); color: var(--text); cursor: pointer;">
                                Heat
                            </button>
                            <button class="btn ac-set-mode-btn" data-mode="DRY" onclick="selectACSetMode('DRY', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--card-bg); border: 2px solid var(--border); color: var(--text); cursor: pointer;">
                                Dry
                            </button>
                            <button class="btn ac-set-mode-btn" data-mode="FAN" onclick="selectACSetMode('FAN', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--card-bg); border: 2px solid var(--border); color: var(--text); cursor: pointer;">
                                Fan
                            </button>
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="applyACSettings()" style="width: 100%; padding: 12px; border-radius: 10px; font-weight: 600;">
                        Apply AC Settings
                    </button>
                </div>

                <div style="margin-top: 15px; font-size: 12px; color: var(--text-secondary); text-align: center;">
                    IRMitsubishiHeavy library — no manual learning needed.
                </div>
            </div>
        </div>

        <!-- Lamp Control Page -->
        <div id="control-lamp" class="page">
            <div class="header">
                <h1>Lamp Control Panel</h1>
                <p>Manual lamp control — 2 Lamps (GPIO 25 & 26), 3 BH1750 Sensors</p>
            </div>

            <!-- ========== LAMP MODE SELECTOR ========== -->
            <div id="lamp-mode-selector" style="margin-bottom: 20px; padding: 20px; border-radius: 16px; border: 2px solid var(--border); background: var(--bg-card);">
                <div style="text-align: center; margin-bottom: 14px; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary);">
                    Lamp Control Mode
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <button id="btn-lamp-adaptive" onclick="setLampMode('ADAPTIVE')" style="padding: 18px 16px; border-radius: 14px; border: 3px solid #0ea5e9; background: linear-gradient(135deg, #0ea5e9, #0284c7); color: white; font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 6px;">
                        <span style="font-size: 24px; font-weight: 800;">A</span>
                        <span>ADAPTIVE</span>
                        <span style="font-size: 11px; font-weight: 400; opacity: 0.9;">PSO controls lamps automatically</span>
                    </button>
                    <button id="btn-lamp-manual" onclick="setLampMode('MANUAL')" style="padding: 18px 16px; border-radius: 14px; border: 3px solid var(--border); background: var(--bg-card); color: var(--text-secondary); font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 6px; opacity: 0.6;">
                        <span style="font-size: 24px; font-weight: 800;">M</span>
                        <span>MANUAL</span>
                        <span style="font-size: 11px; font-weight: 400; opacity: 0.9;">Control lamps manually</span>
                    </button>
                </div>
                <div id="lamp-mode-indicator" style="margin-top: 14px; padding: 10px 16px; border-radius: 10px; text-align: center; font-size: 13px; font-weight: 600; background: rgba(14, 165, 233, 0.1); color: #0ea5e9; border: 1px solid rgba(14, 165, 233, 0.3);">
                    Current mode: <strong>ADAPTIVE</strong> — Lamps controlled automatically by PSO optimization
                </div>
            </div>

            <div class="control-panel">
                <div class="control-title">
                    <span>Lamp Control</span>
                </div>

                <!-- Live Sensor Readings -->
                <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 15px; padding: 12px; background: var(--bg-elevated); border-radius: 10px; border: 1px solid var(--border);">
                    <div style="flex: 1; min-width: 80px; text-align: center;">
                        <div style="font-size: 11px; color: var(--text-secondary);">Sensor 1</div>
                        <div style="font-size: 18px; font-weight: 700; color: #0ea5e9;"><span id="ctrl-lux1">0</span> lx</div>
                    </div>
                    <div style="flex: 1; min-width: 80px; text-align: center;">
                        <div style="font-size: 11px; color: var(--text-secondary);">Sensor 2</div>
                        <div style="font-size: 18px; font-weight: 700; color: #3b82f6;"><span id="ctrl-lux2">0</span> lx</div>
                    </div>
                    <div style="flex: 1; min-width: 80px; text-align: center;">
                        <div style="font-size: 11px; color: var(--text-secondary);">Sensor 3</div>
                        <div style="font-size: 18px; font-weight: 700; color: #2563eb;"><span id="ctrl-lux3">0</span> lx</div>
                    </div>
                    <div style="flex: 1; min-width: 80px; text-align: center;">
                        <div style="font-size: 11px; color: var(--text-secondary);">Motion</div>
                        <div style="font-size: 18px; font-weight: 700;" id="ctrl-motion-status"><span style="color: #1e40af;">IDLE</span></div>
                    </div>
                </div>

                <!-- Manual Controls with Overlay -->
                <div id="lamp-manual-controls" style="position: relative;">
                    <div id="lamp-manual-overlay" style="display: flex; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); border-radius: 12px; z-index: 10; align-items: center; justify-content: center; backdrop-filter: blur(3px);">
                        <div style="text-align: center; color: white;">
                            <div style="font-size: 28px; margin-bottom: 8px;">🤖</div>
                            <div style="font-weight: 700; font-size: 14px;">ADAPTIVE Mode Active</div>
                            <div style="font-size: 12px; opacity: 0.8;">Switch to MANUAL to control lamps</div>
                        </div>
                    </div>
                    <div class="control-group">
                        <label class="control-label">Lamp 1 Brightness (GPIO 25): <span id="brightness-display-1">0</span>%</label>
                        <input type="range" min="0" max="100" value="0" class="slider" id="brightness-slider-1" oninput="updateBrightness(1, this.value)">
                    </div>
                    <div class="control-group">
                        <label class="control-label">Lamp 2 Brightness (GPIO 26): <span id="brightness-display-2">0</span>%</label>
                        <input type="range" min="0" max="100" value="0" class="slider" id="brightness-slider-2" oninput="updateBrightness(2, this.value)">
                    </div>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <button class="btn btn-primary" onclick="applyLampSettings()">
                            Apply Both
                        </button>
                        <button class="btn" onclick="syncAllSliders()" style="background: var(--bg-elevated); color: var(--text); border: 1px solid var(--border);">
                            Sync Lamp 2 to Lamp 1
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- ML Optimization Page -->
        <div id="ml-optimization" class="page">
            <div class="header">
                <h1>Machine Learning Optimization</h1>
                <p>GA &rarr; Adaptive AC | PSO &rarr; Adaptive Lamp | Auto-optimized in ADAPTIVE mode</p>
            </div>

            <!-- ML Summary Cards -->
            <div class="stats-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">GA -> AC</span>
                        <div class="stat-icon" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">GA</div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-ga-fitness" style="color: #3b82f6;">0.00</span></div>
                    <div class="stat-change"><span>Best Fitness</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Optimized Temp</span>
                        <div class="stat-icon" style="background: rgba(30, 64, 175, 0.2); color: #1e40af;">T</div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-ga-temp" style="color: #1e40af;">--</span>°C</div>
                    <div class="stat-change"><span>Fan: <span id="ml-ga-fan" style="font-weight: bold;">--</span> | Mode: <span id="ml-ga-mode" style="font-weight:bold; color:#1e40af;">--</span> | RH: <span id="ml-ga-rh" style="font-weight:bold; color:#3b82f6;">--</span>%</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">PSO -> Lamp</span>
                        <div class="stat-icon" style="background: rgba(14, 165, 233, 0.2); color: #0ea5e9;">PSO</div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-pso-fitness" style="color: #0ea5e9;">0.0</span><span style="font-size:16px;color:#94a3b8;">%</span></div>
                    <div class="stat-change"><span>Lux Accuracy (100% = exactly 350 lux)</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">PWM Optimal</span>
                        <div class="stat-icon" style="background: rgba(37, 99, 235, 0.2); color: #2563eb;">PWM</div>
                    </div>
                    <div class="stat-value" style="font-size: 20px; line-height: 1.8;">
                        PWM1: <span id="ml-pso-pwm1" style="color: #2563eb; font-weight:700;">--</span><small style="color:#94a3b8;">/255</small><br>
                        PWM2: <span id="ml-pso-pwm2" style="color: #2563eb; font-weight:700;">--</span><small style="color:#94a3b8;">/255</small>
                    </div>
                    <div class="stat-change">
                        Lux1: <span id="ml-pso-lux1" style="color:#22d3ee;font-weight:600;">--</span>
                        &nbsp;Lux2: <span id="ml-pso-lux2" style="color:#22d3ee;font-weight:600;">--</span>
                        &nbsp;Lux3: <span id="ml-pso-lux3" style="color:#22d3ee;font-weight:600;">--</span> lx
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Current Conditions</span>
                        <div class="stat-icon" style="background: rgba(37, 99, 235, 0.2); color: #2563eb;">DB</div>
                    </div>
                    <div class="stat-value" style="font-size: 14px; line-height: 1.8;">
                        Temp: <span id="ml-cur-temp">--</span>°C &nbsp; Hum: <span id="ml-cur-hum">--</span>%<br>
                        Lux: <span id="ml-cur-lux">--</span> lux &nbsp; Person: <span id="ml-cur-person">--</span>
                    </div>
                    <div class="stat-change"><span>From InfluxDB</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Optimization Runs</span>
                        <div class="stat-icon" style="background: rgba(37, 99, 235, 0.2); color: #2563eb;">#</div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-opt-runs" style="color: #2563eb;">0</span></div>
                    <div class="stat-change"><span>Total Cycles</span></div>
                </div>
            </div>

            <!-- GA Fitness Convergence Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">GA Fitness Convergence (AC Optimization)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportGAReport()" title="Export GA Report"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="exportChartRange('gaFitness', 'GA Fitness')" title="Export date range">&#128197;</button>
                    </div>
                </div>
                <canvas id="gaFitnessChart" height="80"></canvas>
            </div>

            <!-- PSO Iteration Detail Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">PSO — Detail Iteration (PWM1, PWM2, Lux Avg per Iteration)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportPSOReport()" title="Export PSO Report"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="exportChartRange('psoFitness', 'PSO Fitness')" title="Export date range">&#128197;</button>
                    </div>
                </div>
                <!-- Iteration detail table -->
                <div id="pso-iter-table-wrap" style="overflow-x:auto;margin-bottom:8px;display:none;">
                    <table style="width:100%;border-collapse:collapse;font-size:12px;">
                        <thead>
                            <tr style="background:rgba(14,165,233,0.15);">
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">Iteration</th>
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">PWM 1</th>
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">PWM 2</th>
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">Lux 1</th>
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">Lux 2</th>
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">Lux 3</th>
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">Lux Avg</th>
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">Fitness (%)</th>
                                <th style="padding:4px 8px;text-align:center;border-bottom:1px solid rgba(14,165,233,0.3);">Status</th>
                            </tr>
                        </thead>
                        <tbody id="pso-iter-tbody"></tbody>
                    </table>
                </div>
                <canvas id="psoFitnessChart" height="100"></canvas>
            </div>

            <!-- GA vs PSO Comparison Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">GA vs PSO — Fitness Comparison</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportCompareChart('comparison', 'Fitness')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn" onclick="clearMLCharts()">
                            Clear
                        </button>
                    </div>
                </div>
                <canvas id="comparisonChart" height="80"></canvas>
            </div>

            <!-- Optimization History Table -->
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Optimization History</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportMLHistory()">
                            <i class="fas fa-download"></i> Export CSV
                        </button>
                        <button class="chart-option-btn" onclick="refreshMLHistory()">
                            Refresh
                        </button>
                    </div>
                </div>
                <div style="overflow-x: auto;">
                    <table class="ml-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Time</th>
                                <th>GA Fitness</th>
                                <th>AC Temp</th>
                                <th>Fan Speed</th>
                                <th>PSO Fitness</th>
                                <th>Brightness</th>
                                <th>Combined</th>
                            </tr>
                        </thead>
                        <tbody id="ml-history-body">
                            <tr><td colspan="8" style="text-align: center; color: #94a3b8;">No optimization data yet. Set mode to ADAPTIVE to start automatic optimization.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Parameter Configuration -->
            <div class="stats-grid" style="grid-template-columns: 1fr 1fr;">
                <div class="chart-container" style="margin-bottom: 0;">
                    <div class="chart-header">
                        <div class="chart-title">GA Parameters</div>
                    </div>
                    <div class="ml-param-grid">
                        <div class="ml-param-item">
                            <label>Population Size</label>
                            <input type="number" id="ml-ga-pop" value="15" min="5" max="100" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>Generations</label>
                            <input type="number" id="ml-ga-gen" value="30" min="10" max="500" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>Mutation Rate</label>
                            <input type="number" id="ml-ga-mut" value="0.25" min="0.01" max="0.5" step="0.01" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>Crossover Rate</label>
                            <input type="number" id="ml-ga-cross" value="0.8" min="0.1" max="1.0" step="0.05" class="ml-input">
                        </div>
                    </div>
                </div>

                <div class="chart-container" style="margin-bottom: 0;">
                    <div class="chart-header">
                        <div class="chart-title">PSO Parameters</div>
                    </div>
                    <div class="ml-param-grid">
                        <div class="ml-param-item">
                            <label>Swarm Size</label>
                            <input type="number" id="ml-pso-swarm" value="30" min="5" max="100" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>Iterations</label>
                            <input type="number" id="ml-pso-iter" value="100" min="10" max="500" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>Inertia (w)</label>
                            <input type="number" id="ml-pso-w" value="0.7" min="0.1" max="1.5" step="0.05" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>c1 / c2</label>
                            <input type="number" id="ml-pso-c" value="1.5" min="0.5" max="3.0" step="0.1" class="ml-input">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Logs Page -->
        <div id="logs" class="page">
            <div class="header">
                <h1>System Logs</h1>
                <p>Real-time system events and notifications</p>
                <button class="chart-option-btn" onclick="exportLogs()" style="margin-top: 8px;">
                    <i class="fas fa-download"></i> Export Logs CSV
                </button>
            </div>

            <div class="log-container" id="log-container">
            </div>
        </div>

        <!-- Occupancy Trend & Feedback Page -->
        <div id="occupancy-feedback" class="page">
            <div class="header">
                <h1>Occupancy Trend & Feedback</h1>
                <p>Monitor occupancy trends and submit room comfort ratings (1-5)</p>
            </div>

            <div class="occupancy-top">
                <div class="occupancy-kpi">
                    <div class="kpi-label">Current Occupancy</div>
                    <div class="kpi-value" id="occ-live-count">0</div>
                    <div class="occupancy-mini-note">Currently detected persons</div>
                    <div class="occupancy-mini-note">Confidence: <span id="occ-live-confidence">0%</span></div>
                </div>

                <div class="occupancy-chart-card">
                    <div class="chart-header" style="margin-bottom: 10px;">
                        <div class="chart-title">Occupancy Trend</div>
                        <div class="chart-options">
                            <button class="chart-option-btn" onclick="exportChartData('occupancy', 'Person Count')" title="Export CSV"><i class="fas fa-download"></i></button>
                            <button class="chart-option-btn" onclick="exportChartRange('occupancy', 'Person Count')" title="Export date range">&#128197;</button>
                            <button class="chart-option-btn active" onclick="changeChartRange('occupancy', 1)">1h</button>
                            <button class="chart-option-btn" onclick="changeChartRange('occupancy', 6)">6h</button>
                            <button class="chart-option-btn" onclick="changeChartRange('occupancy', 24)">24h</button>
                        </div>
                    </div>
                    <canvas id="occupancyChart" height="48"></canvas>
                </div>
            </div>

            <div class="feedback-grid">
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Comfort Feedback</div>
                    </div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;">
                        Scale: 1 = Not Satisfied, 5 = Very Satisfied
                    </div>
                    <div class="rating-row" id="rating-row" style="margin-top: 14px;">
                        <button class="rating-btn" onclick="selectFeedbackRating(1)" title="Not satisfied">1</button>
                        <button class="rating-btn" onclick="selectFeedbackRating(2)" title="Less satisfied">2</button>
                        <button class="rating-btn" onclick="selectFeedbackRating(3)" title="Fair">3</button>
                        <button class="rating-btn" onclick="selectFeedbackRating(4)" title="Satisfied">4</button>
                        <button class="rating-btn" onclick="selectFeedbackRating(5)" title="Very satisfied">5</button>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:12px; color:var(--text-secondary);">
                        <span>Not Satisfied</span>
                        <span>Very Satisfied</span>
                    </div>
                    <textarea id="feedback-comment" class="feedback-input" rows="4" placeholder="Enter room user feedback..."></textarea>
                    <input id="google-form-url" class="feedback-input" placeholder="Enter your Google Form URL" />

                    <div style="display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap;">
                        <button class="btn btn-primary" onclick="saveGoogleFormUrl()">
                            Save Form Link
                        </button>
                        <button class="btn btn-success" onclick="openGoogleForm()">
                            Open Google Form
                        </button>
                        <button class="btn btn-warning" onclick="submitOccupancyFeedback()">
                            Submit Feedback
                        </button>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Recent Feedback</div>
                        <div class="chart-options">
                            <button class="chart-option-btn" onclick="exportFeedback()">
                                <i class="fas fa-download"></i> Export CSV
                            </button>
                        </div>
                    </div>
                    <div id="feedback-history"></div>
                </div>
            </div>
        </div>

        <!-- Outlet Control Page -->
        <div id="control-outlet" class="page">
            <div class="header">
                <h1>Outlet Control Panel</h1>
                <p>Smart power outlet control — ON/OFF per outlet via MQTT</p>
            </div>

            <!-- Outlet Status Summary -->
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin-bottom:20px;">
                <div class="stat-card" style="text-align:center;">
                    <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">Outlet</div>
                    <div id="outlet1-power" style="font-size:18px;font-weight:700;color:#2563eb;">--W</div>
                    <div id="outlet1-status-badge" style="margin-top:6px;display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(30,64,175,0.15);color:#1e40af;">OFF</div>
                </div>
            </div>

            <!-- Outlet ON/OFF Cards -->
            <div class="control-panel">
                <div class="control-title">Outlet Switches</div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-top:16px;">

                    <!-- Outlet 1 -->
                    <div style="padding:20px;border-radius:14px;border:2px solid var(--border);background:var(--bg-card);display:flex;flex-direction:column;gap:12px;">
                        <div style="display:flex;align-items:center;gap:10px;">
                            <div id="outlet1-dot" style="width:12px;height:12px;border-radius:50%;background:#1e40af;flex-shrink:0;"></div>
                            <div style="font-size:15px;font-weight:700;color:var(--text-primary);">Outlet</div>
                            <div id="outlet1-label" style="margin-left:auto;font-size:11px;color:var(--text-secondary);">General</div>
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                            <button class="btn btn-success" style="padding:14px;font-size:14px;border-radius:10px;" onclick="outletControl(1,'ON')">ON</button>
                            <button class="btn btn-danger" style="padding:14px;font-size:14px;border-radius:10px;" onclick="outletControl(1,'OFF')">OFF</button>
                        </div>
                    </div>

                </div>

                <div style="margin-top:14px;font-size:12px;color:var(--text-secondary);text-align:center;">
                    MQTT topic: <code style="background:rgba(37,99,235,0.1);padding:2px 7px;border-radius:4px;color:#2563eb;">smartroom/outlet/control</code>
                </div>
            </div>
        </div>

        <!-- Outlet Analysis Page -->
        <div id="outlet-analysis" class="page">
            <div class="header">
                <h1>Outlet Analysis</h1>
                <p>Power consumption monitoring per outlet over time</p>
            </div>

            <!-- MySQL Outlet Energy Consumption Chart -->
            <div class="chart-container" style="margin-bottom:18px;">
                <div class="chart-header">
                    <div class="chart-title">MySQL Outlet Energy Consumption (kWh)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="loadAnalyticsEnergy('outlet','24h',this)">24h</button>
                        <button class="chart-option-btn" onclick="loadAnalyticsEnergy('outlet','7d',this)">7d</button>
                        <button class="chart-option-btn" onclick="loadAnalyticsEnergy('outlet','30d',this)">30d</button>
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:15px;padding:0 10px;">
                    <div>
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;">Latest Interval</div>
                        <div id="outlet-analytics-kwh-last" style="font-size:18px;font-weight:700;color:#059669;">--</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;">Total Period</div>
                        <div id="outlet-analytics-kwh-total" style="font-size:18px;font-weight:700;color:#059669;">--</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;">Average</div>
                        <div id="outlet-analytics-kwh-avg" style="font-size:18px;font-weight:700;color:#059669;">--</div>
                    </div>
                </div>
                <canvas id="outletEnergyKwhMySQLChart" height="90"></canvas>
            </div>

            <!-- Summary Stats -->
            <div class="stats-grid" style="grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin-bottom:20px;">
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Total Power</span>
                        <div class="stat-icon" style="background:rgba(37,99,235,0.15);color:#2563eb;"><i class="fas fa-bolt"></i></div>
                    </div>
                    <div class="stat-value"><span id="outlet-total-power" style="color:#2563eb;">--</span><small style="font-size:13px;color:var(--text-secondary);">W</small></div>
                    <div class="stat-change">All outlets combined</div>
                </div>
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Today Energy</span>
                        <div class="stat-icon" style="background:rgba(14,165,233,0.15);color:#0ea5e9;"><i class="fas fa-chart-bar"></i></div>
                    </div>
                    <div class="stat-value"><span id="outlet-today-kwh" style="color:#0ea5e9;">--</span><small style="font-size:13px;color:var(--text-secondary);">kWh</small></div>
                    <div class="stat-change">Energy used today</div>
                </div>
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Total Energy</span>
                        <div class="stat-icon" style="background:rgba(5,150,105,0.15);color:#059669;"><i class="fas fa-plug"></i></div>
                    </div>
                    <div class="stat-value"><span id="outlet-analytics-total-energy" style="color:#059669;">--</span><small style="font-size:13px;color:var(--text-secondary);">kWh</small></div>
                    <div class="stat-change">Cumulative energy usage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Active Outlets</span>
                        <div class="stat-icon" style="background:rgba(59,130,246,0.15);color:#3b82f6;"><i class="fas fa-plug"></i></div>
                    </div>
                    <div class="stat-value"><span id="outlet-active-count" style="color:#3b82f6;">--</span><small style="font-size:13px;color:var(--text-secondary);">/4</small></div>
                    <div class="stat-change">Currently ON</div>
                </div>
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Peak Power</span>
                        <div class="stat-icon" style="background:rgba(30,64,175,0.15);color:#1e40af;"><i class="fas fa-arrow-up"></i></div>
                    </div>
                    <div class="stat-value"><span id="outlet-peak-power" style="color:#1e40af;">--</span><small style="font-size:13px;color:var(--text-secondary);">W</small></div>
                    <div class="stat-change">Peak in selected period</div>
                </div>
            </div>

            <!-- Live Electrical Parameters -->
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:20px;margin-bottom:20px;">
                <div style="font-size:14px;font-weight:700;color:var(--text);margin-bottom:15px;display:flex;align-items:center;">
                    <i class="fas fa-bolt" style="color:#059669;margin-right:8px;"></i> Live Electrical Parameters
                </div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;">
                    <div style="text-align:center;padding:12px;border-radius:12px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;margin-bottom:4px;">Voltage</div>
                        <div style="font-size:20px;font-weight:700;color:var(--text);"><span id="oa-live-voltage">--</span><small style="font-size:12px;margin-left:2px;color:var(--text-secondary);">V</small></div>
                    </div>
                    <div style="text-align:center;padding:12px;border-radius:12px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;margin-bottom:4px;">Current</div>
                        <div style="font-size:20px;font-weight:700;color:var(--text);"><span id="oa-live-current">--</span><small style="font-size:12px;margin-left:2px;color:var(--text-secondary);">A</small></div>
                    </div>
                    <div style="text-align:center;padding:12px;border-radius:12px;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.15);">
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;margin-bottom:4px;">Apparent</div>
                        <div style="font-size:20px;font-weight:700;color:#3b82f6;"><span id="oa-live-apparent">--</span><small style="font-size:12px;margin-left:2px;color:var(--text-secondary);">VA</small></div>
                    </div>
                    <div style="text-align:center;padding:12px;border-radius:12px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;margin-bottom:4px;">Reactive</div>
                        <div style="font-size:20px;font-weight:700;color:var(--text);"><span id="oa-live-reactive">--</span><small style="font-size:12px;margin-left:2px;color:var(--text-secondary);">VAR</small></div>
                    </div>
                    <div style="text-align:center;padding:12px;border-radius:12px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;margin-bottom:4px;">Frequency</div>
                        <div style="font-size:20px;font-weight:700;color:var(--text);"><span id="oa-live-freq">--</span><small style="font-size:12px;margin-left:2px;color:var(--text-secondary);">Hz</small></div>
                    </div>
                    <div style="text-align:center;padding:12px;border-radius:12px;background:rgba(5,150,105,0.05);border:1px solid rgba(5,150,105,0.1);">
                        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;margin-bottom:4px;">Power Factor</div>
                        <div style="font-size:20px;font-weight:700;color:var(--text);"><span id="oa-live-pf">--</span> <span id="oa-live-pf-quality" style="font-size:10px;padding:2px 6px;border-radius:8px;background:rgba(107,114,128,0.15);color:#6b7280;vertical-align:middle;">--</span></div>
                    </div>
                </div>
            </div>


            <!-- Per-Outlet Breakdown -->
            <div style="margin-top:18px;display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;">
                <div style="padding:16px;border-radius:14px;border:1px solid rgba(37,99,235,0.2);background:rgba(37,99,235,0.04);">
                    <div style="font-size:12px;font-weight:700;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;">Outlet</div>
                    <div style="font-size:22px;font-weight:700;color:#2563eb;"><span id="oa-o1-power">--</span><small style="font-size:12px;color:var(--text-secondary);">W</small></div>
                    <div style="font-size:12px;color:var(--text-secondary);margin-top:4px;">Energy: <span id="oa-o1-kwh" style="font-weight:600;color:#2563eb;">--</span> kWh</div>
                    <div id="oa-o1-bar" style="height:4px;border-radius:2px;background:rgba(37,99,235,0.15);margin-top:8px;"><div id="oa-o1-bar-fill" style="height:100%;border-radius:2px;background:#2563eb;width:0%;"></div></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast" class="toast">
        <div id="toast-message"></div>
    </div>

    <!-- Energy Data Bubble -->
    <div id="energy-bubble" class="energy-bubble">
        <span class="bubble-dot"></span>
        <span id="energy-bubble-text">--</span>
    </div>

    <!-- Detection Alert -->
    <div id="detection-alert" class="detection-alert">
        <button class="detection-close" onclick="closeDetectionAlert()">
            &times;
        </button>
        <div class="detection-alert-header">
            <span style="font-size: 24px;">!</span>
            <span>Person Detected!</span>
        </div>
        <div class="detection-alert-body">
            <div><strong id="alert-person-count">0</strong> person(s) detected</div>
            <div>Confidence: <strong id="alert-person-confidence">0%</strong></div>
            <div style="margin-top: 8px; font-size: 12px; opacity: 0.8;">
                <span id="alert-time">--:--:--</span>
            </div>
        </div>
    </div>

    <script>
        window.onerror = function(msg, url, line, col, error) {
            console.error('[JS ERROR] ' + msg + ' at line ' + line + ':' + col);
            return false;
        };

        var socket = null;
        try {
            socket = io();
            console.log('[OK] Socket.IO connected');
        } catch(e) {
            console.warn('[WARN] Socket.IO not available:', e.message);
            socket = { on: function(){}, emit: function(){} };
        }
        
        var charts = {};

        // ==================== ROLE-BASED ACCESS CONTROL ====================
        var userRole = 'admin'; // default until /api/auth/role responds
        var ADMIN_PAGES = ['ac-analytics','lamp-analytics','camera','energy','ml-optimization','logs','occupancy-feedback','outlet-analysis'];

        function applyRoleRestrictions(role, username) {
            userRole = role;
            var uEl = document.getElementById('role-username');
            var lEl = document.getElementById('role-label');
            if (uEl) uEl.textContent = username || role;
            if (lEl) {
                lEl.textContent = role === 'admin' ? 'ADMIN' : 'USER';
                lEl.style.background = role === 'admin' ? '#2563eb' : '#3b82f6';
            }
            if (role !== 'admin') {
                document.body.classList.add('role-user');
            } else {
                document.body.classList.remove('role-user');
            }
        }
        var chartRanges = {
            temp: 1,
            hum: 1,
            acTemp: 1,
            acPower: 1,
            lampLux: 1,
            lampBright: 1,
            lampPower: 1,
            occupancy: 1
        };

        var selectedFeedbackRating = 0;
        var DEFAULT_GOOGLE_FORM_URL = 'https://docs.google.com/forms/';

        var learnedCodes = {};

        function formatChartValue(v, unit) {
            const n = parseFloat(v || 0);
            if (!Number.isFinite(n)) return '--';
            return (unit ? n.toFixed(2) + ' ' + unit : n.toFixed(2));
        }

        // styleLineChart removed — all styling moved into makeOpts/makeEnergyOpts at chart creation
        // Mutating chart.options post-creation causes _scriptable recursion in Chart.js 4.x

        // Gradients removed — CanvasGradient objects cause _scriptable recursion in Chart.js 4.x
        // Static rgba() backgroundColor set at chart creation provides the fill effect

        // ==================== LOCALSTORAGE PERSISTENCE ====================
        function saveSettings() {
            const getVal = (id, fallback) => {
                const el = document.getElementById(id);
                return el ? el.value : fallback;
            };
            const settings = {
                acTemp: getVal('ac-temp-slider', 24) || 24,
                fanSpeed: getVal('fan-speed-slider', 1) || 1,
                lampBrightness1: getVal('brightness-slider-1', 0) || 0,
                lampBrightness2: getVal('brightness-slider-2', 0) || 0,
            };
            localStorage.setItem('smartroom_settings', JSON.stringify(settings));
        }

        function loadSavedSettings() {
            const saved = localStorage.getItem('smartroom_settings');
            if (saved) {
                try {
                    const settings = JSON.parse(saved);
                    
                    const acTempSlider = document.getElementById('ac-temp-slider');
                    const fanSpeedSlider = document.getElementById('fan-speed-slider');
                    
                    if (acTempSlider) {
                        acTempSlider.value = settings.acTemp || 24;
                        document.getElementById('ac-temp-display').textContent = acTempSlider.value;
                    }
                    
                    if (fanSpeedSlider) {
                        fanSpeedSlider.value = settings.fanSpeed || 1;
                        document.getElementById('fan-speed-display').textContent = fanSpeedSlider.value;
                    }
                    
                    for (let i = 1; i <= 2; i++) {
                        const slider = document.getElementById('brightness-slider-' + i);
                        if (slider) {
                            slider.value = settings['lampBrightness' + i] || 0;
                            document.getElementById('brightness-display-' + i).textContent = slider.value;
                        }
                    }
                } catch (e) {
                    console.error('Error loading settings:', e);
                }
            }
        }

        // Energy-specific options with tooltip formatting and styled axes.
        // This is used both during initial chart creation and later when Energy Usage
        // charts are recreated between line and bar types.
        function makeEnergyOpts(unit) {
            return {
                responsive: true,
                maintainAspectRatio: true,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#f0f7ff',
                        bodyColor: '#bfdbfe',
                        borderColor: 'rgba(148,163,184,0.35)',
                        borderWidth: 1,
                        displayColors: true,
                        callbacks: {
                            label: function(ctx) {
                                var dsLabel = (ctx.dataset && ctx.dataset.label) ? ctx.dataset.label : 'Value';
                                var delta = ctx.parsed.y;
                                // If kWh chart and cumulative arrays exist, show calculation
                                if (unit === 'kWh' && ctx.chart._kwhCumAc) {
                                    var idx = ctx.dataIndex;
                                    var cumArr = [ctx.chart._kwhCumAc, ctx.chart._kwhCumOutlet, ctx.chart._kwhCumLamp, null][ctx.datasetIndex];
                                    if (cumArr && cumArr.length > idx + 1) {
                                        var prev = cumArr[idx].toFixed(4);
                                        var curr = cumArr[idx + 1].toFixed(4);
                                        return dsLabel + ': ' + curr + ' \u2212 ' + prev + ' = ' + delta.toFixed(5) + ' kWh';
                                    }
                                    return dsLabel + ': ' + delta.toFixed(5) + ' kWh';
                                }
                                return dsLabel + ': ' + formatChartValue(delta, unit);
                            },
                            title: function(items) {
                                if (!items || !items[0]) return '';
                                var lbl = items[0].label || '';
                                if (unit === 'kWh') {
                                    return lbl.indexOf('\u2192') >= 0 ? lbl.replace('\u2192', ' \u2192 ') : lbl;
                                }
                                return 'Time: ' + lbl;
                            }
                        }
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(148,163,184,0.12)' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
                    y: { beginAtZero: true, grid: { color: 'rgba(148,163,184,0.14)' }, ticks: { color: '#94a3b8' } }
                }
            };
        }

        // ==================== CHARTS ====================
        function initCharts() {
            function makeOpts(showLegend) {
                var opts = {
                    responsive: true,
                    maintainAspectRatio: true,
                    animation: false,
                    plugins: { legend: { display: !!showLegend } },
                    scales: {
                        y: { beginAtZero: false, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } }
                    }
                };
                if (showLegend) {
                    opts.plugins.legend.labels = { color: '#94a3b8', font: { size: 12 } };
                }
                return opts;
            }

            charts.temp = new Chart(document.getElementById('tempChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Temperature (\u00b0C)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: true }] }
            });

            charts.hum = new Chart(document.getElementById('humChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Humidity (%)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.4, fill: true }] }
            });

            charts.acTemp = new Chart(document.getElementById('acTempChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Set Temperature (\u00b0C)', data: [], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.4, fill: true }] }
            });

            charts.acHum = new Chart(document.getElementById('acHumChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Set RH (%)', data: [], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.4, fill: true }] }
            });

            charts.lampLux = new Chart(document.getElementById('lampLuxChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Light Intensity (lux)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.1)', tension: 0.4, fill: true }] }
            });

            charts.lampBright = new Chart(document.getElementById('lampBrightChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Brightness (%)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.4, fill: true }] }
            });

            charts.acPower = new Chart(document.getElementById('acPowerChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'AC Power (W)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: true, pointRadius: 3, pointHoverRadius: 6 }] }
            });

            charts.lampPower = new Chart(document.getElementById('lampPowerChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Lamp Power (W)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.1)', tension: 0.4, fill: true, pointRadius: 3, pointHoverRadius: 6 }] }
            });

            // AC & Lamp individual energy kWh charts (analytics pages)
            charts.acEnergyKwh = new Chart(document.getElementById('acEnergyKwhChart'), {
                type: 'line',
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f0f7ff',
                            bodyColor: '#bfdbfe', borderColor: 'rgba(59,130,246,0.4)', borderWidth: 1,
                            displayColors: false,
                            callbacks: {
                                label: function(ctx) { return 'Delta: ' + ctx.parsed.y.toFixed(5) + ' kWh'; }
                            }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.10)' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } } },
                        y: { beginAtZero: true, grid: { color: 'rgba(59,130,246,0.10)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'kWh', color: '#64748b', font: { size: 10 } } }
                    }
                },
                data: { labels: [], datasets: [{
                    label: 'AC Energy (kWh)', data: [],
                    borderColor: '#3b82f6', borderWidth: 2, tension: 0.3,
                    backgroundColor: function(ctx) {
                        var c = ctx.chart.ctx; var a = ctx.chart.chartArea; if (!a) return 'rgba(59,130,246,0.1)';
                        var g = c.createLinearGradient(0, a.top, 0, a.bottom);
                        g.addColorStop(0, 'rgba(59,130,246,0.25)'); g.addColorStop(1, 'rgba(59,130,246,0.02)'); return g;
                    },
                    fill: true, pointRadius: 2, pointHoverRadius: 6,
                    pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 2
                }] }
            });

            charts.lampEnergyKwh = new Chart(document.getElementById('lampEnergyKwhChart'), {
                type: 'line',
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f0f7ff',
                            bodyColor: '#bfdbfe', borderColor: 'rgba(37,99,235,0.4)', borderWidth: 1,
                            displayColors: false,
                            callbacks: {
                                label: function(ctx) { return 'Delta: ' + ctx.parsed.y.toFixed(5) + ' kWh'; }
                            }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.10)' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } } },
                        y: { beginAtZero: true, grid: { color: 'rgba(37,99,235,0.10)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'kWh', color: '#64748b', font: { size: 10 } } }
                    }
                },
                data: { labels: [], datasets: [{
                    label: 'Lamp Energy (kWh)', data: [],
                    borderColor: '#2563eb', borderWidth: 2, tension: 0.3,
                    backgroundColor: function(ctx) {
                        var c = ctx.chart.ctx; var a = ctx.chart.chartArea; if (!a) return 'rgba(37,99,235,0.1)';
                        var g = c.createLinearGradient(0, a.top, 0, a.bottom);
                        g.addColorStop(0, 'rgba(37,99,235,0.25)'); g.addColorStop(1, 'rgba(37,99,235,0.02)'); return g;
                    },
                    fill: true, pointRadius: 2, pointHoverRadius: 6,
                    pointBackgroundColor: '#2563eb', pointBorderColor: '#fff', pointBorderWidth: 2
                }] }
            });

            charts.outletEnergyKwhMySQL = new Chart(document.getElementById('outletEnergyKwhMySQLChart'), {
                type: 'line',
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f0f7ff',
                            bodyColor: '#a7f3d0', borderColor: 'rgba(5,150,105,0.4)', borderWidth: 1,
                            displayColors: false,
                            callbacks: {
                                label: function(ctx) { return 'Delta: ' + ctx.parsed.y.toFixed(5) + ' kWh'; }
                            }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.10)' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } } },
                        y: { beginAtZero: true, grid: { color: 'rgba(5,150,105,0.10)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'kWh', color: '#64748b', font: { size: 10 } } }
                    }
                },
                data: { labels: [], datasets: [{
                    label: 'Outlet Energy (kWh)', data: [],
                    borderColor: '#059669', borderWidth: 2, tension: 0.3,
                    backgroundColor: function(ctx) {
                        var c = ctx.chart.ctx; var a = ctx.chart.chartArea; if (!a) return 'rgba(5,150,105,0.1)';
                        var g = c.createLinearGradient(0, a.top, 0, a.bottom);
                        g.addColorStop(0, 'rgba(5,150,105,0.25)'); g.addColorStop(1, 'rgba(5,150,105,0.02)'); return g;
                    },
                    fill: true, pointRadius: 2, pointHoverRadius: 6,
                    pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2
                }] }
            });


            charts.energyPower = new Chart(document.getElementById('energyPowerChart'), {
                type: 'line', options: makeEnergyOpts('W'),
                data: { labels: [], datasets: [
                    { label: 'AC Power (W)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Outlet Power (W)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Lamp Power (W)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Total Power (W)', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.08)', tension: 0.4, fill: false, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#7c3aed', pointBorderColor: '#fff', pointBorderWidth: 2, borderDash: [5,3] }
                ]}
            });

            charts.energyVoltage = new Chart(document.getElementById('energyVoltageChart'), {
                type: 'line', options: makeEnergyOpts('V'),
                data: { labels: [], datasets: [
                    { label: 'AC Voltage (V)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Outlet Voltage (V)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Lamp Voltage (V)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2 }
                ]}
            });

            charts.energyCurrent = new Chart(document.getElementById('energyCurrentChart'), {
                type: 'line', options: makeEnergyOpts('A'),
                data: { labels: [], datasets: [
                    { label: 'AC Current (A)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Outlet Current (A)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Lamp Current (A)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Total Current (A)', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.08)', tension: 0.4, fill: false, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#7c3aed', pointBorderColor: '#fff', pointBorderWidth: 2, borderDash: [5,3] }
                ]}
            });

            charts.energyKwh = new Chart(document.getElementById('energyKwhChart'), {
                type: 'line', options: makeEnergyOpts('kWh'),
                data: { labels: [], datasets: [
                    { label: 'AC Energy (kWh)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.08)', tension: 0.3, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                    { label: 'Outlet Energy (kWh)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.08)', tension: 0.3, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                    { label: 'Lamp Energy (kWh)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.08)', tension: 0.3, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                    { label: 'Total Energy (kWh)', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.06)', tension: 0.3, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#7c3aed', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2, borderDash: [5,3] }
                ]}
            });

            // MySQL Real-time Charts (ring buffer, 30 points)
            charts.mysqlVoltFreq = new Chart(document.getElementById('mysqlVoltFreqChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Frequency (Hz)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.12)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.08)' }, ticks: { color: '#94a3b8', maxTicksLimit: 8, maxRotation: 0, font: { size: 11 } } },
                        y: { title: { display: true, text: 'Hz', color: '#94a3b8', font: { size: 11 } }, suggestedMin: 49, suggestedMax: 51, ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(148,163,184,0.08)' } }
                    }
                }
            });

            charts.mysqlCurrent = new Chart(document.getElementById('mysqlCurrentChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'AC (A)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true },
                        { label: 'Outlet (A)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true },
                        { label: 'Lamp (A)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    plugins: { legend: { display: true, labels: { boxWidth: 12, font: { size: 11 }, color: '#94a3b8', padding: 16 } } },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.08)' }, ticks: { color: '#94a3b8', maxTicksLimit: 8, maxRotation: 0, font: { size: 11 } } },
                        y: { beginAtZero: true, title: { display: true, text: 'A', color: '#94a3b8', font: { size: 11 } }, ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(148,163,184,0.08)' } }
                    }
                }
            });

            // Energy Comparison: 4 separate charts (Before/After x Power/kWh) for side-by-side view
            var compareLineOpts = function(unit) {
                return {
                    responsive: true,
                    maintainAspectRatio: true,
                    animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 }, maxRotation: 45, maxTicksLimit: 12 } },
                        y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { color: '#94a3b8', font: { size: 10 } }, title: { display: true, text: unit, color: '#94a3b8', font: { size: 11 } } }
                    }
                };
            };

            charts.energyCompareBefore = new Chart(document.getElementById('energyCompareBeforeChart'), {
                type: 'line', options: compareLineOpts('W'),
                data: { labels: [], datasets: [{ label: 'Before — Power', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.energyCompareAfter = new Chart(document.getElementById('energyCompareAfterChart'), {
                type: 'line', options: compareLineOpts('W'),
                data: { labels: [], datasets: [{ label: 'After — Power', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.energyCompareKwhBefore = new Chart(document.getElementById('energyCompareKwhBeforeChart'), {
                type: 'line', options: compareLineOpts('kWh'),
                data: { labels: [], datasets: [{ label: 'Before — Energy', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.energyCompareKwhAfter = new Chart(document.getElementById('energyCompareKwhAfterChart'), {
                type: 'line', options: compareLineOpts('kWh'),
                data: { labels: [], datasets: [{ label: 'After — Energy', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });

            // Lamp Comparison Charts
            charts.lampCompareBefore = new Chart(document.getElementById('lampCompareBeforeChart'), {
                type: 'line', options: compareLineOpts('W'),
                data: { labels: [], datasets: [{ label: 'Before — Lamp Power', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.lampCompareAfter = new Chart(document.getElementById('lampCompareAfterChart'), {
                type: 'line', options: compareLineOpts('W'),
                data: { labels: [], datasets: [{ label: 'After — Lamp Power', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.lampCompareKwhBefore = new Chart(document.getElementById('lampCompareKwhBeforeChart'), {
                type: 'line', options: compareLineOpts('kWh'),
                data: { labels: [], datasets: [{ label: 'Before — Lamp Energy', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.lampCompareKwhAfter = new Chart(document.getElementById('lampCompareKwhAfterChart'), {
                type: 'line', options: compareLineOpts('kWh'),
                data: { labels: [], datasets: [{ label: 'After — Lamp Energy', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });

            charts.occupancy = new Chart(document.getElementById('occupancyChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Occupancy (person)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.35, fill: true, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });

            // Energy chart styling is set at creation via makeEnergyOpts — no post-init mutation needed
            // ML charts (gaFitness, psoFitness, comparison) are initialized separately via initMLCharts()
        }

        function initMLCharts() {
            function makeOpts(showLegend) {
                var opts = {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    plugins: { legend: { display: !!showLegend } },
                    scales: {
                        y: { beginAtZero: false, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } }
                    }
                };
                if (showLegend) {
                    opts.plugins.legend.labels = { color: '#94a3b8', font: { size: 12 } };
                }
                return opts;
            }
            if (charts.gaFitness) { try { charts.gaFitness.destroy(); } catch(e){} charts.gaFitness = null; }
            if (charts.psoFitness) { try { charts.psoFitness.destroy(); } catch(e){} charts.psoFitness = null; }
            if (charts.comparison) { try { charts.comparison.destroy(); } catch(e){} charts.comparison = null; }

            charts.gaFitness = new Chart(document.getElementById('gaFitnessChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'GA Best Fitness', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2 }] }
            });

            charts.psoFitness = new Chart(document.getElementById('psoFitnessChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'PWM 1 (0-255)', data: [], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.3, fill: false, pointRadius: 4, yAxisID: 'yPWM' },
                        { label: 'PWM 2 (0-255)', data: [], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.3, fill: false, pointRadius: 4, yAxisID: 'yPWM' },
                        { label: 'Lux Avg (lux)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.3, fill: true, pointRadius: 4, yAxisID: 'yLux', borderDash: [4,2] },
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: true, animation: { duration: 300 },
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: true, labels: { color: '#94a3b8', font: { size: 11 } } },
                        tooltip: { callbacks: {
                            title: items => `Iteration ${items[0].label}`,
                            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}`
                        }}
                    },
                    scales: {
                        x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        yPWM: { type: 'linear', position: 'left', min: 0, max: 255, ticks: { color: '#2563eb', font: { size: 10 } }, title: { display: true, text: 'PWM', color: '#2563eb', font: { size: 10 } }, grid: { color: 'rgba(37,99,235,0.08)' } },
                        yLux: { type: 'linear', position: 'right', min: 0, ticks: { color: '#0ea5e9', font: { size: 10 } }, title: { display: true, text: 'Lux', color: '#0ea5e9', font: { size: 10 } }, grid: { drawOnChartArea: false } },
                    }
                }
            });

            charts.comparison = new Chart(document.getElementById('comparisonChart'), {
                type: 'line', options: makeOpts(true),
                data: {
                    labels: [],
                    datasets: [
                        { label: 'GA (AC)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.4, fill: false, pointRadius: 3 },
                        { label: 'PSO (Lamp)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.1)', tension: 0.4, fill: false, pointRadius: 3 }
                    ]
                }
            });

            // Load any pending ML data that was stored while the ML page was hidden
            if (window.__pendingMLData) {
                var pending = window.__pendingMLData;
                window.__pendingMLData = null;
                Object.keys(pending).forEach(function(chartName) {
                    var pd = pending[chartName];
                    if (pd && pd.history && pd.history.length > 0) {
                        try { updateMLChart(chartName, pd.history, pd.algo); } catch(e) { console.warn('[CHART] pending ML data load failed:', e); }
                    }
                });
            }
        }

        // Ensure charts are available even if init runs before library/page is fully ready.
        function ensureChartsReady() {
            if (typeof Chart === 'undefined') {
                console.warn('[CHART] Chart.js not loaded yet');
                if (!window.__chartFallbackRequested) {
                    window.__chartFallbackRequested = true;
                    try {
                        var s = document.createElement('script');
                        s.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js';
                        s.onload = function() {
                            console.log('[CHART] Fallback Chart.js loaded');
                            try { initCharts(); } catch(e) { console.error('[CHART] init after fallback failed:', e); }
                            try { initMLCharts(); } catch(e) { console.error('[CHART] initML after fallback failed:', e); }
                            try { loadAllEnergyCharts(); } catch(e) { console.error('[CHART] load after fallback failed:', e); }
                        };
                        s.onerror = function() {
                            console.error('[CHART] Fallback Chart.js failed to load');
                        };
                        document.head.appendChild(s);
                    } catch (e) {
                        console.error('[CHART] Fallback loader error:', e);
                    }
                }
                return false;
            }

            // Only call initCharts() if core charts (temp, hum) are also missing — meaning initCharts never ran.
            // If charts.temp already exists, initCharts() already ran and calling it again will crash on duplicate canvases.
            if ((!charts.energyPower || !charts.energyVoltage || !charts.energyKwh || !charts.energyCompareBefore || !charts.energyCompareKwhBefore) && !charts.temp) {
                try {
                    initCharts();
                } catch (e) {
                    console.error('[CHART] initCharts retry failed:', e);
                    return false;
                }
            }

            // ML charts are independent — init separately to avoid destroying already-created canvases
            // Guard: only init when canvas is visible (has positive width), to avoid 0px Chart.js failures
            if (!charts.gaFitness || !charts.psoFitness || !charts.comparison) {
                var mlCanvas = document.getElementById('gaFitnessChart');
                if (mlCanvas && mlCanvas.offsetWidth > 0) {
                    try {
                        initMLCharts();
                    } catch (e) {
                        console.error('[CHART] initMLCharts failed:', e);
                        return false;
                    }
                }
            }

            return true;
        }

        function updateChartData(chartName, hours) {
            let endpoint = '';
            switch(chartName) {
                case 'temp': endpoint = '/api/chart/ac_sensor/temperature/' + hours; break;
                case 'hum': endpoint = '/api/chart/ac_sensor/humidity/' + hours; break;
                case 'acTemp': endpoint = '/api/chart/ac_sensor/ac_temp/' + hours; break;
                case 'acHum': endpoint = '/api/chart/ac_sensor/set_rh/' + hours; break;
                case 'lampLux': endpoint = '/api/chart/lamp_sensor/lux/' + hours; break;
                case 'lampBright': endpoint = '/api/chart/lamp_sensor/brightness/' + hours; break;
                case 'occupancy': endpoint = '/api/chart/camera_detection/person_count/' + hours; break;
                case 'acPower': endpoint = '/api/chart/ac_power/' + hours; break;
                case 'lampPower': endpoint = '/api/chart/lamp_power/' + hours; break;
            }

            fetch(endpoint)
                .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
                .then(data => {
                    if (data && data.length > 0) {
                        charts[chartName].data.labels = data.map(d => d.time);
                        charts[chartName].data.datasets[0].data = data.map(d => d.value);
                        charts[chartName].update('none');
                    }
                })
                .catch(e => console.warn('Chart data unavailable (' + chartName + '):', e.message));
        }

        function changeChartRange(chartName, hours) {
            chartRanges[chartName] = hours;
            localStorage.setItem('chartRanges', JSON.stringify(chartRanges));
            
            const buttons = event.target.parentElement.querySelectorAll('.chart-option-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            updateChartData(chartName, hours);
        }

        // ==================== ENERGY HISTORY (PZEM-016 from InfluxDB) ====================
        var energyChartMap = {
            'power': 'energyPower',
            'voltage': 'energyVoltage',
            'current': 'energyCurrent',
            'energy_kwh': 'energyKwh'
        };

        var energyCanvasMap = {
            'power': 'energyPowerChart',
            'voltage': 'energyVoltageChart',
            'current': 'energyCurrentChart',
            'energy_kwh': 'energyKwhChart'
        };

        var energyColorMap = {
            'power': '#1e40af',
            'voltage': '#3b82f6',
            'current': '#0ea5e9',
            'energy_kwh': '#3b82f6'
        };

        function updateEnergyStats(field, values) {
            if (!values || values.length === 0) return;
            var prefixMap = {'power': 'energy-power', 'voltage': 'energy-voltage', 'current': 'energy-current', 'energy_kwh': 'energy-kwh'};
            const prefix = prefixMap[field] || ('energy-' + field);
            const minV = Math.min(...values);
            const maxV = Math.max(...values);
            const avgV = values.reduce((a, b) => a + b, 0) / values.length;
            const lastV = values[values.length - 1];
            const isKwh = field === 'energy_kwh';
            const totalV = values.reduce((a, b) => a + b, 0);
            const dp = isKwh ? 4 : 2;
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set(prefix + '-latest', lastV.toFixed(dp));
            set(prefix + '-min', minV.toFixed(dp));
            set(prefix + '-max', maxV.toFixed(dp));
            // kWh: show total sum of deltas
            set(prefix + '-avg', isKwh ? totalV.toFixed(4) : avgV.toFixed(2));
        }

        function drawEnergyFallback(field, points) {
            const canvasId = energyCanvasMap[field];
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;

            const rect = canvas.getBoundingClientRect();
            const width = Math.max(320, Math.floor(rect.width || canvas.width || 640));
            const height = Math.max(160, Math.floor(rect.height || canvas.height || 220));
            canvas.width = width;
            canvas.height = height;

            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = 'rgba(148, 163, 184, 0.12)';
            ctx.fillRect(0, 0, width, height);

            ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 4; i++) {
                const y = (height / 5) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            if (!points || points.length === 0) {
                ctx.fillStyle = '#94a3b8';
                ctx.font = '13px sans-serif';
                ctx.fillText('Waiting energy data...', 16, Math.floor(height / 2));
                return;
            }

            const values = points.map(p => parseFloat(p.value || 0)).filter(v => Number.isFinite(v));
            if (values.length === 0) return;

            let minV = Math.min(...values);
            let maxV = Math.max(...values);
            if (minV === maxV) {
                minV -= 1;
                maxV += 1;
            }

            const padX = 26;
            const padY = 18;
            const plotW = width - padX * 2;
            const plotH = height - padY * 2;

            ctx.strokeStyle = energyColorMap[field] || '#2563eb';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const dotColor = energyColorMap[field] || '#2563eb';
            values.forEach((v, i) => {
                const x = padX + (i / Math.max(1, values.length - 1)) * plotW;
                const y = padY + (1 - ((v - minV) / (maxV - minV))) * plotH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Point markers for each incoming sample
            values.forEach((v, i) => {
                const x = padX + (i / Math.max(1, values.length - 1)) * plotW;
                const y = padY + (1 - ((v - minV) / (maxV - minV))) * plotH;
                ctx.beginPath();
                ctx.arc(x, y, 2.8, 0, Math.PI * 2);
                ctx.fillStyle = dotColor;
                ctx.fill();
                ctx.lineWidth = 1;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();
            });

            const last = values[values.length - 1];
            ctx.fillStyle = '#cbd5e1';
            ctx.font = '12px sans-serif';
            ctx.fillText('Latest: ' + last.toFixed(2), 12, 14);
            updateEnergyStats(field, values);
        }

        // Track current period per field so auto-refresh doesn't reset user selection
        var _activePeriod = { power: '24h', voltage: '24h', current: '24h', energy_kwh: '24h' };

        // Energy is computed by integrating high-resolution Power data 
        // to avoid the 1kWh low-resolution stepping from the raw cumulative sensor.
        var _periodMins = {
            '1h': 1, '6h': 5, '24h': 60, '7d': 1440, '30d': 1440, '12mo': 43200, '5y': 525600
        };

        function loadEnergyHistory(field, period, btnElement) {
            // Save current period for this field
            _activePeriod[field] = period;
            // Update button active state
            if (btnElement) {
                const buttons = btnElement.parentElement.querySelectorAll('.chart-option-btn');
                buttons.forEach(btn => btn.classList.remove('active'));
                btnElement.classList.add('active');
            }

            const chartName = energyChartMap[field];
            if (!chartName) return;

            var fetchField = field;
            // Fetch AC, Outlet, and Lamp in parallel
            Promise.all([
                fetch('/api/energy/history?field=' + fetchField + '&period=' + period + '&device=ac').then(r => r.json()),
                fetch('/api/energy/history?field=' + fetchField + '&period=' + period + '&device=outlet').then(r => r.json()),
                fetch('/api/energy/history?field=' + fetchField + '&period=' + period + '&device=lamp').then(r => r.json())
            ])
            .then(function(results) {
                var acResult = results[0];
                var outletResult = results[1];
                var lampResult = results[2];
                var acData = acResult.data || [];
                var outletData = outletResult.data || [];
                var lampData = lampResult.data || [];
                var chart = charts[chartName];

                if (chart && chart.data && chart.data.datasets) {
                    var acValues = acData.map(function(d){ return parseFloat(d.value || 0); });
                    var outletValues = outletData.map(function(d){ return parseFloat(d.value || 0); });
                    var lampValues = lampData.map(function(d){ return parseFloat(d.value || 0); });
                    var acLabels = acData.map(function(d){ return d.time; });
                    var outletLabels = outletData.map(function(d){ return d.time; });
                    var lampLabels = lampData.map(function(d){ return d.time; });
                    
                    // energy_kwh is cumulative — compute delta per interval
                    if (field === 'energy_kwh') {
                        function computeDeltas(cumValues) {
                            var deltas = [];
                            if (cumValues.length <= 1) return cumValues.slice();
                            for (var i = 1; i < cumValues.length; i++) {
                                var diff = cumValues[i] - cumValues[i - 1];
                                deltas.push(parseFloat(Math.max(0, diff).toFixed(5)));
                            }
                            return deltas;
                        }
                        function buildRangeLabels(labels) {
                            var rangeLabels = [];
                            if (labels.length <= 1) return labels.slice();
                            for (var i = 1; i < labels.length; i++) {
                                rangeLabels.push(labels[i - 1] + '\u2192' + labels[i]);
                            }
                            return rangeLabels;
                        }

                        var acCumRaw = acValues.slice();
                        var outletCumRaw = outletValues.slice();
                        var lampCumRaw = lampValues.slice();
                        acValues = computeDeltas(acValues);
                        outletValues = computeDeltas(outletValues);
                        lampValues = computeDeltas(lampValues);

                        // Daily stays as a line; every longer Energy Usage period is a bar chart.
                        var useBarChart = (period !== '24h');
                        if (useBarChart) {
                            if (acLabels.length > 1) acLabels = acLabels.slice(1);
                            if (outletLabels.length > 1) outletLabels = outletLabels.slice(1);
                            if (lampLabels.length > 1) lampLabels = lampLabels.slice(1);
                        } else {
                            acLabels = buildRangeLabels(acLabels);
                            outletLabels = buildRangeLabels(outletLabels);
                            lampLabels = buildRangeLabels(lampLabels);
                        }
                    }

                    // ── Determine chart type: Daily (24h) = line, others = bar ──
                    var useBarForAll = (period !== '24h');
                    var targetType = useBarForAll ? 'bar' : 'line';
                    var unitMap = { power: 'kW', voltage: 'V', current: 'A', energy_kwh: 'kWh' };
                    var unit = unitMap[field] || '';
                    var hasTotal = (field !== 'voltage');  // Voltage has no Total dataset

                    // Recreate chart if type needs to change (line <-> bar)
                    if (chart.config.type !== targetType) {
                        chart.destroy();
                        var canvasId = energyCanvasMap[field];
                        var canvas = document.getElementById(canvasId);
                        var baseOpts = makeEnergyOpts(unit);
                        var datasetsToUse;

                        if (useBarForAll) {
                            baseOpts.plugins.legend = { display: true, labels: { boxWidth: 14, font: { size: 11 }, color: '#94a3b8', padding: 14 } };
                            baseOpts.scales.x.grid = { display: false };
                            baseOpts.scales.x.ticks.maxRotation = 45;
                            baseOpts.scales.y.title = { display: true, text: unit, color: '#94a3b8', font: { size: 11 } };
                            
                            datasetsToUse = [
                                { label: 'AC (' + unit + ')', data: [], backgroundColor: 'rgba(30,64,175,0.6)', borderColor: '#1e40af', borderWidth: 1, borderRadius: 4, barPercentage: 0.6, categoryPercentage: 0.85 },
                                { label: 'Outlet (' + unit + ')', data: [], backgroundColor: 'rgba(5,150,105,0.6)', borderColor: '#059669', borderWidth: 1, borderRadius: 4, barPercentage: 0.6, categoryPercentage: 0.85 },
                                { label: 'Lamp (' + unit + ')', data: [], backgroundColor: 'rgba(234,179,8,0.6)', borderColor: '#eab308', borderWidth: 1, borderRadius: 4, barPercentage: 0.6, categoryPercentage: 0.85 }
                            ];
                            if (hasTotal) {
                                datasetsToUse.push({ label: 'Total (' + unit + ')', data: [], backgroundColor: 'rgba(124,58,237,0.4)', borderColor: '#7c3aed', borderWidth: 1, borderRadius: 4, barPercentage: 0.6, categoryPercentage: 0.85 });
                            }
                        } else {
                            // Line chart (Daily / 24h)
                            datasetsToUse = [
                                { label: 'AC (' + unit + ')', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.08)', tension: 0.3, fill: (field === 'energy_kwh'), pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                                { label: 'Outlet (' + unit + ')', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.08)', tension: 0.3, fill: (field === 'energy_kwh'), pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                                { label: 'Lamp (' + unit + ')', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.08)', tension: 0.3, fill: (field === 'energy_kwh'), pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 }
                            ];
                            if (hasTotal) {
                                datasetsToUse.push({ label: 'Total (' + unit + ')', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.06)', tension: 0.3, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#7c3aed', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2, borderDash: [5,3] });
                            }
                        }
                        
                        try {
                            chart = new Chart(canvas, {
                                type: targetType,
                                options: baseOpts,
                                data: { labels: [], datasets: datasetsToUse }
                            });
                            charts[chartName] = chart;
                        } catch (err) {
                            console.error('Chart.js recreation error:', err);
                        }
                    }

                    if (field === 'energy_kwh') {
                        chart._kwhCumAc = acCumRaw;
                        chart._kwhCumOutlet = outletCumRaw;
                        chart._kwhCumLamp = lampCumRaw;
                    }

                    // Compute Total values (AC + Outlet + Lamp)
                    var maxLen = Math.max(acValues.length, outletValues.length, lampValues.length);
                    var totalValues = [];
                    for (var ti = 0; ti < maxLen; ti++) {
                        totalValues.push((acValues[ti] || 0) + (outletValues[ti] || 0) + (lampValues[ti] || 0));
                    }
                    
                    chart.data.labels = acLabels.length > 0 ? acLabels : (outletLabels.length > 0 ? outletLabels : lampLabels);
                    chart.data.datasets[0].data = acValues;

                    // Voltage chart: datasets = [AC, Outlet, Lamp] (no Total)
                    if (field === 'voltage') {
                        if (chart.data.datasets[1]) chart.data.datasets[1].data = outletValues;
                        if (chart.data.datasets[2]) chart.data.datasets[2].data = lampValues;
                    } else {
                        // Power, Current, kWh: datasets = [AC, Outlet, Lamp, Total]
                        if (chart.data.datasets[1]) chart.data.datasets[1].data = outletValues;
                        if (chart.data.datasets[2]) chart.data.datasets[2].data = lampValues;
                        if (chart.data.datasets[3]) chart.data.datasets[3].data = totalValues;
                    }
                    chart.update('none');
                    updateEnergyStats(field, acValues.filter(function(v){ return Number.isFinite(v); }));
                    updateOutletEnergyStats(field, outletValues.filter(function(v){ return Number.isFinite(v); }));
                    updateLampEnergyStats(field, lampValues.filter(function(v){ return Number.isFinite(v); }));
                } else {
                    drawEnergyFallback(field, acData);
                }

                // Show "No Data" overlay on canvas if empty
                var canvasId = energyCanvasMap[field];
                var noDataId = 'nodata-' + canvasId;
                var existingOverlay = document.getElementById(noDataId);
                if (existingOverlay) existingOverlay.remove();
                if (acData.length === 0 && outletData.length === 0 && lampData.length === 0 && canvasId) {
                    var canvas = document.getElementById(canvasId);
                    if (canvas && canvas.parentElement) {
                        var overlay = document.createElement('div');
                        overlay.id = noDataId;
                        overlay.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:var(--text-secondary);font-size:14px;font-weight:600;pointer-events:none;text-align:center;';
                        overlay.textContent = 'Waiting for MySQL data (polling every 5 seconds)...';
                        canvas.parentElement.style.position = 'relative';
                        canvas.parentElement.appendChild(overlay);
                    }
                }
            })
            .catch(function(e){ console.error('Energy history error:', e); });
        }

        // ── toggleDataset: show/hide chart datasets via visibility checkboxes ──
        function toggleDataset(chartName, datasetIndex, checkbox) {
            var chart = charts[chartName];
            if (!chart || !chart.data || !chart.data.datasets[datasetIndex]) return;
            chart.data.datasets[datasetIndex].hidden = !checkbox.checked;
            chart.update('none');
        }

        function updateOutletEnergyStats(field, values) {
            if (!values || values.length === 0) return;
            var avg = values.reduce(function(a,b){ return a+b; }, 0) / values.length;
            var latest = values[values.length - 1];
            var unit = {power:'kW', current:'A', energy_kwh:'kWh', voltage:'V'}[field] || '';
            var fmt = function(v) {
                return field === 'energy_kwh'
                    ? v.toFixed(4).replace(/\\.0+$/, '').replace(/(\\.\\d*?)0+$/, '$1')
                    : v.toFixed(2);
            };
            var setEl = function(id, v) { var el = document.getElementById(id); if (el) el.textContent = fmt(v) + ' ' + unit; };
            if (field === 'power') { setEl('outlet-power-latest', latest); setEl('outlet-power-avg', avg); }
            else if (field === 'current') { setEl('outlet-current-latest', latest); setEl('outlet-current-avg', avg); }
            else if (field === 'energy_kwh') {
                var total = values.reduce(function(a,b){ return a+b; }, 0);
                setEl('outlet-kwh-latest', latest);
                setEl('outlet-kwh-avg', total);
            }
        }

        function updateLampEnergyStats(field, values) {
            if (!values || values.length === 0) return;
            var avg = values.reduce(function(a,b){ return a+b; }, 0) / values.length;
            var latest = values[values.length - 1];
            var unit = {power:'kW', current:'A', energy_kwh:'kWh', voltage:'V'}[field] || '';
            var fmt = function(v) {
                return field === 'energy_kwh'
                    ? v.toFixed(4).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1')
                    : v.toFixed(2);
            };
            var setEl = function(id, v) { var el = document.getElementById(id); if (el) el.textContent = fmt(v) + ' ' + unit; };
            if (field === 'power') { setEl('lamp-power-latest', latest); setEl('lamp-power-avg', avg); }
            else if (field === 'current') { setEl('lamp-current-latest', latest); setEl('lamp-current-avg', avg); }
            else if (field === 'energy_kwh') {
                var total = values.reduce(function(a,b){ return a+b; }, 0);
                setEl('lamp-kwh-latest', latest);
                setEl('lamp-kwh-avg', total);
            }
        }

        // Analytics page individual energy chart loader
        var _analyticsEnergyPeriod = { ac: '24h', lamp: '24h', outlet: '24h' };
        function loadAnalyticsEnergy(device, period, btnEl) {
            _analyticsEnergyPeriod[device] = period;
            if (btnEl) {
                var btns = btnEl.parentElement.querySelectorAll('.chart-option-btn');
                btns.forEach(function(b){ b.classList.remove('active'); });
                btnEl.classList.add('active');
            }
            fetch('/api/energy/history?field=energy_kwh&period=' + period + '&device=' + device)
                .then(function(r){ return r.json(); })
                .then(function(result) {
                    var raw = result.data || [];
                    var cumValues = raw.map(function(d){ return parseFloat(d.value || 0); });
                    var labels = raw.map(function(d){ return d.time; });
                    // Compute deltas from cumulative values
                    var deltas = [];
                    var rangeLabels = [];
                    for (var i = 1; i < cumValues.length; i++) {
                        var diff = cumValues[i] - cumValues[i - 1];
                        deltas.push(parseFloat(Math.max(0, diff).toFixed(5)));
                        rangeLabels.push(labels[i - 1] + '\u2192' + labels[i]);
                    }
                    var chartKey = device === 'ac' ? 'acEnergyKwh' : (device === 'lamp' ? 'lampEnergyKwh' : 'outletEnergyKwhMySQL');
                    var chart = charts[chartKey];
                    if (chart && chart.data) {
                        chart.data.labels = rangeLabels;
                        chart.data.datasets[0].data = deltas;
                        chart.update('none');
                    }
                    // Update stat cards
                    var prefix = device; // device is 'ac', 'lamp', or 'outlet'
                    var lastEl = document.getElementById(prefix + '-analytics-kwh-last');
                    var totalEl = document.getElementById(prefix + '-analytics-kwh-total');
                    var avgEl = document.getElementById(prefix + '-analytics-kwh-avg');
                    if (deltas.length > 0) {
                        var last = deltas[deltas.length - 1];
                        var total = deltas.reduce(function(a,b){ return a+b; }, 0);
                        var avg = total / deltas.length;
                        if (lastEl) lastEl.textContent = last.toFixed(4) + ' kWh';
                        if (totalEl) totalEl.textContent = total.toFixed(4) + ' kWh';
                        if (avgEl) avgEl.textContent = avg.toFixed(4) + ' kWh';
                    } else {
                        if (lastEl) lastEl.textContent = '--';
                        if (totalEl) totalEl.textContent = '--';
                        if (avgEl) avgEl.textContent = '--';
                    }
                })
                .catch(function(e){ console.error('Analytics energy error (' + device + '):', e); });
        }

        function loadAllEnergyCharts() {
            loadEnergyHistory('power',      _activePeriod.power,      null);
            loadEnergyHistory('voltage',    _activePeriod.voltage,    null);
            loadEnergyHistory('current',    _activePeriod.current,    null);
            loadEnergyHistory('energy_kwh', _activePeriod.energy_kwh, null);
            loadEnergyCompare('power');
            loadEnergyCompare('energy_kwh');
            loadLampEnergyCompare('power');
            loadLampEnergyCompare('energy_kwh');
            loadRecordingState();
            loadLampRecordingState();
            loadDailySummary();
        }

        // ==================== DAILY SUMMARY ====================
        var _dailySummaryTimer = null;
        function loadDailySummary() {
            fetch('/api/energy/daily-summary')
                .then(r => r.json())
                .then(d => {
                    if (d.error) return;
                    var setText = function(id, v) { var el = document.getElementById(id); if (el) el.textContent = v; };
                    var fmtKwh = function(v) {
                        var num = parseFloat(v || 0);
                        if (!Number.isFinite(num)) return '--';
                        return num.toFixed(4).replace(/\\.0+$/, '').replace(/(\\.\\d*?)0+$/, '$1');
                    };
                    setText('daily-summary-date', d.date || '--');
                    setText('ds-ac-kwh',       d.ac ? fmtKwh(d.ac.kwh) : '--');
                    setText('ds-lamp-kwh',     d.lamp ? fmtKwh(d.lamp.kwh) : '--');
                    setText('ds-total-kwh',    d.total_kwh !== undefined ? fmtKwh(d.total_kwh) : '--');
                    setText('ds-cost',         d.cost_rp !== undefined ? d.cost_rp.toLocaleString() : '--');
                    setText('ds-ac-peak',      d.ac ? d.ac.power_peak_w.toFixed(1) : '--');
                    setText('ds-lamp-peak',    d.lamp ? d.lamp.power_peak_w.toFixed(1) : '--');
                    setText('ds-ac-runtime',   d.ac ? d.ac.runtime_h.toFixed(2) : '--');
                    setText('ds-lamp-runtime', d.lamp ? d.lamp.runtime_h.toFixed(2) : '--');
                    if (d.outlet) {
                        setText('outlet-today-kwh', d.outlet.kwh !== undefined ? fmtKwh(d.outlet.kwh) : '--');
                        setText('outlet-peak-power', d.outlet.power_peak_w !== undefined ? d.outlet.power_peak_w.toFixed(1) : '--');
                    }
                })
                .catch(function(e) { console.error('Daily summary error:', e); });
        }
        // Auto-refresh daily summary every 10 minutes
        if (_dailySummaryTimer) clearInterval(_dailySummaryTimer);
        _dailySummaryTimer = setInterval(loadDailySummary, 600000);

        // ==================== TIMEZONE WARNING ====================
        function checkTimezoneWarning() {
            fetch('/api/system/tz-status')
                .then(r => r.json())
                .then(d => {
                    if (!d.ok && d.warning) {
                        var banner = document.getElementById('tz-warning-banner');
                        if (!banner) {
                            banner = document.createElement('div');
                            banner.id = 'tz-warning-banner';
                            banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#0ea5e9;color:#1e293b;padding:8px 20px;font-size:13px;font-weight:600;text-align:center;display:flex;align-items:center;justify-content:center;gap:12px;';
                            banner.innerHTML = '<span>⚠ Timezone Warning: ' + d.warning + '</span><button onclick="this.parentElement.remove()" style="background:rgba(0,0,0,0.15);border:none;padding:3px 10px;border-radius:5px;cursor:pointer;color:#1e293b;">✕</button>';
                            document.body.prepend(banner);
                        }
                    }
                })
                .catch(function() {});
        }
        checkTimezoneWarning();

        // ==================== BEFORE vs AFTER RECORDING ====================
        var energyRecording = {
            before: {active: false, start: null, end: null},
            after: {active: false, start: null, end: null}
        };

        function loadRecordingState() {
            fetch('/api/energy/record')
                .then(r => r.json())
                .then(data => {
                    updateRecordingUI(data.recording);
                })
                .catch(e => console.error('Load recording state error:', e));
        }

        function toggleRecording(phase) {
            var rec = energyRecording[phase];
            var action = rec.active ? 'stop' : 'start';
            fetch('/api/energy/record', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({phase: phase, action: action})
            })
            .then(r => r.json())
            .then(data => {
                updateRecordingUI(data.recording);
                loadEnergyCompare('power');
                loadEnergyCompare('energy_kwh');
                showToast(action === 'start'
                    ? 'Recording ' + phase.toUpperCase() + ' started'
                    : 'Recording ' + phase.toUpperCase() + ' stopped', 'success');
            })
            .catch(e => console.error('Record error:', e));
        }

        function updateRecordingUI(rec) {
            if (!rec) return;
            energyRecording = rec;
            ['before', 'after'].forEach(function(phase) {
                var r = rec[phase];
                var statusEl = document.getElementById('status-' + phase);
                var btnEl = document.getElementById('btn-record-' + phase);
                if (!statusEl || !btnEl) return;

                var baseColor = phase === 'before' ? '#0ea5e9' : '#3b82f6';
                var darkColor = phase === 'before' ? '#0284c7' : '#2563eb';

                if (r.active) {
                    var start = new Date(r.start + 'Z');
                    var elapsed = Math.round((Date.now() - start.getTime()) / 3600000);
                    var days = Math.floor(elapsed / 24);
                    var hours = elapsed % 24;
                    statusEl.innerHTML = '<i class="fas fa-circle" style="color: #1e40af; font-size: 8px; animation: blink 1s infinite;"></i> <strong>Recording</strong> since ' + start.toLocaleDateString() + ' ' + start.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) + '<br>Duration: ' + days + 'd ' + hours + 'h';
                    btnEl.innerHTML = 'STOP RECORDING';
                    btnEl.style.background = 'linear-gradient(135deg, #1e40af, #1d4ed8)';
                } else if (r.start && r.end) {
                    var start = new Date(r.start + 'Z');
                    var end = new Date(r.end + 'Z');
                    var dur = Math.round((end - start) / 3600000);
                    var days = Math.floor(dur / 24);
                    var hours = dur % 24;
                    statusEl.innerHTML = '<strong>Completed</strong><br>' + start.toLocaleDateString() + ' &rarr; ' + end.toLocaleDateString() + '<br>Duration: ' + days + 'd ' + hours + 'h';
                    btnEl.innerHTML = 'RESET & RE-RECORD';
                    btnEl.style.background = 'linear-gradient(135deg, ' + baseColor + ', ' + darkColor + ')';
                } else {
                    statusEl.innerHTML = '&#9675; Not started';
                    btnEl.innerHTML = 'START RECORDING';
                    btnEl.style.background = 'linear-gradient(135deg, ' + baseColor + ', ' + darkColor + ')';
                }
            });
        }

        var compareRange = 'all';  // current compare range selection
        function loadEnergyCompare(field, range) {
            if (range) compareRange = range;
            var beforeChart, afterChart;
            if (field === 'energy_kwh') {
                beforeChart = charts.energyCompareKwhBefore;
                afterChart = charts.energyCompareKwhAfter;
            } else {
                beforeChart = charts.energyCompareBefore;
                afterChart = charts.energyCompareAfter;
            }
            if (!beforeChart || !afterChart) return;

            fetch('/api/energy/compare?field=' + field + '&range=' + compareRange)
                .then(r => r.json())
                .then(result => {
                    var beforeData = result.before || [];
                    var afterData = result.after || [];
                    var summary = result.summary || {};

                    // BEFORE chart — its own labels & data
                    beforeChart.data.labels = beforeData.map(function(d) { return d.label; });
                    beforeChart.data.datasets[0].data = beforeData.map(function(d) { return d.value; });
                    beforeChart.update('none');

                    // AFTER chart — its own labels & data
                    afterChart.data.labels = afterData.map(function(d) { return d.label; });
                    afterChart.data.datasets[0].data = afterData.map(function(d) { return d.value; });
                    afterChart.update('none');

                    // Sync Y-axis scale so both charts have same max for fair visual comparison
                    var allVals = beforeData.map(function(d){return d.value;}).concat(afterData.map(function(d){return d.value;}));
                    if (allVals.length > 0) {
                        var maxVal = Math.max.apply(null, allVals) * 1.1;
                        beforeChart.options.scales.y.max = maxVal;
                        afterChart.options.scales.y.max = maxVal;
                        beforeChart.update('none');
                        afterChart.update('none');
                    }

                    if (field === 'power') {
                        document.getElementById('compare-avg-before').textContent = summary.avg_before || '--';
                        document.getElementById('compare-avg-after').textContent = summary.avg_after || '--';
                        var savingsEl = document.getElementById('compare-savings');
                        savingsEl.textContent = summary.savings_percent || '--';
                        if (summary.savings_percent > 0) {
                            savingsEl.style.color = '#3b82f6';
                        } else if (summary.savings_percent < 0) {
                            savingsEl.style.color = '#1e40af';
                        }
                    }

                    if (result.recording) updateRecordingUI(result.recording);
                })
                .catch(e => console.error('Energy compare error:', e));
        }

        function setCompareRange(range, btn) {
            compareRange = range;
            // Toggle active class on range buttons
            document.querySelectorAll('#compare-range-7d,#compare-range-30d,#compare-range-all').forEach(function(b){ b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            loadEnergyCompare('power');
            loadEnergyCompare('energy_kwh');
        }

        // Listen for recording state changes from server
        socket.on('energy_recording', function(data) {
            updateRecordingUI(data.recording);
        });

        // ==================== LAMP BEFORE/AFTER RECORDING ====================
        var lampRecording = { before: {active:false, start:null, end:null}, after: {active:false, start:null, end:null} };
        var compareLampRange = 'all';

        function loadLampRecordingState() {
            fetch('/api/lamp/energy/record')
                .then(function(r){ return r.json(); })
                .then(function(data){ updateLampRecordingUI(data.recording); })
                .catch(function(e){ console.error('Load lamp recording state error:', e); });
        }

        function toggleLampRecording(phase) {
            var rec = lampRecording[phase];
            var action = rec.active ? 'stop' : 'start';
            fetch('/api/lamp/energy/record', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({phase: phase, action: action})
            })
            .then(function(r){ return r.json(); })
            .then(function(data){
                updateLampRecordingUI(data.recording);
                loadLampEnergyCompare('power');
                loadLampEnergyCompare('energy_kwh');
                showToast(action === 'start'
                    ? 'Lamp recording ' + phase.toUpperCase() + ' started'
                    : 'Lamp recording ' + phase.toUpperCase() + ' stopped', 'success');
            })
            .catch(function(e){ console.error('Lamp record error:', e); });
        }

        function updateLampRecordingUI(rec) {
            if (!rec) return;
            lampRecording = rec;
            ['before', 'after'].forEach(function(phase) {
                var r = rec[phase];
                var statusEl = document.getElementById('lamp-status-' + phase);
                var btnEl = document.getElementById('lamp-btn-record-' + phase);
                if (!statusEl || !btnEl) return;

                var baseColor = phase === 'before' ? '#0ea5e9' : '#3b82f6';
                var darkColor = phase === 'before' ? '#0284c7' : '#2563eb';

                if (r.active) {
                    var start = new Date(r.start + 'Z');
                    var elapsed = Math.round((Date.now() - start.getTime()) / 3600000);
                    statusEl.innerHTML = '<i class="fas fa-circle" style="color:#1e40af;font-size:8px;animation:blink 1s infinite;"></i> <strong>Recording</strong> since ' + start.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) + '<br>Duration: ' + Math.floor(elapsed/24) + 'd ' + (elapsed%24) + 'h';
                    btnEl.textContent = 'STOP RECORDING';
                    btnEl.style.background = 'linear-gradient(135deg,#1e40af,#1d4ed8)';
                } else if (r.start && r.end) {
                    var start = new Date(r.start + 'Z');
                    var end = new Date(r.end + 'Z');
                    var dur = Math.round((end - start) / 3600000);
                    statusEl.innerHTML = '<strong>Completed</strong><br>' + start.toLocaleDateString() + ' &rarr; ' + end.toLocaleDateString() + '<br>Duration: ' + Math.floor(dur/24) + 'd ' + (dur%24) + 'h';
                    btnEl.textContent = 'RESET & RE-RECORD';
                    btnEl.style.background = 'linear-gradient(135deg,' + baseColor + ',' + darkColor + ')';
                } else {
                    statusEl.innerHTML = '&#9675; Not started';
                    btnEl.textContent = 'START RECORDING';
                    btnEl.style.background = 'linear-gradient(135deg,' + baseColor + ',' + darkColor + ')';
                }
            });
        }

        function loadLampEnergyCompare(field, range) {
            if (range) compareLampRange = range;
            var beforeChart, afterChart;
            if (field === 'energy_kwh') {
                beforeChart = charts.lampCompareKwhBefore;
                afterChart = charts.lampCompareKwhAfter;
            } else {
                beforeChart = charts.lampCompareBefore;
                afterChart = charts.lampCompareAfter;
            }
            if (!beforeChart || !afterChart) return;

            fetch('/api/lamp/energy/compare?field=' + field + '&range=' + compareLampRange)
                .then(function(r){ return r.json(); })
                .then(function(result){
                    var beforeData = result.before || [];
                    var afterData = result.after || [];
                    var summary = result.summary || {};

                    beforeChart.data.labels = beforeData.map(function(d){ return d.label; });
                    beforeChart.data.datasets[0].data = beforeData.map(function(d){ return d.value; });
                    beforeChart.update('none');

                    afterChart.data.labels = afterData.map(function(d){ return d.label; });
                    afterChart.data.datasets[0].data = afterData.map(function(d){ return d.value; });
                    afterChart.update('none');

                    var allVals = beforeData.map(function(d){return d.value;}).concat(afterData.map(function(d){return d.value;}));
                    if (allVals.length > 0) {
                        var maxVal = Math.max.apply(null, allVals) * 1.1;
                        beforeChart.options.scales.y.max = maxVal;
                        afterChart.options.scales.y.max = maxVal;
                        beforeChart.update('none');
                        afterChart.update('none');
                    }

                    if (field === 'power') {
                        var el = document.getElementById('lamp-compare-avg-before'); if (el) el.textContent = summary.avg_before || '--';
                        el = document.getElementById('lamp-compare-avg-after'); if (el) el.textContent = summary.avg_after || '--';
                        el = document.getElementById('lamp-compare-savings');
                        if (el) {
                            el.textContent = summary.savings_percent || '--';
                            el.style.color = summary.savings_percent > 0 ? '#3b82f6' : (summary.savings_percent < 0 ? '#1e40af' : '');
                        }
                    }
                    if (result.recording) updateLampRecordingUI(result.recording);
                })
                .catch(function(e){ console.error('Lamp compare error:', e); });
        }

        function setLampCompareRange(range, btn) {
            compareLampRange = range;
            document.querySelectorAll('#lamp-compare-range-7d,#lamp-compare-range-30d,#lamp-compare-range-all').forEach(function(b){ b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            loadLampEnergyCompare('power');
            loadLampEnergyCompare('energy_kwh');
        }

        socket.on('lamp_recording', function(data) {
            updateLampRecordingUI(data.recording);
        });

        // ==================== THEME TOGGLE ====================
        function initTheme() {
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeUI(savedTheme);
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme');
            const newTheme = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeUI(newTheme);
        }

        function updateThemeUI(theme) {
            const icon = document.getElementById('theme-icon');
            const label = document.getElementById('theme-label');
            if (icon && label) {
                if (theme === 'dark') {
                    icon.className = 'fas fa-sun';
                    label.textContent = 'Light Mode';
                } else {
                    icon.className = 'fas fa-moon';
                    label.textContent = 'Dark Mode';
                }
            }
        }

        // Initialize theme on load
        initTheme();

        // ==================== NAVIGATION ====================
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebar-overlay');
            const icon = document.getElementById('hamburger-icon');
            sidebar.classList.toggle('open');
            overlay.classList.toggle('active');
            icon.className = sidebar.classList.contains('open') ? 'fas fa-times' : 'fas fa-bars';
        }

        function showPage(pageId) {
            console.log('[NAV] showPage called:', pageId);
            // Block user role from accessing admin-only pages
            if (userRole !== 'admin' && ADMIN_PAGES.indexOf(pageId) !== -1) {
                console.warn('[NAV] Access denied for role:', userRole, '→ page:', pageId);
                pageId = 'dashboard-ac';
            }
            document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            
            const pageEl = document.getElementById(pageId);
            if (pageEl) {
                pageEl.classList.add('active');
                console.log('[NAV] Page activated:', pageId);
            } else {
                console.error('[NAV] Page not found:', pageId);
            }
            
            // Mark the correct nav-item as active
            document.querySelectorAll('.nav-item').forEach(item => {
                if (item.getAttribute('onclick') && item.getAttribute('onclick').includes("'" + pageId + "'")) {
                    item.classList.add('active');
                }
            });
            
            localStorage.setItem('currentPage', pageId);

            // Close mobile sidebar
            if (window.innerWidth <= 768) {
                const sidebar = document.getElementById('sidebar');
                const overlay = document.getElementById('sidebar-overlay');
                sidebar.classList.remove('open');
                overlay.classList.remove('active');
                document.getElementById('hamburger-icon').className = 'fas fa-bars';
            }

            if (pageId === 'camera') {
                try { checkCameraStatus(); } catch(e) { console.error('[NAV] camera init error:', e); }
            }
            if (pageId === 'ml-optimization') {
                // Use requestAnimationFrame: fires AFTER browser paints the page with display:block
                requestAnimationFrame(function() {
                    // Only create charts if they don't exist yet; otherwise just resize
                    if (!charts.gaFitness || !charts.psoFitness || !charts.comparison) {
                        try { initMLCharts(); } catch(e) { console.error('[NAV] initMLCharts rAF error:', e); }
                    } else {
                        // Charts exist — just resize them for correct dimensions
                        try {
                            ['gaFitness', 'psoFitness', 'comparison'].forEach(function(k) {
                                if (charts[k]) { charts[k].resize(); charts[k].update('none'); }
                            });
                        } catch(e) { console.warn('[NAV] chart resize error:', e); }
                    }
                    try { refreshMLData(); } catch(e) { console.error('[NAV] refreshMLData error:', e); }
                });
            }
            if (pageId === 'ac-analytics') {
                ['temp', 'hum', 'acTemp', 'acHum', 'acPower'].forEach(function(cn) {
                    try { updateChartData(cn, chartRanges[cn] || 1); } catch(e) {}
                });
                try { loadAnalyticsEnergy('ac', _analyticsEnergyPeriod.ac || '24h', null); } catch(e) {}
            }
            if (pageId === 'lamp-analytics') {
                ['lampLux', 'lampBright', 'lampPower'].forEach(function(cn) {
                    try { updateChartData(cn, chartRanges[cn] || 1); } catch(e) {}
                });
                try { loadAnalyticsEnergy('lamp', _analyticsEnergyPeriod.lamp || '24h', null); } catch(e) {}
            }
            if (pageId === 'occupancy-feedback') {
                try { updateChartData('occupancy', chartRanges.occupancy || 1); } catch(e) {}
                try { loadFeedbackHistory(); } catch(e) {}
            }
            if (pageId === 'energy') {
                try { ensureChartsReady(); } catch(e) { console.error('[NAV] ensureChartsReady error:', e); }
                try { loadAllEnergyCharts(); } catch(e) { console.error('[NAV] energy init error:', e); }
                // Chart.js often needs a resize pass when canvas was initialized in a hidden page.
                setTimeout(function() {
                    try {

                        Object.keys(charts).forEach(function(k) {
                            if (charts[k] && typeof charts[k].resize === 'function') {
                                charts[k].resize();
                                charts[k].update('none');
                            }
                        });
                    } catch(e) {
                        console.error('[NAV] chart resize error:', e);
                    }
                }, 120);
            }
            if (pageId === 'control-outlet') {
                try { refreshOutletStatus(); } catch(e) { console.error('[NAV] outlet status error:', e); }
            }
            if (pageId === 'outlet-analysis') {
                requestAnimationFrame(function() {
                    try { loadAnalyticsEnergy('outlet', _analyticsEnergyPeriod.outlet || '24h', null); } catch(e) { console.error(e); }
                    try { initOutletCharts(); } catch(e) { console.error('[NAV] outlet chart init error:', e); }
                    try { loadOutletAnalytics('1h', document.getElementById('outlet-range-1h')); } catch(e) {}
                    try { loadOutletKwhChart('24h', document.getElementById('outlet-kwh-24h')); } catch(e) {}
                    try { refreshOutletStatus(); } catch(e) {}
                });
            }
        }

        // ==================== OUTLET CONTROL ====================
        var outletStates = {1:'OFF', 2:'OFF', 3:'OFF', 4:'OFF'};
        var outletCharts = {};

        function outletControl(num, state) {
            fetch('/api/outlet/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({outlet: num, state: state})
            })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'ok' || data.success) {
                    updateOutletUI(num, state);
                    showToast('Outlet ' + num + ' turned ' + state, state === 'ON' ? 'success' : 'info');
                } else {
                    showToast('Outlet ' + num + ' command failed: ' + (data.message || 'unknown error'), 'error');
                }
            })
            .catch(e => showToast('Outlet control error: ' + e.message, 'error'));
        }

        function outletAllOff() {
            [1,2,3,4].forEach(function(n) { outletControl(n, 'OFF'); });
        }

        function outletAllOn() {
            [1,2,3,4].forEach(function(n) { outletControl(n, 'ON'); });
        }

        function updateOutletUI(num, state) {
            outletStates[num] = state;
            var dot = document.getElementById('outlet'+num+'-dot');
            var badge = document.getElementById('outlet'+num+'-status-badge');
            var isOn = state === 'ON';
            if (dot) { dot.style.background = isOn ? '#2563eb' : '#1e40af'; }
            if (badge) {
                badge.textContent = state;
                badge.style.background = isOn ? 'rgba(37,99,235,0.18)' : 'rgba(30,64,175,0.15)';
                badge.style.color = isOn ? '#2563eb' : '#1e40af';
            }
            // Update active count
            var activeCount = Object.values(outletStates).filter(function(s){ return s==='ON'; }).length;
            var el = document.getElementById('outlet-active-count');
            if (el) el.textContent = activeCount;
        }

        function refreshOutletStatus() {
            fetch('/api/outlet/status')
            .then(r => r.json())
            .then(data => {
                if (!data || !data.outlets) return;
                data.outlets.forEach(function(o) {
                    updateOutletUI(o.id, o.state);
                    var pw = document.getElementById('outlet'+o.id+'-power');
                    if (pw) pw.textContent = (o.power !== undefined ? o.power : '--') + 'W';
                    var oaPw = document.getElementById('oa-o'+o.id+'-power');
                    if (oaPw) oaPw.textContent = (o.power !== undefined ? o.power : '--');
                    var oaKwh = document.getElementById('oa-o'+o.id+'-kwh');
                    if (oaKwh) oaKwh.textContent = (o.energy !== undefined ? o.energy.toFixed(3) : '--');
                });
                if (data.total_power !== undefined) {
                    var tp = document.getElementById('outlet-total-power');
                    if (tp) tp.textContent = data.total_power;
                }
                if (data.today_kwh !== undefined) {
                    var tk = document.getElementById('outlet-today-kwh');
                    if (tk) tk.textContent = data.today_kwh.toFixed(3);
                }
                if (data.peak_power !== undefined) {
                    var pk = document.getElementById('outlet-peak-power');
                    if (pk) pk.textContent = data.peak_power;
                }
                // Update progress bars
                var maxPower = Math.max.apply(null, (data.outlets || []).map(function(o){ return o.power || 0; })) || 1;
                (data.outlets || []).forEach(function(o) {
                    var fill = document.getElementById('oa-o'+o.id+'-bar-fill');
                    if (fill) fill.style.width = (Math.min(100, ((o.power||0)/maxPower)*100)).toFixed(1) + '%';
                });
            })
            .catch(function(){});
        }

        function loadOutletAnalytics(range, btn) {
            document.querySelectorAll('[id^="outlet-range-"]').forEach(function(b){ b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            fetch('/api/outlet/history?range=' + range)
            .then(r => r.json())
            .then(data => {
                if (!outletCharts.power) { initOutletCharts(); }
                var ch = outletCharts.power;
                if (!ch || !data.labels) return;
                ch.data.labels = data.labels;
                ['outlet1','outlet2','outlet3','outlet4'].forEach(function(k, i) {
                    if (ch.data.datasets[i] && data[k]) ch.data.datasets[i].data = data[k];
                });
                ch.update();
            })
            .catch(function(){});
        }

        function loadOutletKwhChart(range, btn) {
            document.querySelectorAll('[id^="outlet-kwh-"]').forEach(function(b){ b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            fetch('/api/outlet/energy?range=' + range)
            .then(r => r.json())
            .then(data => {
                if (!outletCharts.kwh) { initOutletCharts(); }
                var ch = outletCharts.kwh;
                if (!ch || !data.labels) return;
                ch.data.labels = data.labels;
                ['outlet1','outlet2','outlet3','outlet4'].forEach(function(k, i) {
                    if (ch.data.datasets[i] && data[k]) ch.data.datasets[i].data = data[k];
                });
                ch.update();
            })
            .catch(function(){});
        }

        function initOutletCharts() {
            var baseColors = ['#2563eb','#0ea5e9','#3b82f6','#1e40af'];
            var canvasPower = document.getElementById('outletPowerChart');
            var canvasKwh = document.getElementById('outletKwhChart');
            var makeDatasets = function(label_prefix, alpha) {
                return [1,2,3,4].map(function(i) {
                    return {
                        label: 'Outlet ' + i,
                        data: [],
                        borderColor: baseColors[i-1],
                        backgroundColor: baseColors[i-1].replace(')', ', ' + alpha + ')').replace('rgb', 'rgba').replace('#', 'rgba(') || baseColors[i-1],
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3,
                        pointRadius: 2
                    };
                });
            };
            var chartOpts = {
                responsive: true, maintainAspectRatio: true,
                plugins: { legend: { labels: { color: 'var(--text-primary)', font: { size: 11 } } } },
                scales: {
                    x: { ticks: { color: 'var(--text-secondary)', maxRotation: 0, font: { size: 10 } }, grid: { color: 'rgba(37,99,235,0.08)' } },
                    y: { ticks: { color: 'var(--text-secondary)', font: { size: 10 } }, grid: { color: 'rgba(37,99,235,0.08)' } }
                }
            };
            try {
                if (canvasPower) {
                    if (outletCharts.power) outletCharts.power.destroy();
                    outletCharts.power = new Chart(canvasPower.getContext('2d'), {
                        type: 'line', data: { labels: [], datasets: makeDatasets('Outlet', 0.15) }, options: JSON.parse(JSON.stringify(chartOpts))
                    });
                }
                if (canvasKwh) {
                    if (outletCharts.kwh) outletCharts.kwh.destroy();
                    outletCharts.kwh = new Chart(canvasKwh.getContext('2d'), {
                        type: 'bar', data: { labels: [], datasets: makeDatasets('Outlet', 0.7) }, options: JSON.parse(JSON.stringify(chartOpts))
                    });
                }
            } catch(e) { console.error('[Outlet] Chart init error:', e); }
        }

        function exportOutletCSV() {
            window.open('/api/outlet/export/csv', '_blank');
        }

        // ==================== ML OPTIMIZATION ====================
        var mlHistory = [];
        var mlRunCount = 0;

        function refreshMLData() {
            fetch('/api/ml/status')
                .then(r => r.json())
                .then(data => {
                    updateMLDisplay(data);
                    // Update current sensor conditions
                    fetch('/api/data').then(r => r.json()).then(d => {
                        const tempEl = document.getElementById('ml-cur-temp');
                        const humEl = document.getElementById('ml-cur-hum');
                        const luxEl = document.getElementById('ml-cur-lux');
                        const personEl = document.getElementById('ml-cur-person');
                        if (tempEl) {
                            const t = d && d.ac ? parseFloat(d.ac.temperature || 0) : NaN;
                            tempEl.textContent = Number.isFinite(t) ? t.toFixed(1) : '--';
                        }
                        if (humEl) {
                            const h = d && d.ac ? parseFloat(d.ac.humidity || 0) : NaN;
                            humEl.textContent = Number.isFinite(h) ? h.toFixed(1) : '--';
                        }
                        if (luxEl) {
                            const luxRaw = d && d.lamp ? (d.lamp.lux_avg != null ? d.lamp.lux_avg : (d.lamp.lux1 != null ? d.lamp.lux1 : null)) : null;
                            luxEl.textContent = (typeof luxRaw === 'number') ? luxRaw.toFixed(1) : '--';
                        }
                        if (personEl) personEl.textContent = (d && d.camera && d.camera.person_detected) ? 'Yes' : 'No';
                    }).catch(() => {});
                })
                .catch(err => console.error('ML refresh error:', err));
        }

        function updateMLDisplay(data) {
            const setEl = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };

            setEl('ml-ga-fitness', (data.ga_fitness || 0).toFixed(2));
            // PSO fitness is already in percent from server
            setEl('ml-pso-fitness', (data.pso_fitness != null ? parseFloat(data.pso_fitness) : 0).toFixed(1));
            setEl('ml-ga-temp', data.ga_temp || '--');
            setEl('ml-ga-fan', data.ga_fan || '--');
            setEl('ml-ga-mode', data.ga_mode || 'COOL');
            setEl('ml-ga-rh', data.ga_set_rh || '--');
            var p1 = data.pso_pwm1 != null && data.pso_pwm1 > 0 ? data.pso_pwm1 : '--';
            var p2 = data.pso_pwm2 != null && data.pso_pwm2 > 0 ? data.pso_pwm2 : '--';
            setEl('ml-pso-pwm1', p1);
            setEl('ml-pso-pwm2', p2);
            // Sinkron Lamp Dashboard brightness dari PSO result
            if (data.pso_brightness1 != null) setText('dash-bright1', Math.round(parseFloat(data.pso_brightness1)));
            if (data.pso_brightness2 != null) setText('dash-bright2', Math.round(parseFloat(data.pso_brightness2)));
            setEl('ml-opt-runs', data.optimization_runs || 0);

            // GA chart — use server data, or fallback to localStorage
            var gaHistory = (data.ga_history && data.ga_history.length > 0) ? data.ga_history : null;
            if (!gaHistory) {
                try { gaHistory = JSON.parse(localStorage.getItem('ml_ga_history')); } catch(e) {}
            }
            if (gaHistory && gaHistory.length > 0) {
                updateMLChart('gaFitness', gaHistory, 'GA');
            }

            // PSO chart — use server data, or fallback to localStorage
            if (data.pso_iteration_log && data.pso_iteration_log.length > 0) {
                updatePSOIterChart(data.pso_iteration_log);
            } else {
                var savedIterLog = null;
                try { savedIterLog = JSON.parse(localStorage.getItem('ml_pso_iterlog')); } catch(e) {}
                if (savedIterLog && savedIterLog.length > 0) {
                    updatePSOIterChart(savedIterLog);
                } else if (data.pso_history && data.pso_history.length > 0) {
                    // Fallback: if no iteration_log yet, show pso_history as fitness line
                    const chart = charts.psoFitness;
                    if (chart) {
                        chart.data.labels = data.pso_history.map((_, i) => i + 1);
                        chart.data.datasets[0].data = [];
                        chart.data.datasets[1].data = [];
                        chart.data.datasets[2].data = data.pso_history.map(f => Math.max(0, 350 - Math.sqrt(f)));
                        chart.update('none');
                    }
                }
            }
        }

        function updateMLChart(chartName, history, algo) {
            // Init chart if not yet created (e.g. ML tab never opened)
            if (!charts[chartName]) {
                var mlCanvas = document.getElementById('gaFitnessChart');
                if (mlCanvas && mlCanvas.offsetWidth > 0) {
                    try { initMLCharts(); } catch(e) { console.warn('[CHART] initMLCharts failed:', e); return; }
                } else {
                    // ML tab not yet active — store data to render when tab opens
                    window.__pendingMLData = window.__pendingMLData || {};
                    window.__pendingMLData[chartName] = { history, algo };
                    // Also save to localStorage so data survives refresh
                    if (algo === 'GA') {
                        try { localStorage.setItem('ml_ga_history', JSON.stringify(history)); } catch(e) {}
                    }
                    return;
                }
            }
            const chart = charts[chartName];
            if (!chart || !history || history.length === 0) return;

            if (algo === 'GA') {
                chart.data.labels           = history.map((_, i) => 'Gen ' + (i + 1));
                chart.data.datasets[0].data = history;
                chart.update('none');
                // Cache to localStorage for persistence across refresh
                try { localStorage.setItem('ml_ga_history', JSON.stringify(history)); } catch(e) {}
            } else if (algo === 'PSO') {
                // PSO fitness history sebagai fallback jika belum ada iteration_log
                // Tampilkan sebagai Lux Avg estimasi (dataset index 2)
                chart.data.labels               = history.map((_, i) => i + 1);
                chart.data.datasets[0].data     = [];
                chart.data.datasets[1].data     = [];
                chart.data.datasets[2].data     = history.map(f => Math.max(0, 350 - Math.sqrt(Math.max(0, f))));
                chart.update('none');
            }
        }

        function updatePSOIterChart(iterLog) {
            if (!iterLog || iterLog.length === 0) return;
            const chart = charts.psoFitness;
            if (!chart) return;
            chart.data.labels           = iterLog.map(d => d.iter);
            chart.data.datasets[0].data = iterLog.map(d => d.pwm1);
            chart.data.datasets[1].data = iterLog.map(d => d.pwm2);
            chart.data.datasets[2].data = iterLog.map(d => d.lux_avg);
            chart.update('none');
            // Cache to localStorage for persistence across refresh
            try { localStorage.setItem('ml_pso_iterlog', JSON.stringify(iterLog)); } catch(e) {}
            // Update tabel
            const tbody = document.getElementById('pso-iter-tbody');
            const wrap  = document.getElementById('pso-iter-table-wrap');
            if (!tbody) return;
            wrap.style.display = 'block';
            tbody.innerHTML = iterLog.map(d => {
                const inRange     = d.lux_avg >= 315 && d.lux_avg <= 385;
                const statusColor = inRange ? '#3b82f6' : '#0ea5e9';
                const statusText  = inRange ? '✓ Target' : '✗ Not yet';
                const fitPct      = Math.max(0, 100.0 - (d.fitness / 122500.0) * 100.0).toFixed(1);
                return `<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:4px 8px;text-align:center;font-weight:600;">${d.iter}</td>
                    <td style="padding:4px 8px;text-align:center;color:#2563eb;">${d.pwm1}</td>
                    <td style="padding:4px 8px;text-align:center;color:#2563eb;">${d.pwm2}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux1 != null ? d.lux1 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux2 != null ? d.lux2 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux3 != null ? d.lux3 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#0ea5e9;font-weight:600;">${d.lux_avg}</td>
                    <td style="padding:4px 8px;text-align:center;color:#3b82f6;font-weight:600;">${fitPct}%</td>
                    <td style="padding:4px 8px;text-align:center;color:${statusColor};font-weight:600;">${statusText}</td>
                </tr>`;
            }).join('');
        }

        socket.on('pso_iter_progress', function(d) {
            // Local helper — does not depend on outer scope
            const _s = function(id, val) { var e = document.getElementById(id); if (e) e.textContent = val; };

            if (d.status === 'done') {
                // Sinkron Lamp Dashboard brightness
                _s('dash-bright1', Math.round(parseFloat(d.b1 || 0)));
                _s('dash-bright2', Math.round(parseFloat(d.b2 || 0)));
                // Update stat card ML Optimization
                _s('ml-pso-pwm1',  d.pwm1 != null ? d.pwm1 : '--');
                _s('ml-pso-pwm2',  d.pwm2 != null ? d.pwm2 : '--');
                _s('ml-pso-lux1',  d.lux1 != null ? d.lux1 : '--');
                _s('ml-pso-lux2',  d.lux2 != null ? d.lux2 : '--');
                _s('ml-pso-lux3',  d.lux3 != null ? d.lux3 : '--');
                // Fitness %
                const fitPct = Math.max(0, 100.0 - ((d.fitness || 0) / 122500.0) * 100.0).toFixed(1);
                _s('ml-pso-fitness', fitPct);

                // Iteration table — replace waiting row with full data
                const tbody = document.getElementById('pso-iter-tbody');
                const wrap  = document.getElementById('pso-iter-table-wrap');
                if (!tbody) return;
                wrap.style.display = 'block';
                const inRange     = (d.lux_avg || 0) >= 315 && (d.lux_avg || 0) <= 385;
                const statusColor = inRange ? '#3b82f6' : '#0ea5e9';
                const statusText  = inRange ? '✓ Target' : '✗ Not yet';
                const newRow = `<tr data-iter="${d.iter}" style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:4px 8px;text-align:center;font-weight:600;">${d.iter}</td>
                    <td style="padding:4px 8px;text-align:center;color:#2563eb;">${d.pwm1 != null ? d.pwm1 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#2563eb;">${d.pwm2 != null ? d.pwm2 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux1 != null ? d.lux1 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux2 != null ? d.lux2 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux3 != null ? d.lux3 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#0ea5e9;font-weight:600;">${d.lux_avg != null ? d.lux_avg : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#3b82f6;font-weight:600;">${fitPct}%</td>
                    <td style="padding:4px 8px;text-align:center;color:${statusColor};font-weight:600;">${statusText}</td>
                </tr>`;
                const existingRow = tbody.querySelector('tr[data-iter="' + d.iter + '"]');
                if (existingRow) existingRow.outerHTML = newRow;
                else tbody.insertAdjacentHTML('beforeend', newRow);

                // Update chart live
                const chart = charts.psoFitness;
                if (chart) {
                    const idx = chart.data.labels.indexOf(d.iter);
                    if (idx === -1) {
                        chart.data.labels.push(d.iter);
                        chart.data.datasets[0].data.push(d.pwm1);
                        chart.data.datasets[1].data.push(d.pwm2);
                        chart.data.datasets[2].data.push(d.lux_avg);
                    } else {
                        chart.data.datasets[0].data[idx] = d.pwm1;
                        chart.data.datasets[1].data[idx] = d.pwm2;
                        chart.data.datasets[2].data[idx] = d.lux_avg;
                    }
                    chart.update('none');
                }

            } else if (d.status === 'waiting') {
                const tbody = document.getElementById('pso-iter-tbody');
                const wrap  = document.getElementById('pso-iter-table-wrap');
                if (!tbody) return;
                wrap.style.display = 'block';
                // Delete baris lama jika ada, insert baru
                const old = tbody.querySelector('tr[data-iter="' + d.iter + '"]');
                if (old) old.remove();
                tbody.insertAdjacentHTML('beforeend',
                    '<tr data-iter="' + d.iter + '" style="border-bottom:1px solid rgba(255,255,255,0.05);opacity:0.6;">' +
                        '<td style="padding:4px 8px;text-align:center;font-weight:600;">' + d.iter + '</td>' +
                        '<td style="padding:4px 8px;text-align:center;color:#2563eb;">' + (d.pwm1 != null ? d.pwm1 : '--') + '</td>' +
                        '<td style="padding:4px 8px;text-align:center;color:#2563eb;">' + (d.pwm2 != null ? d.pwm2 : '--') + '</td>' +
                        '<td colspan="6" style="padding:4px 8px;text-align:center;color:var(--text-secondary);">⏳ Reading sensors...</td>' +
                    '</tr>');

            } else if (d.status === 'new_cycle') {
                const tbody = document.getElementById('pso-iter-tbody');
                if (tbody) tbody.innerHTML = '';
                const chart = charts.psoFitness;
                if (chart) {
                    chart.data.labels = [];
                    chart.data.datasets.forEach(function(ds) { ds.data = []; });
                    chart.update('none');
                }
            }
        });

        function addToComparisonChart(gaFit, psoFit) {
            const chart = charts.comparison;
            if (!chart) return;

            const now = new Date();
            const label = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0') + ':' + String(now.getSeconds()).padStart(2, '0');

            if (chart.data.labels.length >= 50) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }

            chart.data.labels.push(label);
            chart.data.datasets[0].data.push(gaFit);
            chart.data.datasets[1].data.push(psoFit);
            chart.update('none');
        }

        function addMLHistoryRow(data) {
            mlRunCount++;
            const entry = {
                run: mlRunCount,
                time: new Date().toLocaleTimeString(),
                ga_fitness: data.ga_fitness || 0,
                ga_temp: data.ga_temp || '--',
                ga_fan: data.ga_fan || '--',
                pso_fitness: data.pso_fitness || 0,
                pso_brightness: data.pso_brightness || '--',
                // GA maximize (score) vs PSO minimize (error) — normalize PSO before combining
                combined: (data.ga_fitness || 0) * 0.5 + Math.max(0, 100 - (data.pso_fitness || 0) / 100) * 0.5
            };
            mlHistory.unshift(entry);
            if (mlHistory.length > 50) mlHistory.pop();

            renderMLHistory();
        }

        function renderMLHistory() {
            const tbody = document.getElementById('ml-history-body');
            if (!tbody) return;

            if (mlHistory.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #94a3b8;">No optimization data yet. Run GA or PSO to start.</td></tr>';
                return;
            }

            tbody.innerHTML = mlHistory.map(e => {
                const getBadgeGA  = (f) => f >= 80 ? 'good' : f >= 50 ? 'mid' : 'low';
                const getBadgePSO = (err) => err <= 100 ? 'good' : err <= 500 ? 'mid' : 'low'; // PSO: lower error = better
                return '<tr>' +
                    '<td>' + e.run + '</td>' +
                    '<td>' + e.time + '</td>' +
                    '<td><span class="ml-badge ' + getBadgeGA(e.ga_fitness) + '">' + e.ga_fitness.toFixed(2) + '</span></td>' +
                    '<td>' + e.ga_temp + '\u00b0C</td>' +
                    '<td>' + e.ga_fan + '</td>' +
                    '<td><span class="ml-badge ' + getBadgePSO(e.pso_fitness) + '">' + e.pso_fitness.toFixed(2) + '</span></td>' +
                    '<td>' + e.pso_brightness + '%</td>' +
                    '<td><span class="ml-badge ' + getBadgeGA(e.combined) + '">' + e.combined.toFixed(2) + '</span></td>' +
                '</tr>';
            }).join('');
        }

        function refreshMLHistory() {
            refreshMLData();
            showToast('ML data refreshed', 'success');
        }

        // ==================== EXPORT FUNCTIONS ====================
        function downloadCSV(filename, csvContent) {
            csvContent = "sep=,\\n" + csvContent;
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
            URL.revokeObjectURL(link.href);
        }

        function exportMLHistory() {
            if (!mlHistory || mlHistory.length === 0) {
                showToast('No optimization data to export', 'error');
                return;
            }
            let csv = 'Run,Time,GA Fitness,AC Temp (C),Fan Speed,PSO Fitness,Brightness (%),Combined\\n';
            mlHistory.forEach(e => {
                csv += e.run + ',"' + e.time + '",' + e.ga_fitness.toFixed(2) + ',' + e.ga_temp + ',' + e.ga_fan + ',' + e.pso_fitness.toFixed(2) + ',' + e.pso_brightness + ',' + e.combined.toFixed(2) + '\\n';
            });
            downloadCSV('ml_optimization_history.csv', csv);
            showToast('ML history exported', 'success');
        }

        function exportLogs() {
            const container = document.getElementById('log-container');
            if (!container || !container.children.length) {
                showToast('No logs to export', 'error');
                return;
            }
            let csv = 'Time,Level,Message\\n';
            Array.from(container.children).forEach(entry => {
                const text = entry.textContent || '';
                const timeMatch = text.match(/\\[(.+?)\\]/);
                const time = timeMatch ? timeMatch[1] : '';
                const msg = text.replace(/\\[.+?\\]\\s*/, '').replace(/"/g, '""');
                const level = entry.className.replace('log-entry ', '').trim();
                csv += '"' + time + '","' + level + '","' + msg + '"\\n';
            });
            downloadCSV('system_logs.csv', csv);
            showToast('Logs exported', 'success');
        }

        function exportGAReport() {
            fetch('/api/ga/export-csv')
                .then(function(r) {
                    if (!r.ok) {
                        return r.json().then(function(d) { throw new Error(d.error || 'Export failed'); });
                    }
                    return r.blob();
                })
                .then(function(blob) {
                    var ts = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '');
                    var link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = 'ga_report_' + ts + '.csv';
                    link.click();
                    URL.revokeObjectURL(link.href);
                    showToast('GA report exported successfully!', 'success');
                })
                .catch(function(e) { showToast('GA export failed: ' + e.message, 'error'); });
        }

        function exportPSOReport() {
            fetch('/api/pso/export-csv')
                .then(function(r) {
                    if (!r.ok) {
                        return r.json().then(function(d) { throw new Error(d.error || 'Export failed'); });
                    }
                    return r.blob();
                })
                .then(function(blob) {
                    var ts = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '');
                    var link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = 'pso_report_' + ts + '.csv';
                    link.click();
                    URL.revokeObjectURL(link.href);
                    showToast('PSO report exported successfully!', 'success');
                })
                .catch(function(e) { showToast('PSO export failed: ' + e.message, 'error'); });
        }

        function exportFeedback() {
            fetch('/api/occupancy/feedback/list')
                .then(r => r.json())
                .then(data => {
                    const rows = data.feedback || [];
                    if (rows.length === 0) {
                        showToast('No feedback data to export', 'error');
                        return;
                    }
                    let csv = 'Time,Rating,Occupancy Count,Comment\\n';
                    rows.forEach(item => {
                        const comment = (item.comment || '').replace(/"/g, '""');
                        csv += '"' + item.time + '",' + item.rating + ',' + item.occupancy_count + ',"' + comment + '"\\n';
                    });
                    downloadCSV('occupancy_feedback.csv', csv);
                    showToast('Feedback exported', 'success');
                })
                .catch(() => showToast('Failed to fetch feedback data', 'error'));
        }

        function exportEnergyHistory() {
            const period = '24h';
            fetch('/api/energy/history?period=' + period)
                .then(r => r.json())
                .then(data => {
                    if (!data.power || data.power.length === 0) {
                        showToast('No energy data to export', 'error');
                        return;
                    }
                    let csv = 'Time,Power (W),Voltage (V),Energy (kWh)\\n';
                    const times = data.power.map(p => p.time);
                    const powers = data.power || [];
                    const voltages = data.voltage || [];
                    const energies = data.energy_kwh || [];
                    times.forEach((t, i) => {
                        const pw = powers[i] ? powers[i].value : '';
                        const vl = voltages[i] ? voltages[i].value : '';
                        const en = energies[i] ? energies[i].value : '';
                        csv += '"' + t + '",' + pw + ',' + vl + ',' + en + '\\n';
                    });
                    downloadCSV('energy_history_' + period + '.csv', csv);
                    showToast('Energy data exported (' + period + ')', 'success');
                })
                .catch(() => showToast('Failed to fetch energy history', 'error'));
        }

        // Generic chart data export (single dataset charts)
        function exportChartData(chartName, valueLabel) {
            const chart = charts[chartName];
            if (!chart || !chart.data.labels || chart.data.labels.length === 0) {
                showToast('No data to export for ' + chartName, 'error');
                return;
            }
            let csv = 'Time,' + valueLabel + '\\n';
            chart.data.labels.forEach((label, i) => {
                const val = chart.data.datasets[0].data[i];
                csv += '"' + label + '",' + (val !== null && val !== undefined ? val : '') + '\\n';
            });
            downloadCSV(chartName + '_data.csv', csv);
            showToast(chartName + ' data exported', 'success');
        }

        // Export for multi-dataset comparison charts (Before/After, GA vs PSO)
        function exportCompareChart(chartName, valueLabel) {
            const chart = charts[chartName];
            if (!chart || !chart.data.labels || chart.data.labels.length === 0) {
                showToast('No data to export for ' + chartName, 'error');
                return;
            }
            const dsNames = chart.data.datasets.map(ds => ds.label || valueLabel);
            let csv = 'Time,' + dsNames.join(',') + '\\n';
            chart.data.labels.forEach((label, i) => {
                let row = '"' + label + '"';
                chart.data.datasets.forEach(ds => {
                    const val = ds.data[i];
                    row += ',' + (val !== null && val !== undefined ? val : '');
                });
                csv += row + '\\n';
            });
            downloadCSV(chartName + '_compare.csv', csv);
            showToast(chartName + ' comparison exported', 'success');
        }

        function getMLParams(algo) {
            const getInput = (id, fallback) => {
                const el = document.getElementById(id);
                return el ? el.value : fallback;
            };
            if (algo === 'ga') {
                return {
                    population_size: parseInt(getInput('ml-ga-pop', 15), 10) || 15,
                    generations: parseInt(getInput('ml-ga-gen', 30), 10) || 30,
                    mutation_rate: parseFloat(getInput('ml-ga-mut', 0.25)) || 0.25,
                    crossover_rate: parseFloat(getInput('ml-ga-cross', 0.8)) || 0.8,
                    elitism_ratio: 0.2
                };
            } else {
                return {
                    swarm_size: parseInt(getInput('ml-pso-swarm', 30), 10) || 30,
                    iterations: parseInt(getInput('ml-pso-iter', 100), 10) || 100,
                    w: parseFloat(getInput('ml-pso-w', 0.7)) || 0.7,
                    c: parseFloat(getInput('ml-pso-c', 1.5)) || 1.5
                };
            }
        }

        function runGAOptimization() {
            showToast('Starting GA \u2192 AC optimization...', 'success');
            fetch('/api/ml/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ algorithm: 'ga', params: getMLParams('ga') })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success') {
                    showToast('GA optimization triggered!', 'success');
                } else {
                    showToast('GA error: ' + d.message, 'error');
                }
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function runPSOOptimization() {
            showToast('Starting PSO \u2192 Lamp optimization...', 'success');
            fetch('/api/ml/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ algorithm: 'pso', params: getMLParams('pso') })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success') {
                    showToast('PSO optimization triggered!', 'success');
                } else {
                    showToast('PSO error: ' + d.message, 'error');
                }
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function runBothOptimization() {
            showToast('Starting GA + PSO optimization...', 'success');
            fetch('/api/ml/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    algorithm: 'both',
                    params: { ga: getMLParams('ga'), pso: getMLParams('pso') }
                })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success') {
                    showToast('GA + PSO optimization triggered!', 'success');
                } else {
                    showToast('Error: ' + d.message, 'error');
                }
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function clearMLCharts() {
            ['gaFitness', 'psoFitness', 'comparison'].forEach(name => {
                if (charts[name]) {
                    charts[name].data.labels = [];
                    charts[name].data.datasets.forEach(ds => ds.data = []);
                    charts[name].update('none');
                }
            });
            showToast('Charts cleared', 'success');
        }

        // ==================== CAMERA ====================
        var cameraEnabled = true;

        function toggleCamera() {
            fetch('/api/camera/toggle', { method: 'POST' })
                .then(r => r.json())
                .then(result => {
                    cameraEnabled = result.enabled;
                    updateCameraToggleUI();
                    const img = document.getElementById('camera-img');
                    const error = document.getElementById('camera-error');
                    if (cameraEnabled) {
                        img.src = '/video_feed?' + new Date().getTime();
                        img.style.display = 'block';
                        error.style.display = 'none';
                        showToast('Camera ON', 'success');
                    } else {
                        img.style.display = 'none';
                        error.style.display = 'flex';
                        error.querySelector('h3').textContent = 'Camera OFF';
                        error.querySelector('p').textContent = 'Camera is off. Click Camera ON button to activate.';
                        showToast('Camera OFF', 'info');
                    }
                    checkCameraStatus();
                })
                .catch(e => showToast('Error: ' + e, 'error'));
        }

        function updateCameraToggleUI() {
            const btn = document.getElementById('camera-toggle-btn');
            const text = document.getElementById('camera-toggle-text');
            if (!btn) return;
            if (cameraEnabled) {
                btn.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
                text.textContent = 'Camera ON';
            } else {
                btn.style.background = 'linear-gradient(135deg, #1e40af, #1d4ed8)';
                text.textContent = 'Camera OFF';
            }
        }

        function retryCamera() {
            const img = document.getElementById('camera-img');
            const error = document.getElementById('camera-error');
            
            fetch('/api/camera/restart', { method: 'POST' })
                .then(r => r.json())
                .then(result => {
                    if (result.status === 'success') {
                        img.src = '/video_feed?' + new Date().getTime();
                        img.style.display = 'block';
                        error.style.display = 'none';
                        showToast('Camera restarted successfully!');
                        checkCameraStatus();
                    } else {
                        showToast('Camera restart failed', 'error');
                    }
                })
                .catch(e => {
                    showToast('Camera restart error', 'error');
                });
        }

        function checkCameraStatus() {
            fetch('/api/camera/status')
                .then(r => r.json())
                .then(data => {
                    const statusEl = document.getElementById('cam-status');
                    if (data.status === 'active') {
                        statusEl.textContent = 'Active';
                        statusEl.style.color = '#3b82f6';
                        document.getElementById('cam-resolution').textContent = data.width + ' x ' + data.height;
                        document.getElementById('cam-fps').textContent = data.fps + ' FPS';
                    } else {
                        statusEl.textContent = 'Inactive';
                        statusEl.style.color = '#1e40af';
                    }
                })
                .catch(e => {
                    document.getElementById('cam-status').textContent = 'Error';
                    document.getElementById('cam-status').style.color = '#1e40af';
                });
        }

        function updateCameraTime() {
            const el = document.getElementById('camera-time');
            if (el) el.textContent = new Date().toLocaleTimeString();
        }
        setInterval(updateCameraTime, 1000);

        // ==================== AC CONTROLS ====================
        function updateACTemp(value) {
            document.getElementById('ac-temp-display').textContent = value;
            saveSettings();
        }

        function updateACRH(value) {
            document.getElementById('ac-rh-display').textContent = value;
            saveSettings();
        }

        function updateFanSpeed(value) {
            document.getElementById('fan-speed-display').textContent = value;
            saveSettings();
        }

        function applyACSnapshot(ac) {
            if (!ac) return;
            if (ac.ac_temp !== undefined && ac.ac_temp !== null) {
                var tempValue = parseInt(ac.ac_temp, 10);
                var tempSlider = document.getElementById('ac-temp-slider');
                var tempDisplay = document.getElementById('ac-temp-display');
                if (tempSlider && Number.isFinite(tempValue)) tempSlider.value = tempValue;
                if (tempDisplay && Number.isFinite(tempValue)) tempDisplay.textContent = tempValue;
            }
            if (ac.fan_speed !== undefined && ac.fan_speed !== null) {
                var fanValue = parseInt(ac.fan_speed, 10);
                var fanSlider = document.getElementById('fan-speed-slider');
                var fanDisplay = document.getElementById('fan-speed-display');
                if (fanSlider && Number.isFinite(fanValue)) fanSlider.value = fanValue;
                if (fanDisplay && Number.isFinite(fanValue)) fanDisplay.textContent = fanValue;
            }
            if (ac.set_rh !== undefined && ac.set_rh !== null) {
                var rhValue = parseInt(ac.set_rh, 10);
                var rhSlider = document.getElementById('ac-rh-slider');
                var rhDisplay = document.getElementById('ac-rh-display');
                if (rhSlider && Number.isFinite(rhValue)) rhSlider.value = rhValue;
                if (rhDisplay && Number.isFinite(rhValue)) rhDisplay.textContent = rhValue;
            }
            if (ac.ac_fan_mode) {
                selectedACMode = String(ac.ac_fan_mode).toUpperCase();
                document.querySelectorAll('.ac-set-mode-btn').forEach(function(btn) {
                    var active = btn.getAttribute('data-mode') === selectedACMode;
                    btn.style.background = active ? 'var(--primary)' : 'var(--card-bg)';
                    btn.style.border = active ? '2px solid var(--primary)' : '2px solid var(--border)';
                    btn.style.color = active ? 'white' : 'var(--text)';
                });
            }
            if (typeof updateDashboard === 'function') {
                try { updateDashboard(); } catch (e) {}
            }
            updateExtraFeatureButtons(ac);
        }

        function updateExtraFeatureButtons(ac) {
            if (!ac) return;
            var features = [
                {id: 'btn-swing', statusId: 'swing-status', key: 'swing', color: '#0ea5e9'},
                {id: 'btn-turbo', statusId: 'turbo-status', key: 'turbo', color: '#1e40af'},
                {id: 'btn-econo', statusId: 'econo-status', key: 'econo', color: '#3b82f6'}
            ];
            features.forEach(function(f) {
                var btn = document.getElementById(f.id);
                var status = document.getElementById(f.statusId);
                var isOn = !!ac[f.key];
                if (btn) {
                    btn.style.background = isOn ? f.color : 'var(--bg-card)';
                    btn.style.color = isOn ? 'white' : 'var(--text-secondary)';
                    btn.style.borderColor = isOn ? f.color : 'var(--border)';
                }
                if (status) status.textContent = isOn ? 'ON' : 'OFF';
            });
        }

        function sendACCommand(command) {
            // Block manual commands when in ADAPTIVE mode
            fetch('/api/data').then(r => r.json()).then(data => {
                if (data.ac.mode === 'ADAPTIVE') {
                    showToast('Adaptive mode is active! Switch to Manual for manual control.', 'error');
                    return;
                }
                fetch('/api/ac/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: command })
                })
                .then(r => r.json())
                .then(result => {
                    if (result && result.ac) applyACSnapshot(result.ac);
                    const label = command.replace('_', ' ');
                    showToast('AC: ' + label, result && result.status === 'success' ? 'success' : 'info');
                })
                .catch(e => showToast('Error: ' + (e.message || e), 'error'));
            });
        }

        var selectedACMode = 'COOL';

        function selectACSetMode(mode, btn) {
            selectedACMode = mode;
            document.querySelectorAll('.ac-set-mode-btn').forEach(b => {
                b.style.background = 'var(--card-bg)';
                b.style.border = '2px solid var(--border)';
                b.style.color = 'var(--text)';
            });
            btn.style.background = 'var(--primary)';
            btn.style.border = '2px solid var(--primary)';
            btn.style.color = 'white';
        }

        function applyACSettings() {
            // Block manual commands when in ADAPTIVE mode
            fetch('/api/data').then(r => r.json()).then(data => {
                if (data.ac.mode === 'ADAPTIVE') {
                    showToast('Adaptive mode is active! Switch to Manual for manual control.', 'error');
                    return;
                }
                const temp = document.getElementById('ac-temp-slider').value;
                const rh = document.getElementById('ac-rh-slider').value;
                const fan = document.getElementById('fan-speed-slider').value;

                fetch('/api/ac/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        command: 'SET',
                        temperature: parseInt(temp),
                        set_rh: parseInt(rh),
                        fan_speed: parseInt(fan),
                        mode: selectedACMode
                    })
                })
                .then(r => r.json())
                .then(result => {
                    if (result && result.ac) applyACSnapshot(result.ac);
                    showToast('AC: ' + temp + '°C, RH ' + rh + '%, Fan ' + fan + ', ' + selectedACMode);
                })
                .catch(e => showToast('Error: ' + e, 'error'));
            });
        }

        // ==================== LAMP CONTROLS ====================
        function updateBrightness(lampNum, value) {
            document.getElementById('brightness-display-' + lampNum).textContent = value;
            saveSettings();
        }

        function syncAllSliders() {
            const val = document.getElementById('brightness-slider-1').value;
            document.getElementById('brightness-slider-2').value = val;
            document.getElementById('brightness-display-2').textContent = val;
            saveSettings();
        }

        function applyLampSettings() {
            const b1 = parseInt(document.getElementById('brightness-slider-1').value);
            const b2 = parseInt(document.getElementById('brightness-slider-2').value);
            
            // Switch to MANUAL mode FIRST, then send brightness only after mode confirmed
            fetch('/api/lamp/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'MANUAL' })
            })
            .then(() => {
                applyLampModeUI('MANUAL');
                return fetch('/api/lamp/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ brightness1: b1, brightness2: b2, source: 'dashboard' })
                });
            })
            .then(r => r.json())
            .then(result => showToast('Lamp → MANUAL | L1=' + b1 + '% L2=' + b2 + '%'))
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        // ==================== MODE TOGGLES ====================
        function setACMode(mode) {
            fetch('/api/ac/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            })
            .then(r => r.json())
            .then(result => {
                applyACModeUI(mode);
                showToast('AC Mode: ' + mode, 'success');
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function applyACModeUI(mode) {
            const overlay = document.getElementById('ac-manual-overlay');
            const banner = document.getElementById('adaptive-info-banner');
            const indicator = document.getElementById('ac-mode-indicator');
            const btnAdaptive = document.getElementById('btn-mode-adaptive');
            const btnManual = document.getElementById('btn-mode-manual');

            if (mode === 'ADAPTIVE') {
                // Show overlay on manual controls
                if (overlay) overlay.style.display = 'flex';
                if (banner) banner.style.display = 'block';
                // Indicator
                if (indicator) {
                    indicator.style.background = 'rgba(59, 130, 246, 0.1)';
                    indicator.style.color = '#3b82f6';
                    indicator.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                    indicator.innerHTML = 'Current mode: <strong>ADAPTIVE</strong> — AC controlled automatically by GA optimization';
                }
                // Button styles
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
                    btnAdaptive.style.borderColor = '#3b82f6';
                    btnAdaptive.style.color = 'white';
                    btnAdaptive.style.opacity = '1';
                }
                if (btnManual) {
                    btnManual.style.background = 'var(--bg-card)';
                    btnManual.style.borderColor = 'var(--border)';
                    btnManual.style.color = 'var(--text-secondary)';
                    btnManual.style.opacity = '0.6';
                }
            } else {
                // Hide overlay — allow manual controls
                if (overlay) overlay.style.display = 'none';
                if (banner) banner.style.display = 'none';
                // Indicator
                if (indicator) {
                    indicator.style.background = 'rgba(14, 165, 233, 0.1)';
                    indicator.style.color = '#0ea5e9';
                    indicator.style.borderColor = 'rgba(14, 165, 233, 0.3)';
                    indicator.innerHTML = 'Current mode: <strong>MANUAL</strong> — Control AC manually using buttons below';
                }
                // Button styles
                if (btnManual) {
                    btnManual.style.background = 'linear-gradient(135deg, #0ea5e9, #0284c7)';
                    btnManual.style.borderColor = '#0ea5e9';
                    btnManual.style.color = 'white';
                    btnManual.style.opacity = '1';
                }
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'var(--bg-card)';
                    btnAdaptive.style.borderColor = 'var(--border)';
                    btnAdaptive.style.color = 'var(--text-secondary)';
                    btnAdaptive.style.opacity = '0.6';
                }
            }
            // Update adaptive info cards
            updateAdaptiveInfo();
        }

        function updateAdaptiveInfo() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    const gaTemp = document.getElementById('adaptive-ga-temp');
                    const gaFan = document.getElementById('adaptive-ga-fan');
                    const gaFitness = document.getElementById('adaptive-ga-fitness');
                    if (gaTemp) gaTemp.textContent = (data.system.ga_temp || 0) > 0 ? data.system.ga_temp + '°C' : '--';
                    if (gaFan) gaFan.textContent = (data.system.ga_fan || 0) > 0 ? 'Lv ' + data.system.ga_fan : '--';
                    if (gaFitness) gaFitness.textContent = (data.system.ga_fitness || 0) > 0 ? parseFloat(data.system.ga_fitness).toFixed(2) : '--';
                });
        }

        function updateModeBadges() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    applyACModeUI(data.ac.mode);
                    applyLampModeUI(data.lamp.mode);
                });
        }

        function setLampMode(mode) {
            fetch('/api/lamp/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            })
            .then(r => r.json())
            .then(result => {
                applyLampModeUI(mode);
                showToast('Lamp Mode: ' + mode, 'success');
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function applyLampModeUI(mode) {
            const overlay = document.getElementById('lamp-manual-overlay');
            const indicator = document.getElementById('lamp-mode-indicator');
            const btnAdaptive = document.getElementById('btn-lamp-adaptive');
            const btnManual = document.getElementById('btn-lamp-manual');
            if (mode === 'ADAPTIVE') {
                if (overlay) overlay.style.display = 'flex';
                if (indicator) {
                    indicator.style.background = 'rgba(14, 165, 233, 0.1)';
                    indicator.style.color = '#0ea5e9';
                    indicator.style.borderColor = 'rgba(14, 165, 233, 0.3)';
                    indicator.innerHTML = 'Current mode: <strong>ADAPTIVE</strong> — Lamps controlled automatically by PSO optimization';
                }
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'linear-gradient(135deg, #0ea5e9, #0284c7)';
                    btnAdaptive.style.borderColor = '#0ea5e9';
                    btnAdaptive.style.color = 'white';
                    btnAdaptive.style.opacity = '1';
                }
                if (btnManual) {
                    btnManual.style.background = 'var(--bg-card)';
                    btnManual.style.borderColor = 'var(--border)';
                    btnManual.style.color = 'var(--text-secondary)';
                    btnManual.style.opacity = '0.6';
                }
            } else {
                if (overlay) overlay.style.display = 'none';
                if (indicator) {
                    indicator.style.background = 'rgba(37, 99, 235, 0.1)';
                    indicator.style.color = '#2563eb';
                    indicator.style.borderColor = 'rgba(37, 99, 235, 0.3)';
                    indicator.innerHTML = 'Current mode: <strong>MANUAL</strong> — Control lamps manually using sliders below';
                }
                if (btnManual) {
                    btnManual.style.background = 'linear-gradient(135deg, #2563eb, #1d4ed8)';
                    btnManual.style.borderColor = '#2563eb';
                    btnManual.style.color = 'white';
                    btnManual.style.opacity = '1';
                }
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'var(--bg-card)';
                    btnAdaptive.style.borderColor = 'var(--border)';
                    btnAdaptive.style.color = 'var(--text-secondary)';
                    btnAdaptive.style.opacity = '0.6';
                }
            }
        }

        // ==================== IR REMOTE ====================
        var currentLearningButton = null;
        var learningCheckInterval = null;
        
        function learnIRCode(buttonName, deviceName = 'AC') {
            // Clear any previous learning interval
            if (learningCheckInterval) {
                clearInterval(learningCheckInterval);
                learningCheckInterval = null;
            }
            
            const buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
            const statusElement = document.getElementById('status-' + buttonName);
            
            currentLearningButton = buttonName;
            
            if (buttonElement) {
                buttonElement.classList.add('learning');
            }
            
            if (statusElement) {
                statusElement.textContent = 'Ready! Press remote...';
                statusElement.style.color = '#0ea5e9';
            }
            
            fetch('/api/ir/learn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    button: buttonName,
                    device: deviceName 
                })
            })
            .then(r => r.json())
            .then(result => {
                console.log('Learning mode activated:', result);
                showToast('Press remote button for: ' + buttonName, 'info');
                
                // Auto-check status every 500ms
                let checkCount = 0;
                learningCheckInterval = setInterval(() => {
                    checkCount++;
                    
                    fetch('/api/ir/codes')
                        .then(r => r.json())
                        .then(data => {
                            if (data.codes && data.codes[buttonName]) {
                                // Code learned!
                                clearInterval(learningCheckInterval);
                                
                                if (buttonElement) {
                                    buttonElement.classList.remove('learning');
                                    buttonElement.classList.add('learned');
                                }
                                
                                if (statusElement) {
                                    statusElement.textContent = 'Learned OK';
                                    statusElement.style.color = '#3b82f6';
                                }
                                
                                showToast('IR Code learned: ' + buttonName, 'success');
                                currentLearningButton = null;
                            }
                            
                            // Stop after 120 checks (60 seconds)
                            if (checkCount > 120) {
                                clearInterval(learningCheckInterval);
                                if (buttonElement) buttonElement.classList.remove('learning');
                                if (statusElement) {
                                    statusElement.textContent = 'Timeout';
                                    statusElement.style.color = '#1e40af';
                                }
                                showToast('Learning timeout - no signal', 'error');
                                currentLearningButton = null;
                            }
                        })
                        .catch(e => console.error('Status check error:', e));
                }, 500);
            })
            .catch(e => { 
                if (buttonElement) buttonElement.classList.remove('learning'); 
                if (statusElement) {
                    statusElement.textContent = 'Error';
                    statusElement.style.color = '#1e40af';
                }
                showToast('Error: ' + e, 'error'); 
            });
        }

        function sendIRCode(buttonName) {
            // For IR Learning panel Send buttons - still uses learned codes
            fetch('/api/ir/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ button: buttonName })
            })
            .then(r => r.json())
            .then(result => {
                if (result.status === 'success') {
                    showToast('IR Code sent: ' + buttonName, 'success');
                } else {
                    showToast(result.message || 'Failed to send', 'error');
                }
            })
            .catch(e => showToast('Error: ' + (e.message || e), 'error'));
        }

        function sendACMode(modeName, btnElement) {
            // Block manual commands when in ADAPTIVE mode
            fetch('/api/data').then(r => r.json()).then(data => {
                if (data.ac.mode === 'ADAPTIVE') {
                    showToast('Adaptive mode is active! Switch to Manual for manual control.', 'error');
                    return;
                }
                // Visual feedback
                document.querySelectorAll('.ac-mode-btn').forEach(btn => {
                    btn.style.opacity = '0.6';
                    btn.style.transform = 'scale(0.95)';
                });
                if (btnElement) {
                    btnElement.style.opacity = '1';
                    btnElement.style.transform = 'scale(1.05)';
                }

                // Single endpoint handles state tracking + IR code transmission
                fetch('/api/ac/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: modeName })
                })
                .then(r => r.json())
                .then(result => {
                    if (result && result.ac) applyACSnapshot(result.ac);
                    const modeLabel = modeName.replace('MODE_', '');
                    showToast('AC Mode: ' + modeLabel, result && result.status === 'success' ? 'success' : 'info');
                    setTimeout(() => {
                        document.querySelectorAll('.ac-mode-btn').forEach(btn => {
                            btn.style.opacity = '1';
                            btn.style.transform = 'scale(1)';
                        });
                        if (btnElement) {
                            btnElement.style.transform = 'scale(1.05)';
                            btnElement.style.boxShadow = '0 0 15px rgba(37, 99, 235, 0.5)';
                        }
                    }, 300);
                })
                .catch(e => {
                    showToast('Error: ' + (e.message || e), 'error');
                    document.querySelectorAll('.ac-mode-btn').forEach(btn => {
                        btn.style.opacity = '1';
                        btn.style.transform = 'scale(1)';
                    });
                });
            }); // end fetch /api/data
        }

        function loadIRCodes() {
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    learnedCodes = data.codes || data;
                    const codeCount = Object.keys(learnedCodes).length;
                    
                    // Update summary
                    const summaryEl = document.getElementById('ir-total-learned');
                    if (summaryEl) {
                        summaryEl.textContent = codeCount;
                        summaryEl.style.color = codeCount > 0 ? '#3b82f6' : '#1e40af';
                    }
                    
                    // Update button status
                    Object.keys(learnedCodes).forEach(buttonName => {
                        const buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
                        const statusElement = document.getElementById('status-' + buttonName);
                        if (buttonElement && statusElement) {
                            buttonElement.classList.add('learned');
                            statusElement.textContent = 'Learned [OK]';
                            statusElement.style.color = '#3b82f6';
                        }
                    });
                    
                    if (codeCount > 0) {
                        showToast('IR codes loaded: ' + codeCount + ' button(s)', 'success');
                    }
                })
                .catch(e => {
                    console.error('Error loading IR codes:', e);
                    showToast('Error loading IR codes', 'error');
                });
        }

        function resetAllIRCodes() {
            if (!confirm('[WARNING] Reset all learned IR codes?\\n\\nThis will delete ALL saved remote buttons!')) {
                return;
            }
            
            // Clear UI
            const buttons = ['POWER_ON', 'POWER_OFF', 'TEMP_UP', 'TEMP_DOWN', 'MODE_AUTO', 'MODE_COOL', 'MODE_FAN', 'MODE_DRY'];
            buttons.forEach(buttonName => {
                const buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
                const statusElement = document.getElementById('status-' + buttonName);
                
                if (buttonElement) {
                    buttonElement.classList.remove('learned');
                }
                if (statusElement) {
                    statusElement.textContent = 'Not learned';
                    statusElement.style.color = '#94a3b8';
                }
                
                // Delete from server
                fetch('/api/ir/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ button: buttonName })
                }).catch(e => console.log('Delete error:', e));
            });
            
            // Update summary
            const summaryEl = document.getElementById('ir-total-learned');
            if (summaryEl) {
                summaryEl.textContent = '0';
                summaryEl.style.color = '#1e40af';
            }
            
            learnedCodes = {};
            showToast('All IR codes reset!', 'success');
        }

        function saveAllIRCodes() {
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    const codes = data.codes || data;
                    const count = Object.keys(codes).length;
                    
                    if (count === 0) {
                        showToast('No IR codes to save yet', 'warning');
                        return;
                    }
                    
                    showToast(count + ' IR codes saved to server', 'success');
                    console.log('IR Codes saved:', codes);
                })
                .catch(e => showToast('Save error: ' + e, 'error'));
        }

        function exportIRCodes() {
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    const codes = data.codes || data;
                    const count = Object.keys(codes).length;
                    
                    if (count === 0) {
                        showToast('No IR codes to export', 'warning');
                        return;
                    }
                    
                    // Create JSON file and download
                    const jsonStr = JSON.stringify(codes, null, 2);
                    const blob = new Blob([jsonStr], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'ir_codes_' + new Date().toISOString().slice(0, 10) + '.json';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    showToast('Exported ' + count + ' IR codes', 'success');
                })
                .catch(e => showToast('Export error: ' + e, 'error'));
        }

        function showIRDebugInfo() {
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    const codes = data.codes || data;
                    
                    let debugInfo = '═══════════════════════════════════\\n';
                    debugInfo += '  IR CODES DEBUG INFO\\n';
                    debugInfo += '═══════════════════════════════════\\n\\n';
                    
                    if (Object.keys(codes).length === 0) {
                        debugInfo += '[WARN] No IR codes learned yet\\n\\n';
                        debugInfo += 'Steps to learn:\\n';
                        debugInfo += '1. Select protocol (RAW recommended for Mitsubishi)\\n';
                        debugInfo += '2. Click \"Learn\" button\\n';
                        debugInfo += '3. Press remote button (hold 2-3 seconds)\\n';
                        debugInfo += '4. Wait for \"Learned [OK]\" status\\n';
                        debugInfo += '5. Click \"Send\" to test\\n';
                    } else {
                        Object.keys(codes).forEach(button => {
                            const codeStr = codes[button];
                            const codePreview = codeStr.substring(0, 60) + (codeStr.length > 60 ? '...' : '');
                            debugInfo += '[IR] ' + button + ':\\n';
                            debugInfo += '   Code: ' + codePreview + '\\n';
                            debugInfo += '   Length: ' + codeStr.length + ' chars\\n';
                            
                            // Parse protocol
                            if (codeStr.includes(':')) {
                                const protocol = codeStr.split(':')[0];
                                debugInfo += '   Protocol: ' + protocol + '\\n';
                            }
                            debugInfo += '\\n';
                        });
                        
                        debugInfo += '═══════════════════════════════════\\n';
                        debugInfo += 'Total: ' + Object.keys(codes).length + ' codes\\n';
                    }
                    
                    debugInfo += '\\n[TIP] TROUBLESHOOTING:\\n';
                    debugInfo += '- If AC not responding: Use RAW protocol\\n';
                    debugInfo += '- If IR LED not blinking: Check ESP32 connection\\n';
                    debugInfo += '- If code too short: Press remote longer (2-3s)\\n';
                    debugInfo += '- Distance to AC: 1-3 meters, direct line\\n';
                    
                    alert(debugInfo);
                    console.log('IR Debug Info:', codes);
                })
                .catch(e => showToast('Debug error: ' + e, 'error'));
        }

        function testAllIRCodes() {
            if (!confirm('Test all learned IR codes?\\n\\nThis will send all codes one by one with 2 second delay.')) {
                return;
            }
            
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    const codes = data.codes || data;
                    const buttons = Object.keys(codes);
                    
                    if (buttons.length === 0) {
                        showToast('No codes to test', 'warning');
                        return;
                    }
                    
                    showToast('Testing ' + buttons.length + ' codes...', 'info');
                    
                    let index = 0;
                    const testInterval = setInterval(() => {
                        if (index >= buttons.length) {
                            clearInterval(testInterval);
                            showToast('Test complete!', 'success');
                            return;
                        }
                        
                        const button = buttons[index];
                        console.log('Testing ' + button + '...');
                        sendIRCode(button);
                        showToast('Testing: ' + button, 'info');
                        
                        index++;
                    }, 2000); // 2 second delay between tests
                })
                .catch(e => showToast('Test error: ' + e, 'error'));
        }

        // ==================== DETECTION ALERT ====================
        var lastDetectionTime = 0;
        var DETECTION_COOLDOWN = 60000; // 60 seconds (1 minute) between alerts
        var detectionSoundEnabled = localStorage.getItem('detectionSound') !== 'false';

        // Initialize sound toggle button visual on load
        function updateSoundToggleUI() {
            const btn = document.getElementById('sound-toggle-btn');
            const text = document.getElementById('sound-toggle-text');
            if (!btn) return;
            if (detectionSoundEnabled) {
                btn.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
                if (text) text.textContent = 'Sound ON';
            } else {
                btn.style.background = 'linear-gradient(135deg, #1e40af, #1d4ed8)';
                if (text) text.textContent = 'Sound OFF';
            }
        }

        function playDetectionSound() {
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const now = audioContext.currentTime;

                // Beep 1 — alert tone
                const osc1 = audioContext.createOscillator();
                const gain1 = audioContext.createGain();
                osc1.connect(gain1);
                gain1.connect(audioContext.destination);
                osc1.frequency.value = 880;
                osc1.type = 'sine';
                gain1.gain.setValueAtTime(0.4, now);
                gain1.gain.exponentialRampToValueAtTime(0.01, now + 0.25);
                osc1.start(now);
                osc1.stop(now + 0.25);

                // Beep 2 — higher pitched
                const osc2 = audioContext.createOscillator();
                const gain2 = audioContext.createGain();
                osc2.connect(gain2);
                gain2.connect(audioContext.destination);
                osc2.frequency.value = 1100;
                osc2.type = 'sine';
                gain2.gain.setValueAtTime(0.4, now + 0.3);
                gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.55);
                osc2.start(now + 0.3);
                osc2.stop(now + 0.55);

                // Beep 3 — highest
                const osc3 = audioContext.createOscillator();
                const gain3 = audioContext.createGain();
                osc3.connect(gain3);
                gain3.connect(audioContext.destination);
                osc3.frequency.value = 1320;
                osc3.type = 'sine';
                gain3.gain.setValueAtTime(0.35, now + 0.6);
                gain3.gain.exponentialRampToValueAtTime(0.01, now + 1.0);
                osc3.start(now + 0.6);
                osc3.stop(now + 1.0);
            } catch(e) {
                console.log('Audio notification not available:', e);
            }
        }

        function showDetectionAlert(count, confidence) {
            const now = Date.now();
            if (now - lastDetectionTime < DETECTION_COOLDOWN) return;
            
            lastDetectionTime = now;
            
            document.getElementById('alert-person-count').textContent = count;
            document.getElementById('alert-person-confidence').textContent = confidence + '%';
            document.getElementById('alert-time').textContent = new Date().toLocaleTimeString();
            
            const alertBox = document.getElementById('detection-alert');
            alertBox.classList.add('show');
            
            // Play 3-beep alert sound if enabled
            if (detectionSoundEnabled) {
                playDetectionSound();
            }
            
            // Auto hide after 5 seconds
            setTimeout(() => {
                alertBox.classList.remove('show');
            }, 5000);
            
            console.log('[ALERT] PERSON DETECTED:', {
                count: count,
                confidence: confidence + '%',
                time: new Date().toLocaleTimeString()
            });
        }

        function closeDetectionAlert() {
            document.getElementById('detection-alert').classList.remove('show');
        }

        function toggleDetectionSound() {
            detectionSoundEnabled = !detectionSoundEnabled;
            localStorage.setItem('detectionSound', detectionSoundEnabled);
            updateSoundToggleUI();
            if (detectionSoundEnabled) {
                playDetectionSound();
                showToast('Sound alerts ON — you will hear a sound when a person is detected', 'success');
            } else {
                showToast('Sound alerts OFF — sound notifications disabled', 'info');
            }
        }

        // ==================== TOAST ====================
        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            const toastMessage = document.getElementById('toast-message');
            const icon = type === 'success' ? 'check' : (type === 'info' ? 'info' : 'exclamation');
            
            toastMessage.innerHTML = message;
            toast.classList.add('show');
            setTimeout(() => { toast.classList.remove('show'); }, 3000);
        }

        var energyBubbleTimer = null;
        function showEnergyBubble(power, voltage, current) {
            const bubble = document.getElementById('energy-bubble');
            const text = document.getElementById('energy-bubble-text');
            if (!bubble || !text) return;
            text.textContent = power.toFixed(3) + ' kW | ' + voltage.toFixed(1) + ' V | ' + current.toFixed(2) + ' A';
            bubble.classList.add('show');
            if (energyBubbleTimer) clearTimeout(energyBubbleTimer);
            energyBubbleTimer = setTimeout(() => { bubble.classList.remove('show'); }, 4000);
        }

        // ==================== OCCUPANCY FEEDBACK ====================
        function selectFeedbackRating(value) {
            selectedFeedbackRating = value;
            document.querySelectorAll('#rating-row .rating-btn').forEach((btn, idx) => {
                btn.classList.toggle('active', idx + 1 === value);
            });
        }

        function saveGoogleFormUrl() {
            const url = (document.getElementById('google-form-url').value || '').trim();
            if (!url) {
                showToast('Please enter Google Form URL first', 'error');
                return;
            }
            localStorage.setItem('googleFormUrl', url);
            showToast('Google Form URL saved', 'success');
        }

        function openGoogleForm() {
            const url = (document.getElementById('google-form-url').value || '').trim() || localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;
            window.open(url, '_blank');
        }

        function submitOccupancyFeedback() {
            const comment = (document.getElementById('feedback-comment').value || '').trim();
            const formUrl = (document.getElementById('google-form-url').value || '').trim() || localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;

            if (!selectedFeedbackRating) {
                showToast('Please select a rating 1-5 first', 'error');
                return;
            }

            fetch('/api/occupancy/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rating: selectedFeedbackRating,
                    comment: comment,
                    google_form_url: formUrl,
                    occupancy_count: parseInt((document.getElementById('cam-count') ? document.getElementById('cam-count').textContent : '0') || '0', 10)
                })
            })
            .then(r => r.json())
            .then(result => {
                if (result.status === 'success') {
                    showToast('Feedback submitted successfully', 'success');
                    document.getElementById('feedback-comment').value = '';
                    selectFeedbackRating(0);
                    loadFeedbackHistory();
                    if (formUrl && formUrl !== DEFAULT_GOOGLE_FORM_URL) {
                        window.open(formUrl, '_blank');
                    }
                } else {
                    showToast(result.message || 'Failed to submit feedback', 'error');
                }
            })
            .catch(e => showToast('Error: ' + (e.message || e), 'error'));
        }

        function loadFeedbackHistory() {
            fetch('/api/occupancy/feedback/list')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('feedback-history');
                    if (!container) return;

                    const rows = data.feedback || [];
                    if (rows.length === 0) {
                        container.innerHTML = '<div style="color: var(--text-secondary); font-size: 13px;">No feedback yet.</div>';
                        return;
                    }

                    container.innerHTML = rows.map(item => {
                        const safeComment = (item.comment || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        return '<div class="feedback-history-item">' +
                            '<div style="display:flex; justify-content:space-between; margin-bottom:6px;">' +
                                '<strong>Rating: ' + item.rating + '/5</strong>' +
                                '<span style="font-size:12px; color: var(--text-secondary);">' + item.time + '</span>' +
                            '</div>' +
                            '<div style="font-size: 13px; color: var(--text-secondary);">Occupancy: ' + item.occupancy_count + ' person(s)</div>' +
                            '<div style="margin-top:6px; font-size:13px;">' + (safeComment || '-') + '</div>' +
                        '</div>';
                    }).join('');
                })
                .catch(() => {});
        }

        // ==================== DATA UPDATES ====================
        function updateDashboard() {
            fetch('/api/data', { cache: 'no-store' })
                .then(r => r.json())
                .then(data => { try {
                    data = data || {};
                    const ac = data.ac || {};
                    const lamp = data.lamp || {};
                    const camera = data.camera || {};
                    const system = data.system || {};
                    const energy = data.energy || {};
                    const num = (v, d = 0) => {
                        const n = parseFloat(v);
                        return Number.isFinite(n) ? n : d;
                    };
                    const setText = (id, val) => {
                        const el = document.getElementById(id);
                        if (el) el.textContent = val;
                    };

                    const temperature = num(ac.temperature);
                    setText('dash-temp', temperature.toFixed(1));
                    setText('dash-hum', num(ac.humidity).toFixed(1));
                    // Individual sensor readings — sorted coldest→hottest (S1→S3)
                    var rawTemps = [
                        { t: num(ac.temp1), h: num(ac.hum1) },
                        { t: num(ac.temp2), h: num(ac.hum2) },
                        { t: num(ac.temp3), h: num(ac.hum3) }
                    ];
                    // Sort by temperature ascending (coldest first = S1)
                    rawTemps.sort(function(a, b) { return a.t - b.t; });
                    const t1El = document.getElementById('dash-temp1');
                    const t2El = document.getElementById('dash-temp2');
                    const t3El = document.getElementById('dash-temp3');
                    if (t1El) t1El.textContent = rawTemps[0].t.toFixed(1);
                    if (t2El) t2El.textContent = rawTemps[1].t.toFixed(1);
                    if (t3El) t3El.textContent = rawTemps[2].t.toFixed(1);
                    // Individual humidity readings (sorted same order as temp)
                    const h1El = document.getElementById('dash-hum1');
                    const h2El = document.getElementById('dash-hum2');
                    const h3El = document.getElementById('dash-hum3');
                    if (h1El) h1El.textContent = rawTemps[0].h.toFixed(1);
                    if (h2El) h2El.textContent = rawTemps[1].h.toFixed(1);
                    if (h3El) h3El.textContent = rawTemps[2].h.toFixed(1);
                    
                    // Heat Index
                    const hiEl = document.getElementById('dash-heat-index');
                    if (hiEl) hiEl.textContent = num(ac.heat_index).toFixed(1);
                    
                    // ESP32 Signal & Uptime
                    const rssiEl = document.getElementById('dash-rssi');
                    if (rssiEl) rssiEl.textContent = ac.rssi || 0;
                    const uptEl = document.getElementById('dash-uptime');
                    if (uptEl) {
                        const secs = num(ac.uptime);
                        const h = Math.floor(secs / 3600);
                        const m = Math.floor((secs % 3600) / 60);
                        uptEl.textContent = h > 0 ? h + 'h ' + m + 'm' : m + 'm ' + (secs % 60) + 's';
                    }
                    
                    // AC State - use REAL state from ESP32, not temperature guessing
                    const acStateEl = document.getElementById('dash-ac-state');
                    let acState = ac.ac_state || 'OFF';
                    
                    if (acStateEl) {
                        if (acState === 'ON') {
                            acStateEl.style.color = '#3b82f6'; // Green
                        } else {
                            acStateEl.style.color = '#1e40af'; // Red
                        }
                        acStateEl.textContent = acState;
                    }
                    
                    // AC panel power dot
                    const panelDot = document.getElementById('ac-panel-dot');
                    if (panelDot) {
                        panelDot.style.background = acState === 'ON' ? '#3b82f6' : '#1e40af';
                        panelDot.style.boxShadow = acState === 'ON' ? '0 0 8px rgba(59,130,246,0.5)' : '0 0 8px rgba(30,64,175,0.5)';
                    }
                    
                    // Set Temperature
                    const acTemp = num(ac.ac_temp, 24);
                    setText('dash-ac-temp', acTemp);
                    
                    // Fan Speed with label
                    const fanSpeed = num(ac.fan_speed, 1);
                    setText('dash-ac-fan', fanSpeed);
                    const fanLabel = document.getElementById('dash-ac-fan-label');
                    if (fanLabel) {
                        const fanNames = {1: 'Low', 2: 'Medium', 3: 'High'};
                        fanLabel.textContent = fanNames[fanSpeed] || 'Level ' + fanSpeed;
                    }
                    
                    // AC Mode (COOL/HEAT/DRY/FAN/AUTO) with icon + color — uses ac_fan_mode from ESP32
                    const acMode = ac.ac_fan_mode || 'COOL';
                    setText('dash-ac-mode', acMode);
                    const modeIconEl = document.getElementById('dash-ac-mode-icon');
                    const modeTextEl = document.getElementById('dash-ac-mode');
                    const modeIcons = {
                        'COOL': {text: 'COOL', color: '#0ea5e9'},
                        'HEAT': {text: 'HEAT', color: '#0ea5e9'},
                        'DRY':  {text: 'DRY', color: '#2563eb'},
                        'FAN':  {text: 'FAN', color: '#3b82f6'},
                        'AUTO': {text: 'AUTO', color: '#2563eb'}
                    };
                    const modeInfo = modeIcons[acMode] || modeIcons['COOL'];
                    if (modeIconEl) {
                        modeIconEl.textContent = modeInfo.text;
                        modeIconEl.style.color = modeInfo.color;
                    }
                    if (modeTextEl) modeTextEl.style.color = modeInfo.color;
                    
                    // Operating Mode (ADAPTIVE / MANUAL)
                    const ctrlMode = ac.mode || 'ADAPTIVE';
                    const ctrlIcon = document.getElementById('dash-ac-ctrl-icon');
                    const ctrlText = document.getElementById('dash-ac-ctrl-mode');
                    if (ctrlIcon && ctrlText) {
                        if (ctrlMode === 'ADAPTIVE') {
                            ctrlIcon.textContent = 'A';
                            ctrlIcon.style.color = '#3b82f6';
                            ctrlText.textContent = 'ADAPTIVE';
                            ctrlText.style.color = '#3b82f6';
                        } else {
                            ctrlIcon.textContent = 'M';
                            ctrlIcon.style.color = '#0ea5e9';
                            ctrlText.textContent = 'MANUAL';
                            ctrlText.style.color = '#0ea5e9';
                        }
                    }
                    
                    // Source badge
                    const srcBadge = document.getElementById('dash-ac-source');
                    if (srcBadge) {
                        if (ctrlMode === 'ADAPTIVE') {
                            srcBadge.textContent = 'AI Controlled';
                            srcBadge.style.background = 'rgba(59, 130, 246, 0.12)';
                            srcBadge.style.color = '#3b82f6';
                            srcBadge.style.borderColor = 'rgba(59, 130, 246, 0.25)';
                        } else {
                            srcBadge.textContent = 'Manual Control';
                            srcBadge.style.background = 'rgba(14, 165, 233, 0.12)';
                            srcBadge.style.color = '#0ea5e9';
                            srcBadge.style.borderColor = 'rgba(14, 165, 233, 0.25)';
                        }
                    }
                    
                    // Room environment in AC panel footer
                    const roomTemp = document.getElementById('dash-ac-room-temp');
                    const roomHum = document.getElementById('dash-ac-room-hum');
                    if (roomTemp) roomTemp.textContent = temperature.toFixed(1);
                    if (roomHum) roomHum.textContent = num(ac.humidity).toFixed(1);
                    // Individual sensors in AC panel footer (use sorted rawTemps from above)
                    const acT1 = document.getElementById('dash-ac-temp1');
                    const acT2 = document.getElementById('dash-ac-temp2');
                    const acT3 = document.getElementById('dash-ac-temp3');
                    if (acT1 && rawTemps) acT1.textContent = rawTemps[0].t.toFixed(1);
                    if (acT2 && rawTemps) acT2.textContent = rawTemps[1].t.toFixed(1);
                    if (acT3 && rawTemps) acT3.textContent = rawTemps[2].t.toFixed(1);
                    
                    // Update AC Live Status Bar in control panel
                    const liveDot = document.getElementById('ac-live-dot');
                    const liveState = document.getElementById('ac-live-state');
                    if (liveDot && liveState) {
                        liveState.textContent = acState;
                        liveDot.style.background = acState === 'ON' ? '#3b82f6' : '#1e40af';
                        setText('ac-live-temp', acTemp);
                        setText('ac-live-fan', fanSpeed);
                        setText('ac-live-mode', acMode);
                    }
                    updateExtraFeatureButtons(ac);
                    setText('dash-lux1', num(lamp.lux1).toFixed(0));
                    setText('dash-lux2', num(lamp.lux2).toFixed(0));
                    setText('dash-lux3', num(lamp.lux3).toFixed(0));
                    setText('dash-lux-avg', num(lamp.lux_avg).toFixed(1));
                    setText('dash-bright1', Math.round(num(lamp.brightness1)));
                    setText('dash-bright2', Math.round(num(lamp.brightness2)));
                    setText('dash-motion', lamp.motion ? 'MOTION DETECTED' : 'NO MOTION');
                    // Update lamp control page live readings
                    setText('ctrl-lux1', num(lamp.lux1).toFixed(0));
                    setText('ctrl-lux2', num(lamp.lux2).toFixed(0));
                    setText('ctrl-lux3', num(lamp.lux3).toFixed(0));
                    const ctrlMotion = document.getElementById('ctrl-motion-status');
                    if (ctrlMotion) ctrlMotion.innerHTML = lamp.motion ? '<span style="color:#3b82f6">MOTION</span>' : '<span style="color:#1e40af">IDLE</span>';
                    // Sync lamp mode buttons
                    applyLampModeUI(lamp.mode || 'ADAPTIVE');
                    
                    const personDetected = !!camera.person_detected;
                    const personCount = num(camera.count);
                    const confidence = num(camera.confidence);
                    
                    const camPersonEl = document.getElementById('cam-person');
                    if (camPersonEl) {
                        camPersonEl.textContent = personDetected ? 'Yes' : 'No';
                        camPersonEl.style.color = personDetected ? '#3b82f6' : '#1e40af';
                    }
                    
                    const camCountEl = document.getElementById('cam-count');
                    if (camCountEl) {
                        camCountEl.textContent = personCount;
                        camCountEl.style.color = personCount > 0 ? '#3b82f6' : '#94a3b8';
                    }
                    
                    const camConfEl = document.getElementById('cam-confidence');
                    if (camConfEl) {
                        camConfEl.textContent = confidence + '%';
                        camConfEl.style.color = confidence > 70 ? '#3b82f6' : (confidence > 50 ? '#0ea5e9' : '#1e40af');
                    }
                    
                    // Person detection status badge
                    const camBadge = document.getElementById('cam-status-badge');
                    if (camBadge) {
                        if (personCount > 0) {
                            camBadge.innerHTML = '<i class="fas fa-circle" style="font-size: 7px; vertical-align: middle;"></i> ' + personCount + ' Person Detected';
                            camBadge.style.background = 'rgba(59, 130, 246, 0.12)';
                            camBadge.style.color = '#3b82f6';
                            camBadge.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                        } else {
                            camBadge.innerHTML = '<i class="fas fa-circle" style="font-size: 7px; vertical-align: middle;"></i> No Person';
                            camBadge.style.background = 'rgba(30, 64, 175, 0.12)';
                            camBadge.style.color = '#1e40af';
                            camBadge.style.borderColor = 'rgba(30, 64, 175, 0.3)';
                        }
                    }

                    const occCountEl = document.getElementById('occ-live-count');
                    const occConfEl = document.getElementById('occ-live-confidence');
                    if (occCountEl) occCountEl.textContent = personCount;
                    if (occConfEl) occConfEl.textContent = confidence + '%';
                    
                    // Show detection alert if person detected
                    if (personDetected && personCount > 0) {
                        showDetectionAlert(personCount, confidence);
                    }
                    
                    // Update GA/PSO Fitness (from optimization algorithm)
                    const gaFitness = num(system.ga_fitness);
                    const psoFitness = num(system.pso_fitness);
                    
                    const gaEl = document.getElementById('ga-fitness');
                    const psoEl = document.getElementById('pso-fitness');
                    
                    if (gaEl) {
                        gaEl.textContent = gaFitness.toFixed(2);
                        gaEl.style.color = gaFitness > 0 ? '#3b82f6' : '#94a3b8';
                    }
                    if (psoEl) {
                        psoEl.textContent = psoFitness.toFixed(2);
                        psoEl.style.color = psoFitness > 0 ? '#3b82f6' : '#94a3b8';
                    }
                    
                    // PSO Brightness — pso-brightness element not on main page (only on ML page)
                    const psoBrightEl = document.getElementById('pso-brightness');
                    if (psoBrightEl) {
                        psoBrightEl.textContent = Math.round(num(system.pso_brightness));
                    }
                    
                    // Optimization Runs
                    const optRunsEl = document.getElementById('dash-opt-runs');
                    if (optRunsEl) {
                        optRunsEl.textContent = num(system.optimization_runs);
                    }
                    
                    // Energy data dihandle oleh mysql_energy_update event
                    // (ac-power, lamp-power, total-power, total-current, total-energy-kwh,
                    //  daily-cost diisi langsung dari MySQL via socket)

                    updateModeBadges();
                } catch(e) { var p = document.getElementById('diag-panel'); if (p) { p.style.display='block'; var d = document.getElementById('diag-result'); if (d) d.textContent += '\\n[DASHBOARD ERROR] ' + e.message; } console.error(e); } });
        }

        function updateLogs() {
            fetch('/api/logs')
                .then(r => r.json())
                .then(logs => {
                    const container = document.getElementById('log-container');
                    container.innerHTML = '';
                    logs.reverse().forEach(log => {
                        const entry = document.createElement('div');
                        entry.className = 'log-entry ' + log.level;
                        entry.innerHTML = '<strong>[' + log.time + ']</strong> ' + log.msg;
                        container.appendChild(entry);
                    });
                });
        }

        // ==================== DEVICE STATUS ====================
        function diagLog(msg) {
            var el = document.getElementById('diag-result');
            if (el) el.textContent += msg + '\\n';
        }
        function diagClear(title) {
            var el = document.getElementById('diag-result');
            if (el) el.textContent = '[' + new Date().toLocaleTimeString() + '] ' + title + '\\n';
        }

        function checkMqttStatus(showDetail) {
            fetch('/api/mqtt/status')
                .then(r => r.json())
                .then(data => {
                    var dot = document.getElementById('mqtt-dot');
                    var txt = document.getElementById('mqtt-status-text');
                    if (dot && txt) {
                        if (data.connected) {
                            dot.className = 'device-dot online';
                            txt.textContent = 'Connected' + (data.message_count > 0 ? ' (' + data.message_count + ' msgs)' : '');
                        } else {
                            dot.className = 'device-dot offline';
                            txt.textContent = data.error ? data.error.substring(0, 30) : 'Disconnected';
                        }
                    }
                    if (showDetail) {
                        diagClear('=== MQTT STATUS ===');
                        diagLog('Broker: ' + data.broker);
                        diagLog('Connected: ' + (data.connected ? 'YES' : 'NO'));
                        diagLog('Messages received: ' + data.message_count);
                        diagLog('Last connect: ' + (data.last_connect || 'Never'));
                        diagLog('Last message: ' + (data.last_message || 'Never'));
                        diagLog('Error: ' + (data.error || 'None'));
                        diagLog('Subscriptions: ' + (data.subscriptions || []).join(', '));
                        if (!data.connected) {
                            diagLog('');
                            diagLog('SOLUSI: Pastikan MQTT broker (Mosquitto) berjalan!');
                            diagLog('Windows: net start mosquitto');
                            diagLog('Linux/Pi: sudo systemctl start mosquitto');
                        }
                    }
                })
                .catch(function() {
                    var dot = document.getElementById('mqtt-dot');
                    var txt = document.getElementById('mqtt-status-text');
                    if (dot) dot.className = 'device-dot offline';
                    if (txt) txt.textContent = 'API Error';
                    if (showDetail) diagLog('ERROR: Cannot fetch /api/mqtt/status');
                });
        }

        function runSimulate() {
            diagClear('=== TEST FRONTEND (INJECT DATA DUMMY) ===');
            diagLog('Sending dummy data to server...');
            fetch('/api/simulate', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'ok') {
                        diagLog('SUCCESS! Dummy data injected:');
                        diagLog('  AC Temperature: ' + data.ac_temp + ' C');
                        diagLog('  Lamp Lux1: ' + data.lamp_lux + ' lux');
                        diagLog('');
                        diagLog('If values on dashboard change from 0 -> FRONTEND OK');
                        diagLog('If still 0 -> Problem in JavaScript/DOM');
                        updateDashboard();
                    } else {
                        diagLog('ERROR: ' + data.message);
                    }
                })
                .catch(function(e) { diagLog('FETCH ERROR: ' + e); });
        }

        function runMqttSelftest() {
            diagClear('=== TEST MQTT BROKER (SELF-TEST) ===');
            diagLog('Server will publish to smartroom/ac/sensors...');
            fetch('/api/mqtt/selftest', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'ok') {
                        diagLog('SUCCESS! Test message published to broker.');
                        diagLog(data.message);
                        diagLog('');
                        diagLog('Wait 2 seconds... then check if data appears on dashboard.');
                        diagLog('If visible -> MQTT Broker OK, problem is in ESP32');
                        diagLog('If not visible -> Problem with subscribe/routing topic');
                        setTimeout(function() { updateDashboard(); diagLog('Dashboard refreshed.'); }, 2000);
                    } else {
                        diagLog('FAILED: ' + data.message);
                        diagLog('');
                        diagLog('This means: MQTT Broker is not running or cannot connect!');
                        diagLog('Jalankan Mosquitto terlebih dahulu.');
                    }
                })
                .catch(function(e) { diagLog('FETCH ERROR: ' + e); });
        }

        function runMqttReconnect() {
            diagClear('=== RECONNECT MQTT ===');
            diagLog('Trying to reconnect to MQTT broker...');
            fetch('/api/mqtt/reconnect', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    diagLog('Response: ' + data.message);
                    setTimeout(function() {
                        checkMqttStatus(true);
                        diagLog('Status setelah reconnect dicek.');
                    }, 2000);
                })
                .catch(function(e) { diagLog('FETCH ERROR: ' + e); });
        }

        function updateDeviceStatus() {
            checkMqttStatus(false);
            fetch('/api/device/status')
                .then(r => r.json())
                .then(data => {
                    const mapping = {'esp32_ac': 'ds-esp32-ac', 'esp32_lamp': 'ds-esp32-lamp', 'camera': 'ds-camera'};
                    const timeMapping = {'esp32_ac': 'ds-ac-time', 'esp32_lamp': 'ds-lamp-time', 'camera': 'ds-cam-time'};
                    for (const [devId, info] of Object.entries(data)) {
                        const el = document.getElementById(mapping[devId]);
                        const timeEl = document.getElementById(timeMapping[devId]);
                        if (el) {
                            const dot = el.querySelector('.device-dot');
                            if (dot) {
                                dot.className = 'device-dot ' + info.status;
                            }
                        }
                        if (timeEl) timeEl.textContent = info.last_seen;
                    }
                })
                .catch(() => {});
        }

        // ==================== ALERT SYSTEM ====================
        var alertQueue = [];
        socket.on('alert', function(alert) {
            showAlertBanner(alert);
        });

        function showAlertBanner(alert) {
            const banner = document.createElement('div');
            banner.className = 'alert-banner ' + alert.level;
            banner.innerHTML = '<span>' + alert.message + '</span>' +
                '<button class="alert-close" onclick="this.parentElement.remove()">&times;</button>';
            document.body.appendChild(banner);
            setTimeout(() => { if (banner.parentElement) banner.remove(); }, 8000);
        }

        // ==================== SENSOR HEALTH ====================
        var _sensorHealthState = {};

        socket.on('sensor_health', function(health) {
            _sensorHealthState = health;
            updateSensorHealthBar(health);
        });

        socket.on('sensor_fault', function(data) {
            // Update individual badge immediately on fault/recovery
            var id = data.device;
            if (_sensorHealthState[id]) _sensorHealthState[id].status = data.status;
            else _sensorHealthState[id] = data;
            updateSensorHealthBar(_sensorHealthState);
        });

        function updateSensorHealthBar(health) {
            var bar = document.getElementById('sensor-health-bar');
            if (!bar) return;
            var html = '';
            var statusIcon = {ok: '&#x25CF;', warn: '&#x25CF;', fault: '&#x25CF;'};
            var statusColor = {ok: '#60a5fa', warn: '#0ea5e9', fault: '#1e40af'};
            var statusText  = {ok: 'Online', warn: 'Stale', fault: 'Offline'};
            for (var dev in health) {
                var h = health[dev];
                var col  = statusColor[h.status]  || '#6b7280';
                var icon = statusIcon[h.status]   || '&#x25CF;';
                var txt  = statusText[h.status]   || h.status;
                html += '<span class="sensor-health-badge" style="color:' + col + ';" title="' + h.label + ': ' + h.age + '">'
                      + icon + ' ' + h.label + ' <small style="opacity:.7;">' + txt + '</small></span>';
            }
            bar.innerHTML = html;
        }

        // Fetch initial health on load
        fetch('/api/sensor/health').then(function(r){ return r.json(); }).then(function(h){
            _sensorHealthState = h;
            updateSensorHealthBar(h);
        }).catch(function(){});

        // ==================== SOCKET.IO EVENTS ====================
        // Ring buffer untuk chart MySQL realtime
        var _mysqlBuf = { labels: [], acV: [], outletV: [], lampV: [], freq: [], acP: [], outletP: [], lampP: [], acA: [], outletA: [], lampA: [] };
        var _MYSQL_MAX = 30;

        // ── Fungsi update DOM energi (dipakai socket DAN direct-poll) ──
        var _lastEnergyUpdate = 0;
        function _applyEnergyData(ac, outlet, lamp) {
            outlet = outlet || {};
            lamp = lamp || {};
            _lastEnergyUpdate = Date.now();
            var s = function(id, val) { var el = document.getElementById(id); if (el) el.textContent = val; };
            var f = function(v, d) { return parseFloat(v || 0).toFixed(d !== undefined ? d : 1); };
            var fkwh = function(v) {
                var num = parseFloat(v || 0);
                if (!Number.isFinite(num)) return '0';
                return num.toFixed(4).replace(/\\.0+$/, '').replace(/(\\.\\d*?)0+$/, '$1');
            };

            // ── Overview page badge + mini cards ──
            var badge = document.getElementById('mysql-energy-badge');
            if (badge) {
                badge.innerHTML = '<i class="fas fa-circle" style="font-size:6px;vertical-align:middle;margin-right:4px;"></i> Online';
                badge.style.cssText = 'padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(37,99,235,0.15);color:#2563eb;border:1px solid rgba(37,99,235,0.3);';
            }
            s('mysql-ac-voltage', f(ac.voltage));       s('mysql-ac-current', f(ac.current, 2));
            s('mysql-ac-power',   f(ac.power));         s('mysql-ac-kwh',     fkwh(ac.energy));
            s('mysql-ac-freq',    f(ac.frequency, 2));  s('mysql-ac-pf',      f(ac.pf, 2));
            s('mysql-ac-ts',      ac.updated_at || '--');
            s('mysql-lamp-voltage', f(lamp.voltage));     s('mysql-lamp-current', f(lamp.current, 2));
            s('mysql-lamp-power',   f(lamp.power));       s('mysql-lamp-kwh',     fkwh(lamp.energy));
            s('mysql-lamp-freq',    f(lamp.frequency, 2)); s('mysql-lamp-pf',     f(lamp.pf, 2));
            s('mysql-lamp-ts',      lamp.updated_at || '--');

            // ── Energy Usage page live values ──
            var liveBadge = document.getElementById('mysql-live-badge');
            if (liveBadge) {
                liveBadge.innerHTML = '<i class="fas fa-circle" style="font-size:6px;vertical-align:middle;margin-right:4px;"></i> Live';
                liveBadge.style.cssText = 'padding:4px 14px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(37,99,235,0.15);color:#2563eb;border:1px solid rgba(37,99,235,0.3);';
            }
            s('eu-ac-voltage', f(ac.voltage));       s('eu-ac-current', f(ac.current, 2));
            s('eu-ac-power',   f(ac.power));         s('eu-ac-kwh',     fkwh(ac.energy));
            s('eu-ac-freq',    f(ac.frequency, 2));  s('eu-ac-pf',      f(ac.pf, 2));
            s('eu-ac-ts',      ac.updated_at || '--');
            s('outlet-power',      f(outlet.power));
            s('eu-outlet-voltage', f(outlet.voltage));       s('eu-outlet-current', f(outlet.current, 2));
            s('eu-outlet-kwh',     fkwh(outlet.energy));     s('eu-outlet-freq',    f(outlet.frequency, 2));
            s('eu-outlet-pf',      f(outlet.pf, 2));          s('eu-outlet-ts',      outlet.updated_at || '--');
            
            // Populate Outlet Analysis Live Params
            s('oa-live-voltage', f(outlet.voltage));       s('oa-live-current', f(outlet.current, 2));
            s('oa-live-apparent', f(outlet.apparent_power, 1)); s('oa-live-reactive', f(outlet.reactive_power, 1));
            s('oa-live-freq', f(outlet.frequency, 2));      s('oa-live-pf', f(outlet.pf, 2));
            
            s('eu-lamp-voltage', f(lamp.voltage));     s('eu-lamp-current', f(lamp.current, 2));
            s('eu-lamp-power',   f(lamp.power));       s('eu-lamp-kwh',     fkwh(lamp.energy));
            s('eu-lamp-freq',    f(lamp.frequency, 2)); s('eu-lamp-pf',     f(lamp.pf, 2));
            s('eu-lamp-ts',      lamp.updated_at || '--');

            // ── Reactive / Apparent / PF quality / Power bars ──
            s('eu-ac-reactive',   f(ac.reactive_power,  1));
            s('eu-ac-apparent',   f(ac.apparent_power,  1));
            s('eu-outlet-reactive', f(outlet.reactive_power, 1));
            s('eu-outlet-apparent', f(outlet.apparent_power, 1));
            s('eu-lamp-reactive', f(lamp.reactive_power, 1));
            s('eu-lamp-apparent', f(lamp.apparent_power, 1));
            var _pfBadge = function(pf) {
                var v = parseFloat(pf||0);
                if (v >= 0.95) return {t:'Baik',  bg:'rgba(59,130,246,0.15)', c:'#3b82f6'};
                if (v >= 0.85) return {t:'Cukup', bg:'rgba(14,165,233,0.15)', c:'#0ea5e9'};
                if (v >  0)    return {t:'Buruk', bg:'rgba(30,64,175,0.15)',  c:'#1e40af'};
                return {t:'--', bg:'rgba(107,114,128,0.15)', c:'#6b7280'};
            };
            var _applyBadge = function(id, pf) {
                var q = _pfBadge(pf); var el = document.getElementById(id);
                if (el) { el.textContent = q.t; el.style.background = q.bg; el.style.color = q.c; }
            };
            _applyBadge('eu-ac-pf-quality',   ac.pf);
            _applyBadge('eu-outlet-pf-quality', outlet.pf);
            _applyBadge('oa-live-pf-quality', outlet.pf);
            _applyBadge('eu-lamp-pf-quality', lamp.pf);
            var acBar = document.getElementById('eu-ac-pbar');
            var outletBar = document.getElementById('eu-outlet-pbar');
            var lpBar = document.getElementById('eu-lamp-pbar');
            if (acBar)  acBar.style.width  = Math.min(100, (parseFloat(ac.power||0)   / 3.0 * 100)).toFixed(1) + '%';
            if (outletBar) outletBar.style.width = Math.min(100, (parseFloat(outlet.power||0) / 1.0 * 100)).toFixed(1) + '%';
            if (lpBar)  lpBar.style.width  = Math.min(100, (parseFloat(lamp.power||0) / 0.5 * 100)).toFixed(1) + '%';

            // ── Update power-grid summary cards ──
            var acPow  = parseFloat(ac.power||0);
            var outletPow = parseFloat(outlet.power||0);
            var lampPow = parseFloat(lamp.power||0);
            var acKwh  = parseFloat(ac.energy||0);
            var outletKwh = parseFloat(outlet.energy||0);
            var lampKwh = parseFloat(lamp.energy||0);
            var totPow = acPow + outletPow + lampPow;
            var totKwh = acKwh + outletKwh + lampKwh;
            s('ac-power',   acPow.toFixed(1));
            s('outlet-power', outletPow.toFixed(1));
            s('lamp-power', lampPow.toFixed(1));
            s('total-power',   totPow.toFixed(1));
            s('total-current', (parseFloat(ac.current||0) + parseFloat(outlet.current||0) + parseFloat(lamp.current||0)).toFixed(2));
            s('ac-energy-kwh',   fkwh(acKwh));
            s('outlet-energy-kwh', fkwh(outletKwh));
            s('outlet-analytics-total-energy', fkwh(outletKwh));
            s('outlet-total-power', outletPow.toFixed(1));
            s('lamp-energy-kwh', fkwh(lampKwh));
            s('total-energy-kwh', fkwh(totKwh));
            s('ac-voltage-card',  f(ac.voltage));
            s('ac-current-card',  f(ac.current, 2));
            s('ac-pf-card',       f(ac.pf, 2));
            s('outlet-voltage-card', f(outlet.voltage));
            s('outlet-current-card', f(outlet.current, 2));
            s('outlet-pf-card',      f(outlet.pf, 2));
            s('lamp-voltage-card', f(lamp.voltage));
            s('lamp-current-card', f(lamp.current, 2));
            s('lamp-pf-card',      f(lamp.pf, 2));
            s('total-freq-card',   f(ac.frequency, 2));
            s('energy-last-update', ac.updated_at ? ac.updated_at.substring(11,19) : '--');
            var costEst = Math.round(totKwh * 1500);
            s('daily-cost', costEst > 0 ? costEst.toLocaleString() : '0');

            // ── Ring buffer & chart ──
            var now2 = new Date();
            var ts2  = now2.getHours() + ':' + String(now2.getMinutes()).padStart(2,'0') + ':' + String(now2.getSeconds()).padStart(2,'0');
            _mysqlBuf.labels.push(ts2);
            _mysqlBuf.acV.push(parseFloat(ac.voltage||0));
            _mysqlBuf.outletV.push(parseFloat(outlet.voltage||0));
            _mysqlBuf.lampV.push(parseFloat(lamp.voltage||0));
            _mysqlBuf.freq.push(parseFloat(ac.frequency||0));
            _mysqlBuf.acP.push(parseFloat(ac.power||0));
            _mysqlBuf.outletP.push(parseFloat(outlet.power||0));
            _mysqlBuf.lampP.push(parseFloat(lamp.power||0));
            _mysqlBuf.acA.push(parseFloat(ac.current||0));
            _mysqlBuf.outletA.push(parseFloat(outlet.current||0));
            _mysqlBuf.lampA.push(parseFloat(lamp.current||0));
            if (_mysqlBuf.labels.length > _MYSQL_MAX) {
                ['labels','acV','outletV','lampV','freq','acP','outletP','lampP','acA','outletA','lampA'].forEach(function(k){ _mysqlBuf[k].shift(); });
            }
            if (charts.mysqlVoltFreq) {
                charts.mysqlVoltFreq.data.labels = _mysqlBuf.labels.slice();
                charts.mysqlVoltFreq.data.datasets[0].data = _mysqlBuf.freq.slice();
                charts.mysqlVoltFreq.update('none');
            }
            if (charts.mysqlCurrent) {
                charts.mysqlCurrent.data.labels = _mysqlBuf.labels.slice();
                charts.mysqlCurrent.data.datasets[0].data = _mysqlBuf.acA.slice();
                charts.mysqlCurrent.data.datasets[1].data = _mysqlBuf.outletA.slice();
                charts.mysqlCurrent.data.datasets[2].data = _mysqlBuf.lampA.slice();
                charts.mysqlCurrent.update('none');
            }
        }

        // ── Socket handler ──
        socket.on('mysql_energy_update', function(data) {
            var ac = data.ac || {};
            var outlet = data.outlet || {};
            var lamp = data.lamp || {};
            
            // PROTEKSI GANDA: Pastikan menjadi kWh, jika masih di atas 1000 (Wh), paksa bagi 1000
            if (ac.energy > 1000) ac.energy = ac.energy / 1000.0;
            if (outlet.energy > 1000) outlet.energy = outlet.energy / 1000.0;
            if (lamp.energy > 1000) lamp.energy = lamp.energy / 1000.0;

            _applyEnergyData(ac, outlet, lamp);
        });

        // ── Direct PHP polling fallback ──
        // If socket sends no data for 10 seconds, browser fetches directly from PHP
        var _PHP_ENERGY_URL = 'https://iotlab-uns.com/api_energy.php?key=iotlab_smartroom_2024';
        var _phpPollErr = 0;
        function _phpDirectPoll() {
            if (Date.now() - _lastEnergyUpdate < 10000) return; // socket still active, skip
            fetch(_PHP_ENERGY_URL)
                .then(function(r) {
                    if (!r.ok) { console.error('[PHP-poll] HTTP ' + r.status); return null; }
                    return r.json();
                })
                .then(function(data) {
                    if (!data) { _phpPollErr++; return; }
                    if (data.error) {
                        _phpPollErr++;
                        console.error('[PHP-poll] Server error:', data.error, data.file + ':' + data.line);
                        return;
                    }
                    _phpPollErr = 0;
                    var ac     = data.ac     || {};
                    var outlet = data.outlet || {};
                    var lamp   = data.lamp   || {};
                    // Calculate pf because PHP does not send pf directly
                    var acAp = parseFloat(ac.apparent_power||0);
                    var outletAp = parseFloat(outlet.apparent_power||0);
                    var lpAp = parseFloat(lamp.apparent_power||0);
                    ac.pf   = acAp > 0 ? parseFloat((parseFloat(ac.active_power||0) / acAp).toFixed(3)) : 0;
                    outlet.pf = outletAp > 0 ? parseFloat((parseFloat(outlet.active_power||0) / outletAp).toFixed(3)) : 0;
                    lamp.pf = lpAp > 0 ? parseFloat((parseFloat(lamp.active_power||0) / lpAp).toFixed(3)) : 0;
                    // Adjust field names: active_power → power, frequency → frequency
                    ac.power       = ac.active_power;
                    // total_energy from DB is actually in Wh, divide by 1000 for kWh
                    ac.energy      = parseFloat(ac.total_energy || 0) / 1000.0;
                    ac.frequency   = ac.frequency;
                    ac.voltage     = ac.voltage;
                    ac.current     = ac.arus;
                    outlet.power     = outlet.active_power;
                    outlet.energy    = parseFloat(outlet.total_energy || 0) / 1000.0;
                    outlet.frequency = outlet.frequency;
                    outlet.voltage   = outlet.voltage;
                    outlet.current   = outlet.arus;
                    lamp.power     = lamp.active_power;
                    lamp.energy    = parseFloat(lamp.total_energy || 0) / 1000.0;
                    lamp.frequency = lamp.frequency;
                    lamp.voltage   = lamp.voltage;
                    lamp.current   = lamp.arus;
                    _applyEnergyData(ac, outlet, lamp);
                })
                .catch(function(e) { _phpPollErr++; console.warn('[PHP-poll] error:', e); });
        }
        setInterval(_phpDirectPoll, 6000);
        // Run immediately 3 seconds after load (no wait for socket)
        setTimeout(_phpDirectPoll, 3000);

        // ==================== SESSION STORAGE + SERVER STATE PERSISTENCE ====================
        function _ssSave(key, val) { try { sessionStorage.setItem(key, JSON.stringify(val)); } catch(e) {} }
        function _ssLoad(key, def) { try { var v = sessionStorage.getItem(key); return v !== null ? JSON.parse(v) : def; } catch(e) { return def; } }
        // Push recording state to server so other devices see badges
        function _recStatePush(obj) {
            fetch('/api/rec/state', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(obj) }).catch(function(){});
        }
        // Poll server state every 5 s to update badges on all devices
        function _recStatePoll() {
            fetch('/api/rec/state').then(function(r){ return r.json(); }).then(function(st) {
                // Only update badge if this device is not the one recording (to avoid overwriting active timer)
                if (!_recActive) {
                    var badge = document.getElementById('rec-status-badge');
                    if (badge) {
                        if (st.energy) {
                            badge.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;animation:blink 1s infinite;"></i> Recording (other device)';
                            badge.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(14,165,233,0.15);color:#0ea5e9;border:1px solid rgba(14,165,233,0.4);';
                        } else { _recUpdateBadge(); }
                    }
                }
                if (!_tempActive) {
                    var tb = document.getElementById('temp-rec-badge');
                    if (tb && st.temp) {
                        tb.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;animation:blink 1s infinite;"></i> Recording (other device)';
                        tb.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(14,165,233,0.15);color:#0ea5e9;border:1px solid rgba(14,165,233,0.4);';
                    } else if (tb && !st.temp) { _tempUpdateBadge(); }
                }
                if (!_luxActive) {
                    var lb = document.getElementById('lux-rec-badge');
                    if (lb && st.lux) {
                        lb.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;animation:blink 1s infinite;"></i> Recording (other device)';
                        lb.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(14,165,233,0.15);color:#0ea5e9;border:1px solid rgba(14,165,233,0.4);';
                    } else if (lb && !st.lux) { _luxUpdateBadge(); }
                }
            }).catch(function(){});
        }
        setInterval(_recStatePoll, 5000);

        // ==================== ENERGY RECORDING & CSV EXPORT ====================
        var _recRows    = _ssLoad('_recRows', []);
        var _recActive  = false;  // never auto-resume active state; user must press start
        var _recStartTs = null;
        var _recTimer   = null;
        var _recFilter  = 'all';  // 'all' | 'AC' | 'Lamp'

        function _recFmt(v, d) { return (parseFloat(v)||0).toFixed(d !== undefined ? d : 1); }

        // ── Server-side data sync ──
        function _recServerPush() {
            fetch('/api/rec/data?type=energy', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rows: _recRows})
            }).then(function(r){ return r.json(); }).then(function(d){
                var b = document.getElementById('rec-sync-badge');
                if (b) { b.innerHTML = '<i class="fas fa-cloud-upload-alt" style="margin-right:4px;font-size:9px;"></i>' + d.count + ' baris'; b.style.color='#3b82f6'; b.style.borderColor='rgba(59,130,246,0.3)'; }
            }).catch(function(){});
        }
        function _recServerFetch() {
            fetch('/api/rec/data?type=energy').then(function(r){ return r.json(); }).then(function(d){
                if (!d.rows) return;
                // Always sync from server when not recording — server is source of truth
                // This fixes stale sessionStorage showing wrong data on other devices
                if (d.rows.length !== _recRows.length ||
                    JSON.stringify(d.rows[d.rows.length-1]) !== JSON.stringify(_recRows[_recRows.length-1])) {
                    _recRows = d.rows; _ssSave('_recRows', _recRows); _recUpdatePreview();
                }
                var b = document.getElementById('rec-sync-badge');
                if (b) { b.innerHTML = '<i class="fas fa-cloud" style="margin-right:4px;font-size:9px;"></i>' + d.count + ' baris'; b.style.color='#2563eb'; }
            }).catch(function(){});
        }
        function _recServerDelete() { fetch('/api/rec/data?type=energy', {method:'DELETE'}).catch(function(){}); }
        // Poll server for rows every 5 s on passive devices (matches badge poll speed)
        setInterval(function(){ if (!_recActive) _recServerFetch(); }, 5000);

        function recSetFilter(f) {
            _recFilter = f;
            var map = {all:'rec-tab-all', AC:'rec-tab-ac', Lamp:'rec-tab-lamp'};
            Object.keys(map).forEach(function(k) {
                var el = document.getElementById(map[k]); if (!el) return;
                if (k === f) { el.style.background='#fff'; el.style.fontWeight='700'; el.style.color=k==='AC'?'#1e40af':k==='Lamp'?'#eab308':'#2563eb'; el.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)'; }
                else { el.style.background='transparent'; el.style.fontWeight='600'; el.style.color='var(--text-secondary)'; el.style.boxShadow='none'; }
            });
            _recUpdatePreview();
        }

        function _recUpdateBadge() {
            var badge = document.getElementById('rec-status-badge');
            if (!badge) return;
            if (_recActive) {
                badge.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;animation:blink 1s infinite;"></i> Recording';
                badge.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(30,64,175,0.15);color:#1e40af;border:1px solid rgba(30,64,175,0.4);';
            } else {
                badge.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;"></i> Idle';
                badge.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(107,114,128,0.15);color:#6b7280;border:1px solid rgba(107,114,128,0.3);';
            }
        }

        function _recUpdatePreview() {
            var tbody = document.getElementById('rec-preview-body');
            var countEl = document.getElementById('rec-count');
            if (countEl) countEl.textContent = _recRows.length;
            if (!tbody) return;
            var filtered = _recFilter === 'all' ? _recRows : _recRows.filter(function(r){ return r.device === _recFilter; });
            if (filtered.length === 0) {
                var msg = _recRows.length > 0 ? 'No data for filter <strong>' + _recFilter + '</strong>.' : 'No data yet. Click <strong>Start Record</strong> to begin.';
                tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;padding:16px;color:var(--text-secondary);font-size:12px;">' + msg + '</td></tr>';
                var rngHide = document.getElementById('rec-date-range'); if (rngHide) rngHide.style.display = 'none';
                return;
            }
            // Tampilkan rentang tanggal
            var rng = document.getElementById('rec-date-range'); var rngTxt = document.getElementById('rec-range-text');
            if (rng && rngTxt) { rng.style.display = ''; rngTxt.textContent = filtered[0].ts + ' \u2014 ' + filtered[filtered.length-1].ts; }
            // Tampilkan 100 baris terbaru (scroll)
            var shown = filtered.slice(-100).reverse();
            var shownEl = document.getElementById('rec-shown-count'); if (shownEl) shownEl.textContent = Math.min(filtered.length, 100);
            tbody.innerHTML = shown.map(function(r) {
                return '<tr style="border-bottom:1px solid var(--border);">' +
                    '<td style="padding:6px 10px;white-space:nowrap;color:var(--text-secondary);">' + r.ts + '</td>' +
                    '<td style="padding:6px 10px;text-align:center;font-weight:600;color:' + (r.device==='AC'?'#1e40af':'#eab308') + ';">' + r.device + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:#2563eb;font-weight:600;">' + r.voltage + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:#0ea5e9;font-weight:600;">' + r.arus + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:#1e40af;font-weight:600;">' + r.daya + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:#3b82f6;font-weight:600;">' + r.energi + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text-secondary);">' + r.reaktif + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text-secondary);">' + r.semu + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text-secondary);">' + r.freq + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text-secondary);">' + r.pf + '</td>' +
                '</tr>';
            }).join('');
        }

        function _recTick() {
            if (!_recStartTs) return;
            var sec = Math.floor((Date.now() - _recStartTs) / 1000);
            var m = Math.floor(sec / 60), s = sec % 60;
            var el = document.getElementById('rec-duration');
            if (el) el.textContent = m + ':' + String(s).padStart(2,'0');
        }

        function _recAddRow(ac, outlet, lamp) {
            if (!_recActive) return;
            var now = new Date();
            var ts  = now.toISOString().replace('T',' ').substring(0,19);
            // AC row
            _recRows.push({
                ts: ts, device: 'AC',
                voltage: _recFmt(ac.voltage || ac.voltage),
                arus:     _recFmt(ac.current || ac.arus, 3),
                daya:     _recFmt(ac.power   || ac.active_power, 2),
                energi:   _recFmt(ac.energy  || ac.total_energy, 4),
                reaktif:  _recFmt(ac.reactive_power, 2),
                semu:     _recFmt(ac.apparent_power, 2),
                freq:     _recFmt(ac.frequency || ac.frequency, 2),
                pf:       _recFmt(ac.pf, 3),
            });
            // Outlet row (MySQL id_kwh=2)
            _recRows.push({
                ts: ts, device: 'Outlet',
                voltage: _recFmt(outlet.voltage || outlet.voltage),
                arus:     _recFmt(outlet.current || outlet.arus, 3),
                daya:     _recFmt(outlet.power   || outlet.active_power, 2),
                energi:   _recFmt(outlet.energy  || outlet.total_energy, 4),
                reaktif:  _recFmt(outlet.reactive_power, 2),
                semu:     _recFmt(outlet.apparent_power, 2),
                freq:     _recFmt(outlet.frequency || outlet.frequency, 2),
                pf:       _recFmt(outlet.pf, 3),
            });
            // Lamp row
            _recRows.push({
                ts: ts, device: 'Lamp',
                voltage: _recFmt(lamp.voltage || lamp.voltage),
                arus:     _recFmt(lamp.current || lamp.arus, 3),
                daya:     _recFmt(lamp.power   || lamp.active_power, 2),
                energi:   _recFmt(lamp.energy  || lamp.total_energy, 4),
                reaktif:  _recFmt(lamp.reactive_power, 2),
                semu:     _recFmt(lamp.apparent_power, 2),
                freq:     _recFmt(lamp.frequency || lamp.frequency, 2),
                pf:       _recFmt(lamp.pf, 3),
            });
            _recUpdatePreview();
            _ssSave('_recRows', _recRows);
            _recServerPush();
        }

        function energyRecStart() {
            _recActive  = true;
            _recStartTs = Date.now();
            var startEl = document.getElementById('rec-start-time');
            if (startEl) startEl.textContent = new Date().toLocaleTimeString('id-ID');
            var btnS = document.getElementById('btn-rec-start');
            var btnE = document.getElementById('btn-rec-stop');
            if (btnS) { btnS.disabled = true; btnS.style.opacity = '0.5'; }
            if (btnE) { btnE.disabled = false; btnE.style.opacity = '1'; }
            _recTimer = setInterval(_recTick, 1000);
            _recUpdateBadge();
            _recStatePush({energy: true});
            showToast('Recording started', 'success');
        }

        function energyRecStop() {
            _recActive = false;
            clearInterval(_recTimer);
            var btnS = document.getElementById('btn-rec-start');
            var btnE = document.getElementById('btn-rec-stop');
            if (btnS) { btnS.disabled = false; btnS.style.opacity = '1'; }
            if (btnE) { btnE.disabled = true; btnE.style.opacity = '0.5'; }
            _recUpdateBadge();
            _recStatePush({energy: false});
            _recServerPush();
            showToast('Recording stopped — ' + _recRows.length + ' rows saved', 'info');
        }

        function energyRecClear() {
            _recRows = [];
            _recActive = false;
            clearInterval(_recTimer);
            sessionStorage.removeItem('_recRows');
            _recServerDelete();
            var countEl = document.getElementById('rec-count');
            var durEl   = document.getElementById('rec-duration');
            var startEl = document.getElementById('rec-start-time');
            if (countEl)  countEl.textContent  = '0';
            if (durEl)    durEl.textContent    = '0:00';
            if (startEl)  startEl.textContent  = '--';
            var btnS2 = document.getElementById('btn-rec-start');
            var btnE2 = document.getElementById('btn-rec-stop');
            if (btnS2) { btnS2.disabled = false; btnS2.style.opacity = '1'; }
            if (btnE2) { btnE2.disabled = true; btnE2.style.opacity = '0.5'; }
            _recUpdateBadge();
            _recUpdatePreview();
            var sb = document.getElementById('rec-sync-badge'); if (sb) { sb.innerHTML = '<i class="fas fa-cloud" style="margin-right:4px;font-size:9px;"></i>Sync'; sb.style.color='#2563eb'; sb.style.borderColor='rgba(37,99,235,0.2)'; }
            showToast('Recording data cleared', 'info');
        }

        function energyExportCSV(device) {
            if (_recRows.length === 0) { showToast('No recorded data yet', 'error'); return; }
            var rows = (device && device !== 'all') ? _recRows.filter(function(r){ return r.device === device; }) : _recRows;
            if (rows.length === 0) { showToast('No data for ' + device, 'error'); return; }
            var header = 'Time,Perangkat,Voltage (V),Current (A),Power Active (W),Energy (kWh),Power Reaktif (VAR),Power Semu (VA),Frequency (Hz),Power Factor\\n';
            var body = rows.map(function(r) {
                return '"' + r.ts + '",' + r.device + ',' + r.voltage + ',' + r.arus + ',' +
                       r.daya + ',' + r.energi + ',' + r.reaktif + ',' + r.semu + ',' + r.freq + ',' + r.pf;
            }).join('\\n');
            var now = new Date();
            var suffix = (device && device !== 'all') ? '_' + device.toLowerCase() : '';
            var fname = 'energy' + suffix + '_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
                        String(now.getDate()).padStart(2,'0') + '_' +
                        String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.csv';
            downloadCSV(fname, header + body);
            showToast(rows.length + ' rows exported: ' + fname, 'success');
        }

        // Hook to _applyEnergyData — record every 6 minutes (360 seconds)
        var _lastRecTime = 0;
        var _REC_INTERVAL_MS = 360000; // 6 minutes
        var _origApply = _applyEnergyData;
        _applyEnergyData = function(ac, outlet, lamp) {
            _origApply(ac, outlet, lamp);
            if (Date.now() - _lastRecTime >= _REC_INTERVAL_MS) {
                _lastRecTime = Date.now();
                _recAddRow(ac, outlet, lamp);
            }
        };

        // ==================== RECORD TEMP & HUMIDITY ====================
        function _tempServerPush() {
            fetch('/api/rec/data?type=temp', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rows: _tempRows})
            }).catch(function(){});
        }
        function _tempServerFetch() {
            fetch('/api/rec/data?type=temp').then(function(r){ return r.json(); }).then(function(d){
                if (!d.rows) return;
                if (d.rows.length !== _tempRows.length ||
                    JSON.stringify(d.rows[d.rows.length-1]) !== JSON.stringify(_tempRows[_tempRows.length-1])) {
                    _tempRows = d.rows; _ssSave('_tempRows', _tempRows); _tempUpdatePreview();
                    var c = document.getElementById('temp-rec-count'); if (c) c.textContent = _tempRows.length;
                }
            }).catch(function(){});
        }
        function _tempServerDelete() { fetch('/api/rec/data?type=temp', {method:'DELETE'}).catch(function(){}); }
        setInterval(function(){ if (!_tempActive) _tempServerFetch(); }, 5000);

        var _tempRows = _ssLoad('_tempRows', []), _tempActive = false, _tempStartTs = 0;
        var _tempTimer = null, _lastTempRecTime = 0;
        var _TEMP_REC_MS = 60000; // 1 minute

        function _tempFmt(v, d) { var n = parseFloat(v); return Number.isFinite(n) ? n.toFixed(d != null ? d : 1) : '--'; }

        function _tempAddRow() {
            var ts = new Date().toLocaleString('id-ID');
            var temp = document.getElementById('dash-ac-room-temp') ? document.getElementById('dash-ac-room-temp').textContent : '--';
            var hum  = document.getElementById('dash-ac-room-hum')  ? document.getElementById('dash-ac-room-hum').textContent  : '--';
            var t1   = document.getElementById('dash-ac-temp1')      ? document.getElementById('dash-ac-temp1').textContent      : '--';
            var t2   = document.getElementById('dash-ac-temp2')      ? document.getElementById('dash-ac-temp2').textContent      : '--';
            var t3   = document.getElementById('dash-ac-temp3')      ? document.getElementById('dash-ac-temp3').textContent      : '--';
            var setT = document.getElementById('dash-ac-temp')       ? document.getElementById('dash-ac-temp').textContent       : '--';
            var fan  = document.getElementById('dash-ac-fan-label')  ? document.getElementById('dash-ac-fan-label').textContent  : '--';
            var mode = document.getElementById('dash-ac-mode')       ? document.getElementById('dash-ac-mode').textContent       : '--';
            var ctrl = document.getElementById('dash-ac-ctrl-mode')  ? document.getElementById('dash-ac-ctrl-mode').textContent  : '--';
            _tempRows.push({ ts: ts, temp: temp, hum: hum, t1: t1, t2: t2, t3: t3, setT: setT, fan: fan, mode: mode, ctrl: ctrl });
            _tempUpdatePreview();
            var c = document.getElementById('temp-rec-count');
            if (c) c.textContent = _tempRows.length;
            _ssSave('_tempRows', _tempRows);
            _tempServerPush();
        }

        function _tempTick() {
            if (!_tempActive) return;
            var sec = Math.floor((Date.now() - _tempStartTs) / 1000);
            var m = Math.floor(sec / 60), s = sec % 60;
            var d = document.getElementById('temp-rec-duration');
            if (d) d.textContent = m + ':' + String(s).padStart(2, '0');
            if (Date.now() - _lastTempRecTime >= _TEMP_REC_MS) {
                _lastTempRecTime = Date.now();
                _tempAddRow();
            }
        }

        function _tempUpdateBadge() {
            var b = document.getElementById('temp-rec-badge');
            if (!b) return;
            if (_tempActive) {
                b.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;color:#1e40af;animation:pulse 1s infinite;"></i> Recording...';
                b.style.background = 'rgba(30,64,175,0.12)'; b.style.color = '#1e40af'; b.style.borderColor = 'rgba(30,64,175,0.3)';
            } else {
                b.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;"></i> Idle';
                b.style.background = 'rgba(107,114,128,0.15)'; b.style.color = '#6b7280'; b.style.borderColor = 'rgba(107,114,128,0.3)';
            }
        }

        function _tempUpdatePreview() {
            var tbody = document.getElementById('temp-rec-preview');
            if (!tbody) return;
            if (_tempRows.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:16px;color:var(--text-secondary);">No data yet. Click <strong>Start Record</strong> to begin.</td></tr>';
                return;
            }
            // Tampilkan 100 baris terbaru (scroll)
            var shownT = document.getElementById('temp-shown-count'); if (shownT) shownT.textContent = Math.min(_tempRows.length, 100);
            var last100 = _tempRows.slice(-100).reverse();
            tbody.innerHTML = last100.map(function(r, i) {
                var bg = i % 2 === 0 ? 'background:rgba(59,130,246,0.03);' : '';
                return '<tr style="' + bg + '"><td style="padding:6px 10px;color:var(--text-secondary);white-space:nowrap;">' + r.ts + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#3b82f6;">' + r.temp + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#0ea5e9;">' + r.hum + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text);">' + r.t1 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text);">' + r.t2 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text);">' + r.t3 + '</td></tr>';
            }).join('');
        }

        function tempRecStart() {
            _tempActive = true; _tempStartTs = Date.now(); _lastTempRecTime = 0;
            var s = document.getElementById('temp-rec-start'); if (s) s.textContent = new Date().toLocaleTimeString('id-ID');
            var tS = document.getElementById('btn-temp-start'), tE = document.getElementById('btn-temp-stop');
            if (tS) { tS.disabled = true; tS.style.opacity = '0.5'; }
            if (tE) { tE.disabled = false; tE.style.opacity = '1'; }
            _tempTimer = setInterval(_tempTick, 1000);
            _tempUpdateBadge();
            _recStatePush({temp: true});
            showToast('Temperature & humidity recording started', 'success');
        }

        function tempRecStop() {
            _tempActive = false; clearInterval(_tempTimer);
            var tS = document.getElementById('btn-temp-start'), tE = document.getElementById('btn-temp-stop');
            if (tS) { tS.disabled = false; tS.style.opacity = '1'; }
            if (tE) { tE.disabled = true; tE.style.opacity = '0.5'; }
            _tempUpdateBadge();
            _recStatePush({temp: false});
            _tempServerPush();
            showToast('Recording stopped — ' + _tempRows.length + ' rows saved', 'info');
        }

        function tempRecClear() {
            _tempRows = []; _tempActive = false; clearInterval(_tempTimer);
            sessionStorage.removeItem('_tempRows');
            _tempServerDelete();
            ['temp-rec-count','temp-rec-duration','temp-rec-start'].forEach(function(id) { var e = document.getElementById(id); if (e) e.textContent = id.includes('count') ? '0' : id.includes('duration') ? '0:00' : '--'; });
            var tS2 = document.getElementById('btn-temp-start'), tE2 = document.getElementById('btn-temp-stop');
            if (tS2) { tS2.disabled = false; tS2.style.opacity = '1'; }
            if (tE2) { tE2.disabled = true; tE2.style.opacity = '0.5'; }
            _tempUpdateBadge(); _tempUpdatePreview();
            showToast('Temperature data cleared', 'info');
        }

        function tempExportCSV() {
            if (_tempRows.length === 0) { showToast('No recorded data yet', 'error'); return; }
            var header = 'Time,Temp Rata2 (C),Humidity (%),Sensor T1 (C),Sensor T2 (C),Sensor T3 (C),Set Temp (C),Fan Speed,Mode AC,Control\\n';
            var body = _tempRows.map(function(r) {
                return '"' + r.ts + '",' + r.temp + ',' + r.hum + ',' + r.t1 + ',' + r.t2 + ',' + r.t3 + ',' + r.setT + ',"' + r.fan + '",' + r.mode + ',' + r.ctrl;
            }).join('\\n');
            var now = new Date();
            var fname = 'temp_humidity_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
                        String(now.getDate()).padStart(2,'0') + '_' +
                        String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.csv';
            downloadCSV(fname, header + body);
            showToast(_tempRows.length + ' rows exported: ' + fname, 'success');
        }

        // ==================== REKAM LUX & BRIGHTNESS ====================
        function _luxServerPush() {
            fetch('/api/rec/data?type=lux', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rows: _luxRows})
            }).catch(function(){});
        }
        function _luxServerFetch() {
            fetch('/api/rec/data?type=lux').then(function(r){ return r.json(); }).then(function(d){
                if (!d.rows) return;
                if (d.rows.length !== _luxRows.length ||
                    JSON.stringify(d.rows[d.rows.length-1]) !== JSON.stringify(_luxRows[_luxRows.length-1])) {
                    _luxRows = d.rows; _ssSave('_luxRows', _luxRows); _luxUpdatePreview();
                    var c = document.getElementById('lux-rec-count'); if (c) c.textContent = _luxRows.length;
                }
            }).catch(function(){});
        }
        function _luxServerDelete() { fetch('/api/rec/data?type=lux', {method:'DELETE'}).catch(function(){}); }
        setInterval(function(){ if (!_luxActive) _luxServerFetch(); }, 5000);

        var _luxRows = _ssLoad('_luxRows', []), _luxActive = false, _luxStartTs = 0;
        var _luxTimer = null, _lastLuxRecTime = 0;
        var _LUX_REC_MS = 60000; // 1 minute

        function _luxAddRow() {
            var ts = new Date().toLocaleString('id-ID');
            var get = function(id) { var e = document.getElementById(id); return e ? e.textContent : '--'; };
            _luxRows.push({
                ts: ts,
                lux1: get('dash-lux1'), lux2: get('dash-lux2'), lux3: get('dash-lux3'),
                avg: get('dash-lux-avg'), b1: get('dash-bright1'), b2: get('dash-bright2')
            });
            _luxUpdatePreview();
            var c = document.getElementById('lux-rec-count');
            if (c) c.textContent = _luxRows.length;
            _ssSave('_luxRows', _luxRows);
            _luxServerPush();
        }

        function _luxTick() {
            if (!_luxActive) return;
            var sec = Math.floor((Date.now() - _luxStartTs) / 1000);
            var m = Math.floor(sec / 60), s = sec % 60;
            var d = document.getElementById('lux-rec-duration');
            if (d) d.textContent = m + ':' + String(s).padStart(2, '0');
            if (Date.now() - _lastLuxRecTime >= _LUX_REC_MS) {
                _lastLuxRecTime = Date.now();
                _luxAddRow();
            }
        }

        function _luxUpdateBadge() {
            var b = document.getElementById('lux-rec-badge');
            if (!b) return;
            if (_luxActive) {
                b.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;color:#1e40af;animation:pulse 1s infinite;"></i> Recording...';
                b.style.background = 'rgba(30,64,175,0.12)'; b.style.color = '#1e40af'; b.style.borderColor = 'rgba(30,64,175,0.3)';
            } else {
                b.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;"></i> Idle';
                b.style.background = 'rgba(107,114,128,0.15)'; b.style.color = '#6b7280'; b.style.borderColor = 'rgba(107,114,128,0.3)';
            }
        }

        function _luxUpdatePreview() {
            var tbody = document.getElementById('lux-rec-preview');
            if (!tbody) return;
            if (_luxRows.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:16px;color:var(--text-secondary);">No data yet. Click <strong>Start Record</strong> to begin.</td></tr>';
                return;
            }
            // Tampilkan 100 baris terbaru (scroll)
            var shownL = document.getElementById('lux-shown-count'); if (shownL) shownL.textContent = Math.min(_luxRows.length, 100);
            var last100 = _luxRows.slice(-100).reverse();
            tbody.innerHTML = last100.map(function(r, i) {
                var bg = i % 2 === 0 ? 'background:rgba(234,179,8,0.03);' : '';
                return '<tr style="' + bg + '"><td style="padding:6px 10px;color:var(--text-secondary);white-space:nowrap;">' + r.ts + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#eab308;">' + r.lux1 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#eab308;">' + r.lux2 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#eab308;">' + r.lux3 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#0ea5e9;">' + r.avg + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#fbbf24;">' + r.b1 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#fbbf24;">' + r.b2 + '</td></tr>';
            }).join('');
        }

        function luxRecStart() {
            _luxActive = true; _luxStartTs = Date.now(); _lastLuxRecTime = 0;
            var s = document.getElementById('lux-rec-start'); if (s) s.textContent = new Date().toLocaleTimeString('id-ID');
            var lS = document.getElementById('btn-lux-start'), lE = document.getElementById('btn-lux-stop');
            if (lS) { lS.disabled = true; lS.style.opacity = '0.5'; }
            if (lE) { lE.disabled = false; lE.style.opacity = '1'; }
            _luxTimer = setInterval(_luxTick, 1000);
            _luxUpdateBadge();
            _recStatePush({lux: true});
            showToast('Lux & brightness recording started', 'success');
        }

        function luxRecStop() {
            _luxActive = false; clearInterval(_luxTimer);
            var lS = document.getElementById('btn-lux-start'), lE = document.getElementById('btn-lux-stop');
            if (lS) { lS.disabled = false; lS.style.opacity = '1'; }
            if (lE) { lE.disabled = true; lE.style.opacity = '0.5'; }
            _luxUpdateBadge();
            _recStatePush({lux: false});
            _luxServerPush();
            showToast('Recording stopped — ' + _luxRows.length + ' rows saved', 'info');
        }

        function luxRecClear() {
            _luxRows = []; _luxActive = false; clearInterval(_luxTimer);
            sessionStorage.removeItem('_luxRows');
            _luxServerDelete();
            ['lux-rec-count','lux-rec-duration','lux-rec-start'].forEach(function(id) { var e = document.getElementById(id); if (e) e.textContent = id.includes('count') ? '0' : id.includes('duration') ? '0:00' : '--'; });
            var lS2 = document.getElementById('btn-lux-start'), lE2 = document.getElementById('btn-lux-stop');
            if (lS2) { lS2.disabled = false; lS2.style.opacity = '1'; }
            if (lE2) { lE2.disabled = true; lE2.style.opacity = '0.5'; }
            _luxUpdateBadge(); _luxUpdatePreview();
            showToast('Lux data cleared', 'info');
        }

        function luxExportCSV() {
            if (_luxRows.length === 0) { showToast('No recorded data yet', 'error'); return; }
            var header = 'Time,Lux 1 (lx),Lux 2 (lx),Lux 3 (lx),Avg Lux (lx),Brightness 1 (%),Brightness 2 (%)\\n';
            var body = _luxRows.map(function(r) {
                return '"' + r.ts + '",' + r.lux1 + ',' + r.lux2 + ',' + r.lux3 + ',' + r.avg + ',' + r.b1 + ',' + r.b2;
            }).join('\\n');
            var now = new Date();
            var fname = 'lux_brightness_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
                        String(now.getDate()).padStart(2,'0') + '_' +
                        String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.csv';
            downloadCSV(fname, header + body);
            showToast(_luxRows.length + ' rows exported: ' + fname, 'success');
        }

        // ==================== OCCUPANCY DAILY RECORDING ====================
        var _occRows    = _ssLoad('_occRows', []);   // {ts, hour, count, conf}
        var _occLastHour = -1;                        // last recorded hour (-1 = none yet)
        var _occRangeFilter = 'today';                // range: 'today'|'7d'|'30d'|'all'
        var _OCC_REC_LABEL = Array.from({length:24}, function(_,i){ return String(i).padStart(2,'0') + ':00'; });
        var _occChart   = null;

        // Restore last recorded hour from session
        if (_occRows.length > 0) {
            _occLastHour = _occRows[_occRows.length - 1].hour;
        }

        function _occInitChart() {
            var ctx = document.getElementById('occChart');
            if (!ctx || _occChart) return;
            _occChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: _OCC_REC_LABEL,
                    datasets: [{
                        label: 'Person Count',
                        data: new Array(24).fill(null),
                        backgroundColor: 'rgba(37,99,235,0.5)',
                        borderColor: '#2563eb',
                        borderWidth: 1,
                        borderRadius: 5,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: {
                        callbacks: { label: function(c) { return c.raw !== null ? c.raw + ' persons' : 'Not recorded'; } }
                    }},
                    scales: {
                        x: { ticks: { font: {size:10}, color: 'rgba(100,116,139,0.9)', maxRotation: 0 } },
                        y: { beginAtZero: true, ticks: { font: {size:10}, color: 'rgba(100,116,139,0.9)', stepSize: 1 },
                             title: { display: true, text: 'Person Count', font: {size:10}, color: '#2563eb' } }
                    }
                }
            });
            _occRefreshChart();
        }

        function _occGetFilteredRows() {
            var now = new Date();
            if (_occRangeFilter === 'today') {
                var today = now.toISOString().substring(0,10);
                return _occRows.filter(function(r){ return r.ts.substring(0,10) === today; });
            } else if (_occRangeFilter === '7d') {
                var cutoff = new Date(now - 7*24*60*60*1000).toISOString().substring(0,10);
                return _occRows.filter(function(r){ return r.ts.substring(0,10) >= cutoff; });
            } else if (_occRangeFilter === '30d') {
                var cutoff = new Date(now - 30*24*60*60*1000).toISOString().substring(0,10);
                return _occRows.filter(function(r){ return r.ts.substring(0,10) >= cutoff; });
            }
            return _occRows.slice();
        }

        function occSetRange(r) {
            _occRangeFilter = r;
            ['today','7d','30d','all'].forEach(function(t) {
                var btn = document.getElementById('occ-tab-' + t);
                if (!btn) return;
                if (t === r) {
                    btn.style.background = '#2563eb'; btn.style.color = '#fff'; btn.style.border = 'none';
                    btn.style.boxShadow = '0 1px 4px rgba(37,99,235,0.3)'; btn.style.fontWeight = '700';
                } else {
                    btn.style.background = 'rgba(37,99,235,0.12)'; btn.style.color = '#2563eb';
                    btn.style.border = '1px solid rgba(37,99,235,0.25)'; btn.style.boxShadow = ''; btn.style.fontWeight = '600';
                }
            });
            _occRefreshChart();
        }

        function _occRefreshChart() {
            if (!_occChart) return;
            var filtered = _occGetFilteredRows();
            var total = filtered.reduce(function(s,r){ return s + (r.count||0); }, 0);
            var totalEl = document.getElementById('occ-total-today'); if (totalEl) totalEl.textContent = total;
            var countEl = document.getElementById('occ-count'); if (countEl) countEl.textContent = filtered.length;
            if (_occRangeFilter === 'today') {
                var data = new Array(24).fill(null);
                filtered.forEach(function(r){ if (r.hour >= 0 && r.hour < 24) data[r.hour] = r.count; });
                _occChart.data.labels = _OCC_REC_LABEL;
                _occChart.data.datasets[0].data = data;
                _occChart.data.datasets[0].label = 'Person Count (per hour)';
            } else {
                var dayMap = {};
                filtered.forEach(function(r){ var d = r.ts.substring(0,10); dayMap[d] = (dayMap[d]||0) + (r.count||0); });
                var days = Object.keys(dayMap).sort();
                _occChart.data.labels = days;
                _occChart.data.datasets[0].data = days.map(function(d){ return dayMap[d]; });
                _occChart.data.datasets[0].label = 'Total Persons per Day';
            }
            _occChart.update('none');
            _occUpdatePreview();
        }

        function _occUpdatePreview() {
            var tbody = document.getElementById('occ-preview-body');
            if (!tbody) return;
            var filtered = _occGetFilteredRows();
            if (filtered.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:14px;color:var(--text-secondary);">No data for this range&hellip;</td></tr>';
                return;
            }
            tbody.innerHTML = filtered.slice().reverse().map(function(r) {
                return '<tr style="border-bottom:1px solid var(--border);">' +
                    '<td style="padding:6px 12px;color:var(--text-secondary);">' + r.ts + '</td>' +
                    '<td style="padding:6px 12px;text-align:center;font-weight:700;color:#2563eb;">' + String(r.hour).padStart(2,'0') + ':00</td>' +
                    '<td style="padding:6px 12px;text-align:right;font-weight:700;color:#3b82f6;">' + r.count + '</td>' +
                    '<td style="padding:6px 12px;text-align:right;color:var(--text-secondary);">' + r.conf + '</td>' +
                '</tr>';
            }).join('');
        }

        function _occTryRecord() {
            var now = new Date();
            var h = now.getHours();
            var today = now.toISOString().substring(0, 10);
            var dateHour = today + ' ' + String(h).padStart(2, '0');
            // Already recorded for this hour+date?
            if (_occRows.some(function(r){ return r.ts.substring(0,13) === dateHour; })) return;
            // Prune data older than 30 days
            var cutoff = new Date(now - 30*24*60*60*1000).toISOString().substring(0,10);
            _occRows = _occRows.filter(function(r){ return r.ts.substring(0,10) >= cutoff; });
            var countEl = document.getElementById('cam-count-display');
            var confEl  = document.getElementById('cam-confidence-display');
            var count   = parseInt((countEl ? countEl.textContent : '0') || '0', 10);
            var confStr = confEl ? confEl.textContent.replace('%','').trim() : '0';
            var conf    = parseFloat(confStr) || 0;
            var ts = now.toISOString().replace('T',' ').substring(0,19);
            _occRows.push({ ts: ts, hour: h, count: count, conf: conf.toFixed(1) + '%' });
            _occLastHour = h;
            _ssSave('_occRows', _occRows);
            _occRefreshChart();
            showToast('Occupancy at ' + String(h).padStart(2,'0') + ':00 recorded — ' + count + ' persons', 'info');
        }

        function occClear() {
            _occRows = []; _occLastHour = -1;
            sessionStorage.removeItem('_occRows');
            if (_occChart) { _occChart.data.datasets[0].data = new Array(24).fill(null); _occChart.update('none'); }
            _occUpdatePreview();
            var countEl = document.getElementById('occ-count'); if (countEl) countEl.textContent = '0';
            var totalEl = document.getElementById('occ-total-today'); if (totalEl) totalEl.textContent = '0';
            showToast('Occupancy data cleared', 'info');
        }

        function occExportCSV() {
            if (_occRows.length === 0) { showToast('No occupancy data yet', 'error'); return; }
            var header = 'Time,Hour,Person Count,Confidence\\n';
            var body = _occRows.map(function(r) {
                return '"' + r.ts + '",' + String(r.hour).padStart(2,'0') + ':00,' + r.count + ',' + r.conf;
            }).join('\\n');
            var now = new Date();
            var fname = 'occupancy_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
                        String(now.getDate()).padStart(2,'0') + '.csv';
            downloadCSV(fname, header + body);
            showToast(_occRows.length + ' data exported: ' + fname, 'success');
        }

        // Run occupancy check every 60 seconds; reset and record new hour
        setInterval(function() {
            try { _occTryRecord(); } catch(e) {}
        }, 60000);
        // Update "next record in" countdown every second
        setInterval(function() {
            try {
                var now = new Date();
                var minsLeft = 59 - now.getMinutes();
                var secsLeft = 59 - now.getSeconds();
                var el = document.getElementById('occ-next-in');
                if (el) el.textContent = minsLeft + ' min ' + String(secsLeft).padStart(2,'0') + ' sec';
            } catch(e) {}
        }, 1000);
        // Also init chart once DOM is ready (delayed)
        setTimeout(function() {
            try { _occInitChart(); _occRefreshChart(); } catch(e) {}
        }, 500);

        // ==================== MQTT REAL-TIME UPDATE HANDLER ====================
        socket.on('mqtt_update', function(data) {
            updateDashboard();
            
            if (data.type === 'ac') {
                const now = new Date();
                const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
                
                if (charts.temp && charts.temp.data.labels.length < 50) {
                    charts.temp.data.labels.push(timeStr);
                    charts.temp.data.datasets[0].data.push(data.data.temperature);
                    charts.temp.update('none');
                }
                if (charts.hum && charts.hum.data.labels.length < 50) {
                    charts.hum.data.labels.push(timeStr);
                    charts.hum.data.datasets[0].data.push(data.data.humidity);
                    charts.hum.update('none');
                }
            }
            
            if (data.type === 'lamp') {
                const now = new Date();
                const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
                const lamp = data.data;
                const luxAvg    = parseFloat(lamp.lux_avg)    || 0;
                const brightAvg = parseFloat(lamp.brightness_avg) || 0;
                // local setText — does not depend on outer scope
                const _set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };

                // Update Lamp Dashboard lux dan brightness secara real-time
                _set('dash-lux1',    (lamp.lux1    != null ? parseFloat(lamp.lux1)    : 0).toFixed(0));
                _set('dash-lux2',    (lamp.lux2    != null ? parseFloat(lamp.lux2)    : 0).toFixed(0));
                _set('dash-lux3',    (lamp.lux3    != null ? parseFloat(lamp.lux3)    : 0).toFixed(0));
                _set('dash-lux-avg', luxAvg.toFixed(1));
                _set('ctrl-lux1',    (lamp.lux1    != null ? parseFloat(lamp.lux1)    : 0).toFixed(0));
                _set('ctrl-lux2',    (lamp.lux2    != null ? parseFloat(lamp.lux2)    : 0).toFixed(0));
                _set('ctrl-lux3',    (lamp.lux3    != null ? parseFloat(lamp.lux3)    : 0).toFixed(0));
                _set('dash-bright1', Math.round(parseFloat(lamp.brightness1) || 0));
                _set('dash-bright2', Math.round(parseFloat(lamp.brightness2) || 0));

                if (charts.lampLux) {
                    if (charts.lampLux.data.labels.length >= 50) {
                        charts.lampLux.data.labels.shift();
                        charts.lampLux.data.datasets[0].data.shift();
                    }
                    charts.lampLux.data.labels.push(timeStr);
                    charts.lampLux.data.datasets[0].data.push(luxAvg);
                    charts.lampLux.update('none');
                }
                if (charts.lampBright) {
                    if (charts.lampBright.data.labels.length >= 50) {
                        charts.lampBright.data.labels.shift();
                        charts.lampBright.data.datasets[0].data.shift();
                    }
                    charts.lampBright.data.labels.push(timeStr);
                    charts.lampBright.data.datasets[0].data.push(Math.round(brightAvg));
                    charts.lampBright.update('none');
                }
            }
            
            if (data.type === 'energy') {
                const e = data.data;
                // Tampilkan bubble notifikasi jika ada daya
                const pwr = parseFloat(e.power || 0);
                const vlt = parseFloat(e.voltage || 0);
                const cur = parseFloat(e.current || 0);
                if (pwr > 0 || vlt > 0) {
                    showEnergyBubble(pwr, vlt, cur);
                }
            }

            // Real-time optimization (GA->AC / PSO->Lamp) updates
            if (data.type === 'system') {
                const gaFitness = parseFloat(data.data.ga_fitness) || 0;
                const psoFitness = parseFloat(data.data.pso_fitness) || 0;
                const runs = data.data.optimization_runs || 0;
                const gaTemp = data.data.ga_temp || 0;
                const gaFan = data.data.ga_fan || 0;
                // PWM 0-255 values (actual algorithm output)
                const psoPwm1 = data.data.pso_pwm1 != null ? data.data.pso_pwm1 : 0;
                const psoPwm2 = data.data.pso_pwm2 != null ? data.data.pso_pwm2 : 0;
                // Brightness % values (for MQTT/ESP32)
                const psoBrightness1 = data.data.pso_brightness1 != null ? data.data.pso_brightness1 : 0;
                const psoBrightness2 = data.data.pso_brightness2 != null ? data.data.pso_brightness2 : 0;
                
                // Update GA fitness display
                const gaEl = document.getElementById('ga-fitness');
                if (gaEl) {
                    gaEl.textContent = gaFitness.toFixed(2);
                    gaEl.style.color = gaFitness > 0 ? '#3b82f6' : '#94a3b8';
                    if (gaFitness > 0) gaEl.style.textShadow = '0 0 10px rgba(59, 130, 246, 0.5)';
                }
                
                // Update GA solution details (AC settings)
                const gaTempEl = document.getElementById('ga-temp');
                const gaFanEl = document.getElementById('ga-fan');
                if (gaTempEl && gaTemp > 0) gaTempEl.textContent = gaTemp;
                if (gaFanEl && gaFan > 0) gaFanEl.textContent = gaFan;
                
                // Update PSO fitness display
                const psoEl = document.getElementById('pso-fitness');
                if (psoEl) {
                    psoEl.textContent = psoFitness.toFixed(2);
                    psoEl.style.color = psoFitness > 0 ? '#0ea5e9' : '#94a3b8';
                    if (psoFitness > 0) psoEl.style.textShadow = '0 0 10px rgba(14, 165, 233, 0.5)';
                }
                
                console.log('[ML] Optimization Update:', {
                    ga: gaFitness.toFixed(2), ac_temp: gaTemp, ac_fan: gaFan,
                    pso: psoFitness.toFixed(2), pwm1: psoPwm1, pwm2: psoPwm2,
                    b1: psoBrightness1 + '%', b2: psoBrightness2 + '%', runs: runs
                });
                
                // Show toast with actual PWM values
                if (gaFitness > 0 && psoFitness > 0) {
                    showToast('GA->AC: ' + gaTemp + '°C (' + gaFitness.toFixed(1) + ') | PSO->Lamp: PWM1=' + psoPwm1 + '/255 PWM2=' + psoPwm2 + '/255 (err=' + psoFitness.toFixed(1) + ')', 'success');
                } else if (gaFitness > 0) {
                    showToast('GA->AC: ' + gaTemp + '°C Fan:' + gaFan + ' (Fitness: ' + gaFitness.toFixed(2) + ')', 'success');
                } else if (psoFitness > 0) {
                    showToast('PSO->Lamp: PWM1=' + psoPwm1 + '/255 PWM2=' + psoPwm2 + '/255 (err=' + psoFitness.toFixed(2) + ')', 'success');
                }

                // === Update ML Optimization Page ===
                updateMLDisplay(data.data);
                if (gaFitness > 0 || psoFitness > 0) {
                    addToComparisonChart(gaFitness, psoFitness);
                    addMLHistoryRow({
                        ga_fitness: gaFitness, ga_temp: gaTemp, ga_fan: gaFan,
                        pso_fitness: psoFitness, pso_brightness: Math.round((psoPwm1 + psoPwm2) / 2)
                    });
                }
                // Update GA/PSO convergence charts from history arrays
                if (data.data.ga_history && data.data.ga_history.length > 0) {
                    updateMLChart('gaFitness', data.data.ga_history, 'GA');
                }
                if (data.data.pso_history && data.data.pso_history.length > 0) {
                    updateMLChart('psoFitness', data.data.pso_history, 'PSO');
                }
            }
            
            // Real-time camera/person detection update
            if (data.type === 'camera') {
                const personCount = data.data.count || 0;
                const confidence = data.data.confidence || 0;
                const personDetected = data.data.person_detected || false;
                
                // Update count display with color coding
                const countEl = document.getElementById('cam-count');
                const countDisplayEl = document.getElementById('cam-count-display');
                if (countEl) {
                    countEl.textContent = personCount;
                    if (personCount > 0) {
                        countEl.style.color = '#3b82f6';
                        countEl.style.textShadow = '0 0 10px rgba(59, 130, 246, 0.5)';
                    } else {
                        countEl.style.color = '#94a3b8';
                        countEl.style.textShadow = 'none';
                    }
                }
                if (countDisplayEl) {
                    countDisplayEl.textContent = personCount;
                    countDisplayEl.style.color = personCount > 0 ? '#3b82f6' : '#94a3b8';
                    if (personCount > 0) {
                        countDisplayEl.parentElement.style.background = 'rgba(59, 130, 246, 0.1)';
                        countDisplayEl.parentElement.style.borderColor = '#3b82f6';
                    } else {
                        countDisplayEl.parentElement.style.background = '';
                        countDisplayEl.parentElement.style.borderColor = 'var(--border)';
                    }
                }
                
                // Update confidence with color coding
                const confEl = document.getElementById('cam-confidence');
                const confDisplayEl = document.getElementById('cam-confidence-display');
                const confText = confidence + '%';
                
                [confEl, confDisplayEl].forEach(el => {
                    if (el) {
                        el.textContent = confText;
                        if (confidence > 70) {
                            el.style.color = '#3b82f6';
                        } else if (confidence > 50) {
                            el.style.color = '#0ea5e9';
                        } else {
                            el.style.color = '#1e40af';
                        }
                    }
                });
                
                if (confDisplayEl && personCount > 0) {
                    confDisplayEl.parentElement.style.background = 'rgba(59, 130, 246, 0.1)';
                    confDisplayEl.parentElement.style.borderColor = '#3b82f6';
                } else if (confDisplayEl) {
                    confDisplayEl.parentElement.style.background = '';
                    confDisplayEl.parentElement.style.borderColor = 'var(--border)';
                }
                
                // Update person detected card
                const personEl = document.getElementById('cam-person');
                const personCard = document.getElementById('person-detected-card');
                if (personEl) {
                    personEl.textContent = personDetected ? 'Yes' : 'No';
                    personEl.style.color = personDetected ? '#3b82f6' : '#1e40af';
                }
                if (personCard) {
                    if (personDetected) {
                        personCard.style.background = 'rgba(59, 130, 246, 0.1)';
                        personCard.style.borderColor = '#3b82f6';
                    } else {
                        personCard.style.background = '';
                        personCard.style.borderColor = 'var(--border)';
                    }
                }
                
                // Update overlay badge
                const overlayBadge = document.getElementById('overlay-person-badge');
                if (overlayBadge) {
                    if (personDetected && personCount > 0) {
                        overlayBadge.className = 'person-badge detected';
                        overlayBadge.innerHTML = personCount + ' Person(s) - ' + confidence + '%';
                    } else {
                        overlayBadge.className = 'person-badge not-detected';
                        overlayBadge.innerHTML = 'No Person';
                    }
                }
                
                // Show alert when person detected
                if (personDetected && personCount > 0) {
                    showDetectionAlert(personCount, confidence);
                }
                
                // Update "Last Person Detected" display
                const lastSeenEl = document.getElementById('cam-last-seen');
                const lastSeenLabel = document.getElementById('cam-last-seen-label');
                const lastSeenAgo = data.data.last_seen_ago;
                if (lastSeenEl) {
                    if (personDetected && personCount > 0) {
                        lastSeenEl.textContent = 'Sekarang';
                        lastSeenEl.style.color = '#3b82f6';
                        if (lastSeenLabel) lastSeenLabel.textContent = 'Person currently detected';
                    } else if (lastSeenAgo !== undefined && lastSeenAgo >= 0) {
                        const mins = Math.floor(lastSeenAgo / 60);
                        const secs = lastSeenAgo % 60;
                        lastSeenEl.textContent = mins > 0 ? (mins + 'm ' + secs + 's lalu') : (secs + 's lalu');
                        lastSeenEl.style.color = lastSeenAgo > 300 ? '#1e40af' : '#2563eb';
                        if (lastSeenLabel) lastSeenLabel.textContent = 'Since last detection';
                    } else {
                        lastSeenEl.textContent = '--';
                        lastSeenEl.style.color = '#94a3b8';
                        if (lastSeenLabel) lastSeenLabel.textContent = 'Never detected';
                    }
                }
                
                // Update "Auto-OFF AC" countdown
                const autoOffEl = document.getElementById('cam-auto-off-timer');
                const autoOffLabel = document.getElementById('cam-auto-off-label');
                const autoOffIn = data.data.auto_off_in;
                const autoOffTriggered = data.data.auto_off_triggered;
                if (autoOffEl) {
                    if (autoOffTriggered) {
                        autoOffEl.textContent = 'OFF';
                        autoOffEl.style.color = '#1e40af';
                        if (autoOffLabel) autoOffLabel.textContent = 'AC already auto-OFF';
                    } else if (autoOffIn !== undefined && autoOffIn >= 0 && !personDetected) {
                        const offMins = Math.floor(autoOffIn / 60);
                        const offSecs = autoOffIn % 60;
                        autoOffEl.textContent = offMins + 'm ' + offSecs + 's';
                        autoOffEl.style.color = autoOffIn < 120 ? '#1e40af' : '#0ea5e9';
                        if (autoOffLabel) autoOffLabel.textContent = 'Countdown auto-OFF AC';
                    } else {
                        autoOffEl.textContent = '--';
                        autoOffEl.style.color = '#3b82f6';
                        if (autoOffLabel) autoOffLabel.textContent = 'Person detected, AC safe';
                    }
                }
                
                console.log('[CAM] Camera Update:', {
                    count: personCount,
                    confidence: confidence,
                    detected: personDetected
                });
            }
        });

        // ML Optimization status from main.py
        socket.on('ml_status', function(data) {
            console.log('[ML] ML Status:', data);
            const status = data.status || '';
            const algo = (data.algorithm || '').toUpperCase();
            
            if (status === 'running') {
                showToast(algo + ' optimization running...', 'success');
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                });
                // Reset PSO iteration table when new run starts
                if (algo === 'PSO') {
                    const tbody = document.getElementById('pso-iter-tbody');
                    if (tbody) tbody.innerHTML = '';
                    const chart = charts.psoFitness;
                    if (chart) {
                        chart.data.labels = [];
                        chart.data.datasets.forEach(ds => ds.data = []);
                        chart.update('none');
                    }
                }
            } else if (status === 'completed') {
                var modeLabel = '';
                if (data.ga_solution && data.ga_solution.mode) modeLabel = ' Mode:' + data.ga_solution.mode;
                showToast(algo + ' optimization completed! GA: ' + (data.ga_fitness || 0).toFixed(2) + modeLabel + ', PSO: ' + (data.pso_fitness || 0).toFixed(2), 'success');
                // Re-enable run buttons
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                });
                // Update mode display immediately from solution
                if (data.ga_solution && data.ga_solution.mode) {
                    var modeEl = document.getElementById('ml-ga-mode');
                    if (modeEl) modeEl.textContent = data.ga_solution.mode;
                    var rhEl = document.getElementById('ml-ga-rh');
                    if (rhEl && data.ga_solution.set_rh != null) rhEl.textContent = data.ga_solution.set_rh;
                }
                // Directly update GA chart from socket payload (faster than waiting for refreshMLData fetch)
                if (data.ga_history && data.ga_history.length > 0) {
                    try { updateMLChart('gaFitness', data.ga_history, 'GA'); } catch(e) { console.warn('[ML] Direct GA chart update failed:', e); }
                }
                if (data.pso_history && data.pso_history.length > 0) {
                    try { updateMLChart('psoFitness', data.pso_history, 'PSO'); } catch(e) { console.warn('[ML] Direct PSO chart update failed:', e); }
                }
                // Refresh ML data
                refreshMLData();
            } else if (status === 'error') {
                showToast(algo + ' error: ' + (data.message || 'Unknown error'), 'error');
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                });
            } else if (status === 'busy') {
                showToast('Optimization already in progress', 'warning');
            }
        });

        socket.on('ir_learned', function(data) {
            console.log('[OK] IR Learned event received:', data);
            
            let buttonName = data.button;
            let buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
            let statusElement = document.getElementById('status-' + buttonName);
            
            // Try alternative button names if not found
            if (!buttonElement && currentLearningButton) {
                buttonName = currentLearningButton;
                buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
                statusElement = document.getElementById('status-' + buttonName);
            }
            
            if (data.status === 'error') {
                showToast('[ERROR] IR learning failed: ' + (data.message || 'Unknown error'), 'error');
                if (buttonElement) buttonElement.classList.remove('learning');
                if (statusElement) {
                    statusElement.textContent = 'Failed';
                    statusElement.style.color = '#1e40af';
                }
                return;
            }
            
            // Success!
            if (buttonElement) {
                buttonElement.classList.remove('learning');
                buttonElement.classList.add('learned');
            }
            
            if (statusElement) {
                statusElement.textContent = 'Learned [OK]';
                statusElement.style.color = '#3b82f6';
            }
            
            let message = 'IR Code learned: ' + buttonName;
            if (data.is_toggle) {
                message += ' (Power Toggle)';
            }
            
            showToast(message, 'success');
            learnedCodes[buttonName] = data.code;
            currentLearningButton = null;
        });

        // Debug: Listen to all MQTT messages
        socket.on('mqtt_debug', function(data) {
            console.log('[MQTT Debug]', data.time, '-', data.topic, ':', data.payload);
            
            // If learning mode and IR topic detected, show notification
            if (currentLearningButton && (data.topic.includes('ir') || data.topic.includes('IR'))) {
                console.log('[WARN] IR-related MQTT message while in learning mode!');
            }
        });

        // ==================== INIT ====================
        function loadSavedPreferences() {
            let savedPage = localStorage.getItem('currentPage');
            // Migrate old page IDs to new split pages
            if (savedPage === 'dashboard') savedPage = 'dashboard-ac';
            if (savedPage === 'control') savedPage = 'control-ac';
            if (savedPage === 'power') savedPage = 'energy';
            if (savedPage && document.getElementById(savedPage)) {
                document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
                document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
                
                document.getElementById(savedPage).classList.add('active');
                document.querySelectorAll('.nav-item').forEach(function(item) {
                    const oc = item.getAttribute('onclick') || '';
                    if (oc.indexOf("showPage('" + savedPage + "')") !== -1) {
                        item.classList.add('active');
                    }
                });
            }
            
            const savedRanges = localStorage.getItem('chartRanges');
            if (savedRanges) chartRanges = JSON.parse(savedRanges);
        }

        window.onload = function() {
            console.log('[INIT] Smart Room Dashboard Loading...');
            // Fetch role first so nav is hidden before user sees it
            fetch('/api/auth/role').then(function(r){ return r.json(); }).then(function(d){
                applyRoleRestrictions(d.role, d.username);
            }).catch(function(){ applyRoleRestrictions('user', ''); });
            try {
                var ovInit = document.getElementById('sidebar-overlay');
                if (ovInit) {
                    ovInit.classList.remove('active');
                    ovInit.style.display = 'none';
                    ovInit.style.pointerEvents = 'none';
                }
            } catch(e) {}
            try {
                // Safety net: hidden fixed overlays must not consume clicks.
                var blockers = document.querySelectorAll('body *');
                blockers.forEach(function(el) {
                    if (!el || !el.style) return;
                    if (el.id === 'drp-modal') return; // never touch modal
                    var cs = window.getComputedStyle(el);
                    if (cs.position === 'fixed' && cs.display === 'none') {
                        el.style.pointerEvents = 'none';
                    }
                });
            } catch(e) {}
            try { initCharts(); } catch(e) { console.error('[ERROR] initCharts:', e); }
            // ML charts initialized on first tab open via requestAnimationFrame (canvas must be visible first)
            // Restore recording previews from sessionStorage
            try { _recUpdatePreview(); } catch(e) {}
            try { _tempUpdatePreview(); } catch(e) {}
            try { _luxUpdatePreview(); } catch(e) {}
            // Fetch server-side recorded data (cross-device sync)
            try { _recServerFetch(); } catch(e) {}
            try { _tempServerFetch(); } catch(e) {}
            try { _luxServerFetch(); } catch(e) {}
            try { _occInitChart(); } catch(e) {}
            try { loadSavedPreferences(); } catch(e) { console.error('[ERROR] loadSavedPreferences:', e); }
            try { loadSavedSettings(); } catch(e) { console.error('[ERROR] loadSavedSettings:', e); }
            try { updateSoundToggleUI(); } catch(e) { console.error('[ERROR] updateSoundToggleUI:', e); }
            try { updateCameraToggleUI(); } catch(e) { console.error('[ERROR] updateCameraToggleUI:', e); }
            try { updateDashboard(); } catch(e) { console.error('[ERROR] updateDashboard:', e); }
            try { updateDeviceStatus(); } catch(e) { console.error('[ERROR] updateDeviceStatus:', e); }
            // Auto-show diagnostic panel if MQTT is not connected
            setTimeout(function() {
                fetch('/api/mqtt/status').then(r => r.json()).then(function(d) {
                    if (!d.connected) {
                        var p = document.getElementById('diag-panel');
                        if (p) {
                            p.style.display = 'block';
                            var res = document.getElementById('diag-result');
                            if (res) res.textContent = '[AUTO-DIAGNOSE] MQTT NOT CONNECTED!\\nBroker: ' + d.broker + '\\nError: ' + (d.error || 'Unknown') + '\\n\\nClick "Test Frontend" or "Test MQTT Broker" button below.';
                        }
                    }
                }).catch(function() {});
            }, 2000);
            try { updateLogs(); } catch(e) { console.error('[ERROR] updateLogs:', e); }
            try { checkCameraStatus(); } catch(e) { console.error('[ERROR] checkCameraStatus:', e); }
            try { loadAllEnergyCharts(); } catch(e) { console.error('[ERROR] loadAllEnergyCharts:', e); }
            setTimeout(function() {
                try {
                    if (ensureChartsReady()) {
                        loadAllEnergyCharts();
                    }
                } catch(e) {
                    console.error('[ERROR] delayed energy chart init:', e);
                }
            }, 600);
            const googleFormInput = document.getElementById('google-form-url');
            if (googleFormInput) {
                googleFormInput.value = localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;
            }
            try { loadFeedbackHistory(); } catch(e) { console.error('[ERROR] loadFeedbackHistory:', e); }
            
            // Initialize AC mode UI on page load
            try { updateModeBadges(); } catch(e) { console.error('[ERROR] updateModeBadges:', e); }
            
            Object.keys(chartRanges).forEach(chartName => {
                try { updateChartData(chartName, chartRanges[chartName]); } catch(e) { console.error('[ERROR] updateChartData ' + chartName + ':', e); }
            });
            
            setInterval(function() { try { updateDashboard(); } catch(e) {} }, 1000);
            setInterval(function() { try { updateDeviceStatus(); } catch(e) {} }, 5000);
            setInterval(function() { try { updateLogs(); } catch(e) {} }, 5000);
            
            setInterval(() => {
                Object.keys(chartRanges).forEach(chartName => {
                    try { updateChartData(chartName, chartRanges[chartName]); } catch(e) {}
                });
            }, 30000);

            setInterval(function() {
                try {
                    var energyPage = document.getElementById('energy');
                    if (energyPage && energyPage.classList.contains('active')) {
                        loadAllEnergyCharts();
                    }
                    var outletCtrlPage = document.getElementById('control-outlet');
                    if (outletCtrlPage && outletCtrlPage.classList.contains('active')) {
                        try { refreshOutletStatus(); } catch(e) {}
                    }
                    var outletAnalPage = document.getElementById('outlet-analysis');
                    if (outletAnalPage && outletAnalPage.classList.contains('active')) {
                        try { refreshOutletStatus(); } catch(e) {}
                    }
                } catch(e) {}
            }, 8000);
            
            // ==================== DATE RANGE PICKER ====================
            var _drpCallback = null;
            var _drpAllRows  = null;

            window.showDateRangePicker = function(subtitle, rowsRef, callback) {
                _drpCallback = callback;
                _drpAllRows  = rowsRef;
                var sub = document.getElementById('drp-subtitle');
                if (sub) sub.textContent = subtitle || 'Export data in the selected range';
                var today = new Date().toISOString().substring(0,10);
                var from30 = new Date(Date.now() - 29*24*60*60*1000).toISOString().substring(0,10);
                var fromVal = from30;
                if (rowsRef && rowsRef.length > 0 && rowsRef[0].ts) {
                    var firstDate = rowsRef[0].ts.substring(0,10);
                    fromVal = firstDate < from30 ? firstDate : from30;
                }
                document.getElementById('drp-from').value = fromVal;
                document.getElementById('drp-to').value   = today;
                var modal = document.getElementById('drp-modal');
                modal.style.pointerEvents = 'auto';
                modal.style.display = 'flex';
            };

            window.drpShortcut = function(days) {
                var today = new Date().toISOString().substring(0,10);
                document.getElementById('drp-to').value = today;
                if (days === 0) {
                    var fromVal = today;
                    if (_drpAllRows && _drpAllRows.length > 0 && _drpAllRows[0].ts) {
                        fromVal = _drpAllRows[0].ts.substring(0,10);
                    }
                    document.getElementById('drp-from').value = fromVal;
                } else if (days === 1) {
                    document.getElementById('drp-from').value = today;
                } else {
                    var d = new Date(Date.now() - (days-1)*24*60*60*1000).toISOString().substring(0,10);
                    document.getElementById('drp-from').value = d;
                }
            };

            window.drpConfirm = function() {
                var from = document.getElementById('drp-from').value;
                var to   = document.getElementById('drp-to').value;
                if (!from || !to) { showToast('Please select a date range first', 'error'); return; }
                if (from > to)    { showToast('Start date must be before end date', 'error'); return; }
                var modal = document.getElementById('drp-modal');
                modal.style.display = 'none';
                modal.style.pointerEvents = 'none';
                if (_drpCallback) _drpCallback(from, to);
            };

            window.drpCancel = function() {
                var modal = document.getElementById('drp-modal');
                modal.style.display = 'none';
                modal.style.pointerEvents = 'none';
            };

            function drpFilter(rows, from, to) {
                return rows.filter(function(r){ var d = (r.ts||'').substring(0,10); return d >= from && d <= to; });
            }

            window.energyExportRange = function(device) {
                showDateRangePicker(
                    'Export rekaman energi' + (device && device !== 'all' ? ' (' + device + ')' : ' (All)'),
                    _recRows,
                    function(from, to) {
                        var base = (device && device !== 'all') ? _recRows.filter(function(r){ return r.device === device; }) : _recRows;
                        var rows = drpFilter(base, from, to);
                        if (rows.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                        var header = 'Time,Perangkat,Voltage (V),Current (A),Power Active (W),Energy (kWh),Power Reaktif (VAR),Power Semu (VA),Frequency (Hz),Power Factor\\n';
                        var body = rows.map(function(r) {
                            return '"' + r.ts + '",' + r.device + ',' + r.voltage + ',' + r.arus + ',' +
                                   r.daya + ',' + r.energi + ',' + r.reaktif + ',' + r.semu + ',' + r.freq + ',' + r.pf;
                        }).join('\\n');
                        var suffix = (device && device !== 'all') ? '_' + device.toLowerCase() : '';
                        var fname = 'energy' + suffix + '_' + from + '_to_' + to + '.csv';
                        downloadCSV(fname, header + body);
                        showToast(rows.length + ' rows exported: ' + fname, 'success');
                    }
                );
            };

            window.tempExportRange = function() {
                showDateRangePicker('Export temperature & humidity recording', _tempRows, function(from, to) {
                    var rows = drpFilter(_tempRows, from, to);
                    if (rows.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                    var header = 'Time,Temp Rata2 (C),Humidity (%),Sensor T1 (C),Sensor T2 (C),Sensor T3 (C),Set Temp (C),Fan Speed,Mode AC,Control\\n';
                    var body = rows.map(function(r){ return '"' + r.ts + '",' + r.temp + ',' + r.hum + ',' + r.t1 + ',' + r.t2 + ',' + r.t3 + ',' + r.setT + ',"' + r.fan + '",' + r.mode + ',' + r.ctrl; }).join('\\n');
                    var fname = 'temp_' + from + '_to_' + to + '.csv';
                    downloadCSV(fname, header + body);
                    showToast(rows.length + ' rows exported: ' + fname, 'success');
                });
            };

            window.luxExportRange = function() {
                showDateRangePicker('Export rekaman lux & brightness', _luxRows, function(from, to) {
                    var rows = drpFilter(_luxRows, from, to);
                    if (rows.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                    var header = 'Time,Lux 1 (lx),Lux 2 (lx),Lux 3 (lx),Avg Lux (lx),Brightness 1 (%),Brightness 2 (%)\\n';
                    var body = rows.map(function(r){ return '"' + r.ts + '",' + r.lux1 + ',' + r.lux2 + ',' + r.lux3 + ',' + r.avg + ',' + r.b1 + ',' + r.b2; }).join('\\n');
                    var fname = 'lux_' + from + '_to_' + to + '.csv';
                    downloadCSV(fname, header + body);
                    showToast(rows.length + ' rows exported: ' + fname, 'success');
                });
            };

            window.occExportRange = function() {
                showDateRangePicker('Export rekaman occupancy', _occRows, function(from, to) {
                    var rows = drpFilter(_occRows, from, to);
                    if (rows.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                    var header = 'Time,Hour,Person Count,Confidence\\n';
                    var body = rows.map(function(r){ return '"' + r.ts + '",' + String(r.hour).padStart(2,'0') + ':00,' + r.count + ',' + r.conf; }).join('\\n');
                    var fname = 'occupancy_' + from + '_to_' + to + '.csv';
                    downloadCSV(fname, header + body);
                    showToast(rows.length + ' data exported: ' + fname, 'success');
                });
            };

            window.exportChartRange = function(chartName, valueLabel) {
                var chart = charts[chartName];
                if (!chart || !chart.data.labels || chart.data.labels.length === 0) {
                    showToast('No data for ' + chartName, 'error'); return;
                }
                var pseudoRows = chart.data.labels.map(function(l){ return {ts: l}; });
                showDateRangePicker('Export chart: ' + valueLabel, pseudoRows, function(from, to) {
                    var header = 'Time,' + valueLabel + '\\n';
                    var lines = [];
                    chart.data.labels.forEach(function(label, i) {
                        var d = label.substring(0,10);
                        if (d < from || d > to) return;
                        var val = chart.data.datasets[0].data[i];
                        lines.push('"' + label + '",' + (val !== null && val !== undefined ? val : ''));
                    });
                    if (lines.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                    downloadCSV(chartName + '_' + from + '_to_' + to + '.csv', header + lines.join('\\n') + '\\n');
                    showToast(lines.length + ' points exported', 'success');
                });
            };

            // ==================== DB EXPORT HELPERS ====================
            function _dbDateToday() {
                var d = new Date(); 
                return d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0') + '-' + String(d.getDate()).padStart(2,'0');
            }
            function _dbDateNdAgo(n) {
                var d = new Date(); d.setDate(d.getDate() - n + 1);
                return d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0') + '-' + String(d.getDate()).padStart(2,'0');
            }
            window.dbExportSetToday = function(prefix) {
                var t = _dbDateToday();
                var fi = document.getElementById('db-' + prefix + '-from');
                var ti = document.getElementById('db-' + prefix + '-to');
                if (fi) fi.value = t; if (ti) ti.value = t;
            };
            window.dbExportSet7d = function(prefix) {
                var fi = document.getElementById('db-' + prefix + '-from');
                var ti = document.getElementById('db-' + prefix + '-to');
                if (fi) fi.value = _dbDateNdAgo(7); if (ti) ti.value = _dbDateToday();
            };
            window.dbExportSet30d = function(prefix) {
                var fi = document.getElementById('db-' + prefix + '-from');
                var ti = document.getElementById('db-' + prefix + '-to');
                if (fi) fi.value = _dbDateNdAgo(30); if (ti) ti.value = _dbDateToday();
            };
            window.dbExportCSV = function(type) {
                // prefix mapping: all energy types share the same date inputs (db-energy-from / db-energy-to)
                var prefix = (type === 'energy_ac' || type === 'energy_lamp' || type === 'energy_outlet' || type === 'energy_total') ? 'energy' : type;
                var fromEl = document.getElementById('db-' + prefix + '-from');
                var toEl   = document.getElementById('db-' + prefix + '-to');
                var from = fromEl ? fromEl.value : '';
                var to   = toEl   ? toEl.value   : '';
                if (!from || !to) { showToast('Please select a date range first', 'error'); return; }
                if (from > to) { showToast('Start date must be before end date', 'error'); return; }
                showToast('Fetching data from database...', 'success');
                var url = '/api/export/csv?type=' + type + '&from=' + from + '&to=' + to;
                fetch(url).then(function(r) {
                    if (r.ok && r.headers.get('Content-Type') && r.headers.get('Content-Type').indexOf('text/csv') >= 0) {
                        return r.blob().then(function(blob) {
                            var a = document.createElement('a');
                            a.href = URL.createObjectURL(blob);
                            var fname = type + '_' + from + '_to_' + to + '.csv';
                            a.download = fname;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(a.href);
                        });
                    } else {
                        return r.json().then(function(j) {
                            showToast((j && j.error) ? j.error : 'Failed to export data', 'error');
                        });
                    }
                }).catch(function(e) { showToast('Error: ' + e, 'error'); });
            };

            console.log('[OK] Dashboard Ready!');
        };
    </script>

    <!-- ===== MODAL PILIH RENTANG TANGGAL ===== -->
    <div id="drp-modal" style="display:none;position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.55);align-items:center;justify-content:center;">
        <div style="background:var(--bg-card);border-radius:20px;padding:30px 28px 24px;min-width:310px;max-width:380px;width:90%;box-shadow:0 12px 48px rgba(0,0,0,0.25);border:1px solid var(--border);position:relative;">
            <div style="font-size:16px;font-weight:700;color:var(--text);margin-bottom:6px;display:flex;align-items:center;gap:8px;">
                <span style="font-size:20px;">&#128197;</span> Select Date Range
            </div>
            <div id="drp-subtitle" style="font-size:12px;color:var(--text-secondary);margin-bottom:20px;">Export data in the selected range</div>
            <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:22px;">
                <div>
                    <label style="font-size:11px;font-weight:600;color:var(--text-secondary);display:block;margin-bottom:5px;">From Date</label>
                    <input type="date" id="drp-from" style="width:100%;padding:9px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:13px;box-sizing:border-box;">
                </div>
                <div>
                    <label style="font-size:11px;font-weight:600;color:var(--text-secondary);display:block;margin-bottom:5px;">To Date</label>
                    <input type="date" id="drp-to" style="width:100%;padding:9px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:13px;box-sizing:border-box;">
                </div>
            </div>
            <!-- Shortcut buttons -->
            <div style="display:flex;gap:5px;margin-bottom:20px;flex-wrap:wrap;">
                <button onclick="drpShortcut(1)" style="padding:4px 10px;border-radius:7px;border:1px solid var(--border);background:var(--bg);color:var(--text-secondary);font-size:11px;cursor:pointer;">Today</button>
                <button onclick="drpShortcut(7)" style="padding:4px 10px;border-radius:7px;border:1px solid var(--border);background:var(--bg);color:var(--text-secondary);font-size:11px;cursor:pointer;">7 Days</button>
                <button onclick="drpShortcut(30)" style="padding:4px 10px;border-radius:7px;border:1px solid var(--border);background:var(--bg);color:var(--text-secondary);font-size:11px;cursor:pointer;">30 Days</button>
                <button onclick="drpShortcut(0)" style="padding:4px 10px;border-radius:7px;border:1px solid var(--border);background:var(--bg);color:var(--text-secondary);font-size:11px;cursor:pointer;">All</button>
            </div>
            <div style="display:flex;gap:10px;">
                <button onclick="drpConfirm()" style="flex:1;padding:11px;border-radius:11px;border:none;background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;font-size:13px;font-weight:700;cursor:pointer;"><i class="fas fa-download" style="margin-right:6px;"></i>Export CSV</button>
                <button onclick="drpCancel()" style="padding:11px 18px;border-radius:11px;border:1px solid var(--border);background:var(--bg);color:var(--text-secondary);font-size:13px;font-weight:600;cursor:pointer;">Cancel</button>
            </div>
        </div>
    </div>
</body>
</html>
'''

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

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
