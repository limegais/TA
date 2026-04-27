"""
mysql_energy.py — HTTP Energy Polling untuk Smart Room Dashboard
================================================================
Fetch AC (id_kwh=1) + Lamp (id_kwh=3) dari PHP proxy Jagoan Hosting.

Arsitektur:
  RPi (Flask) -> HTTP GET -> api_energy.php -> MySQL -> JSON -> RPi

Response PHP: {"ac": {...}, "lamp": {...}}
  id_kwh=1 -> AC   : tegangan/arus/active_power/frekuensi/total_energy
  id_kwh=3 -> Lampu: tegangan/arus/active_power/frekuensi/total_energy
"""

import threading
import time
import urllib.request
import urllib.error
import json
from datetime import datetime

# Konfigurasi endpoint PHP
PHP_URL       = 'https://iotlab-uns.com/api_energy.php'
PHP_API_KEY   = 'iotlab_smartroom_2024'
POLL_INTERVAL = 5   # detik
HTTP_TIMEOUT  = 8   # detik

# State internal
_poll_thread = None
_stop_event  = threading.Event()
_last_ts_ac  = None
_last_ts_lamp = None


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _fetch_latest():
    """HTTP GET ke PHP proxy -> return dict {ac: ..., lamp: ...} atau None."""
    url = f'{PHP_URL}?key={PHP_API_KEY}'
    req = urllib.request.Request(url, headers={'User-Agent': 'SmartRoom-RPi/1.0'})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        raw = resp.read().decode('utf-8')
    data = json.loads(raw)
    if not data or 'error' in data:
        return None
    return data


def _build_energy_dict(row, label='ac'):
    """Konversi row PHP JSON -> dict energy."""
    if not row:
        return None
    tegangan        = _safe_float(row.get('tegangan'))
    arus            = _safe_float(row.get('arus'))
    active_power    = _safe_float(row.get('active_power'))
    reactive_power  = _safe_float(row.get('reactive_power'))
    apparent_power  = _safe_float(row.get('apparent_power'))
    total_energy    = _safe_float(row.get('total_energy'))
    frekuensi       = _safe_float(row.get('frekuensi'))
    pf = round(active_power / apparent_power, 3) if apparent_power > 0.001 else 0.0
    return {
        'label':          label,
        'voltage':        tegangan,
        'current':        arus,
        'power':          active_power,
        'reactive_power': reactive_power,
        'apparent_power': apparent_power,
        'energy':         total_energy,
        'frequency':      frekuensi,
        'pf':             pf,
        'connected':      True,
        'source':         'mysql',
        'updated_at':     str(row.get('created_at', '')),
    }


def _poll_loop(ctx):
    global _last_ts_ac, _last_ts_lamp

    mqtt_data              = ctx['mqtt_data']
    energy_runtime_history = ctx['energy_runtime_history']
    get_energy_phase       = ctx['get_energy_phase']
    write_influx           = ctx['write_influx']
    sio                    = ctx['socketio']

    print(f'[MySQL] Energy polling started -- {PHP_URL} interval={POLL_INTERVAL}s')

    consecutive_errors = 0

    while not _stop_event.is_set():
        try:
            data = _fetch_latest()

            if data:
                ac_row   = data.get('ac') or {}
                lamp_row = data.get('lamp') or {}

                ac_dict   = _build_energy_dict(ac_row,   label='AC')
                lamp_dict = _build_energy_dict(lamp_row, label='Lamp')

                # ── AC (id_kwh=1) ──────────────────────────────────────────
                if ac_dict:
                    ts_ac = str(ac_row.get('created_at', ''))
                    if ts_ac != _last_ts_ac:
                        _last_ts_ac = ts_ac

                        # Update mqtt_data['energy'] -> existing display tetap jalan
                        mqtt_data['energy'].update({
                            'voltage':   ac_dict['voltage'],
                            'current':   ac_dict['current'],
                            'power':     ac_dict['power'],
                            'energy':    ac_dict['energy'],
                            'frequency': ac_dict['frequency'],
                            'pf':        ac_dict['pf'],
                            'connected': True,
                            'source':    'mysql',
                        })

                        phase = get_energy_phase()

                        # Tulis ke InfluxDB (tag device=mysql_ac)
                        try:
                            write_influx('energy_monitor', {
                                'voltage':      ac_dict['voltage'],
                                'current':      ac_dict['current'],
                                'power':        ac_dict['power'],
                                'energy_kwh':   ac_dict['energy'],
                                'frequency':    ac_dict['frequency'],
                                'power_factor': ac_dict['pf'],
                            }, tags={'device': 'mysql_ac', 'phase': phase})
                        except Exception as ex:
                            print(f'[MySQL] InfluxDB AC write error: {ex}')

                        # Append ring-buffer
                        try:
                            energy_runtime_history.append({
                                'ts': datetime.now(), 'phase': phase,
                                'voltage': ac_dict['voltage'], 'current': ac_dict['current'],
                                'power': ac_dict['power'], 'energy_kwh': ac_dict['energy'],
                                'frequency': ac_dict['frequency'], 'power_factor': ac_dict['pf'],
                            })
                        except Exception:
                            pass

                # ── Lamp (id_kwh=2) ────────────────────────────────────────
                if lamp_dict:
                    ts_lamp = str(lamp_row.get('created_at', ''))
                    if ts_lamp != _last_ts_lamp:
                        _last_ts_lamp = ts_lamp

                        # Tulis ke InfluxDB (tag device=mysql_lamp)
                        try:
                            write_influx('energy_monitor', {
                                'voltage':      lamp_dict['voltage'],
                                'current':      lamp_dict['current'],
                                'power':        lamp_dict['power'],
                                'energy_kwh':   lamp_dict['energy'],
                                'frequency':    lamp_dict['frequency'],
                                'power_factor': lamp_dict['pf'],
                            }, tags={'device': 'mysql_lamp', 'phase': 'lamp'})
                        except Exception as ex:
                            print(f'[MySQL] InfluxDB Lamp write error: {ex}')

                # ── Emit socket event dengan data AC + Lamp ────────────────
                sio.emit('mqtt_update', {
                    'type': 'energy',
                    'data': mqtt_data['energy']
                })
                sio.emit('mysql_energy_update', {
                    'ac':   ac_dict,
                    'lamp': lamp_dict,
                })

            if consecutive_errors > 0:
                print('[MySQL] Koneksi kembali normal')
            consecutive_errors = 0
            mqtt_data['energy']['connected'] = True

        except urllib.error.URLError as e:
            consecutive_errors += 1
            if consecutive_errors == 1 or consecutive_errors % 12 == 0:
                print(f'[MySQL] HTTP error ({consecutive_errors}x): {e}')
            mqtt_data['energy']['connected'] = False

        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors == 1 or consecutive_errors % 12 == 0:
                print(f'[MySQL] Poll error ({consecutive_errors}x): {e}')
            mqtt_data['energy']['connected'] = False

        time.sleep(POLL_INTERVAL)

    print('[MySQL] Energy polling stopped')


def start_polling(ctx):
    """
    Panggil dari app.py saat startup.

    Contoh di app.py:
        import mysql_energy
        mysql_energy.start_polling({
            'mqtt_data':              mqtt_data,
            'energy_runtime_history': energy_runtime_history,
            'get_energy_phase':       lambda: energy_phase,
            'write_influx':           write_to_influxdb,
            'socketio':               socketio,
        })
    """
    global _poll_thread, _stop_event

    required_keys = ['mqtt_data', 'energy_runtime_history',
                     'get_energy_phase', 'write_influx', 'socketio']
    missing = [k for k in required_keys if k not in ctx]
    if missing:
        print(f'[MySQL] start_polling: missing keys: {missing}')
        return

    _stop_event.clear()
    _poll_thread = threading.Thread(
        target=_poll_loop,
        args=(ctx,),
        daemon=True,
        name='mysql-energy-poll'
    )
    _poll_thread.start()


def stop_polling():
    _stop_event.set()


def test_connection():
    """Test koneksi: python mysql_energy.py"""
    print(f'Mencoba fetch dari {PHP_URL} ...')
    try:
        data = _fetch_latest()
        if not data:
            print('Koneksi OK tapi data kosong.')
            return True
        for label, row in [('AC  (id_kwh=1)', data.get('ac')), ('Lamp (id_kwh=3)', data.get('lamp'))]:
            print(f'\n--- {label} ---')
            if row:
                for k, v in row.items():
                    print(f'  {k:20s} = {v}')
                d = _build_energy_dict(row)
                print(f'  -> power={d["power"]}W  current={d["current"]}A  energy={d["energy"]}kWh')
            else:
                print('  (tidak ada data)')
        return True
    except Exception as e:
        print(f'GAGAL: {e}')
        return False


if __name__ == '__main__':
    test_connection()
