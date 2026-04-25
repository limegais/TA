"""
mysql_energy.py — Background MySQL energy polling untuk Smart Room Dashboard
=============================================================================
Dibuat terpisah dari app.py agar mudah dikelola.

Cara kerja:
  - Diimport oleh app.py
  - app.py memanggil start_polling(ctx) sekali saat startup
  - Thread berjalan di background, polling MySQL setiap POLL_INTERVAL detik
  - Data ditulis ke:
      ctx['mqtt_data']['energy']        → real-time display di dashboard
      ctx['energy_runtime_history']     → fallback buffer untuk chart
      InfluxDB via ctx['write_influx']  → historical chart
      Socket.IO via ctx['socketio']     → push ke browser

Setup di hosting Jagoan Hosting:
  1. cPanel → Remote MySQL → tambahkan IP RPi (atau %)
  2. cPanel → MySQL Databases → buat user, tambahkan ke database dengan ALL PRIVILEGES
  3. Ganti MYSQL_HOST di bawah dengan hostname server hosting
     (biasanya tertulis di cPanel bagian atas, mis: pristine.jagoanhosting.id)
  4. pip install pymysql

Tabel yang dipakai:
  energies     : tegangan, arus, active_power, frekuensi, apparent_power, created_at
  energy_kwh   : total_energy (diambil via JOIN pada id_kwh)
"""

import threading
import time
from datetime import datetime

# ── Konfigurasi MySQL ────────────────────────────────────────────────────────
# ⚠ PENTING:
#   - Kalau Flask jalan di RPi (bukan di server hosting), ganti MYSQL_HOST
#     dengan hostname atau IP eksternal Jagoan Hosting, mis:
#       MYSQL_HOST = 'pristine.jagoanhosting.id'
#   - Kalau Flask jalan DI server hosting yang sama, 127.0.0.1 sudah benar.
MYSQL_HOST     = '127.0.0.1'          # ← ganti ke hostname hosting jika Flask di RPi
MYSQL_PORT     = 3306
MYSQL_DATABASE = 'iotlabun_sbms'
MYSQL_USER     = 'iotlabun_aslab'
MYSQL_PASSWORD = 'Pass2myDB'

POLL_INTERVAL  = 5   # detik — seberapa sering query MySQL

# ── Query SQL ────────────────────────────────────────────────────────────────
# Ambil 1 baris terbaru: JOIN energies + energy_kwh
QUERY_LATEST = """
    SELECT
        e.tegangan,
        e.arus,
        e.active_power,
        e.reactive_power,
        e.apparent_power,
        e.frekuensi,
        e.created_at,
        k.total_energy
    FROM energies e
    LEFT JOIN energy_kwh k ON k.id_kwh = e.id_kwh
    ORDER BY e.id DESC
    LIMIT 1
"""

# ── State internal ────────────────────────────────────────────────────────────
_poll_thread  = None
_stop_event   = threading.Event()
_last_row_id  = None   # track perubahan agar tidak tulis ulang data sama


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _build_energy_dict(row):
    """Konversi row MySQL → dict format mqtt_data['energy']."""
    tegangan      = _safe_float(row['tegangan'])
    arus          = _safe_float(row['arus'])
    active_power  = _safe_float(row['active_power'])
    apparent_power = _safe_float(row['apparent_power'])
    total_energy  = _safe_float(row['total_energy'])
    frekuensi     = _safe_float(row['frekuensi'])

    # Hitung power factor; hindari division by zero
    pf = round(active_power / apparent_power, 3) if apparent_power > 0.001 else 0.0

    return {
        'voltage':   tegangan,
        'current':   arus,
        'power':     active_power,
        'energy':    total_energy,     # kWh
        'frequency': frekuensi,
        'pf':        pf,
        'connected': True,
        'source':    'mysql'
    }


def _poll_loop(ctx):
    """
    Background loop: query MySQL → update mqtt_data → emit Socket.IO → tulis InfluxDB.

    ctx keys yang dibutuhkan (dikirim dari app.py via start_polling):
        mqtt_data              : dict (dari app.py)
        energy_runtime_history : deque (dari app.py)
        get_energy_phase       : callable() → str — supaya ambil nilai terkini
        write_influx           : callable(measurement, fields, tags)
        socketio               : Flask-SocketIO instance
    """
    global _last_row_id

    mqtt_data              = ctx['mqtt_data']
    energy_runtime_history = ctx['energy_runtime_history']
    get_energy_phase       = ctx['get_energy_phase']   # lambda/fungsi, bukan nilai langsung
    write_influx           = ctx['write_influx']
    sio                    = ctx['socketio']

    try:
        import pymysql
        import pymysql.cursors
    except ImportError:
        print('[MySQL] pymysql tidak terinstall. Jalankan: pip install pymysql')
        return

    print(f'[MySQL] Energy polling started — host={MYSQL_HOST}:{MYSQL_PORT} '
          f'db={MYSQL_DATABASE} interval={POLL_INTERVAL}s')

    conn = None

    while not _stop_event.is_set():
        try:
            # Buat koneksi baru kalau belum ada / terputus
            if conn is None or not conn.open:
                conn = pymysql.connect(
                    host=MYSQL_HOST,
                    port=MYSQL_PORT,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    database=MYSQL_DATABASE,
                    connect_timeout=10,
                    read_timeout=10,
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=True,
                    ssl_disabled=True
                )
                print('[MySQL] Koneksi berhasil')

            with conn.cursor() as cur:
                cur.execute(QUERY_LATEST)
                row = cur.fetchone()

            if row:
                # Cek apakah data sudah berubah (bandingkan created_at)
                row_ts = str(row.get('created_at', ''))
                if row_ts == _last_row_id:
                    # Data sama — tidak perlu update
                    time.sleep(POLL_INTERVAL)
                    continue
                _last_row_id = row_ts

                energy_dict = _build_energy_dict(row)
                phase       = get_energy_phase()   # ambil nilai global energy_phase terkini

                # 1. Update mqtt_data (dipakai real-time display)
                mqtt_data['energy'].update(energy_dict)

                # 2. Push ke semua browser via Socket.IO
                sio.emit('mqtt_update', {'type': 'energy', 'data': mqtt_data['energy']})

                # 3. Tulis ke InfluxDB agar chart historis tetap bekerja
                try:
                    write_influx(
                        'energy_monitor',
                        {
                            'voltage':      energy_dict['voltage'],
                            'current':      energy_dict['current'],
                            'power':        energy_dict['power'],
                            'energy_kwh':   energy_dict['energy'],
                            'frequency':    energy_dict['frequency'],
                            'power_factor': energy_dict['pf'],
                        },
                        tags={'device': 'mysql_pzem', 'phase': phase}
                    )
                except Exception as ex:
                    print(f'[MySQL] InfluxDB write error: {ex}')

                # 4. Append ke ring-buffer (fallback chart)
                try:
                    energy_runtime_history.append({
                        'ts':           datetime.now(),
                        'phase':        phase,
                        'voltage':      energy_dict['voltage'],
                        'current':      energy_dict['current'],
                        'power':        energy_dict['power'],
                        'energy_kwh':   energy_dict['energy'],
                        'frequency':    energy_dict['frequency'],
                        'power_factor': energy_dict['pf'],
                    })
                except Exception:
                    pass

        except pymysql.err.OperationalError as e:
            print(f'[MySQL] Koneksi error: {e} — akan retry dalam {POLL_INTERVAL}s')
            mqtt_data['energy']['connected'] = False
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
            conn = None

        except Exception as e:
            print(f'[MySQL] Poll error: {e}')
            mqtt_data['energy']['connected'] = False

        time.sleep(POLL_INTERVAL)

    # Thread dihentikan
    try:
        if conn and conn.open:
            conn.close()
    except Exception:
        pass
    print('[MySQL] Energy polling stopped')


def start_polling(ctx):
    """
    Panggil ini dari app.py saat startup untuk memulai polling MySQL.

    Contoh pemanggilan di app.py:
        import mysql_energy
        mysql_energy.start_polling({
            'mqtt_data':              mqtt_data,
            'energy_runtime_history': energy_runtime_history,
            'get_energy_phase':       lambda: energy_phase,
            'write_influx':           write_to_influxdb,
            'socketio':               socketio,
        })

    ctx harus berisi semua key di atas.
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
    """Hentikan thread polling (opsional, biasanya tidak perlu karena daemon=True)."""
    _stop_event.set()


def test_connection():
    """
    Tes koneksi MySQL — jalankan langsung:
        python mysql_energy.py
    """
    try:
        import pymysql
        import pymysql.cursors
    except ImportError:
        print('ERROR: pymysql tidak terinstall. Jalankan: pip install pymysql')
        return False

    print(f'Mencoba koneksi ke {MYSQL_HOST}:{MYSQL_PORT} db={MYSQL_DATABASE}...')
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST, port=MYSQL_PORT,
            user=MYSQL_USER, password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            connect_timeout=10, charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            ssl_disabled=True
        )
        with conn.cursor() as cur:
            cur.execute(QUERY_LATEST)
            row = cur.fetchone()
        conn.close()

        if row:
            print('Koneksi OK. Data terbaru:')
            for k, v in row.items():
                print(f'  {k:20s} = {v}')
            print()
            print('Energy dict yang akan dikirim ke dashboard:')
            d = _build_energy_dict(row)
            for k, v in d.items():
                print(f'  {k:15s} = {v}')
        else:
            print('Koneksi OK tapi tabel kosong (belum ada data dari ESP32).')
        return True

    except Exception as e:
        print(f'GAGAL: {e}')
        print()
        print('Kemungkinan penyebab:')
        print('  1. Host salah — kalau Flask di RPi, ganti MYSQL_HOST ke hostname hosting')
        print('  2. Remote MySQL belum diaktifkan di cPanel → Remote MySQL → tambah IP RPi')
        print('  3. User belum diberi akses ke database di cPanel → MySQL Databases')
        return False


# ── Test langsung ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_connection()
