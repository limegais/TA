<?php
/**
 * api_energy.php — PHP proxy untuk ambil data energi dari MySQL
 * Upload ke: /home/iotlabun/public_html/api_energy.php
 *
 * Endpoint normal : ?key=iotlab_smartroom_2024
 * Deploy ke RPi   : ?key=iotlab_smartroom_2024&action=deploy  → pipe ke bash
 */

define('API_KEY', 'iotlab_smartroom_2024');

// Izinkan browser fetch langsung (CORS)
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET');

if (($_GET['key'] ?? '') !== API_KEY) {
    http_response_code(403);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'forbidden']);
    exit;
}

// ── Mode deploy: kembalikan shell script untuk dijalankan di RPi ─────────────
if (($_GET['action'] ?? '') === 'deploy') {
    header('Content-Type: text/plain');
    echo <<<'BASH'
#!/bin/bash
set -e
DASHBOARD=/home/iotlab/smartroom/dashboard
VENV=/home/iotlab/smartroom/venv/bin/python3

echo "[1/3] Menulis mysql_energy.py ..."
cat > "$DASHBOARD/mysql_energy.py" << 'PYEOF'
import threading, time, urllib.request, urllib.error, json
from datetime import datetime

PHP_URL       = 'https://iotlab-uns.com/api_energy.php'
PHP_API_KEY   = 'iotlab_smartroom_2024'
POLL_INTERVAL = 5
HTTP_TIMEOUT  = 8
_poll_thread  = None
_stop_event   = threading.Event()
_last_ts_ac   = None
_last_ts_outlet = None
_last_ts_lamp = None

def _safe_float(val, default=0.0):
    try:    return float(val) if val is not None else default
    except: return default

def _fetch_latest():
    url = f"{PHP_URL}?key={PHP_API_KEY}"
    req = urllib.request.Request(url, headers={"User-Agent":"SmartRoom-RPi/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        data = json.loads(r.read().decode("utf-8"))
    return None if (not data or "error" in data) else data

def _build_energy_dict(row, label="ac"):
    if not row: return None
    ap  = _safe_float(row.get("apparent_power"))
    p   = _safe_float(row.get("active_power"))
    pf  = round(p / ap, 3) if ap > 0.001 else 0.0
    return {
        "label":          label,
        "voltage":        _safe_float(row.get("tegangan")),
        "current":        _safe_float(row.get("arus")),
        "power":          p,
        "reactive_power": _safe_float(row.get("reactive_power")),
        "apparent_power": ap,
        "energy":         _safe_float(row.get("total_energy")),
        "frequency":      _safe_float(row.get("frekuensi")),
        "pf":             pf,
        "connected":      True,
        "source":         "mysql",
        "updated_at":     str(row.get("created_at", "")),
    }

def _poll_loop(ctx):
    global _last_ts_ac, _last_ts_outlet, _last_ts_lamp
    mqtt_data = ctx["mqtt_data"]; erh = ctx["energy_runtime_history"]
    get_phase = ctx["get_energy_phase"]; winflux = ctx["write_influx"]; sio = ctx["socketio"]
    print(f"[MySQL] Energy polling started — interval={POLL_INTERVAL}s")
    errs = 0
    while not _stop_event.is_set():
        try:
            data = _fetch_latest()
            if data:
                ac_r = data.get("ac") or {}; ot_r = data.get("outlet") or {}; lp_r = data.get("lamp") or {}
                acd  = _build_energy_dict(ac_r,  "AC")
                otd  = _build_energy_dict(ot_r,  "Outlet")
                lpd  = _build_energy_dict(lp_r, "Lamp")
                if acd:
                    ts = str(ac_r.get("created_at",""))
                    if ts != _last_ts_ac:
                        _last_ts_ac = ts
                        mqtt_data["energy"].update({"voltage":acd["voltage"],"current":acd["current"],
                            "power":acd["power"],"energy":acd["energy"],"frequency":acd["frequency"],
                            "pf":acd["pf"],"connected":True,"source":"mysql"})
                        ph = get_phase()
                        try: winflux("energy_monitor",{"voltage":acd["voltage"],"current":acd["current"],
                            "power":acd["power"],"energy_kwh":acd["energy"],"frequency":acd["frequency"],
                            "power_factor":acd["pf"]},tags={"device":"mysql_ac","phase":ph})
                        except: pass
                        try: erh.append({"ts":datetime.now(),"phase":ph,"voltage":acd["voltage"],
                            "current":acd["current"],"power":acd["power"],"energy_kwh":acd["energy"],
                            "frequency":acd["frequency"],"power_factor":acd["pf"]})
                        except: pass
                if otd:
                    ts = str(ot_r.get("created_at",""))
                    if ts != _last_ts_outlet:
                        _last_ts_outlet = ts
                        try: winflux("energy_monitor",{"voltage":otd["voltage"],"current":otd["current"],
                            "power":otd["power"],"energy_kwh":otd["energy"],"frequency":otd["frequency"],
                            "power_factor":otd["pf"]},tags={"device":"mysql_outlet","phase":"outlet"})
                        except: pass
                if lpd:
                    ts = str(lp_r.get("created_at",""))
                    if ts != _last_ts_lamp:
                        _last_ts_lamp = ts
                        try: winflux("energy_monitor",{"voltage":lpd["voltage"],"current":lpd["current"],
                            "power":lpd["power"],"energy_kwh":lpd["energy"],"frequency":lpd["frequency"],
                            "power_factor":lpd["pf"]},tags={"device":"mysql_lamp","phase":"lamp"})
                        except: pass
                sio.emit("mqtt_update",{"type":"energy","data":mqtt_data["energy"]})
                sio.emit("mysql_energy_update",{"ac":acd,"outlet":otd,"lamp":lpd})
            if errs > 0: print("[MySQL] Koneksi kembali normal")
            errs = 0; mqtt_data["energy"]["connected"] = True
        except urllib.error.URLError as e:
            errs += 1
            if errs == 1 or errs % 12 == 0: print(f"[MySQL] HTTP error ({errs}x): {e}")
            mqtt_data["energy"]["connected"] = False
        except Exception as e:
            errs += 1
            if errs == 1 or errs % 12 == 0: print(f"[MySQL] Poll error ({errs}x): {e}")
            mqtt_data["energy"]["connected"] = False
        time.sleep(POLL_INTERVAL)

def start_polling(ctx):
    global _poll_thread, _stop_event
    _stop_event.clear()
    _poll_thread = threading.Thread(target=_poll_loop,args=(ctx,),daemon=True,name="mysql-energy-poll")
    _poll_thread.start()

def stop_polling():
    _stop_event.set()
PYEOF

echo "[2/3] Merestart Flask ..."
pkill -f "python.*app.py" || true
sleep 1
cd /home/iotlab/smartroom
nohup $VENV dashboard/app.py > /tmp/smartroom.log 2>&1 &
sleep 3

echo "[3/3] Cek status ..."
if pgrep -f "python.*app.py" > /dev/null; then
    echo "Flask BERJALAN OK"
    tail -5 /tmp/smartroom.log
else
    echo "Flask GAGAL — cek: tail -20 /tmp/smartroom.log"
fi
BASH;
    exit;
}

// ── Mode history: kembalikan data rentang tanggal untuk export CSV ────────────
// ?action=history&id_kwh=1&from=YYYY-MM-DD&to=YYYY-MM-DD[&limit=5000]
// id_kwh: 1=AC, 2=Outlet, 3=Lampu
$host = '127.0.0.1';
$db   = 'iotlabun_sbms';
$user = 'iotlabun_aslab';
$pass = 'Pass2myDB';

if (($_GET['action'] ?? '') === 'history') {
    $id_kwh = (int)($_GET['id_kwh'] ?? 1);
    $from   = $_GET['from'] ?? '';
    $to     = $_GET['to']   ?? '';
    $limit  = min((int)($_GET['limit'] ?? 5000), 50000);

    // Validasi format tanggal YYYY-MM-DD
    $date_re = '/^\d{4}-\d{2}-\d{2}$/';
    if (!preg_match($date_re, $from) || !preg_match($date_re, $to)) {
        http_response_code(400);
        header('Content-Type: application/json');
        echo json_encode(['error' => 'from and to required (YYYY-MM-DD)']);
        exit;
    }

    try {
        $pdo = new PDO("mysql:host=$host;dbname=$db;charset=utf8mb4", $user, $pass, [
            PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_TIMEOUT            => 10,
        ]);

        // Ambil data dari tabel energies dalam rentang tanggal
        $stmt = $pdo->prepare("
            SELECT id, id_kwh, tegangan, arus, frekuensi,
                   active_power, reactive_power, apparent_power,
                   created_at
            FROM energies
            WHERE id_kwh = :id_kwh
              AND created_at >= :from AND created_at < DATE_ADD(:to, INTERVAL 1 DAY)
            ORDER BY created_at ASC
            LIMIT :limit
        ");
        $stmt->bindValue(':id_kwh', $id_kwh, PDO::PARAM_INT);
        $stmt->bindValue(':from', $from, PDO::PARAM_STR);
        $stmt->bindValue(':to', $to, PDO::PARAM_STR);
        $stmt->bindValue(':limit', $limit, PDO::PARAM_INT);
        $stmt->execute();
        $rows = $stmt->fetchAll();

        // Ambil nilai energy_kwh kumulatif terakhir sebelum periode ini
        // agar bisa menghitung konsumsi relatif
        $start_kwh = 0.0;
        try {
            $s2 = $pdo->prepare("
                SELECT * FROM energy_kwh WHERE id_kwh = ? ORDER BY id DESC LIMIT 1
            ");
            $s2->execute([$id_kwh]);
            $krow = $s2->fetch();
            if ($krow) {
                foreach (['total_energy','kwh','wh','energy_wh','energy'] as $col) {
                    if (isset($krow[$col]) && (float)$krow[$col] > 0) {
                        $start_kwh = (float)$krow[$col];
                        break;
                    }
                }
            }
        } catch (Exception $e) {}

        // Hitung energy_kwh per baris dari akumulasi daya (trapezoidal)
        $result = [];
        $prev_ts   = null;
        $cumul_wh  = 0.0;
        foreach ($rows as $r) {
            $ap = (float)($r['apparent_power'] ?? 0);
            $p  = (float)($r['active_power']   ?? 0);
            $pf = ($ap > 0.001) ? round($p / $ap, 3) : 0.0;

            // Akumulasi Wh sejak baris pertama (interval antar baris dalam detik)
            $cur_ts = strtotime($r['created_at']);
            if ($prev_ts !== null && $cur_ts > $prev_ts) {
                $dt_h     = ($cur_ts - $prev_ts) / 3600.0;
                $cumul_wh += $p * $dt_h;
            }
            $prev_ts = $cur_ts;

            $result[] = [
                'timestamp'      => $r['created_at'],
                'tegangan'       => (float)($r['tegangan']   ?? 0),
                'arus'           => (float)($r['arus']       ?? 0),
                'frekuensi'      => (float)($r['frekuensi']  ?? 0),
                'active_power'   => $p,
                'reactive_power' => (float)($r['reactive_power']  ?? 0),
                'apparent_power' => $ap,
                'power_factor'   => $pf,
                'energy_kwh'     => round($cumul_wh / 1000.0, 4),   // Wh → kWh relatif
            ];
        }

        header('Content-Type: application/json');
        echo json_encode([
            'id_kwh'  => $id_kwh,
            'from'    => $from,
            'to'      => $to,
            'count'   => count($result),
            'rows'    => $result,
        ]);
    } catch (Exception $e) {
        http_response_code(500);
        header('Content-Type: application/json');
        echo json_encode(['error' => $e->getMessage()]);
    }
    exit;
}

function fetchLatestByKwh($pdo, $id_kwh) {
    // Ambil baris terbaru dari tabel energies untuk data real-time
    $stmt = $pdo->prepare("
        SELECT *
        FROM energies
        WHERE id_kwh = ?
        ORDER BY id DESC
        LIMIT 1
    ");
    $stmt->execute([$id_kwh]);
    $row = $stmt->fetch() ?: [];

    // Ambil total_energy (Wh) dari tabel energy_kwh berdasarkan id_kwh
    // Tabel energy_kwh: id=1 AC, id=3 Lampu — satuan Wh
    $stored_wh = 0.0;
    try {
        // Coba dulu WHERE id_kwh = ? (kalau kolom FK pakai id_kwh)
        $stmt2 = $pdo->prepare("SELECT * FROM energy_kwh WHERE id_kwh = ? ORDER BY id DESC LIMIT 1");
        $stmt2->execute([$id_kwh]);
        $kwh_row = $stmt2->fetch();
        if (!$kwh_row) {
            // Fallback: WHERE id = id_kwh
            $stmt2b = $pdo->prepare("SELECT * FROM energy_kwh WHERE id = ? LIMIT 1");
            $stmt2b->execute([$id_kwh]);
            $kwh_row = $stmt2b->fetch();
        }
        if ($kwh_row) {
            // Coba kolom umum dalam urutan preferensi: total_energy, kwh, wh, energy_wh, energy
            foreach (['total_energy', 'kwh', 'wh', 'energy_wh', 'energy'] as $col) {
                if (isset($kwh_row[$col]) && $kwh_row[$col] !== null && (float)$kwh_row[$col] > 0) {
                    $stored_wh = (float)$kwh_row[$col];
                    break;
                }
            }
        }
    } catch (Exception $e) {
        // energy_kwh mungkin belum ada atau struktur berbeda
    }

    // Fallback: akumulasi harian dari tabel energies (Wh)
    if ($stored_wh <= 0 && !empty($row)) {
        try {
            $stmt3 = $pdo->prepare("
                SELECT COUNT(*) AS cnt,
                       SUM(active_power) AS sum_power,
                       TIMESTAMPDIFF(SECOND, MIN(created_at), MAX(created_at)) AS span_sec
                FROM energies
                WHERE id_kwh = ? AND DATE(created_at) = CURDATE()
            ");
            $stmt3->execute([$id_kwh]);
            $calc = $stmt3->fetch();
            if ($calc && (int)$calc['cnt'] > 1) {
                // Estimasi interval rata-rata antar record (detik)
                $interval   = (float)$calc['span_sec'] / ((int)$calc['cnt'] - 1);
                $interval   = ($interval > 0 && $interval <= 60) ? $interval : 5.0;
                // Akumulasi Wh = SUM(power_W) * interval_s / 3600
                $stored_wh  = round((float)$calc['sum_power'] * $interval / 3600.0, 3);
            } elseif ($calc && (int)$calc['cnt'] === 1) {
                $stored_wh  = round((float)$row['active_power'] * 5.0 / 3600.0, 3);
            }
        } catch (Exception $e) { /* fallback */ }
    }

    // Masukkan ke row — total_energy dalam Wh (JS membagi /1000 untuk kWh)
    $row['total_energy']    = $stored_wh;
    $row['active_power']    = isset($row['active_power'])    ? (float)$row['active_power']    : 0.0;
    $row['reactive_power']  = isset($row['reactive_power'])  ? (float)$row['reactive_power']  : 0.0;
    $row['apparent_power']  = isset($row['apparent_power'])  ? (float)$row['apparent_power']  : 0.0;
    $row['tegangan']        = isset($row['tegangan'])        ? (float)$row['tegangan']        : 0.0;
    $row['arus']            = isset($row['arus'])            ? (float)$row['arus']            : 0.0;
    $row['frekuensi']       = isset($row['frekuensi'])       ? (float)$row['frekuensi']       : 0.0;
    return $row ?: null;
}

try {
    $pdo = new PDO("mysql:host=$host;dbname=$db;charset=utf8mb4", $user, $pass, [
        PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_TIMEOUT            => 5,
    ]);

    $ac     = fetchLatestByKwh($pdo, 1);  // id_kwh=1 → AC
    $outlet = fetchLatestByKwh($pdo, 2);  // id_kwh=2 → Outlet
    $lamp   = fetchLatestByKwh($pdo, 3);  // id_kwh=3 → Lampu

    header('Content-Type: application/json');
    echo json_encode([
        'ac'     => $ac ?: (object)[],
        'outlet' => $outlet ?: (object)[],
        'lamp'   => $lamp ?: (object)[]
    ]);

} catch (Exception $e) {
    http_response_code(500);
    header('Content-Type: application/json');
    echo json_encode(['error' => $e->getMessage(), 'file' => basename($e->getFile()), 'line' => $e->getLine()]);
}
