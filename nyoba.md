from flask import Flask, render_template_string, jsonify, request, Response
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
import json
import cv2
from threading import Lock

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartroom_secret_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# ===================== CONFIGURATION =====================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
mqtt_client = mqtt.Client()

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "rfi_HvWdjwaG8jB3Rqx6g0y5kMWRfSfq_HmLLUvkom1yaHKvwonU9Qfj6nlZjTqb_I0leIREUnMhvQQXtgETfg=="
INFLUX_ORG = "iotlab"
INFLUX_BUCKET = "sensordata"

# Camera Configuration
camera = None
camera_lock = Lock()

# ===================== DATA STORAGE =====================
latest_data = {
    'ac': {'temperature': 0, 'humidity': 0, 'mode': 'MANUAL', 'state': 'OFF', 'set_temp': 24},
    'lamp': {'lux': 0, 'brightness': 0, 'pir': 0, 'mode': 'MANUAL'},
    'power': {'ac_watts': 0, 'lamp_watts': 0, 'total_watts': 0, 'daily_kwh': 0, 'daily_cost': 0}
}

ir_codes = {}
system_logs = []

# ===================== CAMERA FUNCTIONS =====================
def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames():
    while True:
        with camera_lock:
            cam = get_camera()
            success, frame = cam.read()
            if not success:
                # Retry - reinitialize camera
                global camera
                if camera is not None:
                    camera.release()
                    camera = None
                break
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ===================== INFLUXDB FUNCTIONS =====================
def get_influx_data(measurement, field, hours=1):
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["_field"] == "{field}")
          |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        '''
        result = query_api.query(query)
        data = []
        for table in result:
            for record in table.records:
                data.append({
                    'time': record.get_time().isoformat(),
                    'value': round(record.get_value(), 2)
                })
        client.close()
        return data
    except Exception as e:
        print(f"InfluxDB Error: {e}")
        return []

def add_log(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    system_logs.append(log_entry)
    if len(system_logs) > 200:
        system_logs.pop(0)
    socketio.emit('new_log', {'log': log_entry})

# ===================== MQTT CALLBACKS =====================
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker (rc={rc})")
    client.subscribe("smartroom/#")
    add_log("MQTT Connected to broker")

def on_message(client, userdata, msg):
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode())

        if topic == "smartroom/ac/sensors":
            latest_data['ac']['temperature'] = payload.get('temperature', 0)
            latest_data['ac']['humidity'] = payload.get('humidity', 0)
            # Power calculation AC
            if latest_data['ac']['state'] == 'ON':
                latest_data['power']['ac_watts'] = 750
            else:
                latest_data['power']['ac_watts'] = 0
            calculate_power()
            socketio.emit('sensor_update', {'type': 'ac', 'data': payload})

        elif topic == "smartroom/lamp/sensors":
            latest_data['lamp']['lux'] = payload.get('lux', 0)
            latest_data['lamp']['brightness'] = payload.get('brightness', 0)
            latest_data['lamp']['pir'] = payload.get('pir', 0)
            # Power calculation Lamp
            latest_data['power']['lamp_watts'] = round((payload.get('brightness', 0) / 255) * 10, 2)
            calculate_power()
            socketio.emit('sensor_update', {'type': 'lamp', 'data': payload})

        elif topic == "smartroom/ac/mode":
            latest_data['ac']['mode'] = payload.get('mode', 'MANUAL')
            socketio.emit('mode_update', {'device': 'ac', 'mode': payload.get('mode')})
            add_log(f"AC mode changed to {payload.get('mode')}")

        elif topic == "smartroom/lamp/mode":
            latest_data['lamp']['mode'] = payload.get('mode', 'MANUAL')
            socketio.emit('mode_update', {'device': 'lamp', 'mode': payload.get('mode')})
            add_log(f"Lamp mode changed to {payload.get('mode')}")

        elif topic == "smartroom/ac/state":
            latest_data['ac']['state'] = payload.get('state', 'OFF')
            socketio.emit('ac_state_update', {'state': payload.get('state')})
            add_log(f"AC state changed to {payload.get('state')}")

        elif topic == "smartroom/ir/learned":
            button = payload.get('button', '')
            code = payload.get('code', '')
            if button and code:
                ir_codes[button] = code
                socketio.emit('ir_learned', {'button': button, 'code': code})
                add_log(f"IR Code learned: {button}")

        elif topic == "smartroom/ac/optimization":
            socketio.emit('ac_optimization', payload)
            add_log(f"AC Optimization: temp={payload.get('set_temp', 'N/A')}")

        elif topic == "smartroom/lamp/optimization":
            socketio.emit('lamp_optimization', payload)
            add_log(f"Lamp Optimization: brightness={payload.get('brightness', 'N/A')}")

    except Exception as e:
        print(f"MQTT Error: {e}")

def calculate_power():
    ac_w = latest_data['power']['ac_watts']
    lamp_w = latest_data['power']['lamp_watts']
    total = ac_w + lamp_w
    latest_data['power']['total_watts'] = round(total, 2)
    latest_data['power']['daily_kwh'] = round((total / 1000) * 24, 2)
    latest_data['power']['daily_cost'] = round((total / 1000) * 24 * 1500, 0)
    socketio.emit('power_update', latest_data['power'])

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"MQTT Connection failed: {e}")

# ===================== ROUTES =====================
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
        if cam.isOpened():
            return jsonify({'status': 'active', 'resolution': '640x480', 'fps': 30})
        else:
            return jsonify({'status': 'inactive'})
    except:
        return jsonify({'status': 'error'})

@app.route('/api/data')
def get_data():
    return jsonify(latest_data)

@app.route('/api/chart/<measurement>/<field>/<int:hours>')
def get_chart_data(measurement, field, hours):
    data = get_influx_data(measurement, field, hours)
    return jsonify(data)

@app.route('/api/ac/control', methods=['POST'])
def control_ac():
    data = request.json
    mqtt_client.publish('smartroom/ac/control', json.dumps(data))
    add_log(f"AC Control: {data}")
    return jsonify({'status': 'success'})

@app.route('/api/lamp/control', methods=['POST'])
def control_lamp():
    data = request.json
    mqtt_client.publish('smartroom/lamp/control', json.dumps(data))
    add_log(f"Lamp Control: brightness={data.get('brightness', 'N/A')}")
    return jsonify({'status': 'success'})

@app.route('/api/ac/mode', methods=['POST'])
def set_ac_mode():
    data = request.json
    mode = data.get('mode', 'MANUAL')
    mqtt_client.publish('smartroom/ac/mode', json.dumps({'mode': mode}))
    latest_data['ac']['mode'] = mode
    add_log(f"AC mode set to {mode}")
    return jsonify({'status': 'success', 'mode': mode})

@app.route('/api/lamp/mode', methods=['POST'])
def set_lamp_mode():
    data = request.json
    mode = data.get('mode', 'MANUAL')
    mqtt_client.publish('smartroom/lamp/mode', json.dumps({'mode': mode}))
    latest_data['lamp']['mode'] = mode
    add_log(f"Lamp mode set to {mode}")
    return jsonify({'status': 'success', 'mode': mode})

@app.route('/api/ir/learn', methods=['POST'])
def learn_ir():
    data = request.json
    button = data.get('button', '')
    mqtt_client.publish('smartroom/ir/learn', json.dumps({'button': button}))
    add_log(f"IR Learning started for: {button}")
    return jsonify({'status': 'success', 'message': f'Learning IR code for {button}'})

@app.route('/api/ir/send', methods=['POST'])
def send_ir():
    data = request.json
    button = data.get('button', '')
    if button in ir_codes:
        mqtt_client.publish('smartroom/ir/send', json.dumps({'button': button, 'code': ir_codes[button]}))
        add_log(f"IR Code sent: {button}")
        return jsonify({'status': 'success', 'message': f'Sent IR code for {button}'})
    return jsonify({'status': 'error', 'message': 'IR code not found'}), 404

@app.route('/api/ir/codes')
def get_ir_codes():
    return jsonify(ir_codes)

@app.route('/api/logs')
def get_logs():
    return jsonify(system_logs)

# ===================== SOCKETIO EVENTS =====================
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('initial_data', latest_data)
    emit('initial_logs', system_logs[-50:])

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# ===================== HTML TEMPLATE =====================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Room Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --primary: #667eea;
            --primary-dark: #5568d3;
            --secondary: #764ba2;
            --success: #4CAF50;
            --danger: #f44336;
            --warning: #FF9800;
            --info: #2196F3;
            --dark: #333;
            --light: #f5f5f5;
            --white: #fff;
            --shadow: 0 2px 10px rgba(0,0,0,0.1);
            --shadow-hover: 0 5px 20px rgba(0,0,0,0.15);
            --radius: 15px;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            min-height: 100vh;
        }

        .container { display: flex; min-height: 100vh; }

        /* ========== SIDEBAR ========== */
        .sidebar {
            width: 260px;
            background: rgba(255,255,255,0.97);
            padding: 25px 15px;
            box-shadow: 2px 0 15px rgba(0,0,0,0.1);
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            z-index: 100;
        }

        .logo {
            font-size: 26px;
            font-weight: bold;
            color: var(--primary);
            margin-bottom: 10px;
            text-align: center;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }

        .logo-subtitle {
            font-size: 11px;
            color: #999;
            text-align: center;
            margin-bottom: 25px;
        }

        .nav-item {
            padding: 12px 18px;
            margin: 4px 0;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 14px;
            color: #555;
        }

        .nav-item:hover { background: #f0f0f0; color: var(--primary); }
        .nav-item.active { background: var(--primary); color: white; font-weight: 600; }

        .sidebar-footer {
            position: absolute;
            bottom: 15px;
            left: 15px;
            right: 15px;
            text-align: center;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }

        .connection-status {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-size: 12px;
            color: #666;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--danger);
        }

        .status-dot.connected {
            background: var(--success);
            box-shadow: 0 0 6px var(--success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* ========== MAIN CONTENT ========== */
        .main-content {
            flex: 1;
            margin-left: 260px;
            padding: 25px 30px;
            overflow-y: auto;
        }

        .header {
            background: var(--white);
            padding: 22px 30px;
            border-radius: var(--radius);
            margin-bottom: 25px;
            box-shadow: var(--shadow);
        }

        .header h1 { color: var(--dark); font-size: 24px; margin-bottom: 5px; }
        .header p { color: #888; font-size: 14px; }

        .header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .live-badge {
            background: var(--danger);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            animation: pulse 1.5s infinite;
        }

        /* ========== STATS GRID ========== */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }

        .stat-card {
            background: var(--white);
            padding: 22px;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }

        .stat-card:hover { transform: translateY(-5px); box-shadow: var(--shadow-hover); }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }

        .stat-card.temp::before { background: #f44336; }
        .stat-card.humidity::before { background: #2196F3; }
        .stat-card.lux::before { background: #FF9800; }
        .stat-card.brightness::before { background: #4CAF50; }
        .stat-card.power::before { background: #9C27B0; }
        .stat-card.cost::before { background: #E91E63; }

        .stat-card h3 {
            color: #888;
            font-size: 13px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: var(--primary);
            line-height: 1.2;
        }

        .stat-unit { font-size: 16px; color: #aaa; font-weight: normal; }
        .stat-sub { font-size: 12px; color: #aaa; margin-top: 5px; }

        /* ========== CHART ========== */
        .chart-container {
            background: var(--white);
            padding: 25px;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            margin-bottom: 20px;
        }

        .chart-container h3 { color: var(--dark); margin-bottom: 15px; }

        .time-range-selector {
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
            justify-content: center;
        }

        .time-btn {
            padding: 6px 16px;
            border: 2px solid var(--primary);
            background: white;
            color: var(--primary);
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 13px;
            font-weight: 600;
        }

        .time-btn:hover { background: rgba(102,126,234,0.1); }
        .time-btn.active { background: var(--primary); color: white; }

        /* ========== PAGE CONTENT ========== */
        .page-content { display: none; }
        .page-content.active { display: block; }

        /* ========== CONTROL BUTTONS ========== */
        .control-btn {
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            margin: 5px;
        }

        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: var(--primary-dark); transform: translateY(-2px); }
        .btn-success { background: var(--success); color: white; }
        .btn-success:hover { background: #45a049; transform: translateY(-2px); }
        .btn-danger { background: var(--danger); color: white; }
        .btn-danger:hover { background: #da190b; transform: translateY(-2px); }
        .btn-warning { background: var(--warning); color: white; }
        .btn-info { background: var(--info); color: white; }

        /* ========== MODE BADGE ========== */
        .mode-badge {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .mode-adaptive { background: var(--success); color: white; }
        .mode-manual { background: var(--warning); color: white; }
        .mode-badge:hover { opacity: 0.85; transform: scale(1.05); }

        /* ========== IR REMOTE ========== */
        .ir-remote-container {
            background: linear-gradient(145deg, #2c2c2c, #1a1a1a);
            border-radius: 20px;
            padding: 30px 25px;
            max-width: 400px;
            margin: 20px auto;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .ir-remote-title {
            color: #ccc;
            text-align: center;
            font-size: 14px;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .ir-button-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        .ir-button {
            text-align: center;
            padding: 12px 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            transition: all 0.3s;
        }

        .ir-button:hover { background: rgba(255,255,255,0.1); }
        .ir-button h4 { color: #ddd; font-size: 12px; margin-bottom: 8px; }

        .ir-btn {
            width: 100%;
            padding: 8px;
            margin: 3px 0;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 11px;
            transition: all 0.3s;
        }

        .ir-btn.learn { background: var(--info); color: white; }
        .ir-btn.learn:hover { background: #0b7dda; }
        .ir-btn.send { background: var(--success); color: white; }
        .ir-btn.send:hover { background: #45a049; }
        .ir-btn:disabled { background: #555; cursor: not-allowed; color: #999; }

        .ir-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-left: 5px;
            background: #555;
        }

        .ir-status.learned { background: var(--success); box-shadow: 0 0 5px var(--success); }

        /* ========== CAMERA ========== */
        .camera-container {
            text-align: center;
            background: #000;
            padding: 15px;
            border-radius: 12px;
            position: relative;
        }

        .camera-feed {
            width: 100%;
            max-width: 800px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }

        .camera-overlay {
            position: absolute;
            top: 25px;
            left: 25px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .camera-rec {
            background: var(--danger);
            color: white;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            animation: pulse 1s infinite;
        }

        .camera-time {
            color: white;
            font-size: 12px;
            background: rgba(0,0,0,0.5);
            padding: 3px 8px;
            border-radius: 4px;
        }

        /* ========== SLIDER ========== */
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #e0e0e0;
            outline: none;
            margin: 10px 0;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }

        /* ========== LOGS ========== */
        .logs-container {
            max-height: 500px;
            overflow-y: auto;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 8px;
            line-height: 1.6;
        }

        .log-entry { padding: 2px 0; border-bottom: 1px solid #2a2a2a; }
        .log-time { color: #569cd6; }
        .log-mqtt { color: #4ec9b0; }
        .log-ir { color: #ce9178; }
        .log-error { color: #f44747; }

        /* ========== POWER GAUGE ========== */
        .gauge-container {
            text-align: center;
            padding: 20px;
        }

        .gauge-value {
            font-size: 48px;
            font-weight: bold;
            color: var(--primary);
        }

        .gauge-label {
            font-size: 14px;
            color: #888;
            margin-top: 5px;
        }

        .power-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin: 10px 0;
            overflow: hidden;
        }

        .power-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        .power-bar-fill.ac { background: linear-gradient(90deg, #2196F3, #667eea); }
        .power-bar-fill.lamp { background: linear-gradient(90deg, #FF9800, #FFD54F); }

        /* ========== RESPONSIVE ========== */
        @media (max-width: 768px) {
            .sidebar { display: none; }
            .main-content { margin-left: 0; }
            .stats-grid { grid-template-columns: 1fr; }
            .ir-button-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- ========== SIDEBAR ========== -->
        <div class="sidebar">
            <div class="logo">🏠 Smart Room</div>
            <div class="logo-subtitle">IoT Monitoring & Control</div>

            <div class="nav-item active" onclick="showPage('dashboard', this)">📊 Dashboard</div>
            <div class="nav-item" onclick="showPage('ac-analytics', this)">❄️ AC Analytics</div>
            <div class="nav-item" onclick="showPage('lamp-analytics', this)">💡 Lamp Analytics</div>
            <div class="nav-item" onclick="showPage('camera', this)">📷 Camera</div>
            <div class="nav-item" onclick="showPage('power', this)">⚡ Power</div>
            <div class="nav-item" onclick="showPage('control', this)">🎮 Control Panel</div>
            <div class="nav-item" onclick="showPage('logs', this)">📝 Logs</div>

            <div class="sidebar-footer">
                <div class="connection-status">
                    <div class="status-dot" id="mqtt-status"></div>
                    <span id="mqtt-status-text">Connecting...</span>
                </div>
            </div>
        </div>

        <!-- ========== MAIN CONTENT ========== -->
        <div class="main-content">

            <!-- ===== DASHBOARD PAGE ===== -->
            <div id="dashboard-page" class="page-content active">
                <div class="header">
                    <div class="header-row">
                        <div>
                            <h1>Dashboard Overview</h1>
                            <p>Real-time monitoring sistem Smart Room</p>
                        </div>
                        <span class="live-badge">● LIVE</span>
                    </div>
                </div>

                <div class="stats-grid">
                    <div class="stat-card temp">
                        <h3>🌡️ Temperature</h3>
                        <p class="stat-value" id="temp-value">0<span class="stat-unit">°C</span></p>
                        <p class="stat-sub">DHT22 Sensor</p>
                    </div>
                    <div class="stat-card humidity">
                        <h3>💧 Humidity</h3>
                        <p class="stat-value" id="humidity-value">0<span class="stat-unit">%</span></p>
                        <p class="stat-sub">DHT22 Sensor</p>
                    </div>
                    <div class="stat-card lux">
                        <h3>☀️ Light Intensity</h3>
                        <p class="stat-value" id="lux-value">0<span class="stat-unit"> lux</span></p>
                        <p class="stat-sub">BH1750 Sensor</p>
                    </div>
                    <div class="stat-card brightness">
                        <h3>💡 Lamp Brightness</h3>
                        <p class="stat-value" id="brightness-value">0<span class="stat-unit">%</span></p>
                        <p class="stat-sub">DAC Output</p>
                    </div>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>❄️ AC Status</h3>
                        <p class="stat-value" id="ac-state-dash" style="font-size:24px;">OFF</p>
                        <p class="stat-sub">Mode: <span id="ac-mode-dash">MANUAL</span></p>
                    </div>
                    <div class="stat-card">
                        <h3>🚶 Motion</h3>
                        <p class="stat-value" id="pir-value" style="font-size:24px;">No Motion</p>
                        <p class="stat-sub">PIR Sensor</p>
                    </div>
                    <div class="stat-card">
                        <h3>⚡ Total Power</h3>
                        <p class="stat-value" id="total-power-dash" style="font-size:24px;">0<span class="stat-unit"> W</span></p>
                        <p class="stat-sub">Konsumsi saat ini</p>
                    </div>
                    <div class="stat-card">
                        <h3>💰 Daily Cost</h3>
                        <p class="stat-value" id="daily-cost-dash" style="font-size:24px;">Rp 0</p>
                        <p class="stat-sub">Estimasi biaya harian</p>
                    </div>
                </div>
            </div>

            <!-- ===== AC ANALYTICS PAGE ===== -->
            <div id="ac-analytics-page" class="page-content">
                <div class="header">
                    <h1>AC Analytics</h1>
                    <p>Analisis data suhu dan kelembaban - Genetic Algorithm Optimization</p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card temp">
                        <h3>🌡️ Current Temperature</h3>
                        <p class="stat-value" id="ac-temp-detail">0<span class="stat-unit">°C</span></p>
                    </div>
                    <div class="stat-card humidity">
                        <h3>💧 Current Humidity</h3>
                        <p class="stat-value" id="ac-humidity-detail">0<span class="stat-unit">%</span></p>
                    </div>
                    <div class="stat-card">
                        <h3>🎯 Set Temperature</h3>
                        <p class="stat-value" id="ac-settemp" style="font-size:24px;">24<span class="stat-unit">°C</span></p>
                    </div>
                </div>

                <div class="chart-container">
                    <h3>📈 Temperature History</h3>
                    <div class="time-range-selector">
                        <button class="time-btn active" onclick="updateChart('temp', 1, this)">1H</button>
                        <button class="time-btn" onclick="updateChart('temp', 6, this)">6H</button>
                        <button class="time-btn" onclick="updateChart('temp', 24, this)">24H</button>
                    </div>
                    <canvas id="tempChart" height="100"></canvas>
                </div>

                <div class="chart-container">
                    <h3>📈 Humidity History</h3>
                    <div class="time-range-selector">
                        <button class="time-btn active" onclick="updateChart('humidity', 1, this)">1H</button>
                        <button class="time-btn" onclick="updateChart('humidity', 6, this)">6H</button>
                        <button class="time-btn" onclick="updateChart('humidity', 24, this)">24H</button>
                    </div>
                    <canvas id="humidityChart" height="100"></canvas>
                </div>
            </div>

            <!-- ===== LAMP ANALYTICS PAGE ===== -->
            <div id="lamp-analytics-page" class="page-content">
                <div class="header">
                    <h1>Lamp Analytics</h1>
                    <p>Analisis data cahaya dan brightness - PSO Optimization</p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card lux">
                        <h3>☀️ Current Lux</h3>
                        <p class="stat-value" id="lamp-lux-detail">0<span class="stat-unit"> lux</span></p>
                    </div>
                    <div class="stat-card brightness">
                        <h3>💡 Current Brightness</h3>
                        <p class="stat-value" id="lamp-brightness-detail">0<span class="stat-unit">%</span></p>
                    </div>
                    <div class="stat-card">
                        <h3>🚶 Motion Status</h3>
                        <p class="stat-value" id="lamp-pir-detail" style="font-size:24px;">No Motion</p>
                    </div>
                </div>

                <div class="chart-container">
                    <h3>📈 Light Intensity History</h3>
                    <div class="time-range-selector">
                        <button class="time-btn active" onclick="updateChart('lux', 1, this)">1H</button>
                        <button class="time-btn" onclick="updateChart('lux', 6, this)">6H</button>
                        <button class="time-btn" onclick="updateChart('lux', 24, this)">24H</button>
                    </div>
                    <canvas id="luxChart" height="100"></canvas>
                </div>

                <div class="chart-container">
                    <h3>📈 Brightness History</h3>
                    <div class="time-range-selector">
                        <button class="time-btn active" onclick="updateChart('brightness', 1, this)">1H</button>
                        <button class="time-btn" onclick="updateChart('brightness', 6, this)">6H</button>
                        <button class="time-btn" onclick="updateChart('brightness', 24, this)">24H</button>
                    </div>
                    <canvas id="brightnessChart" height="100"></canvas>
                </div>
            </div>

            <!-- ===== CAMERA PAGE ===== -->
            <div id="camera-page" class="page-content">
                <div class="header">
                    <div class="header-row">
                        <div>
                            <h1>Live Camera Feed</h1>
                            <p>Monitoring ruangan secara real-time via USB Camera</p>
                        </div>
                        <span class="live-badge">● REC</span>
                    </div>
                </div>

                <div class="stats-grid">
                    <div class="stat-card" style="grid-column: 1 / -1; padding: 0;">
                        <div class="camera-container">
                            <div class="camera-overlay">
                                <span class="camera-rec">● REC</span>
                                <span class="camera-time" id="camera-time">00:00:00</span>
                            </div>
                            <img id="camera-img" src="/video_feed" class="camera-feed" alt="Camera Feed"
                                 onerror="this.style.display='none'; document.getElementById('camera-error').style.display='block';">
                            <div id="camera-error" style="display:none; padding:80px 20px; color:#aaa;">
                                <h3>📷 Camera Not Available</h3>
                                <p style="margin-top:10px; font-size:14px;">Pastikan kamera USB terhubung dan jalankan ulang server</p>
                                <button class="control-btn btn-primary" style="margin-top:15px;" onclick="retryCamera()">🔄 Retry</button>
                            </div>
                        </div>
                    </div>

                    <div class="stat-card">
                        <h3>📹 Camera Status</h3>
                        <p class="stat-value" id="camera-status-text" style="font-size:22px; color: var(--success);">
                            <span class="status-dot connected" style="display:inline-block; width:10px; height:10px;"></span>
                            ACTIVE
                        </p>
                        <p class="stat-sub">USB Camera #0</p>
                    </div>
                    <div class="stat-card">
                        <h3>📐 Resolution</h3>
                        <p class="stat-value" style="font-size:22px;">640 × 480</p>
                        <p class="stat-sub">VGA Quality</p>
                    </div>
                    <div class="stat-card">
                        <h3>🔄 Frame Rate</h3>
                        <p class="stat-value" style="font-size:22px;">30 FPS</p>
                        <p class="stat-sub">Smooth streaming</p>
                    </div>
                </div>
            </div>

            <!-- ===== POWER PAGE ===== -->
            <div id="power-page" class="page-content">
                <div class="header">
                    <h1>Power Consumption</h1>
                    <p>Monitoring dan estimasi konsumsi daya listrik</p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card power">
                        <h3>❄️ AC Power</h3>
                        <p class="stat-value" id="ac-power">0<span class="stat-unit"> W</span></p>
                        <div class="power-bar"><div class="power-bar-fill ac" id="ac-power-bar" style="width:0%"></div></div>
                        <p class="stat-sub">Max: 750W</p>
                    </div>
                    <div class="stat-card power">
                        <h3>💡 Lamp Power</h3>
                        <p class="stat-value" id="lamp-power">0<span class="stat-unit"> W</span></p>
                        <div class="power-bar"><div class="power-bar-fill lamp" id="lamp-power-bar" style="width:0%"></div></div>
                        <p class="stat-sub">Max: 10W</p>
                    </div>
                    <div class="stat-card power">
                        <h3>⚡ Total Power</h3>
                        <p class="stat-value" id="total-power">0<span class="stat-unit"> W</span></p>
                        <p class="stat-sub">Combined consumption</p>
                    </div>
                    <div class="stat-card cost">
                        <h3>📊 Daily Energy</h3>
                        <p class="stat-value" id="daily-kwh">0<span class="stat-unit"> kWh</span></p>
                        <p class="stat-sub">Estimasi 24 jam</p>
                    </div>
                    <div class="stat-card cost">
                        <h3>💰 Daily Cost</h3>
                        <p class="stat-value" id="daily-cost" style="color: var(--danger);">Rp 0</p>
                        <p class="stat-sub">Tarif: Rp 1.500/kWh</p>
                    </div>
                    <div class="stat-card cost">
                        <h3>📅 Monthly Estimate</h3>
                        <p class="stat-value" id="monthly-cost" style="color: var(--danger); font-size:24px;">Rp 0</p>
                        <p class="stat-sub">Estimasi 30 hari</p>
                    </div>
                </div>
            </div>

            <!-- ===== CONTROL PANEL PAGE ===== -->
            <div id="control-page" class="page-content">
                <div class="header">
                    <h1>Control Panel</h1>
                    <p>Kontrol perangkat Smart Room</p>
                </div>

                <div class="stats-grid">
                    <!-- AC Control -->
                    <div class="stat-card">
                        <h3>❄️ AC Control</h3>
                        <p style="margin:10px 0;">Mode:
                            <span class="mode-badge mode-manual" id="ac-mode-badge" onclick="toggleMode('ac')">MANUAL</span>
                        </p>
                        <p style="margin:10px 0;">State:
                            <span id="ac-state-control" style="font-weight:bold; color:var(--danger);">OFF</span>
                        </p>
                        <div style="margin-top: 15px;">
                            <button class="control-btn btn-success" onclick="controlAC('ON')">⚡ ON</button>
                            <button class="control-btn btn-danger" onclick="controlAC('OFF')">⭕ OFF</button>
                        </div>
                    </div>

                    <!-- Lamp Control -->
                    <div class="stat-card">
                        <h3>💡 Lamp Control</h3>
                        <p style="margin:10px 0;">Mode:
                            <span class="mode-badge mode-manual" id="lamp-mode-badge" onclick="toggleMode('lamp')">MANUAL</span>
                        </p>
                        <div style="margin-top: 15px;">
                            <input type="range" id="brightness-slider" min="0" max="100" value="0"
                                   oninput="document.getElementById('brightness-display').textContent=this.value"
                                   onchange="controlLamp(this.value)">
                            <p style="text-align:center; font-size:18px; font-weight:bold; color:var(--primary);">
                                <span id="brightness-display">0</span>%
                            </p>
                        </div>
                    </div>

                    <!-- IR Remote -->
                    <div class="stat-card" style="grid-column: 1 / -1;">
                        <h3>📡 IR Remote Learning</h3>
                        <p style="color:#888; margin-bottom:10px;">Pelajari dan kirim kode remote AC</p>

                        <div class="ir-remote-container">
                            <div class="ir-remote-title">AC Remote Control</div>
                            <div class="ir-button-grid">
                                <div class="ir-button">
                                    <h4>⏻ POWER <span class="ir-status" id="ir-status-POWER"></span></h4>
                                    <button class="ir-btn learn" onclick="learnIR('POWER')">📡 Learn</button>
                                    <button class="ir-btn send" onclick="sendIR('POWER')">📤 Send</button>
                                </div>
                                <div class="ir-button">
                                    <h4>🔺 TEMP + <span class="ir-status" id="ir-status-TEMP_UP"></span></h4>
                                    <button class="ir-btn learn" onclick="learnIR('TEMP_UP')">📡 Learn</button>
                                    <button class="ir-btn send" onclick="sendIR('TEMP_UP')">📤 Send</button>
                                </div>
                                <div class="ir-button">
                                    <h4>🔻 TEMP - <span class="ir-status" id="ir-status-TEMP_DOWN"></span></h4>
                                    <button class="ir-btn learn" onclick="learnIR('TEMP_DOWN')">📡 Learn</button>
                                    <button class="ir-btn send" onclick="sendIR('TEMP_DOWN')">📤 Send</button>
                                </div>
                                <div class="ir-button">
                                    <h4>🌀 FAN <span class="ir-status" id="ir-status-FAN"></span></h4>
                                    <button class="ir-btn learn" onclick="learnIR('FAN')">📡 Learn</button>
                                    <button class="ir-btn send" onclick="sendIR('FAN')">📤 Send</button>
                                </div>
                                <div class="ir-button">
                                    <h4>🔄 MODE <span class="ir-status" id="ir-status-MODE"></span></h4>
                                    <button class="ir-btn learn" onclick="learnIR('MODE')">📡 Learn</button>
                                    <button class="ir-btn send" onclick="sendIR('MODE')">📤 Send</button>
                                </div>
                                <div class="ir-button">
                                    <h4>↕️ SWING <span class="ir-status" id="ir-status-SWING"></span></h4>
                                    <button class="ir-btn learn" onclick="learnIR('SWING')">📡 Learn</button>
                                    <button class="ir-btn send" onclick="sendIR('SWING')">📤 Send</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ===== LOGS PAGE ===== -->
            <div id="logs-page" class="page-content">
                <div class="header">
                    <div class="header-row">
                        <div>
                            <h1>System Logs</h1>
                            <p>Log aktivitas sistem real-time</p>
                        </div>
                        <button class="control-btn btn-danger" onclick="clearLogs()">🗑️ Clear</button>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="logs-container" id="logs-container">
                        <div class="log-entry"><span class="log-time">[System]</span> Waiting for events...</div>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <script>
        // ===================== SOCKET.IO =====================
        const socket = io();
        let charts = {};
        let chartRanges = { temp: 1, humidity: 1, lux: 1, brightness: 1 };

        socket.on('connect', function() {
            document.getElementById('mqtt-status').classList.add('connected');
            document.getElementById('mqtt-status-text').textContent = 'Connected';
            addLog('System', 'Connected to server');
        });

        socket.on('disconnect', function() {
            document.getElementById('mqtt-status').classList.remove('connected');
            document.getElementById('mqtt-status-text').textContent = 'Disconnected';
        });

        socket.on('initial_data', function(data) {
            if (data.ac) {
                updateACDisplay(data.ac);
            }
            if (data.lamp) {
                updateLampDisplay(data.lamp);
            }
            if (data.power) {
                updatePowerDisplay(data.power);
            }
        });

        socket.on('initial_logs', function(logs) {
            logs.forEach(log => {
                appendLogRaw(log);
            });
        });

        socket.on('sensor_update', function(data) {
            if (data.type === 'ac') {
                updateACDisplay(data.data);
                updateRealtimeChart('tempChart', data.data.temperature);
                updateRealtimeChart('humidityChart', data.data.humidity);
            } else if (data.type === 'lamp') {
                updateLampDisplay(data.data);
                updateRealtimeChart('luxChart', data.data.lux);
                updateRealtimeChart('brightnessChart', data.data.brightness);
            }
        });

        socket.on('mode_update', function(data) {
            const badge = document.getElementById(data.device + '-mode-badge');
            if (badge) {
                badge.textContent = data.mode;
                badge.className = 'mode-badge mode-' + data.mode.toLowerCase();
            }
            if (data.device === 'ac') {
                document.getElementById('ac-mode-dash').textContent = data.mode;
            }
            addLog('Mode', data.device.toUpperCase() + ' → ' + data.mode);
        });

        socket.on('ac_state_update', function(data) {
            document.getElementById('ac-state-dash').textContent = data.state;
            document.getElementById('ac-state-dash').style.color = data.state === 'ON' ? 'var(--success)' : 'var(--danger)';
            document.getElementById('ac-state-control').textContent = data.state;
            document.getElementById('ac-state-control').style.color = data.state === 'ON' ? 'var(--success)' : 'var(--danger)';
        });

        socket.on('power_update', function(data) {
            updatePowerDisplay(data);
        });

        socket.on('ir_learned', function(data) {
            const statusDot = document.getElementById('ir-status-' + data.button);
            if (statusDot) {
                statusDot.classList.add('learned');
            }
            addLog('IR', 'Code learned: ' + data.button);
            alert('✅ IR Code untuk ' + data.button + ' berhasil dipelajari!');
        });

        socket.on('new_log', function(data) {
            appendLogRaw(data.log);
        });

        socket.on('ac_optimization', function(data) {
            if (data.set_temp) {
                document.getElementById('ac-settemp').innerHTML = data.set_temp + '<span class="stat-unit">°C</span>';
            }
        });

        // ===================== DISPLAY UPDATES =====================
        function updateACDisplay(data) {
            let temp = data.temperature || 0;
            let hum = data.humidity || 0;
            document.getElementById('temp-value').innerHTML = temp.toFixed(1) + '<span class="stat-unit">°C</span>';
            document.getElementById('humidity-value').innerHTML = hum.toFixed(1) + '<span class="stat-unit">%</span>';
            document.getElementById('ac-temp-detail').innerHTML = temp.toFixed(1) + '<span class="stat-unit">°C</span>';
            document.getElementById('ac-humidity-detail').innerHTML = hum.toFixed(1) + '<span class="stat-unit">%</span>';

            if (data.mode) {
                document.getElementById('ac-mode-dash').textContent = data.mode;
                let badge = document.getElementById('ac-mode-badge');
                badge.textContent = data.mode;
                badge.className = 'mode-badge mode-' + data.mode.toLowerCase();
            }
        }

        function updateLampDisplay(data) {
            let lux = data.lux || 0;
            let brightness = data.brightness || 0;
            let brightnessPercent = Math.round((brightness / 255) * 100);
            let pir = data.pir || 0;

            document.getElementById('lux-value').innerHTML = lux.toFixed(0) + '<span class="stat-unit"> lux</span>';
            document.getElementById('brightness-value').innerHTML = brightnessPercent + '<span class="stat-unit">%</span>';
            document.getElementById('lamp-lux-detail').innerHTML = lux.toFixed(0) + '<span class="stat-unit"> lux</span>';
            document.getElementById('lamp-brightness-detail').innerHTML = brightnessPercent + '<span class="stat-unit">%</span>';

            document.getElementById('pir-value').textContent = pir ? '🚶 Motion!' : 'No Motion';
            document.getElementById('pir-value').style.color = pir ? 'var(--success)' : '#aaa';
            document.getElementById('lamp-pir-detail').textContent = pir ? '🚶 Motion!' : 'No Motion';
            document.getElementById('lamp-pir-detail').style.color = pir ? 'var(--success)' : '#aaa';

            if (data.mode) {
                let badge = document.getElementById('lamp-mode-badge');
                badge.textContent = data.mode;
                badge.className = 'mode-badge mode-' + data.mode.toLowerCase();
            }
        }

        function updatePowerDisplay(data) {
            let acW = data.ac_watts || 0;
            let lampW = data.lamp_watts || 0;
            let totalW = data.total_watts || 0;
            let dailyKwh = data.daily_kwh || 0;
            let dailyCost = data.daily_cost || 0;
            let monthlyCost = dailyCost * 30;

            document.getElementById('ac-power').innerHTML = acW + '<span class="stat-unit"> W</span>';
            document.getElementById('lamp-power').innerHTML = lampW.toFixed(1) + '<span class="stat-unit"> W</span>';
            document.getElementById('total-power').innerHTML = totalW.toFixed(1) + '<span class="stat-unit"> W</span>';
            document.getElementById('daily-kwh').innerHTML = dailyKwh.toFixed(2) + '<span class="stat-unit"> kWh</span>';
            document.getElementById('daily-cost').textContent = 'Rp ' + dailyCost.toLocaleString('id-ID');
            document.getElementById('monthly-cost').textContent = 'Rp ' + monthlyCost.toLocaleString('id-ID');

            document.getElementById('total-power-dash').innerHTML = totalW.toFixed(1) + '<span class="stat-unit"> W</span>';
            document.getElementById('daily-cost-dash').textContent = 'Rp ' + dailyCost.toLocaleString('id-ID');

            document.getElementById('ac-power-bar').style.width = ((acW / 750) * 100) + '%';
            document.getElementById('lamp-power-bar').style.width = ((lampW / 10) * 100) + '%';
        }

        // ===================== PAGE NAVIGATION =====================
        function showPage(pageName, el) {
            localStorage.setItem('currentPage', pageName);

            document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

            document.getElementById(pageName + '-page').classList.add('active');
            if (el) el.classList.add('active');

            if (pageName === 'ac-analytics') {
                setTimeout(() => {
                    initChart('tempChart', 'Temperature (°C)', 'ac_sensor', 'temperature', chartRanges.temp, '#f44336');
                    initChart('humidityChart', 'Humidity (%)', 'ac_sensor', 'humidity', chartRanges.humidity, '#2196F3');
                }, 150);
            } else if (pageName === 'lamp-analytics') {
                setTimeout(() => {
                    initChart('luxChart', 'Light Intensity (lux)', 'lamp_sensor', 'lux', chartRanges.lux, '#FF9800');
                    initChart('brightnessChart', 'Brightness', 'lamp_sensor', 'brightness', chartRanges.brightness, '#4CAF50');
                }, 150);
            }
        }

        // ===================== CHARTS =====================
        function initChart(canvasId, label, measurement, field, hours, color) {
            const ctx = document.getElementById(canvasId);
            if (!ctx) return;

            fetch('/api/chart/' + measurement + '/' + field + '/' + hours)
                .then(r => r.json())
                .then(data => {
                    const labels = data.map(d => new Date(d.time).toLocaleTimeString());
                    const values = data.map(d => d.value);

                    if (charts[canvasId]) charts[canvasId].destroy();

                    charts[canvasId] = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels.length ? labels : ['No Data'],
                            datasets: [{
                                label: label,
                                data: values.length ? values : [0],
                                borderColor: color,
                                backgroundColor: color + '20',
                                tension: 0.4,
                                fill: true,
                                pointRadius: 2,
                                borderWidth: 2
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: { legend: { display: true, position: 'top' } },
                            scales: {
                                y: { beginAtZero: false, grid: { color: '#f0f0f0' } },
                                x: {
                                    grid: { display: false },
                                    ticks: { maxTicksLimit: 10, maxRotation: 0 }
                                }
                            },
                            interaction: { intersect: false, mode: 'index' }
                        }
                    });
                });
        }

        function updateChart(type, hours, btn) {
            chartRanges[type] = hours;
            localStorage.setItem('chartRanges', JSON.stringify(chartRanges));

            btn.parentElement.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const chartMap = {
                temp: { canvas: 'tempChart', label: 'Temperature (°C)', m: 'ac_sensor', f: 'temperature', c: '#f44336' },
                humidity: { canvas: 'humidityChart', label: 'Humidity (%)', m: 'ac_sensor', f: 'humidity', c: '#2196F3' },
                lux: { canvas: 'luxChart', label: 'Light Intensity (lux)', m: 'lamp_sensor', f: 'lux', c: '#FF9800' },
                brightness: { canvas: 'brightnessChart', label: 'Brightness', m: 'lamp_sensor', f: 'brightness', c: '#4CAF50' }
            };

            const cfg = chartMap[type];
            if (cfg) initChart(cfg.canvas, cfg.label, cfg.m, cfg.f, hours, cfg.c);
        }

        function updateRealtimeChart(chartId, value) {
            if (!charts[chartId]) return;
            const chart = charts[chartId];
            const now = new Date().toLocaleTimeString();

            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(value);

            if (chart.data.labels.length > 30) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            chart.update('none');
        }

        // ===================== CONTROLS =====================
        function controlAC(state) {
            fetch('/api/ac/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state: state })
            }).then(() => addLog('AC', 'Turned ' + state));
        }

        function controlLamp(brightness) {
            document.getElementById('brightness-display').textContent = brightness;
            fetch('/api/lamp/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ brightness: Math.round((brightness / 100) * 255) })
            }).then(() => addLog('Lamp', 'Brightness set to ' + brightness + '%'));
        }

        function toggleMode(device) {
            const badge = document.getElementById(device + '-mode-badge');
            const current = badge.textContent;
            const newMode = current === 'MANUAL' ? 'ADAPTIVE' : 'MANUAL';

            fetch('/api/' + device + '/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: newMode })
            }).then(() => {
                badge.textContent = newMode;
                badge.className = 'mode-badge mode-' + newMode.toLowerCase();
                addLog('Mode', device.toUpperCase() + ' → ' + newMode);
            });
        }

        // ===================== IR REMOTE =====================
        function learnIR(button) {
            fetch('/api/ir/learn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ button: button })
            }).then(() => {
                addLog('IR', 'Learning started: ' + button);
                alert('📡 Arahkan remote ke receiver dan tekan tombol ' + button);
            });
        }

        function sendIR(button) {
            fetch('/api/ir/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ button: button })
            }).then(response => {
                if (response.ok) {
                    addLog('IR', 'Code sent: ' + button);
                } else {
                    alert('❌ Kode IR untuk ' + button + ' belum dipelajari!');
                }
            });
        }

        // ===================== CAMERA =====================
        function retryCamera() {
            const img = document.getElementById('camera-img');
            img.src = '/video_feed?' + new Date().getTime();
            img.style.display = 'block';
            document.getElementById('camera-error').style.display = 'none';
        }

        function updateCameraTime() {
            const el = document.getElementById('camera-time');
            if (el) el.textContent = new Date().toLocaleTimeString();
        }
        setInterval(updateCameraTime, 1000);

        // ===================== LOGS =====================
        function addLog(category, message) {
            const time = new Date().toLocaleTimeString();
            const logHtml = '<span class="log-time">[' + time + ']</span> ' +
                           '<span class="log-mqtt">[' + category + ']</span> ' + message;
            const div = document.createElement('div');
            div.className = 'log-entry';
            div.innerHTML = logHtml;
            const container = document.getElementById('logs-container');
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;

            if (container.children.length > 200) {
                container.removeChild(container.firstChild);
            }
        }

        function appendLogRaw(logText) {
            const div = document.createElement('div');
            div.className = 'log-entry';
            div.textContent = logText;
            const container = document.getElementById('logs-container');
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function clearLogs() {
            document.getElementById('logs-container').innerHTML = '';
            addLog('System', 'Logs cleared');
        }

        // ===================== INIT =====================
        window.onload = function() {
            const savedPage = localStorage.getItem('currentPage');
            const savedRanges = localStorage.getItem('chartRanges');
            if (savedRanges) chartRanges = JSON.parse(savedRanges);

            // Load initial data
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    if (data.ac) updateACDisplay(data.ac);
                    if (data.lamp) updateLampDisplay(data.lamp);
                    if (data.power) updatePowerDisplay(data.power);
                });

            // Load saved IR codes
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(codes => {
                    Object.keys(codes).forEach(button => {
                        const dot = document.getElementById('ir-status-' + button);
                        if (dot) dot.classList.add('learned');
                    });
                });

            // Navigate to saved page
            if (savedPage) {
                const navItems = document.querySelectorAll('.nav-item');
                navItems.forEach(item => {
                    if (item.textContent.toLowerCase().includes(savedPage.split('-')[0])) {
                        showPage(savedPage, item);
                    }
                });
            }
        };
    </script>
</body>
</html>
'''

# ===================== RUN SERVER =====================
if __name__ == '__main__':
    print("=" * 50)
    print("  Smart Room Dashboard Server")
    print("  http://172.20.0.65:5000")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
