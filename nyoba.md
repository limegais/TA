from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import json
from datetime import datetime, timedelta
from collections import deque
from influxdb_client import InfluxDBClient
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartroom-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# InfluxDB Configuration
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "rfi_HvWdjwaG8jB3Rqx6g0y5kMWRfSfq_HmLLUvkom1yaHKvwonU9Qfj6nlZjTqb_I0leIREUnMhvQQXtgETfg=="
INFLUX_ORG = "iotlab"
INFLUX_BUCKET = "sensordata"

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Global data storage
mqtt_data = {
    'ac': {'temperature': 0, 'humidity': 0, 'heat_index': 0, 'ac_state': 'OFF', 'ac_temp': 24, 'fan_speed': 1, 'mode': 'ADAPTIVE', 'rssi': 0, 'uptime': 0},
    'lamp': {'lux': 0, 'motion': False, 'brightness': 0, 'mode': 'ADAPTIVE', 'rssi': 0, 'uptime': 0},
    'camera': {'person_detected': False, 'count': 0, 'confidence': 0},
    'system': {'ga_fitness': 0, 'pso_fitness': 0, 'optimization_runs': 0},
    'ir_codes': {}  # Store learned IR codes
}

log_messages = deque(maxlen=100)
ir_learning_mode = False
ir_learning_button = ""

# MQTT Client Setup
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    log_message = f"MQTT Connected! RC: {rc}"
    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': log_message, 'level': 'success'})
    client.subscribe("smartroom/#")

def on_message(client, userdata, msg):
    global ir_learning_mode, ir_learning_button
    
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        
        if 'ac/sensors' in topic:
            mqtt_data['ac'].update({
                'temperature': payload.get('temperature', 0),
                'humidity': payload.get('humidity', 0),
                'heat_index': payload.get('heat_index', 0),
                'ac_state': payload.get('ac_state', 'OFF'),
                'ac_temp': payload.get('ac_temp', 24),
                'fan_speed': payload.get('fan_speed', 1),
                'rssi': payload.get('rssi', 0),
                'uptime': payload.get('uptime', 0)
            })
            socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
            
        elif 'lamp/sensors' in topic:
            mqtt_data['lamp'].update({
                'lux': payload.get('lux', 0),
                'motion': payload.get('motion', False),
                'brightness': payload.get('brightness', 0),
                'rssi': payload.get('rssi', 0),
                'uptime': payload.get('uptime', 0)
            })
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
            
        elif 'camera/detection' in topic:
            mqtt_data['camera'].update({
                'person_detected': payload.get('person_detected', False),
                'count': payload.get('count', 0),
                'confidence': payload.get('confidence', 0)
            })
            socketio.emit('mqtt_update', {'type': 'camera', 'data': mqtt_data['camera']})
            
        elif 'dashboard/state' in topic:
            mqtt_data['system'].update({
                'ga_fitness': payload.get('ga_best_fitness', 0),
                'pso_fitness': payload.get('pso_best_fitness', 0),
                'optimization_runs': payload.get('optimization_count', 0)
            })
            socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
        
        elif 'ac/mode' in topic:
            mqtt_data['ac']['mode'] = payload.get('mode', 'ADAPTIVE')
            socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
            
        elif 'lamp/mode' in topic:
            mqtt_data['lamp']['mode'] = payload.get('mode', 'ADAPTIVE')
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
        
        elif 'ir/learned' in topic:
            # IR code learned successfully
            button_name = payload.get('button', '')
            ir_code = payload.get('code', '')
            
            if button_name and ir_code:
                mqtt_data['ir_codes'][button_name] = ir_code
                log_messages.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'msg': f'IR Code learned: {button_name}',
                    'level': 'success'
                })
                socketio.emit('ir_learned', {'button': button_name, 'code': ir_code})
                ir_learning_mode = False
                ir_learning_button = ""
            
    except Exception as e:
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'MQTT Error: {str(e)}', 'level': 'error'})

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# InfluxDB Query Function
def get_influx_data(measurement, field, hours=1):
    """Query InfluxDB for historical data"""
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["_field"] == "{field}")
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        '''
        
        result = query_api.query(query=query)
        
        data_points = []
        for table in result:
            for record in table.records:
                data_points.append({
                    'time': record.get_time().strftime('%H:%M'),
                    'value': round(float(record.get_value()), 2)
                })
        
        client.close()
        
        return data_points
        
    except Exception as e:
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'InfluxDB Query Error: {str(e)}', 'level': 'error'})
        return []

# API Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    return jsonify(mqtt_data)

@app.route('/api/chart/<measurement>/<field>/<int:hours>')
def get_chart_data(measurement, field, hours):
    data = get_influx_data(measurement, field, hours)
    return jsonify(data)

@app.route('/api/ac/control', methods=['POST'])
def control_ac():
    try:
        data = request.json
        mqtt_client.publish('smartroom/ac/control', json.dumps(data))
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'AC Control: {data}', 'level': 'info'})
        return jsonify({'status': 'success', 'message': 'AC command sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/lamp/control', methods=['POST'])
def control_lamp():
    try:
        data = request.json
        mqtt_client.publish('smartroom/lamp/control', json.dumps(data))
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Lamp Control: {data}', 'level': 'info'})
        return jsonify({'status': 'success', 'message': 'Lamp command sent'})
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

@app.route('/api/ir/learn', methods=['POST'])
def learn_ir():
    global ir_learning_mode, ir_learning_button
    try:
        data = request.json
        button_name = data.get('button', '')
        
        if not button_name:
            return jsonify({'status': 'error', 'message': 'Button name required'}), 400
        
        ir_learning_mode = True
        ir_learning_button = button_name
        
        # Send MQTT command to ESP32 to start learning
        mqtt_client.publish('smartroom/ir/learn', json.dumps({'button': button_name, 'action': 'start'}))
        
        log_messages.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'msg': f'IR Learning started for: {button_name}',
            'level': 'info'
        })
        
        return jsonify({'status': 'success', 'message': f'Learning mode activated for {button_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ir/send', methods=['POST'])
def send_ir():
    try:
        data = request.json
        button_name = data.get('button', '')
        
        if button_name not in mqtt_data['ir_codes']:
            return jsonify({'status': 'error', 'message': 'IR code not learned yet'}), 400
        
        ir_code = mqtt_data['ir_codes'][button_name]
        
        # Send stored IR code to ESP32
        mqtt_client.publish('smartroom/ir/send', json.dumps({'button': button_name, 'code': ir_code}))
        
        log_messages.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'msg': f'IR Code sent: {button_name}',
            'level': 'info'
        })
        
        return jsonify({'status': 'success', 'message': f'IR code sent for {button_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ir/codes')
def get_ir_codes():
    return jsonify(mqtt_data['ir_codes'])

@app.route('/api/logs')
def get_logs():
    return jsonify(list(log_messages))

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Room IoT Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #334155;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            overflow-x: hidden;
        }

        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 260px;
            height: 100vh;
            background: var(--bg-card);
            border-right: 1px solid var(--border);
            padding: 20px;
            overflow-y: auto;
            z-index: 1000;
        }

        .logo {
            font-size: 24px;
            font-weight: bold;
            color: var(--primary);
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .nav-item {
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 12px;
            color: var(--text-secondary);
        }

        .nav-item:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
        }

        .nav-item.active {
            background: var(--primary);
            color: white;
        }

        .main-content {
            margin-left: 260px;
            padding: 30px;
            min-height: 100vh;
        }

        .page {
            display: none;
        }

        .page.active {
            display: block;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            background: var(--bg-card);
            padding: 20px 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid var(--border);
        }

        .header h1 {
            font-size: 28px;
            margin-bottom: 5px;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: var(--bg-card);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border);
            transition: all 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        }

        .stat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .stat-title {
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
        }

        .stat-icon {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .stat-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 8px;
        }

        .stat-change {
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .stat-change.up { color: var(--success); }
        .stat-change.down { color: var(--danger); }

        .chart-container {
            background: var(--bg-card);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 30px;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .chart-title {
            font-size: 18px;
            font-weight: 600;
        }

        .chart-options {
            display: flex;
            gap: 10px;
        }

        .chart-option-btn {
            padding: 8px 16px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .chart-option-btn:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
        }

        .chart-option-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .mode-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        .mode-badge.adaptive {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
        }

        .mode-badge.manual {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: white;
        }

        .mode-badge:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
        }

        .control-panel {
            background: var(--bg-card);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 20px;
        }

        .control-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .control-group {
            margin-bottom: 20px;
        }

        .control-label {
            display: block;
            color: var(--text-secondary);
            margin-bottom: 10px;
            font-size: 14px;
        }

        .slider {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: var(--border);
            outline: none;
            -webkit-appearance: none;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
        }

        .btn-success {
            background: var(--success);
            color: white;
        }

        .btn-danger {
            background: var(--danger);
            color: white;
        }

        .btn-warning {
            background: var(--warning);
            color: white;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .ir-button-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .ir-button {
            padding: 15px;
            background: var(--bg-dark);
            border: 2px solid var(--border);
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
        }

        .ir-button:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
        }

        .ir-button.learned {
            border-color: var(--success);
        }

        .ir-button.learning {
            border-color: var(--warning);
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .ir-button-name {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }

        .ir-button-icon {
            font-size: 24px;
            margin: 10px 0;
        }

        .ir-status {
            font-size: 10px;
            margin-top: 5px;
        }

        .log-container {
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
            max-height: 500px;
            overflow-y: auto;
        }

        .log-entry {
            padding: 10px;
            margin: 5px 0;
            border-left: 3px solid var(--border);
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }

        .log-entry.success { border-left-color: var(--success); }
        .log-entry.error { border-left-color: var(--danger); }
        .log-entry.info { border-left-color: var(--primary); }

        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: var(--bg-card);
            padding: 16px 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            display: none;
            z-index: 9999;
            animation: slideIn 0.3s;
        }

        @keyframes slideIn {
            from { transform: translateX(400px); }
            to { transform: translateX(0); }
        }

        .toast.show { display: block; }

        .power-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .power-card {
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
            text-align: center;
        }

        .power-value {
            font-size: 36px;
            font-weight: bold;
            color: var(--primary);
            margin: 10px 0;
        }

        .camera-view {
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
            text-align: center;
        }

        .camera-placeholder {
            background: var(--bg-dark);
            height: 400px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-secondary);
            font-size: 18px;
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="logo">
            <i class="fas fa-brain"></i>
            Smart Room
        </div>
        <div class="nav-item active" onclick="showPage('dashboard')">
            <i class="fas fa-home"></i>
            <span>Dashboard</span>
        </div>
        <div class="nav-item" onclick="showPage('ac-analytics')">
            <i class="fas fa-snowflake"></i>
            <span>AC Analytics</span>
        </div>
        <div class="nav-item" onclick="showPage('lamp-analytics')">
            <i class="fas fa-lightbulb"></i>
            <span>Lamp Analytics</span>
        </div>
        <div class="nav-item" onclick="showPage('camera')">
            <i class="fas fa-video"></i>
            <span>Camera/Occupancy</span>
        </div>
        <div class="nav-item" onclick="showPage('power')">
            <i class="fas fa-bolt"></i>
            <span>Power Usage</span>
        </div>
        <div class="nav-item" onclick="showPage('control')">
            <i class="fas fa-sliders-h"></i>
            <span>Control Panel</span>
        </div>
        <div class="nav-item" onclick="showPage('logs')">
            <i class="fas fa-file-alt"></i>
            <span>System Logs</span>
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <!-- Dashboard Page -->
        <div id="dashboard" class="page active">
            <div class="header">
                <h1>Dashboard Overview</h1>
                <p>Real-time monitoring of all systems</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Room Temperature</span>
                        <div class="stat-icon" style="background: rgba(239, 68, 68, 0.2); color: #ef4444;">
                            <i class="fas fa-temperature-high"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-temp">0</span>°C</div>
                    <div class="stat-change up">
                        <i class="fas fa-arrow-up"></i>
                        <span>Real-time</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Humidity</span>
                        <div class="stat-icon" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">
                            <i class="fas fa-tint"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-hum">0</span>%</div>
                    <div class="stat-change">
                        <span>Real-time</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Light Intensity</span>
                        <div class="stat-icon" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">
                            <i class="fas fa-sun"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux">0</span> lx</div>
                    <div class="stat-change">
                        <span>Real-time</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Lamp Brightness</span>
                        <div class="stat-icon" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">
                            <i class="fas fa-lightbulb"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-brightness">0</span>%</div>
                    <div class="stat-change">
                        <span>Real-time</span>
                    </div>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">AC Status</span>
                        <div class="stat-icon" style="background: rgba(99, 102, 241, 0.2); color: #6366f1;">
                            <i class="fas fa-snowflake"></i>
                        </div>
                    </div>
                    <div class="stat-value" style="font-size: 24px;"><span id="dash-ac-state">OFF</span></div>
                    <div class="stat-change">
                        <span>Target: <span id="dash-ac-temp">24</span>°C</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Motion Detection</span>
                        <div class="stat-icon" style="background: rgba(168, 85, 247, 0.2); color: #a855f7;">
                            <i class="fas fa-walking"></i>
                        </div>
                    </div>
                    <div class="stat-value" style="font-size: 24px;"><span id="dash-motion">NO MOTION</span></div>
                    <div class="stat-change">
                        <span>PIR Sensor</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">GA Optimization</span>
                        <div class="stat-icon" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">
                            <i class="fas fa-dna"></i>
                        </div>
                    </div>
                    <div class="stat-value" style="font-size: 24px;"><span id="ga-fitness">0.00</span></div>
                    <div class="stat-change">
                        <span>Best Fitness</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">PSO Optimization</span>
                        <div class="stat-icon" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">
                            <i class="fas fa-chart-line"></i>
                        </div>
                    </div>
                    <div class="stat-value" style="font-size: 24px;"><span id="pso-fitness">0.00</span></div>
                    <div class="stat-change">
                        <span>Best Fitness</span>
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
                    <div class="chart-title">Room Temperature Trend</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="changeChartRange('temp', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('temp', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('temp', 24)">24h</button>
                    </div>
                </div>
                <canvas id="tempChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Humidity Trend</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="changeChartRange('hum', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('hum', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('hum', 24)">24h</button>
                    </div>
                </div>
                <canvas id="humChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">AC Target Temperature</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="changeChartRange('acTemp', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('acTemp', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('acTemp', 24)">24h</button>
                    </div>
                </div>
                <canvas id="acTempChart" height="80"></canvas>
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
                    <div class="chart-title">Light Intensity (Lux)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="changeChartRange('lampLux', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampLux', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampLux', 24)">24h</button>
                    </div>
                </div>
                <canvas id="lampLuxChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Brightness Level</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="changeChartRange('lampBright', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampBright', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampBright', 24)">24h</button>
                    </div>
                </div>
                <canvas id="lampBrightChart" height="80"></canvas>
            </div>
        </div>

        <!-- Camera Page -->
        <div id="camera" class="page">
            <div class="header">
                <h1>Camera & Occupancy</h1>
                <p>Person detection and room occupancy monitoring</p>
            </div>

            <div class="camera-view">
                <div class="camera-placeholder">
                    <div>
                        <i class="fas fa-video" style="font-size: 64px; margin-bottom: 20px;"></i>
                        <p>Camera feed will appear here</p>
                        <p style="font-size: 14px; margin-top: 10px;">Person Detected: <span id="person-detected">No</span></p>
                        <p style="font-size: 14px;">Count: <span id="person-count">0</span></p>
                        <p style="font-size: 14px;">Confidence: <span id="person-confidence">0</span>%</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Power Usage Page -->
        <div id="power" class="page">
            <div class="header">
                <h1>Power Usage</h1>
                <p>Energy consumption monitoring and analysis</p>
            </div>

            <div class="power-grid">
                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">AC Power</div>
                    <div class="power-value"><span id="ac-power">0</span>W</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Real-time</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Lamp Power</div>
                    <div class="power-value"><span id="lamp-power">0</span>W</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Real-time</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Total Power</div>
                    <div class="power-value"><span id="total-power">0</span>W</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Real-time</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Daily Cost</div>
                    <div class="power-value">Rp<span id="daily-cost">0</span></div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">@ Rp 1,500/kWh</div>
                </div>
            </div>

            <div class="chart-container" style="margin-top: 30px;">
                <div class="chart-header">
                    <div class="chart-title">Power Consumption Trend</div>
                </div>
                <canvas id="powerChart" height="80"></canvas>
            </div>
        </div>

        <!-- Control Panel Page -->
        <div id="control" class="page">
            <div class="header">
                <h1>Control Panel</h1>
                <p>Manual control and mode selection</p>
            </div>

            <div class="control-panel">
                <div class="control-title">
                    <span>Air Conditioning Control</span>
                    <div class="mode-badge adaptive" id="ac-mode-badge" onclick="toggleACMode()">ADAPTIVE MODE</div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">AC Power</label>
                    <button class="btn btn-success" onclick="sendACCommand('ON')">
                        <i class="fas fa-power-off"></i> Turn ON
                    </button>
                    <button class="btn btn-danger" onclick="sendACCommand('OFF')" style="margin-left: 10px;">
                        <i class="fas fa-power-off"></i> Turn OFF
                    </button>
                </div>

                <div class="control-group">
                    <label class="control-label">Target Temperature: <span id="ac-temp-display">24</span>°C</label>
                    <input type="range" min="16" max="30" value="24" class="slider" id="ac-temp-slider" oninput="updateACTemp(this.value)">
                </div>

                <div class="control-group">
                    <label class="control-label">Fan Speed: <span id="fan-speed-display">1</span></label>
                    <input type="range" min="1" max="3" value="1" class="slider" id="fan-speed-slider" oninput="updateFanSpeed(this.value)">
                </div>

                <button class="btn btn-primary" onclick="applyACSettings()">
                    <i class="fas fa-check"></i> Apply Settings
                </button>
            </div>

            <div class="control-panel">
                <div class="control-title">
                    <span>IR Remote Learning</span>
                    <button class="btn btn-warning btn-sm" onclick="loadIRCodes()">
                        <i class="fas fa-sync"></i> Refresh
                    </button>
                </div>
                
                <div class="control-label">
                    <i class="fas fa-info-circle"></i> Click "Learn" button, then press the button on your AC remote within 10 seconds
                </div>

                <div class="ir-button-grid" id="ir-button-grid">
                    <!-- Power -->
                    <div class="ir-button" data-button="POWER">
                        <div class="ir-button-name">POWER</div>
                        <div class="ir-button-icon"><i class="fas fa-power-off"></i></div>
                        <button class="btn btn-warning btn-sm" onclick="learnIRCode('POWER')">Learn</button>
                        <button class="btn btn-primary btn-sm" onclick="sendIRCode('POWER')" style="margin-top: 5px;">Send</button>
                        <div class="ir-status" id="status-POWER">Not learned</div>
                    </div>

                    <!-- Temp UP -->
                    <div class="ir-button" data-button="TEMP_UP">
                        <div class="ir-button-name">TEMP +</div>
                        <div class="ir-button-icon"><i class="fas fa-plus"></i></div>
                        <button class="btn btn-warning btn-sm" onclick="learnIRCode('TEMP_UP')">Learn</button>
                        <button class="btn btn-primary btn-sm" onclick="sendIRCode('TEMP_UP')" style="margin-top: 5px;">Send</button>
                        <div class="ir-status" id="status-TEMP_UP">Not learned</div>
                    </div>

                    <!-- Temp DOWN -->
                    <div class="ir-button" data-button="TEMP_DOWN">
                        <div class="ir-button-name">TEMP -</div>
                        <div class="ir-button-icon"><i class="fas fa-minus"></i></div>
                        <button class="btn btn-warning btn-sm" onclick="learnIRCode('TEMP_DOWN')">Learn</button>
                        <button class="btn btn-primary btn-sm" onclick="sendIRCode('TEMP_DOWN')" style="margin-top: 5px;">Send</button>
                        <div class="ir-status" id="status-TEMP_DOWN">Not learned</div>
                    </div>

                    <!-- Fan Speed -->
                    <div class="ir-button" data-button="FAN">
                        <div class="ir-button-name">FAN SPEED</div>
                        <div class="ir-button-icon"><i class="fas fa-fan"></i></div>
                        <button class="btn btn-warning btn-sm" onclick="learnIRCode('FAN')">Learn</button>
                        <button class="btn btn-primary btn-sm" onclick="sendIRCode('FAN')" style="margin-top: 5px;">Send</button>
                        <div class="ir-status" id="status-FAN">Not learned</div>
                    </div>

                    <!-- Mode -->
                    <div class="ir-button" data-button="MODE">
                        <div class="ir-button-name">MODE</div>
                        <div class="ir-button-icon"><i class="fas fa-cog"></i></div>
                        <button class="btn btn-warning btn-sm" onclick="learnIRCode('MODE')">Learn</button>
                        <button class="btn btn-primary btn-sm" onclick="sendIRCode('MODE')" style="margin-top: 5px;">Send</button>
                        <div class="ir-status" id="status-MODE">Not learned</div>
                    </div>

                    <!-- Swing -->
                    <div class="ir-button" data-button="SWING">
                        <div class="ir-button-name">SWING</div>
                        <div class="ir-button-icon"><i class="fas fa-arrows-alt-v"></i></div>
                        <button class="btn btn-warning btn-sm" onclick="learnIRCode('SWING')">Learn</button>
                        <button class="btn btn-primary btn-sm" onclick="sendIRCode('SWING')" style="margin-top: 5px;">Send</button>
                        <div class="ir-status" id="status-SWING">Not learned</div>
                    </div>
                </div>
            </div>

            <div class="control-panel">
                <div class="control-title">
                    <span>Lamp Control</span>
                    <div class="mode-badge adaptive" id="lamp-mode-badge" onclick="toggleLampMode()">ADAPTIVE MODE</div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Brightness: <span id="brightness-display">0</span>%</label>
                    <input type="range" min="0" max="100" value="0" class="slider" id="brightness-slider" oninput="updateBrightness(this.value)">
                </div>

                <button class="btn btn-primary" onclick="applyLampSettings()">
                    <i class="fas fa-check"></i> Apply Brightness
                </button>
            </div>
        </div>

        <!-- Logs Page -->
        <div id="logs" class="page">
            <div class="header">
                <h1>System Logs</h1>
                <p>Real-time system events and notifications</p>
            </div>

            <div class="log-container" id="log-container">
                <!-- Logs will be inserted here -->
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast" class="toast">
        <div id="toast-message"></div>
    </div>

    <script>
        const socket = io();
        
        let charts = {};
        let chartRanges = {
            temp: 1,
            hum: 1,
            acTemp: 1,
            lampLux: 1,
            lampBright: 1
        };

        let learnedCodes = {};

        function initCharts() {
            const chartConfig = {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        }
                    }
                }
            };

            charts.temp = new Chart(document.getElementById('tempChart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Temperature (°C)',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                }
            });

            charts.hum = new Chart(document.getElementById('humChart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Humidity (%)',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                }
            });

            charts.acTemp = new Chart(document.getElementById('acTempChart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'AC Target Temp (°C)',
                        data: [],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                }
            });

            charts.lampLux = new Chart(document.getElementById('lampLuxChart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Light Intensity (lux)',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                }
            });

            charts.lampBright = new Chart(document.getElementById('lampBrightChart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Brightness (%)',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                }
            });

            charts.power = new Chart(document.getElementById('powerChart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Total Power (W)',
                        data: [],
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                }
            });
        }

        function updateChart(chartName, hours) {
            let endpoint = '';
            
            switch(chartName) {
                case 'temp':
                    endpoint = '/api/chart/ac_sensor/temperature/' + hours;
                    break;
                case 'hum':
                    endpoint = '/api/chart/ac_sensor/humidity/' + hours;
                    break;
                case 'acTemp':
                    endpoint = '/api/chart/ac_sensor/ac_temp/' + hours;
                    break;
                case 'lampLux':
                    endpoint = '/api/chart/lamp_sensor/lux/' + hours;
                    break;
                case 'lampBright':
                    endpoint = '/api/chart/lamp_sensor/brightness/' + hours;
                    break;
            }

            fetch(endpoint)
                .then(response => response.json())
                .then(data => {
                    if (data && data.length > 0) {
                        const labels = data.map(d => d.time);
                        const values = data.map(d => d.value);
                        
                        charts[chartName].data.labels = labels;
                        charts[chartName].data.datasets[0].data = values;
                        charts[chartName].update();
                    }
                })
                .catch(error => {
                    console.error('Error fetching chart data:', error);
                });
        }

        function changeChartRange(chartName, hours) {
            chartRanges[chartName] = hours;
            localStorage.setItem('chartRanges', JSON.stringify(chartRanges));
            
            const buttons = event.target.parentElement.querySelectorAll('.chart-option-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            updateChart(chartName, hours);
        }

        function showPage(pageId) {
            document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            
            document.getElementById(pageId).classList.add('active');
            event.target.closest('.nav-item').classList.add('active');
            
            localStorage.setItem('currentPage', pageId);
        }

        function updateACTemp(value) {
            document.getElementById('ac-temp-display').textContent = value;
        }

        function updateFanSpeed(value) {
            document.getElementById('fan-speed-display').textContent = value;
        }

        function sendACCommand(command) {
            const data = { command: command };
            
            fetch('/api/ac/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                showToast(result.message);
            })
            .catch(error => {
                showToast('Error: ' + error, 'error');
            });
        }

        function applyACSettings() {
            const temp = document.getElementById('ac-temp-slider').value;
            const fan = document.getElementById('fan-speed-slider').value;
            
            const data = {
                command: 'SET',
                temperature: parseInt(temp),
                fan_speed: parseInt(fan)
            };
            
            fetch('/api/ac/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                showToast('AC settings applied!');
            })
            .catch(error => {
                showToast('Error: ' + error, 'error');
            });
        }

        function updateBrightness(value) {
            document.getElementById('brightness-display').textContent = value;
        }

        function applyLampSettings() {
            const brightness = document.getElementById('brightness-slider').value;
            
            const data = {
                brightness: parseInt(brightness)
            };
            
            fetch('/api/lamp/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                showToast('Lamp brightness set!');
            })
            .catch(error => {
                showToast('Error: ' + error, 'error');
            });
        }

        function toggleACMode() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    const currentMode = data.ac.mode;
                    const newMode = currentMode === 'ADAPTIVE' ? 'MANUAL' : 'ADAPTIVE';
                    
                    fetch('/api/ac/mode', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode: newMode })
                    })
                    .then(response => response.json())
                    .then(result => {
                        updateModeBadges();
                        showToast('AC Mode: ' + newMode);
                    })
                    .catch(error => {
                        showToast('Error: ' + error, 'error');
                    });
                });
        }

        function toggleLampMode() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    const currentMode = data.lamp.mode;
                    const newMode = currentMode === 'ADAPTIVE' ? 'MANUAL' : 'ADAPTIVE';
                    
                    fetch('/api/lamp/mode', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode: newMode })
                    })
                    .then(response => response.json())
                    .then(result => {
                        updateModeBadges();
                        showToast('Lamp Mode: ' + newMode);
                    })
                    .catch(error => {
                        showToast('Error: ' + error, 'error');
                    });
                });
        }

        function updateModeBadges() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    const acBadge = document.getElementById('ac-mode-badge');
                    const lampBadge = document.getElementById('lamp-mode-badge');
                    
                    acBadge.textContent = data.ac.mode + ' MODE';
                    acBadge.className = 'mode-badge ' + data.ac.mode.toLowerCase();
                    
                    lampBadge.textContent = data.lamp.mode + ' MODE';
                    lampBadge.className = 'mode-badge ' + data.lamp.mode.toLowerCase();
                });
        }

        // IR Learning Functions
        function learnIRCode(buttonName) {
            const buttonElement = document.querySelector(`[data-button="${buttonName}"]`);
            buttonElement.classList.add('learning');
            
            fetch('/api/ir/learn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ button: buttonName })
            })
            .then(response => response.json())
            .then(result => {
                showToast(`Learning ${buttonName}... Press remote button now!`, 'info');
                
                // Auto-remove learning state after 15 seconds
                setTimeout(() => {
                    buttonElement.classList.remove('learning');
                }, 15000);
            })
            .catch(error => {
                buttonElement.classList.remove('learning');
                showToast('Error: ' + error, 'error');
            });
        }

        function sendIRCode(buttonName) {
            fetch('/api/ir/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ button: buttonName })
            })
            .then(response => response.json())
            .then(result => {
                showToast(`IR Code sent: ${buttonName}`);
            })
            .catch(error => {
                showToast('Error: ' + error.message || error, 'error');
            });
        }

        function loadIRCodes() {
            fetch('/api/ir/codes')
                .then(response => response.json())
                .then(codes => {
                    learnedCodes = codes;
                    
                    // Update UI
                    Object.keys(codes).forEach(buttonName => {
                        const buttonElement = document.querySelector(`[data-button="${buttonName}"]`);
                        const statusElement = document.getElementById(`status-${buttonName}`);
                        
                        if (buttonElement && statusElement) {
                            buttonElement.classList.add('learned');
                            statusElement.textContent = 'Learned ✓';
                            statusElement.style.color = '#10b981';
                        }
                    });
                    
                    showToast('IR codes loaded');
                })
                .catch(error => {
                    console.error('Error loading IR codes:', error);
                });
        }

        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            const toastMessage = document.getElementById('toast-message');
            
            toastMessage.innerHTML = `<i class="fas fa-${type === 'success' ? 'check' : (type === 'info' ? 'info' : 'exclamation')}-circle"></i> ${message}`;
            toast.classList.add('show');
            
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('dash-temp').textContent = data.ac.temperature.toFixed(1);
                    document.getElementById('dash-hum').textContent = data.ac.humidity.toFixed(1);
                    document.getElementById('dash-ac-state').textContent = data.ac.ac_state;
                    document.getElementById('dash-ac-temp').textContent = data.ac.ac_temp;
                    
                    document.getElementById('dash-lux').textContent = data.lamp.lux.toFixed(0);
                    document.getElementById('dash-brightness').textContent = Math.round(data.lamp.brightness / 255 * 100);
                    document.getElementById('dash-motion').textContent = data.lamp.motion ? 'MOTION DETECTED' : 'NO MOTION';
                    
                    document.getElementById('person-detected').textContent = data.camera.person_detected ? 'Yes' : 'No';
                    document.getElementById('person-count').textContent = data.camera.count;
                    document.getElementById('person-confidence').textContent = data.camera.confidence.toFixed(0);
                    
                    document.getElementById('ga-fitness').textContent = data.system.ga_fitness.toFixed(2);
                    document.getElementById('pso-fitness').textContent = data.system.pso_fitness.toFixed(2);
                    
                    let acPower = 0;
                    if (data.ac.ac_state === 'ON') {
                        acPower = data.ac.fan_speed === 1 ? 100 : (data.ac.fan_speed === 2 ? 200 : 300);
                    }
                    let lampPower = (data.lamp.brightness / 255) * 10;
                    let totalPower = acPower + lampPower;
                    let dailyCost = (totalPower / 1000) * 24 * 1500;
                    
                    document.getElementById('ac-power').textContent = acPower.toFixed(0);
                    document.getElementById('lamp-power').textContent = lampPower.toFixed(1);
                    document.getElementById('total-power').textContent = totalPower.toFixed(1);
                    document.getElementById('daily-cost').textContent = dailyCost.toFixed(0);
                    
                    updateModeBadges();
                });
        }

        function updateLogs() {
            fetch('/api/logs')
                .then(response => response.json())
                .then(logs => {
                    const container = document.getElementById('log-container');
                    container.innerHTML = '';
                    
                    logs.reverse().forEach(log => {
                        const entry = document.createElement('div');
                        entry.className = 'log-entry ' + log.level;
                        entry.innerHTML = `<strong>[${log.time}]</strong> ${log.msg}`;
                        container.appendChild(entry);
                    });
                });
        }

        socket.on('mqtt_update', function(data) {
            updateDashboard();
            
            if (data.type === 'ac') {
                const now = new Date();
                const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
                
                if (charts.temp && charts.temp.data.labels.length < 50) {
                    charts.temp.data.labels.push(timeStr);
                    charts.temp.data.datasets[0].data.push(data.data.temperature);
                    charts.temp.update();
                }
                
                if (charts.hum && charts.hum.data.labels.length < 50) {
                    charts.hum.data.labels.push(timeStr);
                    charts.hum.data.datasets[0].data.push(data.data.humidity);
                    charts.hum.update();
                }
            }
            
            if (data.type === 'lamp') {
                const now = new Date();
                const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
                
                if (charts.lampLux && charts.lampLux.data.labels.length < 50) {
                    charts.lampLux.data.labels.push(timeStr);
                    charts.lampLux.data.datasets[0].data.push(data.data.lux);
                    charts.lampLux.update();
                }
                
                if (charts.lampBright && charts.lampBright.data.labels.length < 50) {
                    charts.lampBright.data.labels.push(timeStr);
                    charts.lampBright.data.datasets[0].data.push(Math.round(data.data.brightness / 255 * 100));
                    charts.lampBright.update();
                }
            }
        });

        socket.on('ir_learned', function(data) {
            const buttonElement = document.querySelector(`[data-button="${data.button}"]`);
            const statusElement = document.getElementById(`status-${data.button}`);
            
            if (buttonElement && statusElement) {
                buttonElement.classList.remove('learning');
                buttonElement.classList.add('learned');
                statusElement.textContent = 'Learned ✓';
                statusElement.style.color = '#10b981';
            }
            
            showToast(`IR Code learned: ${data.button}`);
            learnedCodes[data.button] = data.code;
        });

        function loadSavedPreferences() {
            const savedPage = localStorage.getItem('currentPage');
            if (savedPage) {
                document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
                document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
                
                document.getElementById(savedPage).classList.add('active');
                const navItem = document.querySelector(`[onclick="showPage('${savedPage}')"]`);
                if (navItem) navItem.classList.add('active');
            }
            
            const savedRanges = localStorage.getItem('chartRanges');
            if (savedRanges) {
                chartRanges = JSON.parse(savedRanges);
            }
        }

        window.onload = function() {
            initCharts();
            loadSavedPreferences();
            updateDashboard();
            updateLogs();
            loadIRCodes();
            
            Object.keys(chartRanges).forEach(chartName => {
                updateChart(chartName, chartRanges[chartName]);
            });
            
            setInterval(updateDashboard, 1000);
            setInterval(updateLogs, 5000);
            
            setInterval(() => {
                Object.keys(chartRanges).forEach(chartName => {
                    updateChart(chartName, chartRanges[chartName]);
                });
            }, 30000);
        };
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("Starting Smart Room Dashboard...")
    print("Dashboard URL: http://172.20.0.65:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
