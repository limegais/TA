#!/usr/bin/env python3
"""Smart Room Dashboard - Premium Analytics & Monitoring System"""

import json
import threading
import time
import os
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request
import paho.mqtt.client as mqtt

app = Flask(__name__)

# --- Global State ---
sensor_data = {
    'ac': {'temperature': '--', 'humidity': '--', 'ac_temp': '--', 'heat_index': '--', 'ac_state': 'off', 'fan_speed': '--', 'last_update': None},
    'lamp': {'lux': '--', 'motion': '--', 'brightness': '--', 'last_update': None},
    'camera': {'person_count': 0, 'occupied': False, 'last_update': None},
    'ac_mode': 'auto',
    'lamp_mode': 'auto',
    'ga_result': {'temp': '--', 'fan': '--', 'energy': '--', 'comfort': '--'},
    'pso_result': {'brightness': '--', 'target_lux': '--', 'efficiency': '--'},
    'system': {'uptime': '--', 'optimization_count': 0, 'mode': 'work'}
}

log_messages = []
system_stats = {
    'total_optimizations': 0,
    'ac_commands': 0,
    'lamp_commands': 0,
    'daily_energy_kwh': 0,
    'monthly_cost_estimate': 0
}

def add_log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_messages.insert(0, {'time': ts, 'message': msg})
    if len(log_messages) > 500:
        log_messages.pop()

# --- MQTT Listener ---
def mqtt_listener():
    def on_connect(client, userdata, flags, rc):
        print(f"[Dashboard MQTT] Connected rc={rc}")
        client.subscribe("smartroom/#")
        print("[Dashboard MQTT] Subscribed to smartroom/#")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic
            ts = datetime.now().strftime("%H:%M:%S")

            if topic == "smartroom/ac/sensors":
                sensor_data['ac']['temperature'] = payload.get('temperature', '--')
                sensor_data['ac']['humidity'] = payload.get('humidity', '--')
                sensor_data['ac']['heat_index'] = payload.get('heat_index', '--')
                sensor_data['ac']['ac_state'] = payload.get('ac_state', 'off')
                sensor_data['ac']['ac_temp'] = payload.get('ac_temp', '--')
                sensor_data['ac']['fan_speed'] = payload.get('fan_speed', '--')
                sensor_data['ac']['last_update'] = ts
                add_log(f"[AC] Temp={payload.get('temperature')}C Hum={payload.get('humidity')}% HI={payload.get('heat_index')}C")

            elif topic == "smartroom/lamp/sensors":
                sensor_data['lamp']['lux'] = payload.get('lux', '--')
                sensor_data['lamp']['motion'] = payload.get('motion', '--')
                sensor_data['lamp']['brightness'] = payload.get('brightness', '--')
                sensor_data['lamp']['last_update'] = ts
                add_log(f"[Lamp] Lux={payload.get('lux')} Motion={'Detected' if payload.get('motion')==1 else 'None'} Brightness={payload.get('brightness')}")

            elif topic == "smartroom/camera/detection":
                sensor_data['camera']['person_count'] = payload.get('person_count', 0)
                sensor_data['camera']['occupied'] = payload.get('occupied', False)
                sensor_data['camera']['last_update'] = ts
                if payload.get('person_count', 0) > 0:
                    add_log(f"[Camera] {payload.get('person_count')} person(s) detected")

            elif topic == "smartroom/ac/status":
                sensor_data['ac']['ac_state'] = payload.get('ac_state', 'off')
                sensor_data['ac']['ac_temp'] = payload.get('ac_temp', '--')
                sensor_data['ac']['fan_speed'] = payload.get('fan_speed', '--')
                if payload.get('temperature'):
                    sensor_data['ac']['temperature'] = payload.get('temperature')
                if payload.get('humidity'):
                    sensor_data['ac']['humidity'] = payload.get('humidity')

            elif topic == "smartroom/lamp/status":
                sensor_data['lamp']['brightness'] = payload.get('brightness', '--')

            elif topic == "smartroom/ac/optimization":
                sensor_data['ga_result'] = {
                    'temp': payload.get('recommended_temp', '--'),
                    'fan': payload.get('fan_speed', '--'),
                    'energy': payload.get('energy_score', '--'),
                    'comfort': payload.get('comfort_score', '--')
                }
                system_stats['total_optimizations'] += 1
                add_log(f"[GA Optimization] Recommend {payload.get('recommended_temp')}C Fan={payload.get('fan_speed')} Energy={payload.get('energy_score'):.2f} Comfort={payload.get('comfort_score'):.2f}")

            elif topic == "smartroom/lamp/optimization":
                sensor_data['pso_result'] = {
                    'brightness': payload.get('brightness', '--'),
                    'target_lux': payload.get('target_lux', '--'),
                    'efficiency': payload.get('efficiency', '--')
                }
                add_log(f"[PSO Optimization] Brightness={payload.get('brightness')} Target={payload.get('target_lux')}lux Efficiency={payload.get('efficiency'):.2f}")

            elif topic == "smartroom/ac/ir_learned":
                status = payload.get('status', '?')
                button = payload.get('button', '?')
                protocol = payload.get('protocol', '?')
                add_log(f"[IR Learning] {status.upper()}: {button} | Protocol: {protocol}")

            elif topic == "smartroom/ac/mode":
                mode = payload.get('mode', 'auto')
                sensor_data['ac_mode'] = mode
                add_log(f"[AC Mode] Changed to {mode.upper()}")

            elif topic == "smartroom/lamp/mode":
                mode = payload.get('mode', 'auto')
                sensor_data['lamp_mode'] = mode
                add_log(f"[Lamp Mode] Changed to {mode.upper()}")

            elif topic == "smartroom/dashboard/state":
                state = payload.get('state', {})
                stats = payload.get('stats', {})
                sensor_data['system']['uptime'] = stats.get('uptime', '--')
                sensor_data['system']['optimization_count'] = stats.get('optimization_count', 0)
                system_stats['ac_commands'] = stats.get('ac_commands_sent', 0)
                system_stats['lamp_commands'] = stats.get('lamp_commands_sent', 0)

        except Exception as e:
            print(f"[Dashboard MQTT] Error: {e}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "dashboard_premium")
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect("localhost", 1883, 60)
        client.loop_forever()
    except Exception as e:
        print(f"[Dashboard MQTT] Connection failed: {e}")

mqtt_thread = threading.Thread(target=mqtt_listener, daemon=True)
mqtt_thread.start()

# --- MQTT Publisher ---
mqtt_pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "dashboard_pub_premium")
try:
    mqtt_pub.connect("localhost", 1883, 60)
    mqtt_pub.loop_start()
except:
    pass

# --- InfluxDB Query ---
def get_influx_data(measurement, field, hours=1):
    try:
        from influxdb_client import InfluxDBClient
        
        INFLUX_URL = "http://localhost:8086"
        INFLUX_TOKEN = "rfi_HvWdjwaG8jB3Rqx6g0y5kMWRfSfq_HmLLUvkom1yaHKvwonU9Qfj6nlZjTqb_I0leIREUnMhvQQXtgETfg=="
        INFLUX_ORG = "iotlab"
        INFLUX_BUCKET = "sensordata"
        
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        query = f'''
        from(bucket:"{INFLUX_BUCKET}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["_field"] == "{field}")
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        '''
        
        result = query_api.query(query)
        
        data_points = []
        for table in result:
            for record in table.records:
                data_points.append({
                    'time': record.get_time().strftime("%H:%M"),
                    'value': round(record.get_value(), 2) if record.get_value() else 0
                })
        
        client.close()
        
        # If no data, return dummy data for testing
        if len(data_points) == 0:
            print(f"[InfluxDB] No data found for {measurement}.{field}")
            # Generate dummy timestamps for last hour
            from datetime import datetime, timedelta
            now = datetime.now()
            for i in range(12):
                t = now - timedelta(minutes=i*5)
                data_points.insert(0, {
                    'time': t.strftime("%H:%M"),
                    'value': 0
                })
        
        return data_points
    except Exception as e:
        print(f"[InfluxDB] Query error: {e}")
        # Return dummy data on error
        from datetime import datetime, timedelta
        data_points = []
        now = datetime.now()
        for i in range(12):
            t = now - timedelta(minutes=i*5)
            data_points.insert(0, {
                'time': t.strftime("%H:%M"),
                'value': 0
            })
        return data_points

# --- HTML Template Premium ---
MAIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Room IoT - Premium Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { margin:0; padding:0; box-sizing:border-box; }
body { 
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif; 
    background: linear-gradient(135deg, #0a0e17 0%, #1a1f35 100%);
    color:#e0e0e0; 
    overflow-x: hidden;
}

/* Animated Background */
.bg-animation {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    opacity: 0.03;
    background: repeating-linear-gradient(
        45deg,
        transparent,
        transparent 10px,
        rgba(56, 189, 248, 0.05) 10px,
        rgba(56, 189, 248, 0.05) 20px
    );
    animation: bgMove 20s linear infinite;
}

@keyframes bgMove {
    0% { transform: translateX(0) translateY(0); }
    100% { transform: translateX(20px) translateY(20px); }
}

.container { display:flex; min-height:100vh; position: relative; z-index: 1; }

/* Sidebar Enhanced */
.sidebar { 
    width:260px; 
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    padding:25px 0; 
    border-right:1px solid rgba(56, 189, 248, 0.1); 
    position:fixed; 
    height:100vh; 
    overflow-y:auto;
    box-shadow: 4px 0 20px rgba(0,0,0,0.3);
}

.sidebar::-webkit-scrollbar { width: 6px; }
.sidebar::-webkit-scrollbar-track { background: #0f172a; }
.sidebar::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

.sidebar-header {
    text-align: center;
    padding: 0 20px 25px 20px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.1);
}

.sidebar h2 { 
    color:#38bdf8; 
    font-size:22px; 
    font-weight:800;
    margin-bottom:5px;
    letter-spacing: 1px;
}

.sidebar .subtitle { 
    color:#64748b; 
    font-size:11px; 
    margin-bottom:15px;
    font-weight: 500;
}

.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
    color: #22c55e;
}

.status-dot {
    width: 6px;
    height: 6px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

.nav-section {
    padding: 20px 0 10px 20px;
    font-size: 10px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.sidebar a { 
    display:flex;
    align-items: center;
    gap: 12px;
    padding:14px 20px; 
    color:#94a3b8; 
    text-decoration:none; 
    font-size:14px;
    font-weight: 500;
    border-left:3px solid transparent;
    transition: all 0.3s ease;
    position: relative;
    cursor: pointer;
}

.sidebar a i {
    width: 20px;
    text-align: center;
    font-size: 16px;
}

.sidebar a:hover { 
    background: rgba(14, 165, 233, 0.05);
    color:#38bdf8; 
    border-left-color: rgba(56, 189, 248, 0.3);
}

.sidebar a.active { 
    background: linear-gradient(90deg, rgba(14, 165, 233, 0.15) 0%, transparent 100%);
    color:#38bdf8; 
    border-left-color:#38bdf8;
    font-weight: 600;
}

.sidebar a.active::before {
    content: '';
    position: absolute;
    right: 20px;
    width: 6px;
    height: 6px;
    background: #38bdf8;
    border-radius: 50%;
}

/* Main Content */
.main { 
    margin-left:260px; 
    flex:1; 
    padding:30px;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.page-header {
    margin-bottom: 30px;
}

.page-title { 
    font-size:32px; 
    font-weight:800; 
    color:#f1f5f9;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-subtitle {
    font-size: 14px;
    color: #64748b;
    font-weight: 500;
}

.breadcrumb {
    display: flex;
    gap: 8px;
    font-size: 12px;
    color: #64748b;
    margin-top: 8px;
}

.breadcrumb span {
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Grid Layouts */
.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }
.grid-3 { display:grid; grid-template-columns:repeat(3, 1fr); gap:20px; margin-bottom:20px; }
.grid-4 { display:grid; grid-template-columns:repeat(4, 1fr); gap:20px; margin-bottom:20px; }

/* Enhanced Cards */
.card { 
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius:16px; 
    padding:24px; 
    border:1px solid rgba(56, 189, 248, 0.1);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.card:hover {
    border-color: rgba(56, 189, 248, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.card:hover::before {
    opacity: 1;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.card-label { 
    font-size:12px; 
    color:#64748b; 
    text-transform:uppercase; 
    font-weight: 700;
    letter-spacing: 1.2px;
}

.card-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.card-icon.blue { background: rgba(14, 165, 233, 0.15); color: #38bdf8; }
.card-icon.green { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
.card-icon.orange { background: rgba(249, 115, 22, 0.15); color: #f97316; }
.card-icon.purple { background: rgba(168, 85, 247, 0.15); color: #a855f7; }
.card-icon.red { background: rgba(239, 68, 68, 0.15); color: #ef4444; }

.card-value-wrapper {
    display: flex;
    align-items: baseline;
    gap: 8px;
}

.card-value { 
    font-size:36px; 
    font-weight:800; 
    color:#f1f5f9;
    line-height: 1;
}

.card-unit { 
    font-size:16px; 
    color:#64748b;
    font-weight: 600;
}

.card-trend {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 12px;
    padding: 6px 10px;
    background: rgba(34, 197, 94, 0.1);
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #22c55e;
    width: fit-content;
}

.card-trend.down {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
}

/* Chart Container */
.chart-container { 
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius:16px; 
    padding:24px; 
    border:1px solid rgba(56, 189, 248, 0.1);
    position:relative;
    transition: all 0.3s ease;
}

.chart-container:hover {
    border-color: rgba(56, 189, 248, 0.3);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.chart-title { 
    font-size:18px; 
    font-weight:700; 
    color:#f1f5f9;
}

.chart-options {
    display: flex;
    gap: 8px;
}

.chart-option-btn {
    padding: 6px 12px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 6px;
    color: #38bdf8;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.chart-option-btn:hover, .chart-option-btn.active {
    background: rgba(14, 165, 233, 0.2);
    border-color: #38bdf8;
}

.chart-canvas {
    height: 320px;
    position: relative;
}

/* Stats Section */
.stats-section {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(6, 78, 59, 0.1) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
}

.stat-item {
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 16px;
    background: rgba(15, 23, 42, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.1);
}

.stat-icon {
    width: 50px;
    height: 50px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.stat-content .label {
    font-size: 11px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stat-content .value {
    font-size: 24px;
    font-weight: 800;
    color: #f1f5f9;
    margin-top: 4px;
}

/* Power Card Special */
.power-mega-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 2px solid rgba(249, 115, 22, 0.3);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.power-mega-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(249, 115, 22, 0.1) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.power-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
    position: relative;
    z-index: 1;
}

.power-title {
    font-size: 24px;
    font-weight: 800;
    color: #f97316;
    display: flex;
    align-items: center;
    gap: 12px;
}

.power-badge {
    padding: 8px 16px;
    background: rgba(249, 115, 22, 0.2);
    border: 1px solid rgba(249, 115, 22, 0.4);
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    color: #f97316;
}

/* Control Panel */
.control-section {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(56, 189, 248, 0.1);
}

.control-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.1);
}

.control-title {
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 10px;
}

.mode-badge {
    padding: 6px 16px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.3s ease;
}

.mode-badge.auto {
    background: rgba(14, 165, 233, 0.2);
    color: #38bdf8;
    border: 1px solid rgba(14, 165, 233, 0.4);
}

.mode-badge.manual {
    background: rgba(249, 115, 22, 0.2);
    color: #f97316;
    border: 1px solid rgba(249, 115, 22, 0.4);
}

.mode-badge:hover {
    transform: scale(1.05);
}

/* Buttons Enhanced */
.btn {
    padding: 12px 24px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    font-weight: 700;
    font-size: 14px;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.btn-primary {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    color: #fff;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
}

.btn-secondary {
    background: rgba(51, 65, 85, 0.5);
    color: #e0e0e0;
    border: 1px solid #334155;
}

.btn-secondary:hover {
    background: rgba(51, 65, 85, 0.8);
    border-color: #475569;
}

.btn-danger {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: #fff;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
}

.btn-success {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: #fff;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
}

/* Slider Enhanced */
.slider-control {
    margin: 20px 0;
}

.slider-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
}

.slider-value {
    font-size: 18px;
    font-weight: 800;
    color: #38bdf8;
}

input[type="range"] {
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    outline: none;
    -webkit-appearance: none;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #38bdf8;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(56, 189, 248, 0.5);
    transition: all 0.3s ease;
}

input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.7);
}

/* Logs Panel */
.logs-container {
    background: #0f172a;
    border-radius: 12px;
    padding: 20px;
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid rgba(56, 189, 248, 0.1);
    font-family: 'Courier New', monospace;
}

.logs-container::-webkit-scrollbar { width: 8px; }
.logs-container::-webkit-scrollbar-track { background: #1e293b; border-radius: 4px; }
.logs-container::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

.log-entry {
    padding: 10px;
    margin-bottom: 8px;
    background: rgba(30, 41, 59, 0.5);
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    font-size: 12px;
    display: flex;
    gap: 12px;
    transition: all 0.2s ease;
}

.log-entry:hover {
    background: rgba(30, 41, 59, 0.8);
    transform: translateX(4px);
}

.log-time {
    color: #64748b;
    font-weight: 600;
    min-width: 80px;
}

.log-message {
    color: #e0e0e0;
}

/* Page Transitions */
.page { 
    display:none; 
    animation: pageSlide 0.4s ease;
}

.page.active { 
    display:block; 
}

@keyframes pageSlide {
    from { 
        opacity: 0; 
        transform: translateX(20px); 
    }
    to { 
        opacity: 1; 
        transform: translateX(0); 
    }
}

/* Responsive */
@media(max-width:1024px){
    .grid-2, .grid-3, .grid-4 { grid-template-columns:1fr; }
    .sidebar { transform: translateX(-100%); }
    .main { margin-left: 0; }
}

/* Loading Animation */
.loading-spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 3px solid rgba(56, 189, 248, 0.3);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Toast Notification */
.toast {
    position: fixed;
    bottom: 30px;
    right: 30px;
    padding: 16px 24px;
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: #fff;
    border-radius: 12px;
    font-weight: 600;
    font-size: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    z-index: 9999;
    opacity: 0;
    transform: translateY(20px);
    transition: all 0.3s ease;
}

.toast.show {
    opacity: 1;
    transform: translateY(0);
}

.toast.error {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.toast.info {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
}
</style>
</head>
<body>
<div class="bg-animation"></div>

<div class="container">
  <div class="sidebar">
    <div class="sidebar-header">
      <h2><i class="fas fa-home"></i> SMART ROOM</h2>
      <div class="subtitle">IoT Analytics Dashboard</div>
      <div class="status-indicator">
        <div class="status-dot"></div>
        SYSTEM ONLINE
      </div>
    </div>

    <div class="nav-section">MONITORING</div>
    <a onclick="showPage('dashboard')" id="nav-dashboard" class="active">
      <i class="fas fa-chart-line"></i> Dashboard
    </a>
    <a onclick="showPage('ac')" id="nav-ac">
      <i class="fas fa-snowflake"></i> AC Analytics
    </a>
    <a onclick="showPage('lamp')" id="nav-lamp">
      <i class="fas fa-lightbulb"></i> Lamp Analytics
    </a>
    <a onclick="showPage('camera')" id="nav-camera">
      <i class="fas fa-video"></i> Occupancy
    </a>

    <div class="nav-section">MANAGEMENT</div>
    <a onclick="showPage('power')" id="nav-power">
      <i class="fas fa-bolt"></i> Power Usage
    </a>
    <a onclick="showPage('control')" id="nav-control">
      <i class="fas fa-sliders-h"></i> Control Panel
    </a>
    <a onclick="showPage('logs')" id="nav-logs">
      <i class="fas fa-terminal"></i> System Logs
    </a>
  </div>

  <div class="main">
    <!-- DASHBOARD PAGE -->
    <div id="dashboard" class="page active">
      <div class="page-header">
        <div class="page-title"><i class="fas fa-chart-line"></i> Dashboard Overview</div>
        <div class="page-subtitle">Real-time monitoring of your smart room environment</div>
        <div class="breadcrumb">
          <span><i class="fas fa-home"></i> Home</span>
          <span><i class="fas fa-chevron-right"></i></span>
          <span>Dashboard</span>
        </div>
      </div>
      
      <div class="stats-section">
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-icon blue"><i class="fas fa-thermometer-half"></i></div>
            <div class="stat-content">
              <div class="label">Room Temperature</div>
              <div class="value"><span id="dash-temp">--</span>°C</div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon green"><i class="fas fa-tint"></i></div>
            <div class="stat-content">
              <div class="label">Humidity</div>
              <div class="value"><span id="dash-hum">--</span>%</div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon orange"><i class="fas fa-sun"></i></div>
            <div class="stat-content">
              <div class="label">Ambient Light</div>
              <div class="value"><span id="dash-lux">--</span> lux</div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon purple"><i class="fas fa-users"></i></div>
            <div class="stat-content">
              <div class="label">Occupancy</div>
              <div class="value"><span id="dash-people">0</span> people</div>
            </div>
          </div>
        </div>
      </div>

      <div class="grid-4">
        <div class="card">
          <div class="card-header">
            <div class="card-label">AC Status</div>
            <div class="card-icon blue"><i class="fas fa-snowflake"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="dash-ac-state">OFF</div>
          </div>
          <div class="card-trend">
            <i class="fas fa-arrow-down"></i>
            <span>Setting: <span id="dash-ac-temp">--</span>°C</span>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Lamp Brightness</div>
            <div class="card-icon orange"><i class="fas fa-lightbulb"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="dash-lamp-bright">--</div>
            <div class="card-unit">/255</div>
          </div>
          <div class="card-trend">
            <i class="fas fa-check-circle"></i>
            <span>Active</span>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Optimizations</div>
            <div class="card-icon green"><i class="fas fa-brain"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="dash-opt-count">0</div>
          </div>
          <div class="card-trend">
            <i class="fas fa-arrow-up"></i>
            <span>AI Running</span>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">System Uptime</div>
            <div class="card-icon purple"><i class="fas fa-clock"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="dash-uptime" style="font-size:20px;">--</div>
          </div>
          <div class="card-trend">
            <i class="fas fa-check-circle"></i>
            <span>Healthy</span>
          </div>
        </div>
      </div>

      <div class="grid-2">
        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title"><i class="fas fa-chart-area"></i> Temperature Trend</div>
            <div class="chart-options">
              <button class="chart-option-btn active" onclick="changeChartRange('temp', 1)">1h</button>
              <button class="chart-option-btn" onclick="changeChartRange('temp', 6)">6h</button>
              <button class="chart-option-btn" onclick="changeChartRange('temp', 24)">24h</button>
            </div>
          </div>
          <div class="chart-canvas">
            <canvas id="tempChart"></canvas>
          </div>
        </div>

        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title"><i class="fas fa-chart-area"></i> Humidity Trend</div>
            <div class="chart-options">
              <button class="chart-option-btn active" onclick="changeChartRange('hum', 1)">1h</button>
              <button class="chart-option-btn" onclick="changeChartRange('hum', 6)">6h</button>
              <button class="chart-option-btn" onclick="changeChartRange('hum', 24)">24h</button>
            </div>
          </div>
          <div class="chart-canvas">
            <canvas id="humChart"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- AC ANALYTICS PAGE -->
    <div id="ac" class="page">
      <div class="page-header">
        <div class="page-title"><i class="fas fa-snowflake"></i> AC System Analytics</div>
        <div class="page-subtitle">Genetic Algorithm optimization & monitoring</div>
      </div>

      <div class="grid-4">
        <div class="card">
          <div class="card-header">
            <div class="card-label">Current Temp</div>
            <div class="card-icon red"><i class="fas fa-thermometer-three-quarters"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="ac-temp">--</div>
            <div class="card-unit">°C</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">AC Setting</div>
            <div class="card-icon blue"><i class="fas fa-temperature-low"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="ac-setting">--</div>
            <div class="card-unit">°C</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Fan Speed</div>
            <div class="card-icon green"><i class="fas fa-fan"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="ac-fan">--</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Heat Index</div>
            <div class="card-icon orange"><i class="fas fa-fire"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="ac-hi">--</div>
            <div class="card-unit">°C</div>
          </div>
        </div>
      </div>

      <div class="grid-2">
        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title"><i class="fas fa-chart-line"></i> Temperature History</div>
            <div class="chart-options">
              <button class="chart-option-btn active" onclick="changeChartRange('acTemp', 1)">1h</button>
              <button class="chart-option-btn" onclick="changeChartRange('acTemp', 6)">6h</button>
              <button class="chart-option-btn" onclick="changeChartRange('acTemp', 24)">24h</button>
            </div>
          </div>
          <div class="chart-canvas">
            <canvas id="acTempChart"></canvas>
          </div>
        </div>

        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title"><i class="fas fa-chart-bar"></i> Humidity History</div>
            <div class="chart-options">
              <button class="chart-option-btn active" onclick="changeChartRange('acHum', 1)">1h</button>
              <button class="chart-option-btn" onclick="changeChartRange('acHum', 6)">6h</button>
              <button class="chart-option-btn" onclick="changeChartRange('acHum', 24)">24h</button>
            </div>
          </div>
          <div class="chart-canvas">
            <canvas id="acHumChart"></canvas>
          </div>
        </div>
      </div>

      <div class="stats-section" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(6, 78, 59, 0.1) 100%); border-color: rgba(34, 197, 94, 0.3);">
        <div style="font-size:18px; font-weight:700; color:#22c55e; margin-bottom:20px;">
          <i class="fas fa-dna"></i> Genetic Algorithm Optimization
        </div>
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-icon green"><i class="fas fa-bullseye"></i></div>
            <div class="stat-content">
              <div class="label">Recommended Temp</div>
              <div class="value"><span id="ga-temp">--</span>°C</div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon orange"><i class="fas fa-bolt"></i></div>
            <div class="stat-content">
              <div class="label">Energy Score</div>
              <div class="value"><span id="ga-energy">--</span></div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon blue"><i class="fas fa-smile"></i></div>
            <div class="stat-content">
              <div class="label">Comfort Score</div>
              <div class="value"><span id="ga-comfort">--</span></div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon purple"><i class="fas fa-fan"></i></div>
            <div class="stat-content">
              <div class="label">Optimal Fan Speed</div>
              <div class="value"><span id="ga-fan">--</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- LAMP ANALYTICS PAGE -->
    <div id="lamp" class="page">
      <div class="page-header">
        <div class="page-title"><i class="fas fa-lightbulb"></i> Lamp System Analytics</div>
        <div class="page-subtitle">PSO optimization & lighting control</div>
      </div>

      <div class="grid-4">
        <div class="card">
          <div class="card-header">
            <div class="card-label">Ambient Light</div>
            <div class="card-icon orange"><i class="fas fa-sun"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="lamp-lux">--</div>
            <div class="card-unit">lux</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Brightness</div>
            <div class="card-icon green"><i class="fas fa-adjust"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="lamp-brightness">--</div>
            <div class="card-unit">/255</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Motion</div>
            <div class="card-icon blue"><i class="fas fa-walking"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="lamp-motion">--</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Power Usage</div>
            <div class="card-icon red"><i class="fas fa-plug"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="lamp-power">--</div>
            <div class="card-unit">W</div>
          </div>
        </div>
      </div>

      <div class="grid-2">
        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title"><i class="fas fa-chart-area"></i> Light Level Trend</div>
            <div class="chart-options">
              <button class="chart-option-btn active" onclick="changeChartRange('lampLux', 1)">1h</button>
              <button class="chart-option-btn" onclick="changeChartRange('lampLux', 6)">6h</button>
              <button class="chart-option-btn" onclick="changeChartRange('lampLux', 24)">24h</button>
            </div>
          </div>
          <div class="chart-canvas">
            <canvas id="lampLuxChart"></canvas>
          </div>
        </div>

        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title"><i class="fas fa-chart-line"></i> Brightness History</div>
            <div class="chart-options">
              <button class="chart-option-btn active" onclick="changeChartRange('lampBright', 1)">1h</button>
              <button class="chart-option-btn" onclick="changeChartRange('lampBright', 6)">6h</button>
              <button class="chart-option-btn" onclick="changeChartRange('lampBright', 24)">24h</button>
            </div>
          </div>
          <div class="chart-canvas">
            <canvas id="lampBrightChart"></canvas>
          </div>
        </div>
      </div>

      <div class="stats-section" style="background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(6, 78, 59, 0.1) 100%); border-color: rgba(14, 165, 233, 0.3);">
        <div style="font-size:18px; font-weight:700; color:#38bdf8; margin-bottom:20px;">
          <i class="fas fa-project-diagram"></i> PSO (Particle Swarm Optimization)
        </div>
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-icon blue"><i class="fas fa-certificate"></i></div>
            <div class="stat-content">
              <div class="label">Optimal Brightness</div>
              <div class="value"><span id="pso-bright">--</span></div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon orange"><i class="fas fa-crosshairs"></i></div>
            <div class="stat-content">
              <div class="label">Target Lux</div>
              <div class="value"><span id="pso-target">--</span> lux</div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon green"><i class="fas fa-chart-line"></i></div>
            <div class="stat-content">
              <div class="label">Efficiency Score</div>
              <div class="value"><span id="pso-eff">--</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- CAMERA/OCCUPANCY PAGE -->
    <div id="camera" class="page">
      <div class="page-header">
        <div class="page-title"><i class="fas fa-video"></i> Occupancy Detection</div>
        <div class="page-subtitle">AI-powered person detection with YOLOv8</div>
      </div>

      <div class="grid-3">
        <div class="card">
          <div class="card-header">
            <div class="card-label">Person Count</div>
            <div class="card-icon purple"><i class="fas fa-users"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="cam-count">0</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Occupancy Status</div>
            <div class="card-icon blue"><i class="fas fa-door-open"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="cam-occupied" style="font-size:20px;">No</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-label">Last Detection</div>
            <div class="card-icon green"><i class="fas fa-clock"></i></div>
          </div>
          <div class="card-value-wrapper">
            <div class="card-value" id="cam-update" style="font-size:16px;">--</div>
          </div>
        </div>
      </div>

      <div class="chart-container" style="height:auto;">
        <div class="chart-header">
          <div class="chart-title"><i class="fas fa-camera"></i> Live Camera Feed</div>
          <div>
            <button class="btn btn-primary" onclick="captureSnapshot()">
              <i class="fas fa-camera"></i> Capture
            </button>
            <button class="btn btn-success" id="auto-btn" onclick="toggleAuto()">
              <i class="fas fa-play"></i> Auto Capture
            </button>
          </div>
        </div>
        <div id="cam-box" style="min-height:400px; background:#000; border-radius:12px; display:flex; align-items:center; justify-content:center; color:#64748b;">
          Click Capture to take snapshot
        </div>
      </div>
    </div>

    <!-- POWER USAGE PAGE -->
    <div id="power" class="page">
      <div class="page-header">
        <div class="page-title"><i class="fas fa-bolt"></i> Power Consumption Analytics</div>
        <div class="page-subtitle">Real-time energy monitoring & cost estimation</div>
      </div>

      <div class="power-mega-card">
        <div class="power-header">
          <div class="power-title">
            <i class="fas fa-bolt"></i>
            Total Power Consumption
          </div>
          <div class="power-badge">LIVE MONITORING</div>
        </div>
        
        <div class="stats-grid" style="position:relative; z-index:1;">
          <div class="stat-item" style="background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.3);">
            <div class="stat-icon red"><i class="fas fa-snowflake"></i></div>
            <div class="stat-content">
              <div class="label">AC Power</div>
              <div class="value" style="color:#ef4444;"><span id="power-ac">0</span> W</div>
            </div>
          </div>

          <div class="stat-item" style="background: rgba(249, 115, 22, 0.1); border-color: rgba(249, 115, 22, 0.3);">
            <div class="stat-icon orange"><i class="fas fa-lightbulb"></i></div>
            <div class="stat-content">
              <div class="label">Lamp Power</div>
              <div class="value" style="color:#f97316;"><span id="power-lamp">0</span> W</div>
            </div>
          </div>

          <div class="stat-item" style="background: rgba(168, 85, 247, 0.1); border-color: rgba(168, 85, 247, 0.3);">
            <div class="stat-icon purple"><i class="fas fa-plug"></i></div>
            <div class="stat-content">
              <div class="label">Total Power</div>
              <div class="value" style="color:#a855f7;"><span id="power-total">0</span> W</div>
            </div>
          </div>

          <div class="stat-item" style="background: rgba(34, 197, 94, 0.1); border-color: rgba(34, 197, 94, 0.3);">
            <div class="stat-icon green"><i class="fas fa-money-bill-wave"></i></div>
            <div class="stat-content">
              <div class="label">Daily Cost (Est.)</div>
              <div class="value" style="color:#22c55e; font-size:18px;"><span id="power-cost">Rp 0</span></div>
            </div>
          </div>
        </div>
      </div>

      <div class="grid-2">
        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title"><i class="fas fa-chart-pie"></i> Power Breakdown</div>
          </div>
          <div class="chart-canvas">
            <canvas id="powerPieChart"></canvas>
          </div>
        </div>

        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title"><i class="fas fa-calculator"></i> Energy Calculation</div>
          </div>
          <div style="padding:20px;">
            <div style="margin:15px 0; padding:15px; background:rgba(14,165,233,0.1); border-radius:10px; border-left:3px solid #38bdf8;">
              <div style="font-size:12px; color:#64748b; margin-bottom:5px;">AC Power Consumption</div>
              <div style="font-size:14px; color:#e0e0e0;">
                Fan Speed 1: <strong>100W</strong> | Fan Speed 2: <strong>200W</strong> | Fan Speed 3: <strong>300W</strong>
              </div>
            </div>

            <div style="margin:15px 0; padding:15px; background:rgba(249,115,22,0.1); border-radius:10px; border-left:3px solid #f97316;">
              <div style="font-size:12px; color:#64748b; margin-bottom:5px;">Lamp Power Consumption</div>
              <div style="font-size:14px; color:#e0e0e0;">
                Power = (Brightness / 255) × 10W
              </div>
            </div>

            <div style="margin:15px 0; padding:15px; background:rgba(34,197,94,0.1); border-radius:10px; border-left:3px solid #22c55e;">
              <div style="font-size:12px; color:#64748b; margin-bottom:5px;">Example Calculation</div>
              <div style="font-size:14px; color:#e0e0e0; line-height:1.8;">
                AC ON (Fan 2) = 200W<br>
                Lamp (Brightness 128) = 5W<br>
                <strong style="color:#22c55e;">Total = 205W × 24h = 4.92 kWh/day</strong><br>
                <strong style="color:#f59e0b;">Cost = 4.92 kWh × Rp 1,500 = Rp 7,380/day</strong>
              </div>
            </div>

            <div style="margin:20px 0; padding:20px; background:rgba(168,85,247,0.1); border-radius:10px; text-align:center; border:2px solid rgba(168,85,247,0.3);">
              <div style="font-size:14px; color:#64748b; margin-bottom:8px;">ESTIMATED MONTHLY COST</div>
              <div style="font-size:32px; font-weight:800; color:#a855f7;">
                Rp <span id="monthly-cost">0</span>
              </div>
              <div style="font-size:12px; color:#64748b; margin-top:5px;">Based on Rp 1,500/kWh × 30 days</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- CONTROL PANEL PAGE -->
    <div id="control" class="page">
      <div class="page-header">
        <div class="page-title"><i class="fas fa-sliders-h"></i> Control Panel</div>
        <div class="page-subtitle">Manual control for AC & Lamp systems</div>
      </div>

      <div class="grid-2">
        <div class="control-section">
          <div class="control-header">
            <div class="control-title">
              <i class="fas fa-snowflake"></i> AC Control
            </div>
            <div class="mode-badge" id="ac-mode-badge" onclick="toggleACMode()">AUTO MODE</div>
          </div>

          <div style="display:flex; gap:10px; margin-bottom:20px;">
            <button class="btn btn-success" onclick="sendACCmd('on')" style="flex:1;">
              <i class="fas fa-power-off"></i> AC ON
            </button>
            <button class="btn btn-danger" onclick="sendACCmd('off')" style="flex:1;">
              <i class="fas fa-power-off"></i> AC OFF
            </button>
          </div>

          <div class="slider-control">
            <div class="slider-label">
              <span><i class="fas fa-temperature-low"></i> Temperature</span>
              <span class="slider-value"><span id="acTempVal">24</span>°C</span>
            </div>
            <input type="range" min="16" max="30" value="24" id="acTemp" 
                   oninput="document.getElementById('acTempVal').textContent=this.value">
          </div>

          <div class="slider-control">
            <div class="slider-label">
              <span><i class="fas fa-fan"></i> Fan Speed</span>
              <span class="slider-value" id="fanVal">2</span>
            </div>
            <input type="range" min="1" max="3" value="2" id="acFan" 
                   oninput="document.getElementById('fanVal').textContent=this.value">
          </div>

          <button class="btn btn-primary" onclick="setACTemp()" style="width:100%; margin-top:15px;">
            <i class="fas fa-check"></i> Apply AC Settings
          </button>
        </div>

        <div class="control-section">
          <div class="control-header">
            <div class="control-title">
              <i class="fas fa-lightbulb"></i> Lamp Control
            </div>
            <div class="mode-badge" id="lamp-mode-badge" onclick="toggleLampMode()">AUTO MODE</div>
          </div>

          <div class="slider-control">
            <div class="slider-label">
              <span><i class="fas fa-adjust"></i> Brightness</span>
              <span class="slider-value"><span id="lampBrightVal">128</span> / 255</span>
            </div>
            <input type="range" min="0" max="255" value="128" id="lampBright" 
                   oninput="document.getElementById('lampBrightVal').textContent=this.value">
          </div>

          <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:20px 0;">
            <button class="btn btn-secondary" onclick="setLampBright(0)">
              <i class="fas fa-power-off"></i> OFF
            </button>
            <button class="btn btn-primary" onclick="setLampBright()">
              <i class="fas fa-check"></i> Apply
            </button>
            <button class="btn btn-success" onclick="setLampBright(255)">
              <i class="fas fa-sun"></i> MAX
            </button>
          </div>

          <div style="margin-top:20px; padding:15px; background:rgba(14,165,233,0.1); border-radius:10px;">
            <div style="font-size:12px; color:#64748b; margin-bottom:8px;">Quick Presets</div>
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px;">
              <button class="btn btn-secondary" onclick="setLampBright(64)" style="padding:8px;">25%</button>
              <button class="btn btn-secondary" onclick="setLampBright(128)" style="padding:8px;">50%</button>
              <button class="btn btn-secondary" onclick="setLampBright(192)" style="padding:8px;">75%</button>
              <button class="btn btn-secondary" onclick="setLampBright(255)" style="padding:8px;">100%</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- LOGS PAGE -->
    <div id="logs" class="page">
      <div class="page-header">
        <div class="page-title"><i class="fas fa-terminal"></i> System Logs</div>
        <div class="page-subtitle">Real-time system events and activities</div>
      </div>

      <div class="control-section" style="margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <span style="font-size:14px; color:#64748b;">Total Logs: </span>
            <span style="font-size:18px; font-weight:700; color:#38bdf8;" id="log-count">0</span>
          </div>
          <div style="display:flex; gap:10px;">
            <button class="btn btn-primary" onclick="refreshLogs()">
              <i class="fas fa-sync"></i> Refresh
            </button>
            <button class="btn btn-danger" onclick="clearLogs()">
              <i class="fas fa-trash"></i> Clear
            </button>
          </div>
        </div>
      </div>

      <div class="logs-container" id="logs-container">
        <!-- Logs will be populated here -->
      </div>
    </div>

  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let charts = {};
let autoCaptureInterval = null;
let chartRanges = {
    'temp': 1,
    'hum': 1,
    'acTemp': 1,
    'acHum': 1,
    'lampLux': 1,
    'lampBright': 1
};

// Navigation
function showPage(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
  document.getElementById(page).classList.add('active');
  document.getElementById('nav-' + page).classList.add('active');
  
  // Save to localStorage
  localStorage.setItem('currentPage', page);
  
  setTimeout(() => {
    Object.values(charts).forEach(chart => {
      if(chart) chart.resize();
    });
  }, 100);

  if(page === 'logs') refreshLogs();
}

// Load page from localStorage on init
function loadSavedPage() {
  let savedPage = localStorage.getItem('currentPage') || 'dashboard';
  showPage(savedPage);
}

// Update Data
function updateData() {
  fetch('/api/data').then(r => r.json()).then(d => {
    let s = (id, v) => { let e = document.getElementById(id); if (e) e.textContent = v; };
    
    // Dashboard
    s('dash-temp', d.ac.temperature);
    s('dash-hum', d.ac.humidity);
    s('dash-lux', d.lamp.lux);
    s('dash-people', d.camera.person_count);
    s('dash-ac-state', (d.ac.ac_state || 'off').toUpperCase());
    s('dash-ac-temp', d.ac.ac_temp);
    s('dash-lamp-bright', d.lamp.brightness);
    s('dash-opt-count', d.system.optimization_count);
    s('dash-uptime', d.system.uptime);
    
    // AC Analytics
    s('ac-temp', d.ac.temperature);
    s('ac-setting', d.ac.ac_temp);
    s('ac-fan', d.ac.fan_speed);
    s('ac-hi', d.ac.heat_index);
    s('ga-temp', d.ga_result.temp);
    s('ga-energy', d.ga_result.energy);
    s('ga-comfort', d.ga_result.comfort);
    s('ga-fan', d.ga_result.fan);
    
    // Lamp Analytics
    s('lamp-lux', d.lamp.lux);
    s('lamp-brightness', d.lamp.brightness);
    s('lamp-motion', d.lamp.motion == 1 ? 'Detected' : 'None');
    s('pso-bright', d.pso_result.brightness);
    s('pso-target', d.pso_result.target_lux);
    s('pso-eff', d.pso_result.efficiency);
    
    // Camera
    s('cam-count', d.camera.person_count);
    s('cam-occupied', d.camera.occupied ? 'Yes' : 'No');
    s('cam-update', d.camera.last_update || '--');
    
    // Mode badges
    updateModeBadges(d.ac_mode, d.lamp_mode);
    
    calculatePower(d);
  });
}

function updateModeBadges(acMode, lampMode) {
  let acBadge = document.getElementById('ac-mode-badge');
  let lampBadge = document.getElementById('lamp-mode-badge');
  
  if(acBadge) {
    acBadge.textContent = acMode.toUpperCase() + ' MODE';
    acBadge.className = 'mode-badge ' + acMode;
  }
  
  if(lampBadge) {
    lampBadge.textContent = lampMode.toUpperCase() + ' MODE';
    lampBadge.className = 'mode-badge ' + lampMode;
  }
}

function calculatePower(d) {
  let acWatt = 0;
  let lampWatt = 0;
  
  if (d.ac.ac_state === 'on' || d.ac.ac_state === 'ON') {
    let fan = parseInt(d.ac.fan_speed) || 1;
    acWatt = fan * 100;
  }
  
  let brightness = parseInt(d.lamp.brightness) || 0;
  lampWatt = Math.round((brightness / 255) * 10 * 10) / 10;
  
  let totalWatt = acWatt + lampWatt;
  let dailyKwh = (totalWatt * 24) / 1000;
  let dailyCost = dailyKwh * 1500;
  let monthlyCost = dailyCost * 30;
  
  document.getElementById('power-ac').textContent = acWatt.toFixed(0);
  document.getElementById('power-lamp').textContent = lampWatt.toFixed(1);
  document.getElementById('power-total').textContent = totalWatt.toFixed(1);
  let lampPowerEl = document.getElementById('lamp-power');
  if(lampPowerEl) lampPowerEl.textContent = lampWatt.toFixed(1);
  document.getElementById('power-cost').textContent = 'Rp ' + Math.round(dailyCost).toLocaleString('id-ID');
  document.getElementById('monthly-cost').textContent = Math.round(monthlyCost).toLocaleString('id-ID');
  
  // Update pie chart
  if(charts.powerPieChart) {
    charts.powerPieChart.data.datasets[0].data = [acWatt, lampWatt];
    charts.powerPieChart.update('none'); // Update without animation
  }
}

// Create Charts
function createChart(canvasId, label, borderColor, data) {
  if (charts[canvasId]) charts[canvasId].destroy();
  
  let ctx = document.getElementById(canvasId);
  if (!ctx) return;
  
  charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => d.time),
      datasets: [{
        label: label,
        data: data.map(d => d.value),
        borderColor: borderColor,
        backgroundColor: borderColor + '30',
        tension: 0.4,
        fill: true,
        pointRadius: 3,
        pointHoverRadius: 6,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(51,65,85,0.3)' },
          ticks: { color: '#94a3b8' }
        },
        x: {
          grid: { color: 'rgba(51,65,85,0.3)' },
          ticks: { color: '#94a3b8', maxRotation: 0 }
        }
      },
      animation: {
        duration: 750
      }
    }
  });
}

function createPieChart() {
  let ctx = document.getElementById('powerPieChart');
  if(!ctx) return;
  
  charts.powerPieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['AC Power', 'Lamp Power'],
      datasets: [{
        data: [0, 0],
        backgroundColor: ['#ef4444', '#f97316'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#e0e0e0', padding: 20, font: {size: 14} }
        }
      }
    }
  });
}

function loadCharts() {
  // Load with saved ranges
  fetch(`/api/chart-data?measurement=ac_sensor&field=temperature&hours=${chartRanges.temp}`)
    .then(r => r.json())
    .then(d => createChart('tempChart', 'Temperature', '#ef4444', d));
  
  fetch(`/api/chart-data?measurement=ac_sensor&field=humidity&hours=${chartRanges.hum}`)
    .then(r => r.json())
    .then(d => createChart('humChart', 'Humidity', '#0ea5e9', d));
  
  fetch(`/api/chart-data?measurement=ac_sensor&field=temperature&hours=${chartRanges.acTemp}`)
    .then(r => r.json())
    .then(d => createChart('acTempChart', 'Temperature', '#ef4444', d));
  
  fetch(`/api/chart-data?measurement=ac_sensor&field=humidity&hours=${chartRanges.acHum}`)
    .then(r => r.json())
    .then(d => createChart('acHumChart', 'Humidity', '#38bdf8', d));
  
  fetch(`/api/chart-data?measurement=lamp_sensor&field=lux&hours=${chartRanges.lampLux}`)
    .then(r => r.json())
    .then(d => createChart('lampLuxChart', 'Lux', '#f59e0b', d));
  
  fetch(`/api/chart-data?measurement=lamp_sensor&field=brightness&hours=${chartRanges.lampBright}`)
    .then(r => r.json())
    .then(d => createChart('lampBrightChart', 'Brightness', '#22c55e', d));
  
  createPieChart();
}

function changeChartRange(chartName, hours) {
  chartRanges[chartName] = hours;
  
  // Save to localStorage
  localStorage.setItem('chartRanges', JSON.stringify(chartRanges));
  
  // Update button active state
  let parentContainer = event.target.closest('.chart-container');
  if(parentContainer) {
    parentContainer.querySelectorAll('.chart-option-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
  }
  
  // Reload specific chart
  let measurement, field, canvasId;
  switch(chartName) {
    case 'temp':
      measurement = 'ac_sensor';
      field = 'temperature';
      canvasId = 'tempChart';
      break;
    case 'hum':
      measurement = 'ac_sensor';
      field = 'humidity';
      canvasId = 'humChart';
      break;
    case 'acTemp':
      measurement = 'ac_sensor';
      field = 'temperature';
      canvasId = 'acTempChart';
      break;
    case 'acHum':
      measurement = 'ac_sensor';
      field = 'humidity';
      canvasId = 'acHumChart';
      break;
    case 'lampLux':
      measurement = 'lamp_sensor';
      field = 'lux';
      canvasId = 'lampLuxChart';
      break;
    case 'lampBright':
      measurement = 'lamp_sensor';
      field = 'brightness';
      canvasId = 'lampBrightChart';
      break;
  }
  
  fetch(`/api/chart-data?measurement=${measurement}&field=${field}&hours=${hours}`)
    .then(r => r.json())
    .then(d => {
      let color = charts[canvasId].data.datasets[0].borderColor.substring(0, 7);
      createChart(canvasId, field, color, d);
      showToast(`Chart updated to ${hours}h range`, 'info');
    });
}

// Load saved chart ranges
function loadSavedChartRanges() {
  let saved = localStorage.getItem('chartRanges');
  if(saved) {
    chartRanges = JSON.parse(saved);
  }
}

// AC Control
function sendACCmd(cmd) {
  let action = cmd === 'on' ? 'turn_on' : 'turn_off';
  fetch('/api/ac/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: action })
  }).then(() => {
    showToast(`AC ${cmd.toUpperCase()}`, 'success');
    setTimeout(updateData, 500);
  });
}

function setACTemp() {
  let temp = document.getElementById('acTemp').value;
  let fan = document.getElementById('acFan').value;
  fetch('/api/ac/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'set_temp', temperature: parseInt(temp), fan_speed: parseInt(fan) })
  }).then(() => {
    showToast(`AC set to ${temp}°C, Fan Speed ${fan}`, 'success');
    setTimeout(updateData, 500);
  });
}

function toggleACMode() {
  fetch('/api/data').then(r => r.json()).then(d => {
    let newMode = d.ac_mode === 'auto' ? 'manual' : 'auto';
    fetch('/api/ac/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: newMode })
    }).then(() => {
      showToast(`AC Mode: ${newMode.toUpperCase()}`, newMode === 'auto' ? 'info' : 'success');
      setTimeout(updateData, 500);
    });
  });
}

// Lamp Control
function setLampBright() {
  let bright = arguments[0] !== undefined ? arguments[0] : document.getElementById('lampBright').value;
  fetch('/api/lamp/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ brightness: parseInt(bright) })
  }).then(() => {
    if(arguments[0] !== undefined) {
      document.getElementById('lampBright').value = bright;
      document.getElementById('lampBrightVal').textContent = bright;
    }
    showToast(`Lamp brightness set to ${bright}`, 'success');
    setTimeout(updateData, 500);
  });
}

function toggleLampMode() {
  fetch('/api/data').then(r => r.json()).then(d => {
    let newMode = d.lamp_mode === 'auto' ? 'manual' : 'auto';
    fetch('/api/lamp/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: newMode })
    }).then(() => {
      showToast(`Lamp Mode: ${newMode.toUpperCase()}`, newMode === 'auto' ? 'info' : 'success');
      setTimeout(updateData, 500);
    });
  });
}

// Camera
function captureSnapshot() {
  let box = document.getElementById('cam-box');
  box.innerHTML = '<div class="loading-spinner"></div> <span style="margin-left:10px;">Capturing...</span>';
  
  fetch('/api/camera/snapshot').then(r => r.json()).then(d => {
    if(d.image) {
      box.innerHTML = '<img src="data:image/jpeg;base64,' + d.image + '" style="max-width:100%; border-radius:12px;">';
    } else {
      box.innerHTML = '<span style="color:#ef4444;">Error: ' + (d.error || 'No camera') + '</span>';
    }
    updateData();
  });
}

function toggleAuto() {
  let btn = document.getElementById('auto-btn');
  if(autoCaptureInterval) {
    clearInterval(autoCaptureInterval);
    autoCaptureInterval = null;
    btn.innerHTML = '<i class="fas fa-play"></i> Auto Capture';
    btn.className = 'btn btn-success';
  } else {
    captureSnapshot();
    autoCaptureInterval = setInterval(captureSnapshot, 3000);
    btn.innerHTML = '<i class="fas fa-stop"></i> Stop Auto';
    btn.className = 'btn btn-danger';
  }
}

// Logs
function refreshLogs() {
  fetch('/api/logs').then(r => r.json()).then(d => {
    let container = document.getElementById('logs-container');
    let count = document.getElementById('log-count');
    if(!container) return;
    
    if(count) count.textContent = d.logs.length;
    
    if(d.logs.length === 0) {
      container.innerHTML = '<div style="text-align:center; padding:50px; color:#64748b;">No logs available</div>';
      return;
    }
    
    container.innerHTML = d.logs.map(l => 
      `<div class="log-entry">
        <div class="log-time">${l.time.split(' ')[1]}</div>
        <div class="log-message">${l.message}</div>
      </div>`
    ).join('');
  });
}

function clearLogs() {
  fetch('/api/logs/clear', {method:'POST'}).then(() => {
    showToast('Logs cleared', 'info');
    refreshLogs();
  });
}

// Toast notification
function showToast(message, type) {
  let toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = 'toast show ' + type;
  setTimeout(() => {
    toast.className = 'toast';
  }, 3000);
}

// Initialize
loadSavedChartRanges();
loadSavedPage();
updateData();
loadCharts();
setInterval(updateData, 1000);
setInterval(loadCharts, 60000);
</script>
</body>
</html>
"""

# --- Routes ---
@app.route('/')
def index():
    return render_template_string(MAIN_HTML)

@app.route('/api/data')
def api_data():
    return jsonify(sensor_data)

@app.route('/api/chart-data')
def api_chart_data():
    measurement = request.args.get('measurement', 'ac_sensor')
    field = request.args.get('field', 'temperature')
    hours = int(request.args.get('hours', 1))
    
    data = get_influx_data(measurement, field, hours)
    return jsonify(data)

@app.route('/api/ac/mode', methods=['POST'])
def api_ac_mode():
    data = request.get_json()
    mode = data.get('mode', 'auto')
    sensor_data['ac_mode'] = mode
    mqtt_pub.publish("smartroom/ac/mode", json.dumps({"mode": mode}))
    add_log(f"[Control Panel] AC Mode changed to {mode.upper()}")
    return jsonify({"status": "ok", "mode": mode})

@app.route('/api/ac/control', methods=['POST'])
def api_ac_control():
    data = request.get_json()
    mqtt_pub.publish("smartroom/ac/control", json.dumps(data))
    system_stats['ac_commands'] += 1
    add_log(f"[Control] AC command sent: {data}")
    return jsonify({"status": "ok"})

@app.route('/api/lamp/mode', methods=['POST'])
def api_lamp_mode():
    data = request.get_json()
    mode = data.get('mode', 'auto')
    sensor_data['lamp_mode'] = mode
    mqtt_pub.publish("smartroom/lamp/mode", json.dumps({"mode": mode}))
    add_log(f"[Control Panel] Lamp Mode changed to {mode.upper()}")
    return jsonify({"status": "ok", "mode": mode})

@app.route('/api/lamp/control', methods=['POST'])
def api_lamp_control():
    data = request.get_json()
    mqtt_pub.publish("smartroom/lamp/control", json.dumps(data))
    system_stats['lamp_commands'] += 1
    add_log(f"[Control] Lamp command sent: {data}")
    return jsonify({"status": "ok"})

@app.route('/api/camera/snapshot')
def api_camera_snapshot():
    try:
        import cv2
        import base64

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({"error": "Cannot capture from camera"})

        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            results = model(frame, verbose=False)
            person_count = 0
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        person_count += 1
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (56,189,248), 3)
                        cv2.putText(frame, f"Person {conf:.1%}", (x1,y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56,189,248), 2)
            sensor_data['camera']['person_count'] = person_count
            sensor_data['camera']['occupied'] = person_count > 0
            sensor_data['camera']['last_update'] = datetime.now().strftime("%H:%M:%S")
        except ImportError:
            cv2.putText(frame, "YOLOv8 not installed", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"image": img_base64, "person_count": sensor_data['camera']['person_count']})

    except ImportError:
        return jsonify({"error": "OpenCV not installed"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/logs')
def api_logs():
    return jsonify({"logs": log_messages[:200]})

@app.route('/api/logs/clear', methods=['POST'])
def api_logs_clear():
    log_messages.clear():
    add_log("[System] Logs cleared")
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("=" * 70)
    print("   SMART ROOM IoT - PREMIUM ANALYTICS DASHBOARD")
    print("  " + "=" * 68)
    print("   URL: http://localhost:5000")
    print("   Features: Real-time Monitoring | AI Optimization | Power Analytics")
    print("   Pages: Dashboard | AC | Lamp | Camera | Power | Control | Logs")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False)
