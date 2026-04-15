from flask import Flask, render_template_string, jsonify, request, Response, session, redirect, url_for
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import json
from datetime import datetime, timedelta
from collections import deque
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartroom-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Authentication Credentials
AUTH_USERNAME = "admin"
AUTH_PASSWORD = "smartroom2024"

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
    'no_person_timeout': {'timeout_minutes': 10, 'enabled': True, 'last_person_seen': None}
}
active_alerts = deque(maxlen=50)

# Occupancy Feedback
GOOGLE_FORM_URL = "https://docs.google.com/forms/"
occupancy_feedback = deque(maxlen=200)

# Energy Phase: 'before' = sebelum adaptive AC, 'after' = sesudah adaptive AC
energy_phase = 'before'

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
NO_PERSON_TIMEOUT_SECONDS = 600  # 10 menit no person -> auto OFF
AUTO_OFF_COOLDOWN = 120          # 2 menit cooldown: block auto-ON setelah auto-OFF
_last_person_confirmed_time = 0.0  # Unix time when person last confirmed (0 = never since startup)
_auto_off_triggered = False        # Prevents repeated POWER_OFF from firing
_auto_off_time = 0.0               # Unix time when auto-OFF last fired (for cooldown)

# Background detection thread state
_latest_frame_bytes = None
_latest_frame_lock = threading.Lock()
_detection_thread_running = False

# Adaptive apply debounce: prevent duplicate AC/Lamp commands within 5 seconds
_last_adaptive_ac_apply = 0
_last_adaptive_lamp_apply = 0

# Global data storage
mqtt_data = {
    'ac': {'temperature': 0, 'humidity': 0, 'heat_index': 0, 'ac_state': 'OFF', 'ac_temp': 24, 'fan_speed': 1, 'mode': 'ADAPTIVE', 'ac_fan_mode': 'COOL', 'rssi': 0, 'uptime': 0, 'temp1': 0, 'hum1': 0, 'temp2': 0, 'hum2': 0, 'temp3': 0, 'hum3': 0},
    'lamp': {'lux1': 0, 'lux2': 0, 'lux3': 0, 'lux_avg': 0, 'motion': False, 'brightness1': 0, 'brightness2': 0, 'brightness3': 0, 'brightness_avg': 0, 'mode': 'ADAPTIVE', 'rssi': 0, 'uptime': 0},
    'camera': {'person_detected': False, 'count': 0, 'confidence': 0, 'status': 'inactive'},
    'energy': {'voltage': 0, 'current': 0, 'power': 0, 'energy': 0, 'frequency': 0, 'pf': 0, 'connected': False, 'ac_state': 'OFF'},
    'system': {'ga_fitness': 0, 'pso_fitness': 0, 'optimization_runs': 0, 'ga_temp': 0, 'ga_fan': 0, 'pso_brightness': 0, 'ga_history': [], 'pso_history': []},
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

# Debug counter for /api/data response snapshots
api_data_debug_counter = 0

# ==================== INFLUXDB WRITE FUNCTIONS ====================
def write_to_influxdb(measurement, fields, tags=None):
    """Write data point to InfluxDB"""
    try:
        print(f"[DB] Attempting to write to InfluxDB: {measurement}")
        print(f"   URL: {INFLUX_URL}, ORG: {INFLUX_ORG}, BUCKET: {INFLUX_BUCKET}")
        
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        point = Point(measurement).time(datetime.utcnow(), WritePrecision.NS)
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                point = point.tag(key, str(value))
        
        # Add fields
        for key, value in fields.items():
            if isinstance(value, (int, float)):
                point = point.field(key, float(value))
            elif isinstance(value, bool):
                point = point.field(key, value)
            else:
                point = point.field(key, str(value))
        
        print(f"   Data: {fields}")
        
        # Write to InfluxDB
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        write_api.close()
        client.close()
        print(f"[OK] Successfully written to InfluxDB: {measurement}")
        return True
    except Exception as e:
        print(f"[ERROR] InfluxDB Write Error ({measurement}): {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def save_sensor_data(temperature, humidity, heat_index):
    """Save temperature and humidity data to InfluxDB"""
    try:
        result = write_to_influxdb(
            measurement="ac_sensor",
            fields={
                "temperature": float(temperature),
                "humidity": float(humidity),
                "heat_index": float(heat_index)
            },
            tags={"device": "esp32_ac", "location": "room"}
        )
        if result:
            print(f"[OK] Sensor data saved: {temperature}°C, {humidity}%")
    except Exception as e:
        print(f"[ERROR] Error saving sensor data: {e}")
        import traceback
        traceback.print_exc()

def save_lamp_data(lux1, lux2, lux3, brightness1, brightness2, brightness3, motion):
    """Save lamp sensor data to InfluxDB (3 lamps)"""
    try:
        lux_avg = (lux1 + lux2 + lux3) / 3.0
        bright_avg = (brightness1 + brightness2 + brightness3) / 3.0
        result = write_to_influxdb(
            measurement="lamp_sensor",
            fields={
                "lux1": float(lux1), "lux2": float(lux2), "lux3": float(lux3), "lux_avg": float(lux_avg),
                "brightness1": float(brightness1), "brightness2": float(brightness2), "brightness3": float(brightness3), "brightness_avg": float(bright_avg),
                "motion": bool(motion)
            },
            tags={"device": "esp32_lamp", "location": "room"}
        )
        if result:
            print(f"[OK] Lamp data saved: lux=[{lux1},{lux2},{lux3}] avg={lux_avg:.1f}, bright=[{brightness1},{brightness2},{brightness3}]")
    except Exception as e:
        print(f"[ERROR] Error saving lamp data: {e}")
        import traceback
        traceback.print_exc()

def save_person_detection(person_count, confidence):
    """Save person detection data to InfluxDB"""
    try:
        write_to_influxdb(
            measurement="camera_detection",
            fields={
                "person_count": int(person_count),
                "confidence": float(confidence),
                "person_detected": bool(person_count > 0)
            },
            tags={"device": "camera_yolo", "model": "yolov8n"}
        )
        if person_count > 0:
            print(f"[OK] Detection saved: {person_count} person(s), {confidence:.2f} confidence")
    except Exception as e:
        print(f"[ERROR] Error saving detection data: {e}")

def save_ir_command(device, command, signal_length):
    """Save IR remote command to InfluxDB"""
    try:
        write_to_influxdb(
            measurement="ir_remote",
            fields={
                "command": str(command),
                "signal_length": int(signal_length),
                "learned": True
            },
            tags={"device": str(device), "type": "ir_code"}
        )
        print(f"[OK] IR command saved: {device} - {command}")
    except Exception as e:
        print(f"[ERROR] Error saving IR command: {e}")

def save_ac_control(ac_temp, fan_speed, ac_state):
    """Save AC control settings to InfluxDB"""
    try:
        write_to_influxdb(
            measurement="ac_sensor",
            fields={
                "ac_temp": float(ac_temp),
                "fan_speed": int(fan_speed),
                "ac_state": str(ac_state)
            },
            tags={"device": "esp32_ac", "type": "control"}
        )
        print(f"[OK] AC control saved: {ac_temp}°C, Fan: {fan_speed}, State: {ac_state}")
    except Exception as e:
        print(f"[ERROR] Error saving AC control: {e}")

# ==================== YOLO INITIALIZATION ====================
def load_yolo_model():
    global yolo_model
    try:
        import os
        
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
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
        return frame, 0, 0.0, []

def _person_present_recently():
    """True if person confirmed within NO_PERSON_TIMEOUT_SECONDS — blocks adaptive SET when room is empty"""
    return _last_person_confirmed_time > 0 and \
           (time.time() - _last_person_confirmed_time) < NO_PERSON_TIMEOUT_SECONDS

# ==================== SMART PERSON-BASED CONTROL ====================
def handle_person_based_control(person_count):
    """Smart auto ON/OFF: turn ON AC when person confirmed, OFF after 10 min empty"""
    global _person_consecutive_frames, _no_person_start_time, _auto_off_triggered, _last_person_confirmed_time, _auto_off_time
    
    # Only act in ADAPTIVE mode
    if mqtt_data['ac'].get('mode') != 'ADAPTIVE':
        _person_consecutive_frames = 0
        _no_person_start_time = None
        _auto_off_triggered = False
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
            _auto_off_triggered = False  # Allow auto-OFF to fire again if person leaves later
            _auto_off_time = 0.0  # Clear cooldown
            print(f"[OK] Person RE-CONFIRMED after {_person_consecutive_frames} frames (cooldown cleared)")
        elif _person_consecutive_frames >= required_frames and in_cooldown:
            cooldown_left = AUTO_OFF_COOLDOWN - (time.time() - _auto_off_time)
            if int(_person_consecutive_frames) % 30 == 0:  # Print every ~20s
                print(f"[WAIT] Person detected but cooldown active ({int(cooldown_left)}s left) — need sustained presence")
        
        # Auto ON: person confirmed AND cooldown elapsed
        if _person_consecutive_frames >= required_frames and not in_cooldown and not _auto_off_triggered:
            if mqtt_data['ac'].get('ac_state') == 'OFF':
                print(f"[ON] Person confirmed ({_person_consecutive_frames} frames, req={required_frames}) -> Auto turning ON AC")
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
            _person_consecutive_frames = required_frames  # Cap to avoid overflow
    else:
        _person_consecutive_frames = 0
        
        # Start no-person timer
        if _no_person_start_time is None:
            _no_person_start_time = time.time()
        
        # Auto OFF: no person for NO_PERSON_TIMEOUT_SECONDS
        elapsed = time.time() - _no_person_start_time
        if elapsed >= NO_PERSON_TIMEOUT_SECONDS and not _auto_off_triggered:
            _auto_off_triggered = True  # Always set flag to block adaptive SET
            _auto_off_time = time.time()  # Start cooldown timer
            if mqtt_data['ac'].get('ac_state') != 'OFF':
                print(f"[OFF] No person for {int(elapsed)}s -> Auto turning OFF AC (cooldown {AUTO_OFF_COOLDOWN}s starts)")
                try:
                    mqtt_client.publish("smartroom/ac/control", json.dumps({
                        "action": "POWER_OFF",
                        "source": "camera_auto"
                    }))
                    mqtt_data['ac']['ac_state'] = 'OFF'
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'),
                                       'msg': f'Auto OFF: No person for {int(elapsed/60)} min', 'level': 'warning'})
                    socketio.emit('alert', {
                        'type': 'auto_off', 'level': 'warning',
                        'message': f'No person for {int(elapsed/60)} min — AC turned OFF automatically',
                        'time': datetime.now().strftime('%H:%M:%S')
                    })
                except Exception as e:
                    print(f"[ERROR] Auto OFF error: {e}")

# ==================== CAMERA FUNCTIONS ====================
def get_camera():
    global camera
    if camera is None:
        # Auto-detect camera: try index 0-4
        for idx in range(5):
            print(f"[CAM] Trying camera index {idx}...")
            cam = cv2.VideoCapture(idx)
            if cam.isOpened():
                # Verify it can actually capture a frame
                ret, test_frame = cam.read()
                if ret and test_frame is not None:
                    camera = cam
                    print(f"[OK] Camera found at index {idx}")
                    break
                else:
                    cam.release()
                    print(f"   Index {idx}: opened but can't read frames")
            else:
                cam.release()
                print(f"   Index {idx}: not available")
        
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
            print(f"[OK] Camera initialized: {actual_w}x{actual_h} @ {actual_fps}fps")
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
    retry_delay = 2
    frame_count = 0
    YOLO_EVERY_N = 4  # Run YOLO every 4th frame — other frames just capture+encode
    last_person_count = 0
    last_confidence = 0.0
    last_boxes = []  # Cache bounding boxes to draw on non-YOLO frames
    fps_counter = 0
    fps_timer = time.time()
    
    print("[CAM] Camera detection background thread started (runs 24/7)")
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
                        print("[WARN] Camera read failed, will retry...")
            
            # If no frame captured, wait and retry (lock is released)
            if frame is None:
                time.sleep(retry_delay)
                continue
            
            mqtt_data['camera']['status'] = 'active'
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
            
            # Debug: print timer status every 60 seconds
            if _no_person_start_time is not None:
                elapsed = time.time() - _no_person_start_time
                if int(elapsed) % 60 < 1:
                    print(f"[TIMER] No person timer: {int(elapsed)}s / {NO_PERSON_TIMEOUT_SECONDS}s | AC: {mqtt_data['ac'].get('ac_state')} | Mode: {mqtt_data['ac'].get('mode')}")
            
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
            import traceback
            traceback.print_exc()
            time.sleep(retry_delay)
    
    print("[STOP] Camera detection background thread stopped")

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
    print("\n" + "="*70)
    print("  [MQTT] FLASK MQTT CONNECTION EVENT")
    print("="*70)
    
    is_success = False
    if hasattr(reason_code, 'is_failure'):
        is_success = not reason_code.is_failure
    else:
        is_success = (int(reason_code) == 0)
    
    if is_success:
        mqtt_status['connected'] = True
        mqtt_status['last_connect_time'] = datetime.now().strftime('%H:%M:%S')
        mqtt_status['error'] = None
        print("[OK] MQTT CONNECTED SUCCESSFULLY!")
        
        client.subscribe("smartroom/#")
        client.subscribe("ir/#")
        client.subscribe("IR/#")
        client.subscribe("+/ir/#")
        print("[OK] Subscribed to: smartroom/#, ir/#, IR/#, +/ir/#")
        print("="*70 + "\n")
        
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'MQTT Connected!', 'level': 'success'})
    else:
        mqtt_status['connected'] = False
        mqtt_status['error'] = f'Connection failed: {reason_code}'
        print(f"[ERROR] MQTT CONNECTION FAILED! RC={reason_code}")
        print("="*70 + "\n")

def on_message(client, userdata, msg):
    global ir_learning_mode, ir_learning_button, ir_learning_device, mqtt_status
    mqtt_status['last_message_time'] = datetime.now().strftime('%H:%M:%S')
    mqtt_status['message_count'] = mqtt_status.get('message_count', 0) + 1
    
    try:
        topic = msg.topic
        
        # Debug: Print ALL incoming MQTT messages with timestamp
        print("\n" + "─"*70)
        print(f"[MSG] [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] MQTT Message Received")
        print(f"Topic: {topic}")
        print(f"Payload Length: {len(msg.payload)} bytes")
        print(f"QoS: {msg.qos} | Retain: {msg.retain}")
        
        # Broadcast to frontend for debugging
        try:
            payload_str = msg.payload.decode()[:100]
            socketio.emit('mqtt_debug', {
                'topic': topic,
                'payload': payload_str,
                'time': datetime.now().strftime('%H:%M:%S')
            })
        except Exception as e:
            pass
        
        # Try to parse as JSON first
        try:
            full_payload_str = msg.payload.decode()  # Decode ONCE, use full string
            payload = json.loads(full_payload_str)
            # Don't print full payload (RAW IR codes are 1000+ chars), show summary
            if isinstance(payload, dict) and 'code' in payload:
                code_val = payload['code']
                code_len = len(code_val) if isinstance(code_val, str) else 0
                raw_count = code_val.count(',') + 1 if isinstance(code_val, str) and code_val.startswith('RAW:') else 0
                print(f"   Data (JSON): code length={code_len} chars, RAW values={raw_count}")
                print(f"   Code preview: {code_val[:80]}..." if code_len > 80 else f"   Code: {code_val}")
            else:
                print(f"   Data (JSON): {payload}")
        except Exception as json_err:
            print(f"[JSON ERROR] Failed to parse JSON: {json_err} - Payload: {full_payload_str}")
            # If not JSON, treat as plain text
            payload_text = msg.payload.decode()
            print(f"   Data (Text): {payload_text[:200]}..." if len(payload_text) > 200 else f"   Data (Text): {payload_text}")
            payload = {'raw': payload_text}
        
        # DEBUG: Check which condition will match
        print(f"\n[ROUTE] Topic Routing Check:")
        print(f"   'ac/sensors' in topic: {'ac/sensors' in topic}")
        print(f"   'lamp/sensors' in topic: {'lamp/sensors' in topic}")
        print(f"   'camera/detection' in topic: {'camera/detection' in topic}")
        print(f"   'dashboard/state' in topic: {'dashboard/state' in topic}")
        print(f"   'ml/result' in topic: {'ml/result' in topic}")
        print(f"   'ac/mode' in topic: {'ac/mode' in topic}")
        print(f"   'lamp/mode' in topic: {'lamp/mode' in topic}")
        print(f"   'ir/learned' in topic: {'ir/learned' in topic}")
        print(f"   'IR/learned' in topic.lower(): {'IR/learned' in topic.lower()}")
        
        if 'ac/sensors' in topic:
            print("[SENSOR] Processing AC Sensor Data:")
            print(f"   Temperature: {payload.get('temperature', 0)}°C")
            print(f"   Humidity: {payload.get('humidity', 0)}%")
            print(f"   Heat Index: {payload.get('heat_index', 0)}°C")
            
            mqtt_data['ac'].update({
                'temperature': payload.get('temperature', 0),
                'humidity': payload.get('humidity', 0),
                'heat_index': payload.get('heat_index', 0),
                'ac_state': payload.get('ac_state', 'OFF'),
                'ac_temp': payload.get('ac_temp', 24),
                'fan_speed': payload.get('fan_speed', 1),
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
            })
            # Save sensor data to InfluxDB
            print("   [DB] Saving to InfluxDB...")
            save_sensor_data(
                mqtt_data['ac']['temperature'],
                mqtt_data['ac']['humidity'],
                mqtt_data['ac']['heat_index']
            )
            # Save AC control state
            save_ac_control(
                mqtt_data['ac']['ac_temp'],
                mqtt_data['ac']['fan_speed'],
                mqtt_data['ac']['ac_state']
            )
            print("   [OK] AC data updated in memory & InfluxDB")
            socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
            print("   [WS] Sent to frontend via WebSocket")
            # Track device status & check alerts
            device_last_seen['esp32_ac']['last_seen'] = datetime.now()
            device_last_seen['esp32_ac']['status'] = 'online'
            check_alert_rules()
            
        elif 'lamp/sensors' in topic:
            l1 = payload.get('lux1', payload.get('lux', 0))
            l2 = payload.get('lux2', l1)
            l3 = payload.get('lux3', l1)
            b1 = payload.get('brightness1', payload.get('brightness', 0))
            b2 = payload.get('brightness2', b1)
            b3 = payload.get('brightness3', b1)
            mqtt_data['lamp'].update({
                'lux1': l1, 'lux2': l2, 'lux3': l3,
                'lux_avg': round((l1 + l2 + l3) / 3.0, 1),
                'motion': payload.get('motion', False),
                'brightness1': b1, 'brightness2': b2, 'brightness3': b3,
                'brightness_avg': round((b1 + b2 + b3) / 3.0, 1),
                'rssi': payload.get('rssi', 0),
                'uptime': payload.get('uptime', 0)
            })
            # Save lamp data to InfluxDB
            save_lamp_data(l1, l2, l3, b1, b2, b3, mqtt_data['lamp']['motion'])
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
            # Track device status
            device_last_seen['esp32_lamp']['last_seen'] = datetime.now()
            device_last_seen['esp32_lamp']['status'] = 'online'
            
        elif 'energy/data' in topic:
            mqtt_data['energy'].update({
                'voltage': payload.get('voltage', 0),
                'current': payload.get('current', 0),
                'power': payload.get('power', 0),
                'energy': payload.get('energy', 0),
                'frequency': payload.get('frequency', 0),
                'pf': payload.get('pf', 0),
                'connected': True
            })
            print(f"   [ENERGY] Energy: {mqtt_data['energy']['voltage']}V {mqtt_data['energy']['current']}A {mqtt_data['energy']['power']}W {mqtt_data['energy']['energy']}kWh")
            # Save to InfluxDB for historical data (30 day retention)
            write_to_influxdb('energy_monitor', {
                'voltage': float(mqtt_data['energy']['voltage']),
                'current': float(mqtt_data['energy']['current']),
                'power': float(mqtt_data['energy']['power']),
                'energy_kwh': float(mqtt_data['energy']['energy']),
                'frequency': float(mqtt_data['energy']['frequency']),
                'power_factor': float(mqtt_data['energy']['pf'])
            }, tags={'device': 'pzem016', 'phase': energy_phase})

            # Keep runtime ring-buffer so charts still work even if Influx has no data yet.
            try:
                energy_runtime_history.append({
                    'ts': datetime.now(),
                    'phase': energy_phase,
                    'voltage': float(mqtt_data['energy']['voltage']),
                    'current': float(mqtt_data['energy']['current']),
                    'power': float(mqtt_data['energy']['power']),
                    'energy_kwh': float(mqtt_data['energy']['energy']),
                    'frequency': float(mqtt_data['energy']['frequency']),
                    'power_factor': float(mqtt_data['energy']['pf'])
                })
            except Exception:
                pass

            socketio.emit('mqtt_update', {'type': 'energy', 'data': mqtt_data['energy']})
            # Track device status
            device_last_seen['esp32_energy'] = {'last_seen': datetime.now(), 'status': 'online'}

        elif 'camera/detection' in topic:
            mqtt_data['camera'].update({
                'person_detected': payload.get('person_detected', False),
                'count': payload.get('count', 0),
                'confidence': payload.get('confidence', 0)
            })
            socketio.emit('mqtt_update', {'type': 'camera', 'data': mqtt_data['camera']})
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
            print(f"[ML] Optimization Update: GA={mqtt_data['system']['ga_fitness']:.2f} (AC: {mqtt_data['system']['ga_temp']}°C Fan:{mqtt_data['system']['ga_fan']}), PSO={mqtt_data['system']['pso_fitness']:.2f} (Lamp: {mqtt_data['system']['pso_brightness']}%)")
            socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
            # Write optimization history to InfluxDB
            write_to_influxdb('optimization_result', {
                'ga_fitness': float(mqtt_data['system']['ga_fitness']),
                'pso_fitness': float(mqtt_data['system']['pso_fitness']),
                'ga_temp': float(mqtt_data['system']['ga_temp']),
                'ga_fan': float(mqtt_data['system']['ga_fan']),
                'pso_brightness': float(mqtt_data['system']['pso_brightness']),
                'combined_fitness': float(mqtt_data['system']['ga_fitness'] + mqtt_data['system']['pso_fitness']) / 2
            })
            # AUTO-APPLY: If AC is in ADAPTIVE mode, send optimized settings to ESP32
            # Skip if camera auto-OFF is active (no person detected)
            if mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE' and _person_present_recently():
                opt_temp = mqtt_data['system'].get('ga_temp', 0)
                opt_fan = mqtt_data['system'].get('ga_fan', 0)
                if opt_temp >= 16 and opt_temp <= 30 and opt_fan >= 1:
                    global _last_adaptive_ac_apply
                    now = time.time()
                    if now - _last_adaptive_ac_apply >= 5:
                        _last_adaptive_ac_apply = now
                        ac_cmd = {'command': 'SET', 'temperature': int(opt_temp), 'fan_speed': int(opt_fan), 'mode': 'COOL', 'source': 'adaptive'}
                        client.publish('smartroom/ac/control', json.dumps(ac_cmd))
                        print(f"[ADAPTIVE] -> Applied to AC: {opt_temp}°C Fan:{opt_fan}")
                    else:
                        print(f"[ADAPTIVE] -> AC apply debounced ({5 - (now - _last_adaptive_ac_apply):.1f}s remaining)")
            elif mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE':
                print(f"[ADAPTIVE] -> BLOCKED: No person confirmed in last {NO_PERSON_TIMEOUT_SECONDS//60} min")
            # AUTO-APPLY: If Lamp is in ADAPTIVE mode
            if mqtt_data['lamp'].get('mode', 'MANUAL') == 'ADAPTIVE':
                opt_brightness = mqtt_data['system'].get('pso_brightness', 0)
                if opt_brightness > 0:
                    global _last_adaptive_lamp_apply
                    now = time.time()
                    if now - _last_adaptive_lamp_apply >= 5:
                        _last_adaptive_lamp_apply = now
                        client.publish('smartroom/lamp/control', json.dumps({'brightness1': int(opt_brightness), 'brightness2': int(opt_brightness), 'brightness3': int(opt_brightness), 'source': 'adaptive'}))
                        print(f"[ADAPTIVE] -> Applied to All Lamps: {opt_brightness}%")
                    else:
                        print(f"[ADAPTIVE] -> Lamp apply debounced")
        
        elif 'ml/status' in topic:
            # ML optimization status from main.py (running/completed/error/busy)
            print(f"[ML] ML Status: {payload.get('status', 'unknown')} ({payload.get('algorithm', '')})")
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
                        'pso_brightness': pso_sol.get('brightness', payload.get('pso_brightness', mqtt_data['system'].get('pso_brightness', 0))),
                        'ga_history': payload.get('ga_history', mqtt_data['system'].get('ga_history', [])),
                        'pso_history': payload.get('pso_history', mqtt_data['system'].get('pso_history', []))
                    })
                    print(f"[ML] ML Completed -> Updated: GA={mqtt_data['system']['ga_fitness']:.2f}, PSO={mqtt_data['system']['pso_fitness']:.2f}")
                    socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
        
        elif 'ac/mode' in topic:
            mqtt_data['ac']['mode'] = payload.get('mode', 'ADAPTIVE')
            socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
            
        elif 'lamp/mode' in topic:
            mqtt_data['lamp']['mode'] = payload.get('mode', 'ADAPTIVE')
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
        
        elif 'ir/learned' in topic or 'IR/learned' in topic.lower():
            print("\n" + "="*70)
            print("[IR] IR SIGNAL RECEIVED FROM ESP32!")
            print("="*70)
            print(f"Topic received: {topic}")
            print(f"ir_learning_mode: {ir_learning_mode}")
            print(f"ir_learning_button: {ir_learning_button}")
            print(f"ir_learning_device: {ir_learning_device}")
            
            # Determine if this is a forwarded signal (ESP32 not in learn mode)
            esp32_status = payload.get('status', '') if isinstance(payload, dict) else ''
            esp32_learning = payload.get('learning_mode', True) if isinstance(payload, dict) else True
            print(f"ESP32 status: {esp32_status}")
            print(f"ESP32 learning_mode: {esp32_learning}")
            print("="*70)
            
            # Handle different payload formats
            ir_code = ''
            device = ir_learning_device or 'remote'
            
            # CRITICAL: Always use Flask's ir_learning_button (not ESP32's)
            # because ESP32 might send "auto_captured" when not in its own learn mode
            button_name = ir_learning_button  # Flask always knows the correct button
            
            # Format 1: {"button": "...", "code": "..."}
            if isinstance(payload, dict):
                # Only use ESP32's button name if Flask doesn't have one set
                if not button_name:
                    button_name = payload.get('button', '')
                ir_code = payload.get('code', payload.get('ir_code', payload.get('raw', '')))
                if not ir_learning_device:
                    device = payload.get('device', device)
            # Format 2: Plain text IR code
            elif isinstance(payload, str):
                ir_code = payload
            # Format 3: Raw data
            else:
                ir_code = str(payload)
            
            # Count actual RAW values to verify completeness
            raw_value_count = 0
            if ir_code.startswith('RAW:'):
                raw_part = ir_code[4:]  # Skip 'RAW:'
                raw_value_count = raw_part.count(',') + 1 if raw_part else 0
            print(f"   Parsed - Button: {button_name}, Device: {device}")
            print(f"   Code length: {len(ir_code)} chars")
            if raw_value_count > 0:
                print(f"   [OK] RAW values count: {raw_value_count} (need 200+ for Mitsubishi AC)")
                if raw_value_count < 100:
                    print(f"   [WARN] Only {raw_value_count} values! Signal mungkin tidak lengkap!")
                    print(f"   [WARN] Mitsubishi SRK AC butuh ~200+ values (2 frame)")
            
            # CHECK: Is Flask in learning mode?
            if not ir_learning_mode:
                print(f"[WARN] Flask NOT in learning mode — ignoring signal")
                print(f"   (Signal has {len(ir_code)} chars, {raw_value_count} RAW values)")
                print(f"   To capture: click Learn button on dashboard first")
                print("="*70)
            elif button_name and ir_code and len(ir_code) > 0:
                print(f"[OK] Flask IS in learning mode — SAVING signal!")
                
                # Check if this is a power toggle (same code for ON/OFF)
                is_power_toggle = False
                if 'power' in button_name.lower():
                    # Check if we already have a power code for this device
                    existing_power_codes = {
                        k: v for k, v in mqtt_data['ir_codes'].items() 
                        if 'power' in k.lower() and device in k
                    }
                    
                    if existing_power_codes and ir_code in existing_power_codes.values():
                        # Same code detected - this is a toggle button
                        is_power_toggle = True
                        button_name = f"{device}_power_toggle"
                        mqtt_data['ir_states'][button_name] = 'OFF'  # Initialize state
                        print(f"[WARN] Power toggle detected for {device}: Same code for ON/OFF")
                
                # Save to memory
                mqtt_data['ir_codes'][button_name] = ir_code
                print(f"[OK] IR code saved to memory: {button_name}")
                
                # Verify code integrity
                if ir_code.startswith('RAW:'):
                    raw_cnt = ir_code[4:].count(',') + 1
                    print(f"   [OK] Verified RAW signal: {raw_cnt} values stored for {button_name}")
                else:
                    print(f"   [OK] Non-RAW code stored: {len(ir_code)} chars")
                
                # Auto-save to InfluxDB immediately
                save_ir_command(device, button_name, len(ir_code))
                
                # Save to file for persistence
                try:
                    import os
                    ir_file = os.path.join(os.path.dirname(__file__), 'ir_codes.json')
                    with open(ir_file, 'w') as f:
                        json.dump(mqtt_data['ir_codes'], f, indent=2)
                    print(f"[SAVE] IR codes saved to file: {ir_file}")
                    
                    # Verify file write
                    with open(ir_file, 'r') as f:
                        verify = json.load(f)
                    if button_name in verify:
                        saved_code = verify[button_name]
                        if saved_code == ir_code:
                            print(f"   [OK] File verified: {button_name} saved correctly ({len(saved_code)} chars)")
                        else:
                            print(f"   [WARN] File mismatch! Memory={len(ir_code)} vs File={len(saved_code)}")
                    else:
                        print(f"   [WARN] Button {button_name} NOT found in saved file!")
                except Exception as e:
                    print(f"[ERROR] Error saving IR codes to file: {e}")
                
                log_messages.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'msg': f'IR Code learned: {button_name}{" (TOGGLE)" if is_power_toggle else ""}',
                    'level': 'success'
                })
                
                # Emit to frontend
                print("\n[WS] Emitting 'ir_learned' event to frontend via WebSocket...")
                print(f"   Event data: button={button_name}, device={device}, is_toggle={is_power_toggle}")
                
                socketio.emit('ir_learned', {
                    'button': button_name, 
                    'code': ir_code[:50] + '...' if len(ir_code) > 50 else ir_code,  # Truncate for display
                    'device': device,
                    'is_toggle': is_power_toggle,
                    'status': 'success'
                })
                
                print("[OK] WebSocket event emitted successfully!")
                print("   Frontend should update NOW!")
                print("="*70)
                
                print(f"[OK] IR learning completed for: {button_name}")
                
                ir_learning_mode = False
                ir_learning_button = ""
                ir_learning_device = ""
            else:
                print(f"[ERROR] IR learning failed: Missing button name or code")
                socketio.emit('ir_learned', {
                    'status': 'error',
                    'message': 'Invalid IR data received'
                })
        
        else:
            # No handler matched this topic
            print(f"[WARN] UNHANDLED TOPIC: {topic}")
            print(f"   No matching handler found for this topic!")
            print(f"   Available handlers: ac/sensors, lamp/sensors, camera/detection, ir/learned, etc.")
            
        print("─"*70 + "\n")
        
    except Exception as e:
        print(f"[ERROR] MQTT Message Handler Error: {str(e)}")
        import traceback
        traceback.print_exc()
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'MQTT Error: {str(e)}', 'level': 'error'})

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

print(f"[INIT] MQTT Client connecting to {MQTT_BROKER}:{MQTT_PORT}...")

def start_mqtt():
    """Start MQTT connection in background thread with retry"""
    global mqtt_status
    mqtt_status['broker'] = f'{MQTT_BROKER}:{MQTT_PORT}'
    retries = 0
    while retries < 5:
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
            print(f"[OK] MQTT loop started (attempt {retries + 1})")
            time.sleep(2)
            if mqtt_client.is_connected():
                print("[OK] MQTT Connected!")
            else:
                print("[WARN] MQTT loop started, waiting for connection...")
            return
        except Exception as e:
            retries += 1
            mqtt_status['error'] = str(e)
            print(f"[WARN] MQTT connect attempt {retries}/5 failed: {e}")
            if retries < 5:
                time.sleep(3)
    print("[ERROR] MQTT: All connection attempts failed. Dashboard will run without MQTT.")
    print("[ERROR] Make sure mosquitto/MQTT broker is running: sudo systemctl start mosquitto")

# Start MQTT in a background thread so it doesn't block app startup
mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()

# ==================== INFLUXDB ====================
def get_influx_data(measurement, field, hours=1):
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

# ==================== AUTHENTICATION ====================
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
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            session['logged_in'] = True
            return redirect('/')
        return render_template_string(LOGIN_TEMPLATE, error='Invalid credentials')
    return render_template_string(LOGIN_TEMPLATE, error=None)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect('/login')

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
    global api_data_debug_counter
    api_data_debug_counter += 1
    if api_data_debug_counter % 5 == 0:
        try:
            print(f"[API DATA] ac.temp={mqtt_data.get('ac', {}).get('temperature', 0)} hum={mqtt_data.get('ac', {}).get('humidity', 0)} energy.power={mqtt_data.get('energy', {}).get('power', 0)}")
        except Exception:
            pass

    response = jsonify(mqtt_data)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

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
        return jsonify({'status': 'error', 'message': 'Broker IP tidak boleh kosong'}), 400
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
            print(f"[CONFIG] MQTT reconnected to {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            mqtt_status['error'] = str(e)
            print(f"[ERROR] MQTT connect to {MQTT_BROKER}:{MQTT_PORT} failed: {e}")
    threading.Thread(target=reconnect_thread, daemon=True).start()
    print(f"[CONFIG] MQTT broker changed: {old_broker} -> {MQTT_BROKER}:{MQTT_PORT}")
    return jsonify({'status': 'ok', 'message': f'Mencoba connect ke {MQTT_BROKER}:{MQTT_PORT}', 'broker': MQTT_BROKER, 'port': MQTT_PORT})

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
    print(f"[SIMULATE] Dummy data injected into mqtt_data successfully")
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
        print(f"[SELFTEST] Published test AC message to smartroom/ac/sensors")
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

@app.route('/api/chart/<measurement>/<field>/<int:hours>')
def get_chart_data(measurement, field, hours):
    data = get_influx_data(measurement, field, hours)
    return jsonify(data)

@app.route('/api/energy/phase', methods=['GET', 'POST'])
def energy_phase_api():
    """Get or set energy monitoring phase (before/after adaptive AC)"""
    global energy_phase
    if request.method == 'POST':
        new_phase = request.json.get('phase', '').lower()
        if new_phase not in ('before', 'after'):
            return jsonify({'error': 'Phase must be before or after'}), 400
        energy_phase = new_phase
        print(f"[ENERGY] Energy phase changed to: {energy_phase}")
        socketio.emit('energy_phase', {'phase': energy_phase})
        return jsonify({'phase': energy_phase, 'message': f'Phase set to {energy_phase}'})
    return jsonify({'phase': energy_phase})

@app.route('/api/energy/compare')
def energy_compare():
    """Compare energy data between before and after adaptive AC phases"""
    period = request.args.get('period', '7d')
    field = request.args.get('field', 'power')
    
    period_map = {
        '30m': {'range': '-30m', 'window': '30s'},
        '24h': {'range': '-24h', 'window': '15m'},
        '7d':  {'range': '-7d',  'window': '1h'},
        '30d': {'range': '-30d', 'window': '6h'}
    }
    
    if period not in period_map:
        return jsonify({'error': 'Invalid period. Use: 30m, 24h, 7d, 30d'}), 400
    
    allowed_fields = ['voltage', 'current', 'power', 'energy_kwh', 'frequency', 'power_factor']
    if field not in allowed_fields:
        return jsonify({'error': f'Invalid field. Use: {", ".join(allowed_fields)}'}), 400
    
    p = period_map[period]
    time_format = '%H:%M:%S' if period == '30m' else ('%H:%M' if period == '24h' else '%m/%d %H:%M')
    
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        results = {}
        for phase in ['before', 'after']:
            query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: {p['range']})
              |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> filter(fn: (r) => r["phase"] == "{phase}")
              |> aggregateWindow(every: {p['window']}, fn: mean, createEmpty: false)
              |> yield(name: "mean")
            '''
            result = query_api.query(query=query)
            data_points = []
            for table in result:
                for record in table.records:
                    data_points.append({
                        'time': record.get_time().strftime(time_format),
                        'value': round(float(record.get_value()), 2)
                    })
            results[phase] = data_points

        # Fallback to runtime buffer if Influx doesn't have comparison data yet.
        if not results['before'] and not results['after']:
            lookback = {
                '30m': timedelta(minutes=30),
                '24h': timedelta(hours=24),
                '7d': timedelta(days=7),
                '30d': timedelta(days=30),
            }[period]
            cutoff = datetime.now() - lookback
            runtime = [r for r in energy_runtime_history if r.get('ts') and r['ts'] >= cutoff]
            if runtime:
                step = max(1, len(runtime) // 120)
                sampled = runtime[::step]
                for phase in ['before', 'after']:
                    results[phase] = [
                        {
                            'time': r['ts'].strftime(time_format),
                            'value': round(float(r.get(field, 0)), 2)
                        }
                        for r in sampled if r.get('phase') == phase
                    ]
        
        # Calculate averages for summary
        before_vals = [d['value'] for d in results['before']] if results['before'] else []
        after_vals = [d['value'] for d in results['after']] if results['after'] else []
        avg_before = round(sum(before_vals) / len(before_vals), 2) if before_vals else 0
        avg_after = round(sum(after_vals) / len(after_vals), 2) if after_vals else 0
        savings_pct = round((1 - avg_after / avg_before) * 100, 1) if avg_before > 0 else 0
        
        client.close()
        return jsonify({
            'period': period,
            'field': field,
            'before': results['before'],
            'after': results['after'],
            'summary': {
                'avg_before': avg_before,
                'avg_after': avg_after,
                'savings_percent': savings_pct
            }
        })
        
    except Exception as e:
        print(f"[ERROR] Energy compare error: {e}")
        return jsonify({'error': str(e), 'before': [], 'after': [], 'summary': {}}), 500

@app.route('/api/energy/history')
def energy_history():
    """Get PZEM energy history from InfluxDB — supports 1h to 30 days"""
    period = request.args.get('period', '24h')  # 1h, 6h, 24h, 7d, 30d
    field = request.args.get('field')  # voltage, current, power, energy_kwh, frequency, power_factor
    
    # Map period to InfluxDB range and aggregation window
    period_map = {
        '1h':  {'range': '-1h',  'window': '1m'},
        '6h':  {'range': '-6h',  'window': '5m'},
        '24h': {'range': '-24h', 'window': '15m'},
        '7d':  {'range': '-7d',  'window': '1h'},
        '30d': {'range': '-30d', 'window': '6h'}
    }
    
    if period not in period_map:
        return jsonify({'error': 'Invalid period. Use: 1h, 6h, 24h, 7d, 30d'}), 400
    
    allowed_fields = ['voltage', 'current', 'power', 'energy_kwh', 'frequency', 'power_factor']
    if field and field not in allowed_fields:
        return jsonify({'error': f'Invalid field. Use: {", ".join(allowed_fields)}'}), 400
    
    p = period_map[period]
    
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        # Use different time format based on period
        time_format = '%H:%M' if period in ('1h', '6h', '24h') else '%m/%d %H:%M'
        
        def query_field_points(field_name):
            query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: {p['range']})
              |> filter(fn: (r) => r["_measurement"] == "energy_monitor")
              |> filter(fn: (r) => r["_field"] == "{field_name}")
              |> aggregateWindow(every: {p['window']}, fn: mean, createEmpty: false)
              |> yield(name: "mean")
            '''
            result = query_api.query(query=query)
            points = []
            for table in result:
                for record in table.records:
                    points.append({
                        'time': record.get_time().strftime(time_format),
                        'value': round(float(record.get_value()), 2)
                    })
            return points

        # Support both modes:
        # 1) /api/energy/history?field=power&period=1h -> {data:[...]}
        # 2) /api/energy/history?period=24h -> {power:[...], voltage:[...], energy_kwh:[...]}
        if field:
            data_points = query_field_points(field)
            if not data_points:
                now = datetime.now()
                lookback = {
                    '1h': timedelta(hours=1),
                    '6h': timedelta(hours=6),
                    '24h': timedelta(hours=24),
                    '7d': timedelta(days=7),
                    '30d': timedelta(days=30),
                }[period]
                cutoff = now - lookback
                runtime = [r for r in energy_runtime_history if r.get('ts') and r['ts'] >= cutoff]
                if runtime:
                    step = max(1, len(runtime) // 120)
                    data_points = [
                        {'time': r['ts'].strftime(time_format), 'value': round(float(r.get(field, 0)), 2)}
                        for r in runtime[::step]
                    ]

            client.close()
            return jsonify({'period': period, 'field': field, 'data': data_points})

        power_points = query_field_points('power')
        voltage_points = query_field_points('voltage')
        kwh_points = query_field_points('energy_kwh')

        if not power_points and not voltage_points and not kwh_points:
            now = datetime.now()
            lookback = {
                '1h': timedelta(hours=1),
                '6h': timedelta(hours=6),
                '24h': timedelta(hours=24),
                '7d': timedelta(days=7),
                '30d': timedelta(days=30),
            }[period]
            cutoff = now - lookback
            runtime = [r for r in energy_runtime_history if r.get('ts') and r['ts'] >= cutoff]
            if runtime:
                step = max(1, len(runtime) // 120)
                sampled = runtime[::step]
                power_points = [{'time': r['ts'].strftime(time_format), 'value': round(float(r.get('power', 0)), 2)} for r in sampled]
                voltage_points = [{'time': r['ts'].strftime(time_format), 'value': round(float(r.get('voltage', 0)), 2)} for r in sampled]
                kwh_points = [{'time': r['ts'].strftime(time_format), 'value': round(float(r.get('energy_kwh', 0)), 3)} for r in sampled]

        client.close()
        return jsonify({
            'period': period,
            'power': power_points,
            'voltage': voltage_points,
            'energy_kwh': kwh_points
        })
        
    except Exception as e:
        print(f"[ERROR] Energy history query error: {e}")
        # Keep response shape compatible with frontend callers.
        if field:
            return jsonify({'error': str(e), 'period': period, 'field': field, 'data': []}), 500
        return jsonify({'error': str(e), 'period': period, 'power': [], 'voltage': [], 'energy_kwh': []}), 500

@app.route('/api/ml/status')
def ml_status():
    """Return current ML optimization state for the ML page"""
    return jsonify({
        'ga_fitness': mqtt_data['system'].get('ga_fitness', 0),
        'pso_fitness': mqtt_data['system'].get('pso_fitness', 0),
        'ga_temp': mqtt_data['system'].get('ga_temp', 0),
        'ga_fan': mqtt_data['system'].get('ga_fan', 0),
        'pso_brightness': mqtt_data['system'].get('pso_brightness', 0),
        'optimization_runs': mqtt_data['system'].get('optimization_runs', 0),
        'ga_history': mqtt_data['system'].get('ga_history', []),
        'pso_history': mqtt_data['system'].get('pso_history', [])
    })

@app.route('/api/ml/run', methods=['POST'])
def ml_run():
    """Trigger optimization run (GA, PSO, or both) via MQTT"""
    try:
        data = request.json
        algo = data.get('algorithm', 'both')  # 'ga', 'pso', or 'both'
        params = data.get('params', {})
        
        cmd = {
            'action': 'run_optimization',
            'algorithm': algo,
            'params': params
        }
        mqtt_client.publish('smartroom/ml/command', json.dumps(cmd))
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'ML Run triggered: {algo}', 'level': 'info'})
        return jsonify({'status': 'success', 'message': f'{algo} optimization triggered'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ac/control', methods=['POST'])
def control_ac():
    try:
        data = request.json
        command = data.get('command', data.get('action', ''))

        # Send command to ESP32 — ESP32 now uses IRMitsubishiAC library
        # to construct and send proper Mitsubishi protocol frames directly.
        # No need to send RAW IR codes from Flask anymore!
        mqtt_client.publish('smartroom/ac/control', json.dumps(data))
        print(f"[AC Control] Sent command '{command}' to ESP32 (Mitsubishi AC library handles IR)")

        # Track AC operating mode locally for instant dashboard update
        mode_map = {'MODE_COOL': 'COOL', 'MODE_HEAT': 'HEAT', 'MODE_DRY': 'DRY', 'MODE_FAN': 'FAN', 'MODE_AUTO': 'AUTO'}
        if command in mode_map:
            mqtt_data['ac']['ac_fan_mode'] = mode_map[command]

        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'AC Control: {command}', 'level': 'info'})
        return jsonify({'status': 'success', 'message': f'AC command sent: {command}'})
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
            'ga_history': data.get('ga_history', mqtt_data['system'].get('ga_history', [])),
            'pso_history': data.get('pso_history', mqtt_data['system'].get('pso_history', []))
        })
        
        # Broadcast to all connected clients
        socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
        
        print(f"[ML] Optimization Update: GA={data.get('ga_fitness', 0):.2f} (AC:{mqtt_data['system']['ga_temp']}°C), PSO={data.get('pso_fitness', 0):.2f} (Lamp:{mqtt_data['system']['pso_brightness']}%)")
        
        # AUTO-APPLY: If AC is in ADAPTIVE mode, send optimized settings to ESP32
        # Skip if camera auto-OFF is active (no person detected)
        if mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE' and _person_present_recently():
            opt_temp = mqtt_data['system'].get('ga_temp', 0)
            opt_fan = mqtt_data['system'].get('ga_fan', 0)
            if opt_temp >= 16 and opt_temp <= 30 and opt_fan >= 1:
                global _last_adaptive_ac_apply
                now = time.time()
                if now - _last_adaptive_ac_apply >= 5:
                    _last_adaptive_ac_apply = now
                    ac_cmd = {'command': 'SET', 'temperature': int(opt_temp), 'fan_speed': int(opt_fan), 'mode': 'COOL', 'source': 'adaptive'}
                    mqtt_client.publish('smartroom/ac/control', json.dumps(ac_cmd))
                    print(f"[ADAPTIVE] (HTTP) -> Applied to AC: {opt_temp}°C Fan:{opt_fan}")
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Adaptive AC: {opt_temp}°C Fan:{opt_fan}', 'level': 'success'})
                else:
                    print(f"[ADAPTIVE] (HTTP) -> AC apply debounced (MQTT already handled it)")
        elif mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE':
            print(f"[ADAPTIVE] (HTTP) -> BLOCKED: No person confirmed in last {NO_PERSON_TIMEOUT_SECONDS//60} min")
        
        # AUTO-APPLY: If Lamp is in ADAPTIVE mode, send optimized brightness
        if mqtt_data['lamp'].get('mode', 'MANUAL') == 'ADAPTIVE':
            opt_brightness = mqtt_data['system'].get('pso_brightness', 0)
            if opt_brightness > 0:
                global _last_adaptive_lamp_apply
                now = time.time()
                if now - _last_adaptive_lamp_apply >= 5:
                    _last_adaptive_lamp_apply = now
                    lamp_cmd = {'brightness': int(opt_brightness), 'source': 'adaptive'}
                    mqtt_client.publish('smartroom/lamp/control', json.dumps(lamp_cmd))
                    print(f"[ADAPTIVE] (HTTP) -> Applied to Lamp: {opt_brightness}%")
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Adaptive Lamp: {opt_brightness}%', 'level': 'success'})
                else:
                    print(f"[ADAPTIVE] (HTTP) -> Lamp apply debounced")
        
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
        print("\n" + "="*60)
        print("[IR] FLASK: PUBLISHING IR LEARN COMMAND")
        print("="*60)
        print(f"Topic    : smartroom/ir/learn")
        print(f"Button   : {button_name}")
        print(f"Device   : {device_name}")
        print(f"Payload  : {mqtt_payload_str}")
        print(f"Length   : {len(mqtt_payload_str)} bytes")
        print(f"MQTT Connected: {mqtt_client.is_connected()}")
        print("="*60)
        
        # PUBLISH WITH RESULT CHECK
        result = mqtt_client.publish('smartroom/ir/learn', mqtt_payload_str)
        
        print(f"Publish Result: {result.rc}")
        if result.rc == 0:
            print("[OK] MQTT PUBLISH SUCCESS!")
        else:
            print(f"[ERROR] MQTT PUBLISH FAILED! RC={result.rc}")
        print("="*60 + "\n")
        
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
        print(f"[ERROR] EXCEPTION in learn_ir(): {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ir/send', methods=['POST'])
def send_ir():
    try:
        data = request.json
        button_name = data.get('button', '')
        
        print("\n" + "="*70)
        print("[IR] IR SEND REQUEST RECEIVED")
        print("="*70)
        print(f"Button requested: {button_name}")
        print(f"Available codes: {list(mqtt_data['ir_codes'].keys())}")
        
        if button_name not in mqtt_data['ir_codes']:
            print(f"[ERROR] Button '{button_name}' not found in learned codes!")
            print("="*70 + "\n")
            return jsonify({'status': 'error', 'message': 'IR code not learned yet'}), 400
        
        ir_code = mqtt_data['ir_codes'][button_name]
        
        # Verify code completeness before sending
        raw_value_count = 0
        if isinstance(ir_code, str) and ir_code.startswith('RAW:'):
            raw_part = ir_code[4:]
            raw_value_count = raw_part.count(',') + 1 if raw_part else 0
        
        print(f"[OK] IR code found!")
        print(f"Code length: {len(ir_code)} chars")
        if raw_value_count > 0:
            print(f"RAW values: {raw_value_count} values")
            if raw_value_count < 100:
                print(f"[WARN] Only {raw_value_count} RAW values - signal mungkin tidak lengkap!")
        print(f"Code preview: {ir_code[:100]}..." if len(ir_code) > 100 else f"Code: {ir_code}")
        
        # Handle toggle buttons (power ON/OFF with same code)
        action_suffix = ''
        if 'toggle' in button_name.lower() or button_name in mqtt_data['ir_states']:
            # Toggle the state
            current_state = mqtt_data['ir_states'].get(button_name, 'OFF')
            new_state = 'ON' if current_state == 'OFF' else 'OFF'
            mqtt_data['ir_states'][button_name] = new_state
            action_suffix = f' ({new_state})'
            print(f"[IR] Toggle button: {current_state} -> {new_state}")
        
        # MQTT Payload - send COMPLETE code, no truncation!
        mqtt_payload = {
            'button': button_name, 
            'code': ir_code  # FULL code, no slicing!
        }
        mqtt_payload_str = json.dumps(mqtt_payload)
        
        # Verify the JSON payload still contains the full code
        verify_payload = json.loads(mqtt_payload_str)
        verify_code = verify_payload.get('code', '')
        verify_raw_count = 0
        if verify_code.startswith('RAW:'):
            verify_raw_count = verify_code[4:].count(',') + 1
        
        print(f"\n[PUB] Publishing to MQTT...")
        print(f"Topic: smartroom/ir/send")
        print(f"Payload size: {len(mqtt_payload_str)} bytes")
        if verify_raw_count > 0:
            print(f"RAW values in payload: {verify_raw_count} (verified after JSON encode)")
        print(f"MQTT Connected: {mqtt_client.is_connected()}")
        
        result = mqtt_client.publish('smartroom/ir/send', mqtt_payload_str)
        
        print(f"Publish Result: {result.rc}")
        if result.rc == 0:
            print("[OK] MQTT PUBLISH SUCCESS!")
            print("ESP32 should transmit IR signal NOW!")
        else:
            print(f"[ERROR] MQTT PUBLISH FAILED! RC={result.rc}")
        print("="*70 + "\n")
        
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
        print(f"[ERROR] EXCEPTION in send_ir(): {e}")
        import traceback
        traceback.print_exc()
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
                import os
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
            import os
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
    return jsonify({k: {'status': v['status'], 'last_seen': v['last_seen'].strftime('%H:%M:%S') if v['last_seen'] else 'Never'} for k, v in device_last_seen.items()})

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
    now = datetime.now()
    temp = mqtt_data['ac'].get('temperature', 0)
    humidity = mqtt_data['ac'].get('humidity', 0)
    person = mqtt_data['camera'].get('person_detected', False)
    
    # High temperature alert
    rule = alert_rules['high_temp']
    if rule['enabled'] and temp > rule['threshold']:
        if not rule['triggered']:
            rule['triggered'] = True
            alert = {'type': 'high_temp', 'message': f'Temperature {temp:.1f}°C exceeds {rule["threshold"]}°C!', 'level': 'danger', 'time': now.strftime('%H:%M:%S')}
            active_alerts.append(alert)
            socketio.emit('alert', alert)
    else:
        rule['triggered'] = False
    
    # High humidity alert
    rule = alert_rules['high_humidity']
    if rule['enabled'] and humidity > rule['threshold']:
        if not rule['triggered']:
            rule['triggered'] = True
            alert = {'type': 'high_humidity', 'message': f'Humidity {humidity:.1f}% exceeds {rule["threshold"]}%!', 'level': 'warning', 'time': now.strftime('%H:%M:%S')}
            active_alerts.append(alert)
            socketio.emit('alert', alert)
    else:
        rule['triggered'] = False
    
    # No person timeout -> suggest turning off AC
    rule = alert_rules['no_person_timeout']
    if rule['enabled']:
        if person:
            rule['last_person_seen'] = now
        elif rule['last_person_seen'] is not None:
            elapsed = (now - rule['last_person_seen']).total_seconds() / 60
            if elapsed > rule['timeout_minutes'] and mqtt_data['ac'].get('ac_state') != 'OFF':
                alert = {'type': 'no_person', 'message': f'No person detected for {int(elapsed)} min. Consider turning off AC.', 'level': 'warning', 'time': now.strftime('%H:%M:%S')}
                active_alerts.append(alert)
                socketio.emit('alert', alert)
                rule['last_person_seen'] = now  # Reset to avoid spamming

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
        .login-logo i { font-size: 48px; color: #6366f1; }
        .login-logo h1 { color: #f1f5f9; font-size: 24px; margin-top: 10px; }
        .login-logo p { color: #94a3b8; font-size: 14px; margin-top: 5px; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; color: #94a3b8; font-size: 13px; margin-bottom: 6px; font-weight: 500; }
        .form-group input { width: 100%; padding: 12px 16px; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #f1f5f9; font-size: 14px; transition: border-color 0.3s; }
        .form-group input:focus { outline: none; border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,0.2); }
        .login-btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s; }
        .login-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(99,102,241,0.4); }
        .error-msg { background: rgba(239,68,68,0.15); border: 1px solid #ef4444; color: #ef4444; padding: 10px; border-radius: 8px; margin-bottom: 20px; font-size: 13px; text-align: center; }
    </style>
</head>
<body>
    <div class="login-box">
        <div class="login-logo">
            <i class="fas fa-brain"></i>
            <h1>Smart Room IoT</h1>
            <p>Login to access dashboard</p>
        </div>
        {% if error %}
        <div class="error-msg"><i class="fas fa-exclamation-circle"></i> {{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label><i class="fas fa-user"></i> Username</label>
                <input type="text" name="username" required autocomplete="username" placeholder="Enter username">
            </div>
            <div class="form-group">
                <label><i class="fas fa-lock"></i> Password</label>
                <input type="password" name="password" required autocomplete="current-password" placeholder="Enter password">
            </div>
            <button type="submit" class="login-btn"><i class="fas fa-sign-in-alt"></i> Login</button>
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
            --bg-dark: #f1f5f9;
            --bg-card: #ffffff;
            --bg-card-hover: #f8fafc;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border: #e2e8f0;
            --shadow: rgba(0, 0, 0, 0.08);
            --input-bg: #f8fafc;
        }

        [data-theme="dark"] {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #334155;
            --shadow: rgba(0, 0, 0, 0.3);
            --input-bg: #0f172a;
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
            padding: 22px;
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
            margin-bottom: 16px;
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
            gap: 14px;
            margin-bottom: 16px;
        }

        .dashboard-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            align-items: stretch;
        }

        .dashboard-grid .stat-card {
            min-height: 185px;
        }

        .feedback-grid {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 20px;
        }

        .occupancy-top {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .occupancy-kpi {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }

        .occupancy-kpi .kpi-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .occupancy-kpi .kpi-value {
            font-size: 36px;
            font-weight: 700;
            color: #06b6d4;
            line-height: 1;
        }

        .occupancy-mini-note {
            margin-top: 8px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .occupancy-chart-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
        }

        .occupancy-chart-card canvas {
            max-height: 160px;
        }

        .rating-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        .rating-btn {
            border: 1px solid var(--border);
            background: var(--bg-card);
            color: var(--text-primary);
            border-radius: 14px;
            min-width: 74px;
            min-height: 74px;
            padding: 10px 12px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 22px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .rating-btn.active {
            background: var(--primary);
            border-color: var(--primary);
            color: #fff;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.35);
        }

        .feedback-input {
            width: 100%;
            background: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-primary);
            padding: 10px 12px;
            margin-top: 10px;
        }

        .feedback-history-item {
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 10px;
            margin-bottom: 10px;
            background: var(--bg-card-hover);
        }

        .stat-card {
            background: var(--bg-card);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid var(--border);
            transition: all 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px var(--shadow);
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

        /* Energy page premium chart styling */
        #energy .power-card {
            border-radius: 16px;
            border: 1px solid rgba(99, 102, 241, 0.16);
            background: linear-gradient(150deg, rgba(99,102,241,0.08), rgba(15,23,42,0.03));
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
        }

        #energy .chart-container {
            border-radius: 16px;
            border: 1px solid rgba(99, 102, 241, 0.18);
            background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(241,245,249,0.9));
            box-shadow: 0 10px 34px rgba(15, 23, 42, 0.08);
            padding: 18px 20px 16px;
            margin-bottom: 22px;
        }

        #energy .chart-header {
            margin-bottom: 14px;
            gap: 10px;
            flex-wrap: wrap;
        }

        #energy .chart-title {
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.2px;
            color: var(--text-primary);
        }

        #energy .chart-options {
            gap: 8px;
            flex-wrap: wrap;
        }

        #energy .chart-option-btn {
            border-radius: 999px;
            padding: 6px 13px;
            font-size: 12px;
            font-weight: 700;
            border: 1px solid rgba(99, 102, 241, 0.22);
            background: rgba(99, 102, 241, 0.06);
            color: #475569;
        }

        #energy .chart-option-btn:hover {
            transform: translateY(-1px);
            background: rgba(99, 102, 241, 0.12);
            color: #1e293b;
        }

        #energy .chart-option-btn.active {
            background: linear-gradient(135deg, #4f46e5, #6366f1);
            color: #ffffff;
            border-color: transparent;
            box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35);
        }

        #energy canvas {
            height: 250px !important;
            max-height: 250px;
        }

        /* ML Optimization Page Styles */
        .ml-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .ml-table th, .ml-table td {
            padding: 10px 14px;
            text-align: center;
            border-bottom: 1px solid var(--border);
        }
        .ml-table th {
            background: rgba(99, 102, 241, 0.1);
            color: var(--primary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }
        .ml-table tr:hover {
            background: var(--bg-card-hover);
        }
        .ml-param-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        .ml-param-item label {
            display: block;
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        .ml-input {
            width: 100%;
            padding: 8px 12px;
            background: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 14px;
            transition: border-color 0.3s;
        }
        .ml-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        .ml-action-btn {
            padding: 8px 18px !important;
            font-weight: 600 !important;
            cursor: pointer;
            transition: all 0.3s !important;
        }
        .ml-action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .ml-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .ml-badge.good { background: rgba(16, 185, 129, 0.2); color: #10b981; }
        .ml-badge.mid { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
        .ml-badge.low { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

        @media (max-width: 768px) {
            .ml-param-grid { grid-template-columns: 1fr; }
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

        .btn-sm {
            padding: 6px 12px;
            font-size: 12px;
        }

        .ir-button-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .ir-button {
            padding: 15px;
            background: var(--input-bg);
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

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.2; }
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
            box-shadow: 0 10px 30px var(--shadow);
            display: none;
            z-index: 9999;
            animation: slideIn 0.3s;
        }

        @keyframes slideIn {
            from { transform: translateX(400px); }
            to { transform: translateX(0); }
        }

        .toast.show { display: block; }

        /* Energy data bubble notification */
        .energy-bubble {
            position: fixed;
            top: 80px;
            right: 30px;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.95), rgba(5, 150, 105, 0.95));
            color: #fff;
            padding: 10px 18px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            z-index: 9998;
            pointer-events: none;
            opacity: 0;
            transform: translateY(-10px) scale(0.9);
            transition: opacity 0.3s, transform 0.3s;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.4);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .energy-bubble.show {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
        .energy-bubble .bubble-dot {
            width: 8px;
            height: 8px;
            background: #fff;
            border-radius: 50%;
            animation: bubblePulse 1s infinite;
        }
        @keyframes bubblePulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
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

        /* ========== CAMERA STYLES ========== */
        .camera-view {
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
            text-align: center;
            margin-bottom: 20px;
        }

        .camera-feed-container {
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            max-width: 100%;
        }

        .camera-feed-container img {
            width: 100%;
            max-height: 800px;
            object-fit: contain;
            display: block;
            margin: 0 auto;
        }

        .camera-overlay-bar {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(180deg, rgba(0,0,0,0.7) 0%, transparent 100%);
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .camera-rec-badge {
            background: var(--danger);
            color: white;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            animation: pulse 1.5s infinite;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .camera-time-badge {
            color: white;
            font-size: 12px;
            background: rgba(0,0,0,0.5);
            padding: 3px 10px;
            border-radius: 4px;
            font-family: monospace;
        }

        .camera-error {
            padding: 80px 20px;
            color: var(--text-secondary);
        }

        .camera-error i {
            font-size: 64px;
            margin-bottom: 20px;
            color: var(--border);
        }

        .camera-info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .camera-info-card {
            background: var(--input-bg);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .camera-info-label {
            color: var(--text-secondary);
            font-size: 12px;
            margin-bottom: 5px;
        }

        .camera-info-value {
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
        }

        .detection-alert {
            position: fixed;
            top: 100px;
            right: 30px;
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            padding: 20px 25px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(239, 68, 68, 0.5);
            display: none;
            z-index: 8888;
            animation: slideInRight 0.5s, pulse 2s infinite;
            max-width: 300px;
        }

        .detection-alert.show {
            display: block;
        }

        @keyframes slideInRight {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .detection-alert-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 18px;
            font-weight: bold;
        }

        .detection-alert-icon {
            font-size: 24px;
            animation: bounce 1s infinite;
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }

        .detection-alert-body {
            font-size: 14px;
            opacity: 0.9;
        }

        .detection-close {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .person-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin: 2px;
            animation: fadeIn 0.3s;
        }

        .person-badge.detected {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        }

        .person-badge.not-detected {
            background: linear-gradient(135deg, #64748b, #475569);
            color: rgba(255,255,255,0.7);
        }

        @media (max-width: 768px) {
            .sidebar { 
                transform: translateX(-100%);
                transition: transform 0.3s ease;
                z-index: 2000;
            }
            .sidebar.open { transform: translateX(0); }
            .main-content { margin-left: 0; padding: 15px; padding-top: 60px; }
            .stats-grid { grid-template-columns: 1fr; }
            .dashboard-grid { grid-template-columns: 1fr; }
            .feedback-grid { grid-template-columns: 1fr; }
            .occupancy-top { grid-template-columns: 1fr; }
            .hamburger-btn { display: flex !important; }
        }

        @media (min-width: 769px) and (max-width: 1200px) {
            .dashboard-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .occupancy-top { grid-template-columns: 1fr; }
        }

        @media (min-width: 1201px) {
            .dashboard-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        .hamburger-btn {
            display: none;
            position: fixed;
            top: 12px;
            left: 12px;
            z-index: 1998;
            width: 44px;
            height: 44px;
            border-radius: 10px;
            background: var(--primary);
            color: white;
            border: none;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(99,102,241,0.4);
        }

        .sidebar-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1999;
            pointer-events: none;
        }

        .sidebar-overlay.active {
            display: block;
            pointer-events: auto;
        }

        /* Device Status Indicators */
        .device-status-item {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            background: var(--bg-card-hover);
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }
        .device-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .device-dot.online {
            background: #10b981;
            box-shadow: 0 0 6px #10b981;
            animation: pulse-green 2s infinite;
        }
        .device-dot.offline {
            background: #ef4444;
        }
        .device-time {
            font-size: 10px;
            color: var(--text-secondary);
            opacity: 0.7;
        }
        @keyframes pulse-green {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Alert Notification Banner */
        .alert-banner {
            position: fixed;
            top: 15px;
            right: 15px;
            z-index: 3000;
            max-width: 380px;
            padding: 14px 20px;
            border-radius: 12px;
            color: white;
            font-size: 13px;
            font-weight: 500;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            animation: slideInRight 0.4s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .alert-banner.danger { background: linear-gradient(135deg, #ef4444, #dc2626); }
        .alert-banner.warning { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .alert-banner .alert-close { background: none; border: none; color: white; cursor: pointer; font-size: 16px; margin-left: auto; opacity: 0.8; }
        .alert-banner .alert-close:hover { opacity: 1; }
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Theme Toggle */
        .theme-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            color: var(--text-secondary);
            border: 1px solid var(--border);
            background: transparent;
            width: 100%;
            font-size: 14px;
        }
        .theme-toggle:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
        }
        .theme-toggle i {
            font-size: 16px;
            width: 20px;
            text-align: center;
        }
        .theme-divider {
            border: none;
            border-top: 1px solid var(--border);
            margin: 15px 0;
        }

        /* Light theme adjustments */
        .slider {
            background: var(--border);
        }
        select.slider {
            background: var(--input-bg);
            color: var(--text-primary);
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
            <i class="fas fa-brain"></i>
            Smart Room
        </div>
        <div class="nav-item active" onclick="showPage('dashboard-ac')">
            <i class="fas fa-snowflake"></i>
            <span>AC Dashboard</span>
        </div>
        <div class="nav-item" onclick="showPage('dashboard-lamp')">
            <i class="fas fa-lightbulb"></i>
            <span>Lamp Dashboard</span>
        </div>
        <div class="nav-item" onclick="showPage('ac-analytics')">
            <i class="fas fa-chart-line"></i>
            <span>AC Analytics</span>
        </div>
        <div class="nav-item" onclick="showPage('lamp-analytics')">
            <i class="fas fa-chart-bar"></i>
            <span>Lamp Analytics</span>
        </div>
        <div class="nav-item" onclick="showPage('camera')">
            <i class="fas fa-video"></i>
            <span>Camera</span>
        </div>
        <div class="nav-item" onclick="showPage('energy')">
            <i class="fas fa-bolt"></i>
            <span>Energy Usage</span>
        </div>
        <div class="nav-item" onclick="showPage('control-ac')">
            <i class="fas fa-sliders-h"></i>
            <span>AC Control</span>
        </div>
        <div class="nav-item" onclick="showPage('control-lamp')">
            <i class="fas fa-adjust"></i>
            <span>Lamp Control</span>
        </div>
        <div class="nav-item" onclick="showPage('ml-optimization')">
            <i class="fas fa-brain"></i>
            <span>ML Optimization</span>
        </div>
        <div class="nav-item" onclick="showPage('logs')">
            <i class="fas fa-file-alt"></i>
            <span>System Logs</span>
        </div>
        <div class="nav-item" onclick="showPage('occupancy-feedback')">
            <i class="fas fa-clipboard-check"></i>
            <span>Occupancy Trend & Feedback</span>
        </div>
        <hr class="theme-divider">
        <button class="theme-toggle" onclick="toggleTheme()" id="theme-toggle-btn">
            <i class="fas fa-moon" id="theme-icon"></i>
            <span id="theme-label">Dark Mode</span>
        </button>
        <a href="/logout" class="theme-toggle" style="text-decoration: none; color: var(--danger); border-color: var(--danger);">
            <i class="fas fa-sign-out-alt"></i>
            <span>Logout</span>
        </a>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <!-- AC Dashboard Page -->
        <div id="dashboard-ac" class="page active">
            <div class="header">
                <h1><i class="fas fa-snowflake"></i> AC Dashboard</h1>
                <p>Air Conditioning monitoring & status <button onclick="document.getElementById('diag-panel').style.display='block'" style="margin-left: 10px; padding: 3px 10px; font-size: 11px; background: #f59e0b; border: none; color: white; border-radius: 6px; cursor: pointer;"><i class="fas fa-stethoscope"></i> Diagnostik</button></p>
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
            <div id="diag-panel" style="background: var(--card-bg); border: 2px solid #f59e0b; border-radius: 12px; padding: 16px; margin-bottom: 16px; display: none;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <i class="fas fa-stethoscope" style="color: #f59e0b; font-size: 18px;"></i>
                    <strong style="color: #f59e0b;">Diagnostic Mode</strong>
                    <button onclick="document.getElementById('diag-panel').style.display='none'" style="margin-left: auto; background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 18px;">&times;</button>
                </div>
                <div id="diag-result" style="font-family: monospace; font-size: 12px; background: var(--bg-secondary); padding: 10px; border-radius: 8px; margin-bottom: 12px; min-height: 60px; white-space: pre-wrap; color: var(--text-primary);">Klik tombol di bawah untuk diagnosa...</div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button onclick="runSimulate()" style="padding: 8px 16px; font-size: 13px; cursor: pointer; background: #10b981; border: none; color: white; border-radius: 8px; border-radius: 8px;">
                        <i class="fas fa-vial"></i> Test Frontend (Inject Data Dummy)
                    </button>
                    <button onclick="runMqttSelftest()" style="padding: 8px 16px; font-size: 13px; cursor: pointer; background: #3b82f6; border: none; color: white; border-radius: 8px;">
                        <i class="fas fa-satellite-dish"></i> Test MQTT Broker (Self-Test)
                    </button>
                    <button onclick="runMqttReconnect()" style="padding: 8px 16px; font-size: 13px; cursor: pointer; background: #8b5cf6; border: none; color: white; border-radius: 8px;">
                        <i class="fas fa-sync"></i> Reconnect MQTT
                    </button>
                    <button onclick="checkMqttStatus(true)" style="padding: 8px 16px; font-size: 13px; cursor: pointer; background: #6b7280; border: none; color: white; border-radius: 8px;">
                        <i class="fas fa-info-circle"></i> MQTT Status Detail
                    </button>
                </div>
            </div>

            <div class="stats-grid dashboard-grid">
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Room Temperature</span>
                        <div class="stat-icon" style="background: rgba(239, 68, 68, 0.2); color: #ef4444;">
                            <i class="fas fa-temperature-high"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-temp">0</span>°C</div>
                    <div class="stat-change up">
                        <i class="fas fa-robot"></i>
                        <span>Avg 3×DHT22 — Real-time</span>
                    </div>
                    <div style="display: flex; gap: 6px; margin-top: 8px; font-size: 11px; color: var(--text-secondary);">
                        <span style="flex:1; text-align:center; background: rgba(239,68,68,0.08); border-radius: 6px; padding: 3px 0;">S1: <strong id="dash-temp1">0</strong>°C</span>
                        <span style="flex:1; text-align:center; background: rgba(239,68,68,0.08); border-radius: 6px; padding: 3px 0;">S2: <strong id="dash-temp2">0</strong>°C</span>
                        <span style="flex:1; text-align:center; background: rgba(239,68,68,0.08); border-radius: 6px; padding: 3px 0;">S3: <strong id="dash-temp3">0</strong>°C</span>
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
                        <i class="fas fa-robot"></i>
                        <span>Avg 3×DHT22 — Real-time</span>
                    </div>
                    <div style="display: flex; gap: 6px; margin-top: 8px; font-size: 11px; color: var(--text-secondary);">
                        <span style="flex:1; text-align:center; background: rgba(59,130,246,0.08); border-radius: 6px; padding: 3px 0;">S1: <strong id="dash-hum1">0</strong>%</span>
                        <span style="flex:1; text-align:center; background: rgba(59,130,246,0.08); border-radius: 6px; padding: 3px 0;">S2: <strong id="dash-hum2">0</strong>%</span>
                        <span style="flex:1; text-align:center; background: rgba(59,130,246,0.08); border-radius: 6px; padding: 3px 0;">S3: <strong id="dash-hum3">0</strong>%</span>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Heat Index</span>
                        <div class="stat-icon" style="background: rgba(249, 115, 22, 0.2); color: #f97316;">
                            <i class="fas fa-sun"></i>
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
                        <div class="stat-icon" style="background: rgba(99, 102, 241, 0.2); color: #6366f1;">
                            <i class="fas fa-wifi"></i>
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
                    <div id="ac-panel-header" style="padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(79, 70, 229, 0.08)); border-bottom: 1px solid var(--border);">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(99, 102, 241, 0.2); display: flex; align-items: center; justify-content: center;">
                                <i class="fas fa-snowflake" style="font-size: 18px; color: #6366f1;"></i>
                            </div>
                            <div>
                                <div style="font-size: 15px; font-weight: 700; color: var(--text);">AC Status</div>
                                <div style="font-size: 11px; color: var(--text-secondary);">Mitsubishi Heavy Industries — Real-time</div>
                            </div>
                        </div>
                        <div id="ac-panel-power" style="display: flex; align-items: center; gap: 8px;">
                            <div id="ac-panel-dot" style="width: 12px; height: 12px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);"></div>
                            <span id="dash-ac-state" style="font-size: 16px; font-weight: 800; color: #ef4444;">OFF</span>
                        </div>
                    </div>
                    <!-- Body Grid -->
                    <div style="padding: 16px 20px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                        <!-- Set Temperature -->
                        <div style="text-align: center; padding: 14px 8px; background: rgba(59, 130, 246, 0.06); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-thermometer-half"></i> Set Temperature
                            </div>
                            <div style="font-size: 28px; font-weight: 800; color: #3b82f6;" id="dash-ac-temp">24</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">°C</div>
                        </div>
                        <!-- Fan Speed -->
                        <div style="text-align: center; padding: 14px 8px; background: rgba(139, 92, 246, 0.06); border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-fan"></i> Fan Speed
                            </div>
                            <div style="font-size: 28px; font-weight: 800; color: #8b5cf6;" id="dash-ac-fan">1</div>
                            <div style="font-size: 12px; color: var(--text-secondary);" id="dash-ac-fan-label">Low</div>
                        </div>
                        <!-- AC Mode (COOL/HEAT/DRY/FAN/AUTO) -->
                        <div style="text-align: center; padding: 14px 8px; background: rgba(14, 165, 233, 0.06); border-radius: 12px; border: 1px solid rgba(14, 165, 233, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-cog"></i> Mode AC
                            </div>
                            <div style="font-size: 22px; font-weight: 800; color: #0ea5e9;" id="dash-ac-mode-icon"><i class="fas fa-snowflake"></i></div>
                            <div style="font-size: 14px; font-weight: 700; color: #0ea5e9; margin-top: 2px;" id="dash-ac-mode">COOL</div>
                        </div>
                        <!-- Operating Mode (ADAPTIVE/MANUAL) -->
                        <div style="text-align: center; padding: 14px 8px; background: rgba(16, 185, 129, 0.06); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-sliders-h"></i> Kontrol
                            </div>
                            <div style="font-size: 22px; font-weight: 800; color: #10b981;" id="dash-ac-ctrl-icon"><i class="fas fa-robot"></i></div>
                            <div style="font-size: 14px; font-weight: 700; color: #10b981; margin-top: 2px;" id="dash-ac-ctrl-mode">ADAPTIVE</div>
                        </div>
                    </div>
                    <!-- Footer: Room Environment + extra info -->
                    <div style="padding: 10px 20px 14px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border); font-size: 12px; color: var(--text-secondary);">
                        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                            <span><i class="fas fa-temperature-high" style="color: #f97316;"></i> Avg: <strong id="dash-ac-room-temp" style="color: var(--text);">0</strong>°C</span>
                            <span style="font-size: 11px; color: var(--text-secondary);">S1: <strong id="dash-ac-temp1" style="color: var(--text);">0</strong>°C</span>
                            <span style="font-size: 11px; color: var(--text-secondary);">S2: <strong id="dash-ac-temp2" style="color: var(--text);">0</strong>°C</span>
                            <span style="font-size: 11px; color: var(--text-secondary);">S3: <strong id="dash-ac-temp3" style="color: var(--text);">0</strong>°C</span>
                            <span><i class="fas fa-tint" style="color: #3b82f6;"></i> Avg: <strong id="dash-ac-room-hum" style="color: var(--text);">0</strong>%</span>
                        </div>
                        <div id="dash-ac-source" style="padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.25);">
                            <i class="fas fa-robot"></i> AI Controlled
                        </div>
                    </div>
                </div>

                <!-- Energy Monitor Panel (PZEM-016) -->
                <div class="stat-card" id="energy-panel" style="grid-column: 1 / -1; background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 0; overflow: hidden;">
                    <div style="padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(217, 119, 6, 0.08)); border-bottom: 1px solid var(--border);">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(245, 158, 11, 0.2); display: flex; align-items: center; justify-content: center;">
                                <i class="fas fa-bolt" style="font-size: 18px; color: #f59e0b;"></i>
                            </div>
                            <div>
                                <div style="font-size: 15px; font-weight: 700; color: var(--text);">Energy Monitor</div>
                                <div style="font-size: 11px; color: var(--text-secondary);">PZEM-016 + CT — AC Energy Usage</div>
                            </div>
                        </div>
                        <div id="energy-status-badge" style="padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; background: rgba(107, 114, 128, 0.15); color: #6b7280; border: 1px solid rgba(107, 114, 128, 0.3);">
                            <i class="fas fa-circle" style="font-size: 6px; vertical-align: middle; margin-right: 4px;"></i> Offline
                        </div>
                    </div>
                    <div style="padding: 16px 20px;">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px;">
                            <div style="background: var(--bg-elevated); border-radius: 12px; padding: 14px; text-align: center;">
                                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;"><i class="fas fa-plug" style="color: #f59e0b;"></i> Voltage</div>
                                <div style="font-size: 22px; font-weight: 700; color: var(--text);"><span id="energy-voltage">0</span><span style="font-size: 12px; color: var(--text-secondary);"> V</span></div>
                            </div>
                            <div style="background: var(--bg-elevated); border-radius: 12px; padding: 14px; text-align: center;">
                                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;"><i class="fas fa-wave-square" style="color: #ef4444;"></i> Current</div>
                                <div style="font-size: 22px; font-weight: 700; color: var(--text);"><span id="energy-current">0</span><span style="font-size: 12px; color: var(--text-secondary);"> A</span></div>
                            </div>
                            <div style="background: var(--bg-elevated); border-radius: 12px; padding: 14px; text-align: center;">
                                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;"><i class="fas fa-fire" style="color: #f97316;"></i> Power</div>
                                <div style="font-size: 22px; font-weight: 700; color: #f59e0b;"><span id="energy-power">0</span><span style="font-size: 12px; color: var(--text-secondary);"> W</span></div>
                            </div>
                            <div style="background: var(--bg-elevated); border-radius: 12px; padding: 14px; text-align: center;">
                                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;"><i class="fas fa-battery-half" style="color: #10b981;"></i> Energy</div>
                                <div style="font-size: 22px; font-weight: 700; color: #10b981;"><span id="energy-kwh">0</span><span style="font-size: 12px; color: var(--text-secondary);"> kWh</span></div>
                            </div>
                            <div style="background: var(--bg-elevated); border-radius: 12px; padding: 14px; text-align: center;">
                                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;"><i class="fas fa-signal" style="color: #6366f1;"></i> Frequency</div>
                                <div style="font-size: 22px; font-weight: 700; color: var(--text);"><span id="energy-freq">0</span><span style="font-size: 12px; color: var(--text-secondary);"> Hz</span></div>
                            </div>
                            <div style="background: var(--bg-elevated); border-radius: 12px; padding: 14px; text-align: center;">
                                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;"><i class="fas fa-tachometer-alt" style="color: #8b5cf6;"></i> Power Factor</div>
                                <div style="font-size: 22px; font-weight: 700; color: var(--text);"><span id="energy-pf">0</span></div>
                            </div>
                        </div>
                        <!-- Estimasi biaya harian -->
                        <div style="margin-top: 12px; padding: 10px 14px; background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(217, 119, 6, 0.05)); border-radius: 10px; display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 12px; color: var(--text-secondary);"><i class="fas fa-coins" style="color: #f59e0b;"></i> Cost Estimate (Rp 1.444,70/kWh)</div>
                            <div style="font-size: 14px; font-weight: 700; color: #f59e0b;">Rp <span id="energy-cost">0</span></div>
                        </div>
                    </div>
                </div>

                <!-- Person Detection - Enhanced Card -->
                <div class="stat-card" style="grid-column: 1 / -1; padding: 0; overflow: hidden;">
                    <div style="padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(220, 38, 38, 0.05)); border-bottom: 1px solid var(--border);">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(239, 68, 68, 0.2); display: flex; align-items: center; justify-content: center;">
                                <i class="fas fa-user-friends" style="font-size: 18px; color: #ef4444;"></i>
                            </div>
                            <div>
                                <div style="font-size: 15px; font-weight: 700; color: var(--text);">Person Detection</div>
                                <div style="font-size: 11px; color: var(--text-secondary);">YOLOv8n — Camera Real-time</div>
                            </div>
                        </div>
                        <div id="cam-status-badge" style="padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; background: rgba(239, 68, 68, 0.12); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3);">
                            <i class="fas fa-circle" style="font-size: 7px; vertical-align: middle;"></i> No Person
                        </div>
                    </div>
                    <div style="padding: 16px 20px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                        <div style="text-align: center; padding: 14px 8px; background: rgba(239, 68, 68, 0.06); border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-users"></i> Person Count
                            </div>
                            <div style="font-size: 32px; font-weight: 800; color: #ef4444;" id="cam-count">0</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">person(s)</div>
                        </div>
                        <div style="text-align: center; padding: 14px 8px; background: rgba(245, 158, 11, 0.06); border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-crosshairs"></i> Confidence
                            </div>
                            <div style="font-size: 32px; font-weight: 800; color: #f59e0b;" id="cam-confidence">0%</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">detection accuracy</div>
                        </div>
                        <div style="text-align: center; padding: 14px 8px; background: rgba(16, 185, 129, 0.06); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-bolt"></i> Auto Control
                            </div>
                            <div style="font-size: 22px; font-weight: 800; color: #10b981;" id="cam-auto-status"><i class="fas fa-check-circle"></i></div>
                            <div style="font-size: 12px; color: var(--text-secondary);" id="cam-auto-label">Active (10 min timeout)</div>
                        </div>
                    </div>
                    <!-- Last Person Detection + Auto-OFF Timer -->
                    <div style="padding: 8px 20px 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div style="padding: 12px; background: rgba(99, 102, 241, 0.06); border-radius: 10px; border: 1px solid rgba(99, 102, 241, 0.15);">
                            <div style="font-size: 11px; color: #818cf8; font-weight: 600; margin-bottom: 4px;">
                                <i class="fas fa-clock"></i> Last Person Detection
                            </div>
                            <div style="font-size: 18px; font-weight: 800; color: #6366f1;" id="cam-last-seen">--</div>
                            <div style="font-size: 11px; color: var(--text-secondary);" id="cam-last-seen-label">Not detected yet</div>
                        </div>
                        <div style="padding: 12px; background: rgba(239, 68, 68, 0.06); border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.15);">
                            <div style="font-size: 11px; color: #f87171; font-weight: 600; margin-bottom: 4px;">
                                <i class="fas fa-power-off"></i> Auto-OFF AC
                            </div>
                            <div style="font-size: 18px; font-weight: 800; color: #ef4444;" id="cam-auto-off-timer">--</div>
                            <div style="font-size: 11px; color: var(--text-secondary);" id="cam-auto-off-label">Not active</div>
                        </div>
                    </div>
                </div>

                <!-- GA + PSO Optimization - Full Width -->
                <div class="stat-card" style="grid-column: 1 / -1; padding: 0; overflow: hidden;">
                    <div style="padding: 14px 20px; display: flex; align-items: center; gap: 10px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.05)); border-bottom: 1px solid var(--border);">
                        <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(16, 185, 129, 0.2); display: flex; align-items: center; justify-content: center;">
                            <i class="fas fa-brain" style="font-size: 18px; color: #10b981;"></i>
                        </div>
                        <div>
                            <div style="font-size: 15px; font-weight: 700; color: var(--text);">ML Optimization Status</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Genetic Algorithm -> AC | PSO -> Lamp</div>
                        </div>
                    </div>
                    <div style="padding: 16px 20px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
                        <!-- GA Section -->
                        <div style="padding: 16px; background: rgba(16, 185, 129, 0.04); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.15);">
                            <div style="font-size: 12px; font-weight: 700; color: #10b981; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
                                <i class="fas fa-dna"></i> GA -> AC Control
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                                <div>
                                    <div style="font-size: 10px; color: var(--text-secondary); text-transform: uppercase;">Fitness</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #10b981;" id="ga-fitness">0.00</div>
                                </div>
                                <div>
                                    <div style="font-size: 10px; color: var(--text-secondary); text-transform: uppercase;">Temp</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #3b82f6;" id="ga-temp">--</div>
                                    <div style="font-size: 10px; color: var(--text-secondary);">°C</div>
                                </div>
                                <div>
                                    <div style="font-size: 10px; color: var(--text-secondary); text-transform: uppercase;">Fan</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #8b5cf6;" id="ga-fan">--</div>
                                    <div style="font-size: 10px; color: var(--text-secondary);">speed</div>
                                </div>
                            </div>
                        </div>
                        <!-- PSO Section -->
                        <div style="padding: 16px; background: rgba(245, 158, 11, 0.04); border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
                            <div style="font-size: 12px; font-weight: 700; color: #f59e0b; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
                                <i class="fas fa-lightbulb"></i> PSO -> Lamp Control
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                                <div>
                                    <div style="font-size: 10px; color: var(--text-secondary); text-transform: uppercase;">Fitness</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #f59e0b;" id="pso-fitness">0.00</div>
                                </div>
                                <div>
                                    <div style="font-size: 10px; color: var(--text-secondary); text-transform: uppercase;">Brightness</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #eab308;" id="pso-brightness">--</div>
                                    <div style="font-size: 10px; color: var(--text-secondary);">%</div>
                                </div>
                                <div>
                                    <div style="font-size: 10px; color: var(--text-secondary); text-transform: uppercase;">Runs</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #a855f7;" id="dash-opt-runs">0</div>
                                    <div style="font-size: 10px; color: var(--text-secondary);">cycle</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Lamp Dashboard Page -->
        <div id="dashboard-lamp" class="page">
            <div class="header">
                <h1><i class="fas fa-lightbulb"></i> Lamp Dashboard</h1>
                <p>Lighting monitoring & status — 3 Lamps</p>
                <div style="display: flex; gap: 15px; margin-top: 12px; flex-wrap: wrap;">
                    <div class="device-status-item" id="ds-esp32-lamp">
                        <span class="device-dot offline"></span>
                        <span>ESP32-Lamp</span>
                        <span class="device-time" id="ds-lamp-time">Never</span>
                    </div>
                </div>
            </div>

            <div class="stats-grid dashboard-grid">
                <!-- Lamp 1 -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Lamp 1 — Lux</span>
                        <div class="stat-icon" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">
                            <i class="fas fa-sun"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux1">0</span> lx</div>
                    <div class="stat-change">
                        <span>Brightness: <span id="dash-bright1">0</span>%</span>
                    </div>
                </div>

                <!-- Lamp 2 -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Lamp 2 — Lux</span>
                        <div class="stat-icon" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">
                            <i class="fas fa-sun"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux2">0</span> lx</div>
                    <div class="stat-change">
                        <span>Brightness: <span id="dash-bright2">0</span>%</span>
                    </div>
                </div>

                <!-- Lamp 3 -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Lamp 3 — Lux</span>
                        <div class="stat-icon" style="background: rgba(99, 102, 241, 0.2); color: #6366f1;">
                            <i class="fas fa-sun"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux3">0</span> lx</div>
                    <div class="stat-change">
                        <span>Brightness: <span id="dash-bright3">0</span>%</span>
                    </div>
                </div>

                <!-- Average -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Average (3 Lamps)</span>
                        <div class="stat-icon" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">
                            <i class="fas fa-chart-bar"></i>
                        </div>
                    </div>
                    <div class="stat-value"><span id="dash-lux-avg">0</span> lx</div>
                    <div class="stat-change">
                        <span>Avg Brightness: <span id="dash-bright-avg">0</span>%</span>
                    </div>
                </div>

                <!-- Motion Detection -->
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

                <!-- PSO Control -->
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">PSO -> Lamp Control</span>
                        <div class="stat-icon" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">
                            <i class="fas fa-chart-line"></i>
                        </div>
                    </div>
                    <div class="stat-value" style="font-size: 24px;"><span id="pso-fitness">0.00</span></div>
                    <div class="stat-change" style="display: flex; flex-direction: column; gap: 4px;">
                        <span>Fitness Score</span>
                        <span style="font-size: 11px; color: #94a3b8;"><i class="fas fa-lightbulb"></i> Brightness: <span id="pso-brightness" style="color: #f59e0b; font-weight: bold;">--</span>%</span>
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
                        <button class="chart-option-btn" onclick="exportChartData('temp', 'Temperature (C)')" title="Export CSV"><i class="fas fa-download"></i></button>
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
                        <button class="chart-option-btn" onclick="exportChartData('hum', 'Humidity (%)')" title="Export CSV"><i class="fas fa-download"></i></button>
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
                        <button class="chart-option-btn" onclick="exportChartData('acTemp', 'AC Target Temp (C)')" title="Export CSV"><i class="fas fa-download"></i></button>
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
                    <div class="chart-title">Average Light Intensity (Lux) — 3 Lamps</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('lampLux', 'Lux')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn active" onclick="changeChartRange('lampLux', 1)">1h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampLux', 6)">6h</button>
                        <button class="chart-option-btn" onclick="changeChartRange('lampLux', 24)">24h</button>
                    </div>
                </div>
                <canvas id="lampLuxChart" height="80"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Average Brightness Level — 3 Lamps</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('lampBright', 'Brightness (%)')" title="Export CSV"><i class="fas fa-download"></i></button>
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
                <h1><i class="fas fa-video"></i> Live Camera Feed - YOLOv8 Detection</h1>
                <p>Real-time person detection using YOLOv8n</p>
            </div>

            <div class="camera-view">
                <div class="camera-feed-container" id="camera-feed-container">
                    <div class="camera-overlay-bar">
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span class="camera-rec-badge"><i class="fas fa-circle"></i> LIVE</span>
                            <span class="person-badge not-detected" id="overlay-person-badge">
                                <i class="fas fa-user-slash"></i> No Person
                            </span>
                        </div>
                        <span class="camera-time-badge" id="camera-time">00:00:00</span>
                    </div>
                    <img id="camera-img" src="/video_feed" alt="Camera Feed"
                         onerror="this.style.display='none'; document.getElementById('camera-error').style.display='flex';">
                    <div id="camera-error" class="camera-error" style="display:none; flex-direction:column; align-items:center;">
                        <i class="fas fa-video-slash"></i>
                        <h3>Camera Not Available</h3>
                        <p style="margin-top:10px; font-size:14px; color:var(--text-secondary);">
                            Make sure USB camera is properly connected
                        </p>
                        <button class="btn btn-primary" style="margin-top:15px;" onclick="retryCamera()">
                            <i class="fas fa-sync"></i> Retry Connection
                        </button>
                    </div>
                </div>

                <div class="camera-info-grid">
                    <div class="camera-info-card">
                        <div class="camera-info-label"><i class="fas fa-circle" style="color: var(--success);"></i> Status</div>
                        <div class="camera-info-value" id="cam-status">Connecting...</div>
                    </div>
                    <div class="camera-info-card">
                        <div class="camera-info-label"><i class="fas fa-expand"></i> Resolution</div>
                        <div class="camera-info-value" id="cam-resolution">Loading...</div>
                    </div>
                    <div class="camera-info-card">
                        <div class="camera-info-label"><i class="fas fa-tachometer-alt"></i> Frame Rate</div>
                        <div class="camera-info-value" id="cam-fps">Loading...</div>
                    </div>
                    <div class="camera-info-card" id="person-detected-card" style="transition: all 0.3s;">
                        <div class="camera-info-label"><i class="fas fa-walking"></i> Person Detected</div>
                        <div class="camera-info-value" id="cam-person" style="color: var(--danger);">No</div>
                    </div>
                    <div class="camera-info-card" style="transition: all 0.3s;">
                        <div class="camera-info-label"><i class="fas fa-users"></i> Person Count</div>
                        <div class="camera-info-value" id="cam-count-display">0</div>
                    </div>
                    <div class="camera-info-card" style="transition: all 0.3s;">
                        <div class="camera-info-label"><i class="fas fa-percentage"></i> Confidence</div>
                        <div class="camera-info-value" id="cam-confidence-display">0%</div>
                    </div>
                </div>
                
                <div style="margin-top: 20px; text-align: center; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
                    <button class="btn" id="camera-toggle-btn" onclick="toggleCamera()" style="padding: 12px 24px; border-radius: 12px; font-weight: 600; background: linear-gradient(135deg, #10b981, #059669); color: white; border: none; transition: all 0.3s;">
                        <i class="fas fa-video" id="camera-toggle-icon"></i> <span id="camera-toggle-text">Camera ON</span>
                    </button>
                    <button class="btn" id="sound-toggle-btn" onclick="toggleDetectionSound()" style="padding: 12px 24px; border-radius: 12px; font-weight: 600; background: linear-gradient(135deg, #10b981, #059669); color: white; border: none; transition: all 0.3s;">
                        <i class="fas fa-volume-up" id="sound-toggle-icon"></i> <span id="sound-toggle-text">Sound ON</span>
                    </button>
                    <button class="btn btn-success" onclick="retryCamera()" style="padding: 12px 24px; border-radius: 12px; font-weight: 600;">
                        <i class="fas fa-sync"></i> Refresh Feed
                    </button>
                </div>
            </div>
        </div>

        <!-- Energy Usage Page -->
        <div id="energy" class="page">
            <div class="header">
                <h1><i class="fas fa-bolt"></i> Energy Usage</h1>
                <p>Real-time & historical energy monitoring from PZEM-016</p>
            </div>

            <!-- Real-time summary cards -->
            <div class="power-grid">
                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">AC Energy / Day</div>
                    <div class="power-value"><span id="ac-energy-kwh">0</span> kWh</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Power: <span id="ac-power">0</span> W</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Lamp Energy / Day</div>
                    <div class="power-value"><span id="lamp-energy-kwh">0</span> kWh</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Power: <span id="lamp-power">0</span> W</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Total Energy / Day</div>
                    <div class="power-value"><span id="total-energy-kwh">0</span> kWh</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Total Power: <span id="total-power">0</span> W</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Daily Cost</div>
                    <div class="power-value">Rp<span id="daily-cost">0</span></div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">@ Rp 1,500/kWh</div>
                </div>
            </div>

            <!-- Historical Power Chart -->
            <div class="chart-container" style="margin-top: 30px;">
                <div class="chart-header">
                    <div class="chart-title"><i class="fas fa-chart-area"></i> Power Consumption (W)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="loadEnergyHistory('power', '1h', this)">1h</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('power', '6h', this)">6h</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('power', '24h', this)">24h</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('power', '7d', this)">7d</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('power', '30d', this)">30d</button>
                    </div>
                </div>
                <canvas id="energyPowerChart" height="80"></canvas>
                <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; font-size:12px; color:var(--text-secondary);">
                    <span>Latest: <strong id="energy-power-latest" style="color:var(--text-primary);">--</strong></span>
                    <span>Min: <strong id="energy-power-min" style="color:var(--text-primary);">--</strong></span>
                    <span>Max: <strong id="energy-power-max" style="color:var(--text-primary);">--</strong></span>
                    <span>Avg: <strong id="energy-power-avg" style="color:var(--text-primary);">--</strong></span>
                </div>
            </div>

            <!-- Historical Voltage Chart -->
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-header">
                    <div class="chart-title"><i class="fas fa-bolt"></i> Voltage (V)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="loadEnergyHistory('voltage', '1h', this)">1h</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('voltage', '6h', this)">6h</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('voltage', '24h', this)">24h</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('voltage', '7d', this)">7d</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('voltage', '30d', this)">30d</button>
                    </div>
                </div>
                <canvas id="energyVoltageChart" height="80"></canvas>
                <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; font-size:12px; color:var(--text-secondary);">
                    <span>Latest: <strong id="energy-voltage-latest" style="color:var(--text-primary);">--</strong></span>
                    <span>Min: <strong id="energy-voltage-min" style="color:var(--text-primary);">--</strong></span>
                    <span>Max: <strong id="energy-voltage-max" style="color:var(--text-primary);">--</strong></span>
                    <span>Avg: <strong id="energy-voltage-avg" style="color:var(--text-primary);">--</strong></span>
                </div>
            </div>

            <!-- Historical Energy kWh Chart -->
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-header">
                    <div class="chart-title"><i class="fas fa-battery-half"></i> Cumulative Energy (kWh)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn active" onclick="loadEnergyHistory('energy_kwh', '24h', this)">24h</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('energy_kwh', '7d', this)">7d</button>
                        <button class="chart-option-btn" onclick="loadEnergyHistory('energy_kwh', '30d', this)">30d</button>
                    </div>
                </div>
                <canvas id="energyKwhChart" height="80"></canvas>
                <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; font-size:12px; color:var(--text-secondary);">
                    <span>Latest: <strong id="energy-kwh-latest" style="color:var(--text-primary);">--</strong></span>
                    <span>Min: <strong id="energy-kwh-min" style="color:var(--text-primary);">--</strong></span>
                    <span>Max: <strong id="energy-kwh-max" style="color:var(--text-primary);">--</strong></span>
                    <span>Avg: <strong id="energy-kwh-avg" style="color:var(--text-primary);">--</strong></span>
                </div>
            </div>

            <!-- Estimated daily energy (real-time) -->
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-header">
                    <div class="chart-title"><i class="fas fa-chart-line"></i> Real-time Energy Trend (kWh/day estimate)</div>
                </div>
                <canvas id="energyChart" height="80"></canvas>
            </div>

            <!-- ===== BEFORE vs AFTER Adaptive AC Comparison ===== -->
            <div style="margin-top: 34px; padding: 20px; border-radius: 18px; border: 1px solid rgba(99,102,241,0.2); background: linear-gradient(160deg, rgba(99,102,241,0.08), rgba(241,245,249,0.92)); box-shadow: 0 12px 34px rgba(15,23,42,0.1);">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2 style="font-size: 20px; font-weight: 700; color: var(--text-primary); margin: 0 0 8px 0;">
                        <i class="fas fa-exchange-alt"></i> Before vs After Adaptive AC
                    </h2>
                    <p style="color: var(--text-secondary); font-size: 13px; margin: 0;">Compare energy usage before and after installing the adaptive AC system</p>
                </div>

                <!-- Phase Selector -->
                <div style="display: flex; justify-content: center; gap: 12px; margin-bottom: 20px;">
                    <button id="btn-phase-before" onclick="setEnergyPhase('before')" style="padding: 14px 28px; border-radius: 12px; border: 3px solid #f59e0b; background: linear-gradient(135deg, #f59e0b, #d97706); color: white; font-size: 14px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; align-items: center; gap: 8px;">
                        <i class="fas fa-clock"></i> BEFORE (Recording)
                    </button>
                    <button id="btn-phase-after" onclick="setEnergyPhase('after')" style="padding: 14px 28px; border-radius: 12px; border: 3px solid var(--border); background: var(--bg-card); color: var(--text-secondary); font-size: 14px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; align-items: center; gap: 8px; opacity: 0.6;">
                        <i class="fas fa-robot"></i> AFTER (Recording)
                    </button>
                </div>
                <div id="phase-indicator" style="text-align: center; padding: 10px; border-radius: 10px; font-size: 13px; font-weight: 600; margin-bottom: 20px; background: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3);">
                    <i class="fas fa-circle" style="font-size: 8px; animation: blink 1s infinite;"></i> Currently recording: <strong>BEFORE</strong> adaptive AC
                </div>

                <!-- Savings Summary Cards -->
                <div style="display: grid; grid-template-columns: repeat(3, minmax(130px,1fr)); gap: 12px; margin-bottom: 20px;">
                    <div style="padding: 14px; border-radius: 14px; text-align: center; background: rgba(245, 158, 11, 0.10); border: 1px solid rgba(245, 158, 11, 0.26); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.45);">
                        <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 6px;">Avg Before</div>
                        <div style="font-size: 22px; font-weight: 700; color: #f59e0b;"><span id="compare-avg-before">--</span> W</div>
                    </div>
                    <div style="padding: 14px; border-radius: 14px; text-align: center; background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16, 185, 129, 0.26); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.45);">
                        <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 6px;">Avg After</div>
                        <div style="font-size: 22px; font-weight: 700; color: #10b981;"><span id="compare-avg-after">--</span> W</div>
                    </div>
                    <div style="padding: 14px; border-radius: 14px; text-align: center; background: rgba(99, 102, 241, 0.10); border: 1px solid rgba(99, 102, 241, 0.26); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.45);">
                        <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 6px;">Energy Savings</div>
                        <div style="font-size: 22px; font-weight: 700; color: #6366f1;"><span id="compare-savings">--</span>%</div>
                    </div>
                </div>

                <!-- Comparison Chart -->
                <div class="chart-container" style="border: none; padding: 0;">
                    <div class="chart-header">
                        <div class="chart-title"><i class="fas fa-chart-bar"></i> Power Comparison (W)</div>
                        <div class="chart-options">
                            <button class="chart-option-btn active" onclick="loadEnergyCompare('power', '30m', this)" title="Test: 5 menit before + 5 menit after">TEST 30m</button>
                            <button class="chart-option-btn" onclick="loadEnergyCompare('power', '24h', this)">24h</button>
                            <button class="chart-option-btn" onclick="loadEnergyCompare('power', '7d', this)">7d</button>
                            <button class="chart-option-btn" onclick="loadEnergyCompare('power', '30d', this)">30d</button>
                        </div>
                    </div>
                    <canvas id="energyCompareChart" height="100"></canvas>
                </div>

                <!-- Energy kWh Comparison -->
                <div class="chart-container" style="border: none; padding: 0; margin-top: 16px;">
                    <div class="chart-header">
                        <div class="chart-title"><i class="fas fa-battery-half"></i> Energy (kWh) Comparison</div>
                        <div class="chart-options">
                            <button class="chart-option-btn active" onclick="loadEnergyCompare('energy_kwh', '30m', this)" title="Test: 5 menit before + 5 menit after">TEST 30m</button>
                            <button class="chart-option-btn" onclick="loadEnergyCompare('energy_kwh', '24h', this)">24h</button>
                            <button class="chart-option-btn" onclick="loadEnergyCompare('energy_kwh', '7d', this)">7d</button>
                            <button class="chart-option-btn" onclick="loadEnergyCompare('energy_kwh', '30d', this)">30d</button>
                        </div>
                    </div>
                    <canvas id="energyCompareKwhChart" height="100"></canvas>
                </div>
            </div>
        </div>

        <!-- Control Panel Page -->
        <!-- AC Control Page -->
        <div id="control-ac" class="page">
            <div class="header">
                <h1><i class="fas fa-snowflake"></i> AC Control Panel</h1>
                <p>AC Control using IRMitsubishiHeavy library</p>
            </div>

            <!-- ========== MODE SELECTOR (PROMINENT) ========== -->
            <div id="ac-mode-selector" style="margin-bottom: 20px; padding: 20px; border-radius: 16px; border: 2px solid var(--border); background: var(--bg-card);">
                <div style="text-align: center; margin-bottom: 14px; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary);">
                    <i class="fas fa-cog"></i> Control Mode
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <button id="btn-mode-adaptive" onclick="setACMode('ADAPTIVE')" style="padding: 18px 16px; border-radius: 14px; border: 3px solid #10b981; background: linear-gradient(135deg, #10b981, #059669); color: white; font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 6px;">
                        <i class="fas fa-robot" style="font-size: 28px;"></i>
                        <span>ADAPTIVE</span>
                        <span style="font-size: 11px; font-weight: 400; opacity: 0.9;">AI controls AC automatically</span>
                    </button>
                    <button id="btn-mode-manual" onclick="setACMode('MANUAL')" style="padding: 18px 16px; border-radius: 14px; border: 3px solid var(--border); background: var(--bg-card); color: var(--text-secondary); font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 6px; opacity: 0.6;">
                        <i class="fas fa-hand-paper" style="font-size: 28px;"></i>
                        <span>MANUAL</span>
                        <span style="font-size: 11px; font-weight: 400; opacity: 0.9;">Control AC manually</span>
                    </button>
                </div>
                <!-- Current mode indicator -->
                <div id="ac-mode-indicator" style="margin-top: 14px; padding: 10px 16px; border-radius: 10px; text-align: center; font-size: 13px; font-weight: 600; background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);">
                    <i class="fas fa-robot"></i> Current mode: <strong>ADAPTIVE</strong> — AC controlled automatically by AI (GA optimization)
                </div>
            </div>

            <!-- ========== ADAPTIVE INFO BANNER (shown when ADAPTIVE) ========== -->
            <div id="adaptive-info-banner" style="margin-bottom: 20px; padding: 20px; border-radius: 14px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.12)); border: 2px solid rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, #10b981, #059669); display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-brain" style="color: white; font-size: 22px;"></i>
                    </div>
                    <div>
                        <div style="font-size: 16px; font-weight: 700; color: #10b981;">Adaptive Mode Active</div>
                        <div style="font-size: 12px; color: var(--text-secondary);">GA Optimization controls temperature & fan speed automatically</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 13px;">
                    <div style="padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; text-align: center;">
                        <div style="color: var(--text-secondary); font-size: 11px;">GA Temp</div>
                        <div style="font-weight: 700; color: #10b981; font-size: 18px;" id="adaptive-ga-temp">--</div>
                    </div>
                    <div style="padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; text-align: center;">
                        <div style="color: var(--text-secondary); font-size: 11px;">GA Fan</div>
                        <div style="font-weight: 700; color: #10b981; font-size: 18px;" id="adaptive-ga-fan">--</div>
                    </div>
                    <div style="padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; text-align: center;">
                        <div style="color: var(--text-secondary); font-size: 11px;">GA Fitness</div>
                        <div style="font-weight: 700; color: #10b981; font-size: 18px;" id="adaptive-ga-fitness">--</div>
                    </div>
                </div>
                <div style="margin-top: 12px; padding: 10px; background: rgba(245, 158, 11, 0.1); border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.3); font-size: 12px; color: #f59e0b; text-align: center;">
                    <i class="fas fa-lock"></i> Manual control is disabled in Adaptive mode. Switch to Manual to control AC manually.
                </div>
            </div>

            <div class="control-panel" style="position: relative;">
                <!-- OVERLAY: blocks manual controls when ADAPTIVE -->
                <div id="ac-manual-overlay" style="display: block; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.45); backdrop-filter: blur(3px); border-radius: 12px; z-index: 10; display: flex; align-items: center; justify-content: center; cursor: not-allowed;">
                    <div style="text-align: center; color: white;">
                        <i class="fas fa-lock" style="font-size: 36px; margin-bottom: 10px; opacity: 0.8;"></i>
                        <div style="font-size: 14px; font-weight: 600;">Adaptive Mode Active</div>
                        <div style="font-size: 12px; opacity: 0.7;">Switch to Manual to use these controls</div>
                    </div>
                </div>

                <div class="control-title">
                    <span>Air Conditioning Control</span>
                    <div class="mode-badge manual" id="ac-mode-badge" style="display:none;">MANUAL MODE</div>
                </div>
                
                <!-- AC Realtime Status Bar -->
                <div id="ac-live-status" style="margin-top: 12px; padding: 12px 16px; background: rgba(99, 102, 241, 0.08); border: 1px solid var(--border); border-radius: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 13px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div id="ac-live-dot" style="width: 10px; height: 10px; border-radius: 50%; background: #ef4444;"></div>
                        <span style="font-weight: 600; color: var(--text);">AC <span id="ac-live-state">OFF</span></span>
                    </div>
                    <div style="display: flex; gap: 16px; color: var(--text-secondary);">
                        <span><i class="fas fa-thermometer-half"></i> <span id="ac-live-temp">24</span>°C</span>
                        <span><i class="fas fa-fan"></i> Fan <span id="ac-live-fan">1</span></span>
                        <span><i class="fas fa-cog"></i> <span id="ac-live-mode">COOL</span></span>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <button class="btn btn-success" style="padding: 20px; font-size: 16px; border-radius: 12px;" onclick="sendACCommand('POWER_ON')">
                        <i class="fas fa-power-off" style="font-size: 24px; display: block; margin-bottom: 8px;"></i> AC ON
                    </button>
                    <button class="btn btn-danger" style="padding: 20px; font-size: 16px; border-radius: 12px;" onclick="sendACCommand('POWER_OFF')">
                        <i class="fas fa-power-off" style="font-size: 24px; display: block; margin-bottom: 8px;"></i> AC OFF
                    </button>
                    <button class="btn btn-primary" style="padding: 20px; font-size: 16px; border-radius: 12px;" onclick="sendACCommand('TEMP_UP')">
                        <i class="fas fa-temperature-high" style="font-size: 24px; display: block; margin-bottom: 8px;"></i> TEMP +
                    </button>
                    <button class="btn btn-primary" style="padding: 20px; font-size: 16px; border-radius: 12px;" onclick="sendACCommand('TEMP_DOWN')">
                        <i class="fas fa-temperature-low" style="font-size: 24px; display: block; margin-bottom: 8px;"></i> TEMP -
                    </button>
                </div>

                <!-- AC Mode Buttons -->
                <div style="margin-top: 20px;">
                    <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 10px; font-weight: 600;">
                        <i class="fas fa-cog"></i> AC Mode
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;">
                        <button class="btn ac-mode-btn" id="mode-btn-auto" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: var(--primary); color: white; flex-direction: column;" onclick="sendACMode('MODE_AUTO', this)">
                            <i class="fas fa-magic" style="font-size: 20px; display: block; margin-bottom: 6px;"></i> Auto
                        </button>
                        <button class="btn ac-mode-btn" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: #0ea5e9; color: white; flex-direction: column;" onclick="sendACMode('MODE_COOL', this)">
                            <i class="fas fa-snowflake" style="font-size: 20px; display: block; margin-bottom: 6px;"></i> Cool
                        </button>
                        <button class="btn ac-mode-btn" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: #8b5cf6; color: white; flex-direction: column;" onclick="sendACMode('MODE_FAN', this)">
                            <i class="fas fa-fan" style="font-size: 20px; display: block; margin-bottom: 6px;"></i> Fan
                        </button>
                        <button class="btn ac-mode-btn" style="padding: 15px 10px; font-size: 13px; border-radius: 12px; background: #f97316; color: white; flex-direction: column;" onclick="sendACMode('MODE_DRY', this)">
                            <i class="fas fa-tint-slash" style="font-size: 20px; display: block; margin-bottom: 6px;"></i> Dry
                        </button>
                    </div>
                </div>

                <!-- AC Temperature & Fan Speed Sliders -->
                <div style="margin-top: 20px; padding: 20px; background: rgba(99, 102, 241, 0.05); border-radius: 12px; border: 1px solid var(--border);">
                    <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 15px; font-weight: 600;">
                        <i class="fas fa-sliders-h"></i> AC Settings (Direct MQTT)
                    </div>
                    <div class="control-group" style="margin-bottom: 15px;">
                        <label class="control-label">Temperature: <span id="ac-temp-display" style="color: var(--primary); font-weight: bold;">24</span>°C</label>
                        <input type="range" min="16" max="30" value="24" class="slider" id="ac-temp-slider" oninput="updateACTemp(this.value)" style="width: 100%;">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
                            <span>16°C</span><span>20°C</span><span>24°C</span><span>28°C</span><span>30°C</span>
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
                                <i class="fas fa-magic"></i><br>Auto
                            </button>
                            <button class="btn ac-set-mode-btn active" data-mode="COOL" onclick="selectACSetMode('COOL', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--primary); border: 2px solid var(--primary); color: white; cursor: pointer;">
                                <i class="fas fa-snowflake"></i><br>Cool
                            </button>
                            <button class="btn ac-set-mode-btn" data-mode="HEAT" onclick="selectACSetMode('HEAT', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--card-bg); border: 2px solid var(--border); color: var(--text); cursor: pointer;">
                                <i class="fas fa-fire"></i><br>Heat
                            </button>
                            <button class="btn ac-set-mode-btn" data-mode="DRY" onclick="selectACSetMode('DRY', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--card-bg); border: 2px solid var(--border); color: var(--text); cursor: pointer;">
                                <i class="fas fa-tint-slash"></i><br>Dry
                            </button>
                            <button class="btn ac-set-mode-btn" data-mode="FAN" onclick="selectACSetMode('FAN', this)" style="padding: 8px; font-size: 11px; border-radius: 8px; background: var(--card-bg); border: 2px solid var(--border); color: var(--text); cursor: pointer;">
                                <i class="fas fa-fan"></i><br>Fan
                            </button>
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="applyACSettings()" style="width: 100%; padding: 12px; border-radius: 10px; font-weight: 600;">
                        <i class="fas fa-paper-plane"></i> Apply AC Settings
                    </button>
                </div>

                <div style="margin-top: 15px; font-size: 12px; color: var(--text-secondary); text-align: center;">
                    <i class="fas fa-info-circle"></i> Using IRMitsubishiHeavy library — no manual learning needed.
                </div>
            </div>
        </div>

        <!-- Lamp Control Page -->
        <div id="control-lamp" class="page">
            <div class="header">
                <h1><i class="fas fa-lightbulb"></i> Lamp Control Panel</h1>
                <p>Manual lamp control and brightness settings — 3 Lamps</p>
            </div>

            <div class="control-panel">
                <div class="control-title">
                    <span>Lamp Control</span>
                    <div class="mode-badge adaptive" id="lamp-mode-badge" onclick="toggleLampMode()">ADAPTIVE MODE</div>
                </div>
                
                <div class="control-group">
                    <label class="control-label"><i class="fas fa-lightbulb" style="color: #f59e0b;"></i> Lamp 1 Brightness: <span id="brightness-display-1">0</span>%</label>
                    <input type="range" min="0" max="100" value="0" class="slider" id="brightness-slider-1" oninput="updateBrightness(1, this.value)">
                </div>

                <div class="control-group">
                    <label class="control-label"><i class="fas fa-lightbulb" style="color: #10b981;"></i> Lamp 2 Brightness: <span id="brightness-display-2">0</span>%</label>
                    <input type="range" min="0" max="100" value="0" class="slider" id="brightness-slider-2" oninput="updateBrightness(2, this.value)">
                </div>

                <div class="control-group">
                    <label class="control-label"><i class="fas fa-lightbulb" style="color: #6366f1;"></i> Lamp 3 Brightness: <span id="brightness-display-3">0</span>%</label>
                    <input type="range" min="0" max="100" value="0" class="slider" id="brightness-slider-3" oninput="updateBrightness(3, this.value)">
                </div>

                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button class="btn btn-primary" onclick="applyLampSettings()">
                        <i class="fas fa-check"></i> Apply All
                    </button>
                    <button class="btn" onclick="syncAllSliders()" style="background: var(--bg-elevated); color: var(--text); border: 1px solid var(--border);">
                        <i class="fas fa-sync"></i> Sync All to Lamp 1
                    </button>
                </div>
            </div>
        </div>

        <!-- ML Optimization Page -->
        <div id="ml-optimization" class="page">
            <div class="header">
                <h1><i class="fas fa-brain"></i> Machine Learning Optimization</h1>
                <p>GA -> Adaptive AC | PSO -> Adaptive Lamp | Data from InfluxDB</p>
            </div>

            <!-- ML Summary Cards -->
            <div class="stats-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">GA -> AC</span>
                        <div class="stat-icon" style="background: rgba(16, 185, 129, 0.2); color: #10b981;"><i class="fas fa-dna"></i></div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-ga-fitness" style="color: #10b981;">0.00</span></div>
                    <div class="stat-change"><span>Best Fitness</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Optimized Temp</span>
                        <div class="stat-icon" style="background: rgba(239, 68, 68, 0.2); color: #ef4444;"><i class="fas fa-thermometer-half"></i></div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-ga-temp" style="color: #ef4444;">--</span>°C</div>
                    <div class="stat-change"><span>Fan: <span id="ml-ga-fan" style="font-weight: bold;">--</span></span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">PSO -> Lamp</span>
                        <div class="stat-icon" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;"><i class="fas fa-chart-line"></i></div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-pso-fitness" style="color: #f59e0b;">0.00</span></div>
                    <div class="stat-change"><span>Best Fitness</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Optimized Brightness</span>
                        <div class="stat-icon" style="background: rgba(99, 102, 241, 0.2); color: #6366f1;"><i class="fas fa-lightbulb"></i></div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-pso-brightness" style="color: #6366f1;">--</span>%</div>
                    <div class="stat-change"><span>Lamp Level</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">Current Conditions</span>
                        <div class="stat-icon" style="background: rgba(99, 102, 241, 0.2); color: #6366f1;"><i class="fas fa-database"></i></div>
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
                        <div class="stat-icon" style="background: rgba(168, 85, 247, 0.2); color: #a855f7;"><i class="fas fa-sync-alt"></i></div>
                    </div>
                    <div class="stat-value" style="font-size: 28px;"><span id="ml-opt-runs" style="color: #a855f7;">0</span></div>
                    <div class="stat-change"><span>Total Cycles</span></div>
                </div>
            </div>

            <!-- GA Fitness Convergence Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title"><i class="fas fa-dna" style="color: #10b981;"></i> GA Fitness Convergence (AC Optimization)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('gaFitness', 'GA Fitness')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn ml-action-btn" onclick="runGAOptimization()" style="background: linear-gradient(135deg, #10b981, #059669); color: white; border: none;">
                            <i class="fas fa-play"></i> Run GA
                        </button>
                    </div>
                </div>
                <canvas id="gaFitnessChart" height="80"></canvas>
            </div>

            <!-- PSO Fitness Convergence Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title"><i class="fas fa-chart-line" style="color: #f59e0b;"></i> PSO Fitness Convergence (Lamp Optimization)</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportChartData('psoFitness', 'PSO Fitness')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn ml-action-btn" onclick="runPSOOptimization()" style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; border: none;">
                            <i class="fas fa-play"></i> Run PSO
                        </button>
                    </div>
                </div>
                <canvas id="psoFitnessChart" height="80"></canvas>
            </div>

            <!-- GA vs PSO Comparison Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title"><i class="fas fa-balance-scale" style="color: #6366f1;"></i> GA vs PSO — Fitness Comparison</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportCompareChart('comparison', 'Fitness')" title="Export CSV"><i class="fas fa-download"></i></button>
                        <button class="chart-option-btn ml-action-btn" onclick="runBothOptimization()" style="background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; border: none;">
                            <i class="fas fa-play"></i> Run Both
                        </button>
                        <button class="chart-option-btn" onclick="clearMLCharts()">
                            <i class="fas fa-eraser"></i> Clear
                        </button>
                    </div>
                </div>
                <canvas id="comparisonChart" height="80"></canvas>
            </div>

            <!-- Optimization History Table -->
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title"><i class="fas fa-history" style="color: #a855f7;"></i> Optimization History</div>
                    <div class="chart-options">
                        <button class="chart-option-btn" onclick="exportMLHistory()">
                            <i class="fas fa-download"></i> Export CSV
                        </button>
                        <button class="chart-option-btn" onclick="refreshMLHistory()">
                            <i class="fas fa-sync"></i> Refresh
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
                            <tr><td colspan="8" style="text-align: center; color: #94a3b8;">No optimization data yet. Run GA or PSO to start.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Parameter Configuration -->
            <div class="stats-grid" style="grid-template-columns: 1fr 1fr;">
                <div class="chart-container" style="margin-bottom: 0;">
                    <div class="chart-header">
                        <div class="chart-title"><i class="fas fa-cog" style="color: #10b981;"></i> GA Parameters</div>
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
                        <div class="chart-title"><i class="fas fa-cog" style="color: #f59e0b;"></i> PSO Parameters</div>
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
                <h1><i class="fas fa-users"></i> Occupancy Trend & Feedback</h1>
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
                        <div class="chart-title"><i class="fas fa-star"></i> Comfort Feedback</div>
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
                            <i class="fas fa-save"></i> Save Form Link
                        </button>
                        <button class="btn btn-success" onclick="openGoogleForm()">
                            <i class="fas fa-external-link-alt"></i> Open Google Form
                        </button>
                        <button class="btn btn-warning" onclick="submitOccupancyFeedback()">
                            <i class="fas fa-paper-plane"></i> Submit Feedback
                        </button>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title"><i class="fas fa-history"></i> Recent Feedback</div>
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
            <i class="fas fa-times"></i>
        </button>
        <div class="detection-alert-header">
            <i class="fas fa-exclamation-triangle detection-alert-icon"></i>
            <span>Person Detected!</span>
        </div>
        <div class="detection-alert-body">
            <div><strong id="alert-person-count">0</strong> person(s) detected</div>
            <div>Confidence: <strong id="alert-person-confidence">0%</strong></div>
            <div style="margin-top: 8px; font-size: 12px; opacity: 0.8;">
                <i class="fas fa-clock"></i> <span id="alert-time">--:--:--</span>
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
        var chartRanges = {
            temp: 1,
            hum: 1,
            acTemp: 1,
            lampLux: 1,
            lampBright: 1,
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

        function styleLineChart(chart, unit, showLegend) {
            if (!chart) return;
            chart.options = chart.options || {};
            chart.options.responsive = true;
            chart.options.maintainAspectRatio = true;
            chart.options.interaction = { mode: 'index', intersect: false };
            chart.options.plugins = chart.options.plugins || {};
            chart.options.plugins.legend = {
                display: !!showLegend,
                labels: {
                    color: '#94a3b8',
                    usePointStyle: true,
                    pointStyle: 'circle',
                    padding: 14,
                    font: { size: 12, weight: '600' }
                }
            };
            chart.options.plugins.tooltip = {
                enabled: true,
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                titleColor: '#f8fafc',
                bodyColor: '#e2e8f0',
                borderColor: 'rgba(148,163,184,0.35)',
                borderWidth: 1,
                displayColors: true,
                callbacks: {
                    label: function(ctx) {
                        const label = (ctx.dataset && ctx.dataset.label) ? ctx.dataset.label : 'Value';
                        return label + ': ' + formatChartValue(ctx.parsed.y, unit);
                    },
                    title: function(items) {
                        return items && items[0] ? ('Time: ' + items[0].label) : 'Time';
                    }
                }
            };
            chart.options.scales = chart.options.scales || {};
            chart.options.scales.x = chart.options.scales.x || {};
            chart.options.scales.y = chart.options.scales.y || {};
            chart.options.scales.x.grid = { color: 'rgba(148,163,184,0.12)' };
            chart.options.scales.x.ticks = { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 };
            chart.options.scales.y.grid = { color: 'rgba(148,163,184,0.14)' };
            chart.options.scales.y.ticks = {
                color: '#94a3b8',
                callback: function(value) {
                    return unit ? (Number(value).toFixed(1) + ' ' + unit) : Number(value).toFixed(1);
                }
            };

            if (chart.data && Array.isArray(chart.data.datasets)) {
                chart.data.datasets.forEach(function(ds) {
                    ds.borderWidth = 2.8;
                    ds.tension = 0.35;
                    ds.pointRadius = 4.4;
                    ds.pointHoverRadius = 7.5;
                    ds.pointBorderWidth = 1.6;
                    ds.pointBorderColor = '#ffffff';
                    ds.hitRadius = 10;
                    ds.hoverBorderWidth = 2;
                });
            }
        }

        function applyChartGradients() {
            const setGradient = function(chart, datasetIndex, top, bottom) {
                if (!chart || !chart.ctx || !chart.chartArea || !chart.data || !chart.data.datasets || !chart.data.datasets[datasetIndex]) return;
                const g = chart.ctx.createLinearGradient(0, chart.chartArea.top, 0, chart.chartArea.bottom);
                g.addColorStop(0, top);
                g.addColorStop(1, bottom);
                chart.data.datasets[datasetIndex].backgroundColor = g;
            };

            setGradient(charts.energyPower, 0, 'rgba(239,68,68,0.34)', 'rgba(239,68,68,0.03)');
            setGradient(charts.energyVoltage, 0, 'rgba(59,130,246,0.34)', 'rgba(59,130,246,0.03)');
            setGradient(charts.energyKwh, 0, 'rgba(16,185,129,0.34)', 'rgba(16,185,129,0.03)');
            setGradient(charts.energy, 0, 'rgba(168,85,247,0.34)', 'rgba(168,85,247,0.03)');
            setGradient(charts.energyCompare, 0, 'rgba(245,158,11,0.26)', 'rgba(245,158,11,0.02)');
            setGradient(charts.energyCompare, 1, 'rgba(16,185,129,0.26)', 'rgba(16,185,129,0.02)');
            setGradient(charts.energyCompareKwh, 0, 'rgba(245,158,11,0.26)', 'rgba(245,158,11,0.02)');
            setGradient(charts.energyCompareKwh, 1, 'rgba(16,185,129,0.26)', 'rgba(16,185,129,0.02)');
        }

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
                lampBrightness3: getVal('brightness-slider-3', 0) || 0
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
                    
                    for (let i = 1; i <= 3; i++) {
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

        // ==================== CHARTS ====================
        function initCharts() {
            function makeOpts(showLegend) {
                var opts = {
                    responsive: true,
                    maintainAspectRatio: true,
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
                data: { labels: [], datasets: [{ label: 'Temperature (\u00b0C)', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', tension: 0.4, fill: true }] }
            });

            charts.hum = new Chart(document.getElementById('humChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Humidity (%)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.4, fill: true }] }
            });

            charts.acTemp = new Chart(document.getElementById('acTempChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'AC Target Temp (\u00b0C)', data: [], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)', tension: 0.4, fill: true }] }
            });

            charts.lampLux = new Chart(document.getElementById('lampLuxChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Light Intensity (lux)', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.4, fill: true }] }
            });

            charts.lampBright = new Chart(document.getElementById('lampBrightChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Brightness (%)', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', tension: 0.4, fill: true }] }
            });

            charts.energy = new Chart(document.getElementById('energyChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Total Energy (kWh/day)', data: [], borderColor: '#a855f7', backgroundColor: 'rgba(168,85,247,0.1)', tension: 0.4, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#a855f7', pointBorderColor: '#fff', pointBorderWidth: 2 }] }
            });

            charts.energyPower = new Chart(document.getElementById('energyPowerChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Power (W)', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', tension: 0.4, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#ef4444', pointBorderColor: '#fff', pointBorderWidth: 2 }] }
            });

            charts.energyVoltage = new Chart(document.getElementById('energyVoltageChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Voltage (V)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.4, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 2 }] }
            });

            charts.energyKwh = new Chart(document.getElementById('energyKwhChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Energy (kWh)', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', tension: 0.4, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#10b981', pointBorderColor: '#fff', pointBorderWidth: 2 }] }
            });

            charts.energyCompare = new Chart(document.getElementById('energyCompareChart'), {
                type: 'line', options: makeOpts(true),
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Before Adaptive AC', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.4, fill: true, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#f59e0b', pointBorderColor: '#fff', pointBorderWidth: 1 },
                        { label: 'After Adaptive AC', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', tension: 0.4, fill: true, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#10b981', pointBorderColor: '#fff', pointBorderWidth: 1 }
                    ]
                }
            });

            charts.energyCompareKwh = new Chart(document.getElementById('energyCompareKwhChart'), {
                type: 'line', options: makeOpts(true),
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Before Adaptive AC', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.4, fill: true, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#f59e0b', pointBorderColor: '#fff', pointBorderWidth: 1 },
                        { label: 'After Adaptive AC', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', tension: 0.4, fill: true, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#10b981', pointBorderColor: '#fff', pointBorderWidth: 1 }
                    ]
                }
            });

            charts.occupancy = new Chart(document.getElementById('occupancyChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Occupancy (person)', data: [], borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.15)', tension: 0.35, fill: true, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#06b6d4', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });

            charts.gaFitness = new Chart(document.getElementById('gaFitnessChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'GA Best Fitness', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.15)', tension: 0.4, fill: true, pointRadius: 2 }] }
            });

            charts.psoFitness = new Chart(document.getElementById('psoFitnessChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'PSO Best Fitness', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.15)', tension: 0.4, fill: true, pointRadius: 2 }] }
            });

            charts.comparison = new Chart(document.getElementById('comparisonChart'), {
                type: 'line', options: makeOpts(true),
                data: {
                    labels: [],
                    datasets: [
                        { label: 'GA (AC)', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', tension: 0.4, fill: false, pointRadius: 3 },
                        { label: 'PSO (Lamp)', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.4, fill: false, pointRadius: 3 }
                    ]
                }
            });

            // Make energy charts visually richer and more informative.
            styleLineChart(charts.energyPower, 'W', false);
            styleLineChart(charts.energyVoltage, 'V', false);
            styleLineChart(charts.energyKwh, 'kWh', false);
            styleLineChart(charts.energy, 'kWh/day', false);
            styleLineChart(charts.energyCompare, 'W', true);
            styleLineChart(charts.energyCompareKwh, 'kWh', true);

            // Keep comparison lines visually distinct.
            if (charts.energyCompare && charts.energyCompare.data && charts.energyCompare.data.datasets) {
                charts.energyCompare.data.datasets[0].borderDash = [6, 4];
                charts.energyCompare.data.datasets[1].borderDash = [];
            }
            if (charts.energyCompareKwh && charts.energyCompareKwh.data && charts.energyCompareKwh.data.datasets) {
                charts.energyCompareKwh.data.datasets[0].borderDash = [6, 4];
                charts.energyCompareKwh.data.datasets[1].borderDash = [];
            }

            // Apply gradients after chart area is computed.
            setTimeout(function() {
                try {
                    applyChartGradients();
                    [
                        charts.energyPower,
                        charts.energyVoltage,
                        charts.energyKwh,
                        charts.energy,
                        charts.energyCompare,
                        charts.energyCompareKwh
                    ].forEach(function(c) { if (c) c.update('none'); });
                } catch (e) {
                    console.error('[CHART] gradient apply error:', e);
                }
            }, 60);
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

            if (!charts.energyPower || !charts.energyVoltage || !charts.energyKwh || !charts.energyCompare || !charts.energyCompareKwh) {
                try {
                    initCharts();
                } catch (e) {
                    console.error('[CHART] initCharts retry failed:', e);
                    return false;
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
                case 'lampLux': endpoint = '/api/chart/lamp_sensor/lux/' + hours; break;
                case 'lampBright': endpoint = '/api/chart/lamp_sensor/brightness/' + hours; break;
                case 'occupancy': endpoint = '/api/chart/camera_detection/person_count/' + hours; break;
            }

            fetch(endpoint)
                .then(r => r.json())
                .then(data => {
                    if (data && data.length > 0) {
                        charts[chartName].data.labels = data.map(d => d.time);
                        charts[chartName].data.datasets[0].data = data.map(d => d.value);
                        charts[chartName].update();
                    }
                })
                .catch(e => console.error('Chart error:', e));
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
            'energy_kwh': 'energyKwh'
        };

        var energyCanvasMap = {
            'power': 'energyPowerChart',
            'voltage': 'energyVoltageChart',
            'energy_kwh': 'energyKwhChart'
        };

        var energyColorMap = {
            'power': '#ef4444',
            'voltage': '#3b82f6',
            'energy_kwh': '#10b981'
        };

        function updateEnergyStats(field, values) {
            if (!values || values.length === 0) return;
            const prefix = field === 'power' ? 'energy-power' : (field === 'voltage' ? 'energy-voltage' : 'energy-kwh');
            const minV = Math.min(...values);
            const maxV = Math.max(...values);
            const avgV = values.reduce((a, b) => a + b, 0) / values.length;
            const lastV = values[values.length - 1];
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set(prefix + '-latest', lastV.toFixed(2));
            set(prefix + '-min', minV.toFixed(2));
            set(prefix + '-max', maxV.toFixed(2));
            set(prefix + '-avg', avgV.toFixed(2));
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

            ctx.strokeStyle = energyColorMap[field] || '#a855f7';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const dotColor = energyColorMap[field] || '#a855f7';
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

        function loadEnergyHistory(field, period, btnElement) {
            // Update button active state
            if (btnElement) {
                const buttons = btnElement.parentElement.querySelectorAll('.chart-option-btn');
                buttons.forEach(btn => btn.classList.remove('active'));
                btnElement.classList.add('active');
            }

            const chartName = energyChartMap[field];
            if (!chartName) return;

            fetch('/api/energy/history?field=' + field + '&period=' + period)
                .then(r => r.json())
                .then(result => {
                    const data = result.data || [];
                    const chart = charts[chartName];

                    if (chart && chart.data && chart.data.datasets && chart.data.datasets[0]) {
                        chart.data.labels = data.map(d => d.time);
                        const values = data.map(d => parseFloat(d.value || 0));
                        chart.data.datasets[0].data = values;
                        try { applyChartGradients(); } catch(e) {}
                        chart.update();
                        updateEnergyStats(field, values.filter(v => Number.isFinite(v)));
                    } else {
                        drawEnergyFallback(field, data);
                    }

                    if (data.length === 0) {
                        console.log('No energy data for ' + field + ' / ' + period);
                    }
                })
                .catch(e => console.error('Energy history error:', e));
        }

        function loadAllEnergyCharts() {
            loadEnergyHistory('power', '1h', null);
            loadEnergyHistory('voltage', '1h', null);
            loadEnergyHistory('energy_kwh', '24h', null);
            loadEnergyCompare('power', '30m', null);
            loadEnergyCompare('energy_kwh', '30m', null);
            loadCurrentPhase();
        }

        // ==================== BEFORE vs AFTER COMPARISON ====================
        var currentEnergyPhase = 'before';

        function loadCurrentPhase() {
            fetch('/api/energy/phase')
                .then(r => r.json())
                .then(data => {
                    currentEnergyPhase = data.phase;
                    updatePhaseUI(data.phase);
                })
                .catch(e => console.error('Phase load error:', e));
        }

        function setEnergyPhase(phase) {
            fetch('/api/energy/phase', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({phase: phase})
            })
            .then(r => r.json())
            .then(data => {
                currentEnergyPhase = data.phase;
                updatePhaseUI(data.phase);
            })
            .catch(e => console.error('Phase set error:', e));
        }

        function updatePhaseUI(phase) {
            const btnBefore = document.getElementById('btn-phase-before');
            const btnAfter = document.getElementById('btn-phase-after');
            const indicator = document.getElementById('phase-indicator');

            if (phase === 'before') {
                btnBefore.style.border = '3px solid #f59e0b';
                btnBefore.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
                btnBefore.style.color = 'white';
                btnBefore.style.opacity = '1';
                btnAfter.style.border = '3px solid var(--border)';
                btnAfter.style.background = 'var(--bg-card)';
                btnAfter.style.color = 'var(--text-secondary)';
                btnAfter.style.opacity = '0.6';
                indicator.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; animation: blink 1s infinite;"></i> Currently recording: <strong>BEFORE</strong> adaptive AC';
                indicator.style.background = 'rgba(245, 158, 11, 0.1)';
                indicator.style.color = '#f59e0b';
                indicator.style.borderColor = 'rgba(245, 158, 11, 0.3)';
            } else {
                btnAfter.style.border = '3px solid #10b981';
                btnAfter.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                btnAfter.style.color = 'white';
                btnAfter.style.opacity = '1';
                btnBefore.style.border = '3px solid var(--border)';
                btnBefore.style.background = 'var(--bg-card)';
                btnBefore.style.color = 'var(--text-secondary)';
                btnBefore.style.opacity = '0.6';
                indicator.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; animation: blink 1s infinite;"></i> Currently recording: <strong>AFTER</strong> adaptive AC';
                indicator.style.background = 'rgba(16, 185, 129, 0.1)';
                indicator.style.color = '#10b981';
                indicator.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            }
        }

        function loadEnergyCompare(field, period, btnElement) {
            if (btnElement) {
                const buttons = btnElement.parentElement.querySelectorAll('.chart-option-btn');
                buttons.forEach(btn => btn.classList.remove('active'));
                btnElement.classList.add('active');
            }

            const chartName = field === 'energy_kwh' ? 'energyCompareKwh' : 'energyCompare';
            if (!charts[chartName]) return;

            fetch('/api/energy/compare?field=' + field + '&period=' + period)
                .then(r => r.json())
                .then(result => {
                    const beforeData = result.before || [];
                    const afterData = result.after || [];
                    const summary = result.summary || {};

                    // Use the longer dataset's time labels
                    var timeSet = {};
                    beforeData.forEach(function(d) { timeSet[d.time] = true; });
                    afterData.forEach(function(d) { timeSet[d.time] = true; });
                    var allTimes = Object.keys(timeSet).sort();

                    const beforeMap = {};
                    beforeData.forEach(d => beforeMap[d.time] = d.value);
                    const afterMap = {};
                    afterData.forEach(d => afterMap[d.time] = d.value);

                    const chart = charts[chartName];
                    chart.data.labels = allTimes;
                    chart.data.datasets[0].data = allTimes.map(t => beforeMap[t] !== undefined ? beforeMap[t] : null);
                    chart.data.datasets[1].data = allTimes.map(t => afterMap[t] !== undefined ? afterMap[t] : null);
                    chart.options.spanGaps = true;
                    chart.update();

                    // Update summary cards (only for power comparison)
                    if (field === 'power') {
                        document.getElementById('compare-avg-before').textContent = summary.avg_before || '--';
                        document.getElementById('compare-avg-after').textContent = summary.avg_after || '--';
                        const savingsEl = document.getElementById('compare-savings');
                        savingsEl.textContent = summary.savings_percent || '--';
                        if (summary.savings_percent > 0) {
                            savingsEl.style.color = '#10b981';
                        } else if (summary.savings_percent < 0) {
                            savingsEl.style.color = '#ef4444';
                        }
                    }
                })
                .catch(e => console.error('Energy compare error:', e));
        }

        // Listen for phase changes from server
        socket.on('energy_phase', function(data) {
            currentEnergyPhase = data.phase;
            updatePhaseUI(data.phase);
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
                try { refreshMLData(); } catch(e) { console.error('[NAV] ML init error:', e); }
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
                        applyChartGradients();
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
                            const lux = d && d.lamp ? (d.lamp.lux_avg || d.lamp.lux1 || d.lamp.lux || '--') : '--';
                            luxEl.textContent = lux;
                        }
                        if (personEl) personEl.textContent = (d && d.camera && d.camera.person_detected) ? 'Yes' : 'No';
                    }).catch(() => {});
                })
                .catch(err => console.error('ML refresh error:', err));
        }

        function updateMLDisplay(data) {
            const setEl = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };

            setEl('ml-ga-fitness', (data.ga_fitness || 0).toFixed(2));
            setEl('ml-pso-fitness', (data.pso_fitness || 0).toFixed(2));
            setEl('ml-ga-temp', data.ga_temp || '--');
            setEl('ml-ga-fan', data.ga_fan || '--');
            setEl('ml-pso-brightness', data.pso_brightness || '--');
            setEl('ml-opt-runs', data.optimization_runs || 0);

            // Update convergence charts if history available
            if (data.ga_history && data.ga_history.length > 0) {
                updateMLChart('gaFitness', data.ga_history, 'GA');
            }
            if (data.pso_history && data.pso_history.length > 0) {
                updateMLChart('psoFitness', data.pso_history, 'PSO');
            }
        }

        function updateMLChart(chartName, history, algo) {
            const chart = charts[chartName];
            if (!chart) return;

            chart.data.labels = history.map((_, i) => algo === 'GA' ? ('Gen ' + (i + 1)) : ('Iter ' + (i + 1)));
            chart.data.datasets[0].data = history;
            chart.update();
        }

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
            chart.update();
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
                combined: ((data.ga_fitness || 0) + (data.pso_fitness || 0)) / 2
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
                const getBadge = (f) => f >= 80 ? 'good' : f >= 50 ? 'mid' : 'low';
                return '<tr>' +
                    '<td>' + e.run + '</td>' +
                    '<td>' + e.time + '</td>' +
                    '<td><span class="ml-badge ' + getBadge(e.ga_fitness) + '">' + e.ga_fitness.toFixed(2) + '</span></td>' +
                    '<td>' + e.ga_temp + '\u00b0C</td>' +
                    '<td>' + e.ga_fan + '</td>' +
                    '<td><span class="ml-badge ' + getBadge(e.pso_fitness) + '">' + e.pso_fitness.toFixed(2) + '</span></td>' +
                    '<td>' + e.pso_brightness + '%</td>' +
                    '<td><span class="ml-badge ' + getBadge(e.combined) + '">' + e.combined.toFixed(2) + '</span></td>' +
                '</tr>';
            }).join('');
        }

        function refreshMLHistory() {
            refreshMLData();
            showToast('ML data refreshed', 'success');
        }

        // ==================== EXPORT FUNCTIONS ====================
        function downloadCSV(filename, csvContent) {
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
                    charts[name].update();
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
            const icon = document.getElementById('camera-toggle-icon');
            const text = document.getElementById('camera-toggle-text');
            if (!btn) return;
            if (cameraEnabled) {
                btn.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                icon.className = 'fas fa-video';
                text.textContent = 'Camera ON';
            } else {
                btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
                icon.className = 'fas fa-video-slash';
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
                        statusEl.style.color = '#10b981';
                        document.getElementById('cam-resolution').textContent = data.width + ' x ' + data.height;
                        document.getElementById('cam-fps').textContent = data.fps + ' FPS';
                    } else {
                        statusEl.textContent = 'Inactive';
                        statusEl.style.color = '#ef4444';
                    }
                })
                .catch(e => {
                    document.getElementById('cam-status').textContent = 'Error';
                    document.getElementById('cam-status').style.color = '#ef4444';
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

        function updateFanSpeed(value) {
            document.getElementById('fan-speed-display').textContent = value;
            saveSettings();
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
                    const label = command.replace('_', ' ');
                    showToast('AC: ' + label, result.ir_sent ? 'success' : 'info');
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
                const fan = document.getElementById('fan-speed-slider').value;

                fetch('/api/ac/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        command: 'SET',
                        temperature: parseInt(temp),
                        fan_speed: parseInt(fan),
                        mode: selectedACMode
                    })
                })
                .then(r => r.json())
                .then(result => showToast('AC: ' + temp + '°C, Fan ' + fan + ', ' + selectedACMode))
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
            document.getElementById('brightness-slider-3').value = val;
            document.getElementById('brightness-display-2').textContent = val;
            document.getElementById('brightness-display-3').textContent = val;
            saveSettings();
        }

        function applyLampSettings() {
            const b1 = parseInt(document.getElementById('brightness-slider-1').value);
            const b2 = parseInt(document.getElementById('brightness-slider-2').value);
            const b3 = parseInt(document.getElementById('brightness-slider-3').value);
            
            fetch('/api/lamp/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ brightness1: b1, brightness2: b2, brightness3: b3 })
            })
            .then(r => r.json())
            .then(result => showToast('Lamp brightness applied: L1=' + b1 + '% L2=' + b2 + '% L3=' + b3 + '%'))
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
                    indicator.style.background = 'rgba(16, 185, 129, 0.1)';
                    indicator.style.color = '#10b981';
                    indicator.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                    indicator.innerHTML = '<i class="fas fa-robot"></i> Current mode: <strong>ADAPTIVE</strong> — AC controlled automatically by AI (GA optimization)';
                }
                // Button styles
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                    btnAdaptive.style.borderColor = '#10b981';
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
                    indicator.style.background = 'rgba(245, 158, 11, 0.1)';
                    indicator.style.color = '#f59e0b';
                    indicator.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                    indicator.innerHTML = '<i class="fas fa-hand-paper"></i> Current mode: <strong>MANUAL</strong> — Control AC manually using buttons below';
                }
                // Button styles
                if (btnManual) {
                    btnManual.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
                    btnManual.style.borderColor = '#f59e0b';
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

        // Keep old function name for backward compat (called elsewhere)
        function toggleACMode() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    const newMode = data.ac.mode === 'ADAPTIVE' ? 'MANUAL' : 'ADAPTIVE';
                    setACMode(newMode);
                });
        }

        function updateModeBadges() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    applyACModeUI(data.ac.mode);
                    
                    const lampBadge = document.getElementById('lamp-mode-badge');
                    if (lampBadge) {
                        lampBadge.textContent = data.lamp.mode + ' MODE';
                        lampBadge.className = 'mode-badge ' + data.lamp.mode.toLowerCase();
                    }
                });
        }

        function toggleLampMode() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    const newMode = data.lamp.mode === 'ADAPTIVE' ? 'MANUAL' : 'ADAPTIVE';
                    fetch('/api/lamp/mode', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode: newMode })
                    })
                    .then(r => r.json())
                    .then(result => { updateModeBadges(); showToast('Lamp Mode: ' + newMode); })
                    .catch(e => showToast('Error: ' + e, 'error'));
                });
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
                statusElement.style.color = '#f59e0b';
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
                                    statusElement.style.color = '#10b981';
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
                                    statusElement.style.color = '#ef4444';
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
                    statusElement.style.color = '#ef4444';
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
                const modeLabel = modeName.replace('MODE_', '');
                showToast('AC Mode: ' + modeLabel, result.ir_sent ? 'success' : 'info');
                setTimeout(() => {
                    document.querySelectorAll('.ac-mode-btn').forEach(btn => {
                        btn.style.opacity = '1';
                        btn.style.transform = 'scale(1)';
                    });
                    if (btnElement) {
                        btnElement.style.transform = 'scale(1.05)';
                        btnElement.style.boxShadow = '0 0 15px rgba(99, 102, 241, 0.5)';
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
                        summaryEl.style.color = codeCount > 0 ? '#10b981' : '#ef4444';
                    }
                    
                    // Update button status
                    Object.keys(learnedCodes).forEach(buttonName => {
                        const buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
                        const statusElement = document.getElementById('status-' + buttonName);
                        if (buttonElement && statusElement) {
                            buttonElement.classList.add('learned');
                            statusElement.textContent = 'Learned [OK]';
                            statusElement.style.color = '#10b981';
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
                summaryEl.style.color = '#ef4444';
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
            const icon = document.getElementById('sound-toggle-icon');
            const text = document.getElementById('sound-toggle-text');
            if (!btn) return;
            if (detectionSoundEnabled) {
                btn.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                if (icon) icon.className = 'fas fa-volume-up';
                if (text) text.textContent = 'Sound ON';
            } else {
                btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
                if (icon) icon.className = 'fas fa-volume-mute';
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
            
            toastMessage.innerHTML = '<i class="fas fa-' + icon + '-circle"></i> ' + message;
            toast.classList.add('show');
            setTimeout(() => { toast.classList.remove('show'); }, 3000);
        }

        var energyBubbleTimer = null;
        function showEnergyBubble(power, voltage, current) {
            const bubble = document.getElementById('energy-bubble');
            const text = document.getElementById('energy-bubble-text');
            if (!bubble || !text) return;
            text.textContent = power.toFixed(1) + ' W | ' + voltage.toFixed(1) + ' V | ' + current.toFixed(2) + ' A';
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
                    // Individual sensor readings
                    const t1El = document.getElementById('dash-temp1');
                    const t2El = document.getElementById('dash-temp2');
                    const t3El = document.getElementById('dash-temp3');
                    if (t1El) t1El.textContent = num(ac.temp1).toFixed(1);
                    if (t2El) t2El.textContent = num(ac.temp2).toFixed(1);
                    if (t3El) t3El.textContent = num(ac.temp3).toFixed(1);
                    // Individual humidity readings
                    const h1El = document.getElementById('dash-hum1');
                    const h2El = document.getElementById('dash-hum2');
                    const h3El = document.getElementById('dash-hum3');
                    if (h1El) h1El.textContent = num(ac.hum1).toFixed(1);
                    if (h2El) h2El.textContent = num(ac.hum2).toFixed(1);
                    if (h3El) h3El.textContent = num(ac.hum3).toFixed(1);
                    
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
                            acStateEl.style.color = '#10b981'; // Green
                        } else {
                            acStateEl.style.color = '#ef4444'; // Red
                        }
                        acStateEl.textContent = acState;
                    }
                    
                    // AC panel power dot
                    const panelDot = document.getElementById('ac-panel-dot');
                    if (panelDot) {
                        panelDot.style.background = acState === 'ON' ? '#10b981' : '#ef4444';
                        panelDot.style.boxShadow = acState === 'ON' ? '0 0 8px rgba(16,185,129,0.5)' : '0 0 8px rgba(239,68,68,0.5)';
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
                        'COOL': {icon: 'fa-snowflake', color: '#0ea5e9'},
                        'HEAT': {icon: 'fa-fire', color: '#f97316'},
                        'DRY':  {icon: 'fa-tint-slash', color: '#a855f7'},
                        'FAN':  {icon: 'fa-fan', color: '#8b5cf6'},
                        'AUTO': {icon: 'fa-magic', color: '#6366f1'}
                    };
                    const modeInfo = modeIcons[acMode] || modeIcons['COOL'];
                    if (modeIconEl) {
                        modeIconEl.innerHTML = '<i class="fas ' + modeInfo.icon + '"></i>';
                        modeIconEl.style.color = modeInfo.color;
                    }
                    if (modeTextEl) modeTextEl.style.color = modeInfo.color;
                    
                    // Operating Mode (ADAPTIVE / MANUAL)
                    const ctrlMode = ac.mode || 'ADAPTIVE';
                    const ctrlIcon = document.getElementById('dash-ac-ctrl-icon');
                    const ctrlText = document.getElementById('dash-ac-ctrl-mode');
                    if (ctrlIcon && ctrlText) {
                        if (ctrlMode === 'ADAPTIVE') {
                            ctrlIcon.innerHTML = '<i class="fas fa-robot"></i>';
                            ctrlIcon.style.color = '#10b981';
                            ctrlText.textContent = 'ADAPTIVE';
                            ctrlText.style.color = '#10b981';
                        } else {
                            ctrlIcon.innerHTML = '<i class="fas fa-hand-paper"></i>';
                            ctrlIcon.style.color = '#f59e0b';
                            ctrlText.textContent = 'MANUAL';
                            ctrlText.style.color = '#f59e0b';
                        }
                    }
                    
                    // Source badge
                    const srcBadge = document.getElementById('dash-ac-source');
                    if (srcBadge) {
                        if (ctrlMode === 'ADAPTIVE') {
                            srcBadge.innerHTML = '<i class="fas fa-robot"></i> AI Controlled';
                            srcBadge.style.background = 'rgba(16, 185, 129, 0.12)';
                            srcBadge.style.color = '#10b981';
                            srcBadge.style.borderColor = 'rgba(16, 185, 129, 0.25)';
                        } else {
                            srcBadge.innerHTML = '<i class="fas fa-hand-paper"></i> Manual Control';
                            srcBadge.style.background = 'rgba(245, 158, 11, 0.12)';
                            srcBadge.style.color = '#f59e0b';
                            srcBadge.style.borderColor = 'rgba(245, 158, 11, 0.25)';
                        }
                    }
                    
                    // Room environment in AC panel footer
                    const roomTemp = document.getElementById('dash-ac-room-temp');
                    const roomHum = document.getElementById('dash-ac-room-hum');
                    if (roomTemp) roomTemp.textContent = temperature.toFixed(1);
                    if (roomHum) roomHum.textContent = num(ac.humidity).toFixed(1);
                    // Individual sensors in AC panel footer
                    const acT1 = document.getElementById('dash-ac-temp1');
                    const acT2 = document.getElementById('dash-ac-temp2');
                    const acT3 = document.getElementById('dash-ac-temp3');
                    if (acT1) acT1.textContent = num(ac.temp1).toFixed(1);
                    if (acT2) acT2.textContent = num(ac.temp2).toFixed(1);
                    if (acT3) acT3.textContent = num(ac.temp3).toFixed(1);
                    
                    // Update AC Live Status Bar in control panel
                    const liveDot = document.getElementById('ac-live-dot');
                    const liveState = document.getElementById('ac-live-state');
                    if (liveDot && liveState) {
                        liveState.textContent = acState;
                        liveDot.style.background = acState === 'ON' ? '#10b981' : '#ef4444';
                        setText('ac-live-temp', acTemp);
                        setText('ac-live-fan', fanSpeed);
                        setText('ac-live-mode', acMode);
                    }
                    setText('dash-lux1', num(lamp.lux1).toFixed(0));
                    setText('dash-lux2', num(lamp.lux2).toFixed(0));
                    setText('dash-lux3', num(lamp.lux3).toFixed(0));
                    setText('dash-lux-avg', num(lamp.lux_avg).toFixed(1));
                    setText('dash-bright1', Math.round(num(lamp.brightness1) / 255 * 100));
                    setText('dash-bright2', Math.round(num(lamp.brightness2) / 255 * 100));
                    setText('dash-bright3', Math.round(num(lamp.brightness3) / 255 * 100));
                    setText('dash-bright-avg', Math.round(num(lamp.brightness_avg) / 255 * 100));
                    setText('dash-motion', lamp.motion ? 'MOTION DETECTED' : 'NO MOTION');
                    
                    const personDetected = !!camera.person_detected;
                    const personCount = num(camera.count);
                    const confidence = num(camera.confidence);
                    
                    const camPersonEl = document.getElementById('cam-person');
                    if (camPersonEl) {
                        camPersonEl.textContent = personDetected ? 'Yes' : 'No';
                        camPersonEl.style.color = personDetected ? '#10b981' : '#ef4444';
                    }
                    
                    const camCountEl = document.getElementById('cam-count');
                    if (camCountEl) {
                        camCountEl.textContent = personCount;
                        camCountEl.style.color = personCount > 0 ? '#10b981' : '#94a3b8';
                    }
                    
                    const camConfEl = document.getElementById('cam-confidence');
                    if (camConfEl) {
                        camConfEl.textContent = confidence + '%';
                        camConfEl.style.color = confidence > 70 ? '#10b981' : (confidence > 50 ? '#f59e0b' : '#ef4444');
                    }
                    
                    // Person detection status badge
                    const camBadge = document.getElementById('cam-status-badge');
                    if (camBadge) {
                        if (personCount > 0) {
                            camBadge.innerHTML = '<i class="fas fa-circle" style="font-size: 7px; vertical-align: middle;"></i> ' + personCount + ' Person Detected';
                            camBadge.style.background = 'rgba(16, 185, 129, 0.12)';
                            camBadge.style.color = '#10b981';
                            camBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                        } else {
                            camBadge.innerHTML = '<i class="fas fa-circle" style="font-size: 7px; vertical-align: middle;"></i> No Person';
                            camBadge.style.background = 'rgba(239, 68, 68, 0.12)';
                            camBadge.style.color = '#ef4444';
                            camBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
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
                        gaEl.style.color = gaFitness > 0 ? '#10b981' : '#94a3b8';
                    }
                    if (psoEl) {
                        psoEl.textContent = psoFitness.toFixed(2);
                        psoEl.style.color = psoFitness > 0 ? '#10b981' : '#94a3b8';
                    }
                    
                    // PSO Brightness
                    const psoBrightEl = document.getElementById('pso-brightness');
                    if (psoBrightEl) {
                        const psoBright = num(system.pso_brightness);
                        psoBrightEl.textContent = Math.round(psoBright / 255 * 100);
                    }
                    
                    // Optimization Runs
                    const optRunsEl = document.getElementById('dash-opt-runs');
                    if (optRunsEl) {
                        optRunsEl.textContent = num(system.optimization_runs);
                    }
                    
                    // Calculate AC power based on temperature logic
                    let acPower = 0;
                    const actualACState = temperature < 30 ? 'ON' : 'OFF';
                    
                    if (actualACState === 'ON') {
                        acPower = fanSpeed === 1 ? 100 : (fanSpeed === 2 ? 200 : 300);
                    }
                    let lampPower = (num(lamp.brightness1) + num(lamp.brightness2) + num(lamp.brightness3)) / 255 * 10;
                    let totalPower = acPower + lampPower;

                    // Use real PZEM data if available, otherwise estimate
                    let realPower = null;
                    let realEnergyKwh = null;
                    if (energy && energy.connected) {
                        realPower = num(energy.power);
                        realEnergyKwh = num(energy.energy);
                    }

                    let acEnergyKwh = (acPower / 1000) * 24;
                    let lampEnergyKwh = (lampPower / 1000) * 24;
                    let totalEnergyKwh = realEnergyKwh !== null ? realEnergyKwh : (acEnergyKwh + lampEnergyKwh);
                    let displayPower = realPower !== null ? realPower : totalPower;
                    let dailyCost = totalEnergyKwh * 1500;
                    
                    document.getElementById('ac-power').textContent = acPower.toFixed(0);
                    document.getElementById('lamp-power').textContent = lampPower.toFixed(1);
                    document.getElementById('total-power').textContent = displayPower.toFixed(1);
                    document.getElementById('ac-energy-kwh').textContent = acEnergyKwh.toFixed(2);
                    document.getElementById('lamp-energy-kwh').textContent = lampEnergyKwh.toFixed(2);
                    document.getElementById('total-energy-kwh').textContent = totalEnergyKwh.toFixed(3);
                    document.getElementById('daily-cost').textContent = dailyCost.toFixed(0);

                    if (charts.energy) {
                        const now = new Date();
                        const timeLabel = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
                        if (charts.energy.data.labels.length >= 50) {
                            charts.energy.data.labels.shift();
                            charts.energy.data.datasets[0].data.shift();
                        }
                        charts.energy.data.labels.push(timeLabel);
                        charts.energy.data.datasets[0].data.push(parseFloat(totalEnergyKwh.toFixed(2)));
                        charts.energy.update();
                    }
                    
                    // Energy Monitor (PZEM-016) data
                    if (energy) {
                        const e = energy;
                        const eVolt = document.getElementById('energy-voltage');
                        const eCurr = document.getElementById('energy-current');
                        const ePow = document.getElementById('energy-power');
                        const eKwh = document.getElementById('energy-kwh');
                        const eFreq = document.getElementById('energy-freq');
                        const ePf = document.getElementById('energy-pf');
                        const eCost = document.getElementById('energy-cost');
                        const eBadge = document.getElementById('energy-status-badge');
                        if (eVolt) eVolt.textContent = parseFloat(e.voltage || 0).toFixed(1);
                        if (eCurr) eCurr.textContent = parseFloat(e.current || 0).toFixed(2);
                        if (ePow) ePow.textContent = parseFloat(e.power || 0).toFixed(1);
                        if (eKwh) eKwh.textContent = parseFloat(e.energy || 0).toFixed(3);
                        if (eFreq) eFreq.textContent = parseFloat(e.frequency || 0).toFixed(1);
                        if (ePf) ePf.textContent = parseFloat(e.pf || 0).toFixed(2);
                        if (eCost) {
                            const cost = parseFloat(e.energy || 0) * 1500;
                            eCost.textContent = cost.toLocaleString('id-ID', {minimumFractionDigits: 0, maximumFractionDigits: 0});
                        }
                        if (eBadge) {
                            if (e.connected) {
                                eBadge.innerHTML = '<i class="fas fa-circle" style="font-size: 6px; vertical-align: middle; margin-right: 4px;"></i> Online';
                                eBadge.style.background = 'rgba(16, 185, 129, 0.15)';
                                eBadge.style.color = '#10b981';
                            } else {
                                eBadge.innerHTML = '<i class="fas fa-circle" style="font-size: 6px; vertical-align: middle; margin-right: 4px;"></i> Offline';
                                eBadge.style.background = 'rgba(107, 114, 128, 0.15)';
                                eBadge.style.color = '#6b7280';
                            }
                        }
                    }

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
                    if (showDetail) diagLog('ERROR: Tidak bisa fetch /api/mqtt/status');
                });
        }

        function runSimulate() {
            diagClear('=== TEST FRONTEND (INJECT DATA DUMMY) ===');
            diagLog('Mengirim data dummy ke server...');
            fetch('/api/simulate', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'ok') {
                        diagLog('BERHASIL! Data dummy sudah diinjek:');
                        diagLog('  AC Temperature: ' + data.ac_temp + ' C');
                        diagLog('  Lamp Lux1: ' + data.lamp_lux + ' lux');
                        diagLog('');
                        diagLog('Jika nilai di dashboard berubah dari 0 -> FRONTEND OK');
                        diagLog('Jika tetap 0 -> Ada masalah di JavaScript/DOM');
                        updateDashboard();
                    } else {
                        diagLog('ERROR: ' + data.message);
                    }
                })
                .catch(function(e) { diagLog('FETCH ERROR: ' + e); });
        }

        function runMqttSelftest() {
            diagClear('=== TEST MQTT BROKER (SELF-TEST) ===');
            diagLog('Server akan publish ke smartroom/ac/sensors...');
            fetch('/api/mqtt/selftest', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'ok') {
                        diagLog('BERHASIL! Pesan test berhasil dipublish ke broker.');
                        diagLog(data.message);
                        diagLog('');
                        diagLog('Tunggu 2 detik... lalu cek apakah data muncul di dashboard.');
                        diagLog('Jika muncul -> MQTT Broker OK, masalah ada di ESP32');
                        diagLog('Jika tidak muncul -> Ada masalah subscribe/routing topic');
                        setTimeout(function() { updateDashboard(); diagLog('Dashboard refreshed.'); }, 2000);
                    } else {
                        diagLog('GAGAL: ' + data.message);
                        diagLog('');
                        diagLog('Artinya: MQTT Broker tidak berjalan atau tidak bisa terkoneksi!');
                        diagLog('Jalankan Mosquitto terlebih dahulu.');
                    }
                })
                .catch(function(e) { diagLog('FETCH ERROR: ' + e); });
        }

        function runMqttReconnect() {
            diagClear('=== RECONNECT MQTT ===');
            diagLog('Mencoba reconnect ke MQTT broker...');
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
            banner.innerHTML = '<i class="fas fa-' + (alert.level === 'danger' ? 'exclamation-triangle' : 'exclamation-circle') + '"></i>' +
                '<span>' + alert.message + '</span>' +
                '<button class="alert-close" onclick="this.parentElement.remove()"><i class="fas fa-times"></i></button>';
            document.body.appendChild(banner);
            setTimeout(() => { if (banner.parentElement) banner.remove(); }, 8000);
        }

        // ==================== SOCKET.IO EVENTS ====================
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
                
                const luxAvg = data.data.lux_avg || 0;
                const brightAvg = data.data.brightness_avg || 0;
                
                if (charts.lampLux && charts.lampLux.data.labels.length < 50) {
                    charts.lampLux.data.labels.push(timeStr);
                    charts.lampLux.data.datasets[0].data.push(luxAvg);
                    charts.lampLux.update();
                }
                if (charts.lampBright && charts.lampBright.data.labels.length < 50) {
                    charts.lampBright.data.labels.push(timeStr);
                    charts.lampBright.data.datasets[0].data.push(Math.round(brightAvg / 255 * 100));
                    charts.lampBright.update();
                }
            }
            
            if (data.type === 'energy') {
                const e = data.data;
                const voltEl = document.getElementById('energy-voltage');
                const currEl = document.getElementById('energy-current');
                const powEl = document.getElementById('energy-power');
                const kwhEl = document.getElementById('energy-kwh');
                const freqEl = document.getElementById('energy-freq');
                const pfEl = document.getElementById('energy-pf');
                const costEl = document.getElementById('energy-cost');
                const badge = document.getElementById('energy-status-badge');
                
                if (voltEl) voltEl.textContent = parseFloat(e.voltage || 0).toFixed(1);
                if (currEl) currEl.textContent = parseFloat(e.current || 0).toFixed(2);
                if (powEl) powEl.textContent = parseFloat(e.power || 0).toFixed(1);
                if (kwhEl) kwhEl.textContent = parseFloat(e.energy || 0).toFixed(3);
                if (freqEl) freqEl.textContent = parseFloat(e.frequency || 0).toFixed(1);
                if (pfEl) pfEl.textContent = parseFloat(e.pf || 0).toFixed(2);
                
                // Estimasi biaya: tarif PLN R1/1300VA = Rp 1.444,70/kWh
                if (costEl) {
                    const cost = parseFloat(e.energy || 0) * 1444.70;
                    costEl.textContent = cost.toLocaleString('id-ID', {minimumFractionDigits: 0, maximumFractionDigits: 0});
                }
                
                if (badge) {
                    if (e.connected) {
                        badge.innerHTML = '<i class="fas fa-circle" style="font-size: 6px; vertical-align: middle; margin-right: 4px;"></i> Online';
                        badge.style.background = 'rgba(16, 185, 129, 0.15)';
                        badge.style.color = '#10b981';
                        badge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                    } else {
                        badge.innerHTML = '<i class="fas fa-circle" style="font-size: 6px; vertical-align: middle; margin-right: 4px;"></i> Offline';
                        badge.style.background = 'rgba(107, 114, 128, 0.15)';
                        badge.style.color = '#6b7280';
                        badge.style.borderColor = 'rgba(107, 114, 128, 0.3)';
                    }
                }
                
                // Show energy data bubble notification
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
                const psoBrightness = data.data.pso_brightness || 0;
                
                // Update GA fitness display
                const gaEl = document.getElementById('ga-fitness');
                if (gaEl) {
                    gaEl.textContent = gaFitness.toFixed(2);
                    gaEl.style.color = gaFitness > 0 ? '#10b981' : '#94a3b8';
                    if (gaFitness > 0) gaEl.style.textShadow = '0 0 10px rgba(16, 185, 129, 0.5)';
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
                    psoEl.style.color = psoFitness > 0 ? '#f59e0b' : '#94a3b8';
                    if (psoFitness > 0) psoEl.style.textShadow = '0 0 10px rgba(245, 158, 11, 0.5)';
                }
                
                // Update PSO solution details (Lamp brightness)
                const psoBrightEl = document.getElementById('pso-brightness');
                if (psoBrightEl && psoBrightness > 0) psoBrightEl.textContent = psoBrightness;
                
                console.log('[ML] Optimization Update:', {
                    ga: gaFitness.toFixed(2), ac_temp: gaTemp, ac_fan: gaFan,
                    pso: psoFitness.toFixed(2), lamp_brightness: psoBrightness,
                    runs: runs
                });
                
                // Show toast with specific results
                if (gaFitness > 0 && psoFitness > 0) {
                    showToast('GA->AC: ' + gaTemp + '°C (' + gaFitness.toFixed(1) + ') | PSO->Lamp: ' + psoBrightness + '% (' + psoFitness.toFixed(1) + ')', 'success');
                } else if (gaFitness > 0) {
                    showToast('GA->AC: ' + gaTemp + '°C Fan:' + gaFan + ' (Fitness: ' + gaFitness.toFixed(2) + ')', 'success');
                } else if (psoFitness > 0) {
                    showToast('PSO->Lamp: ' + psoBrightness + '% (Fitness: ' + psoFitness.toFixed(2) + ')', 'success');
                }

                // === Update ML Optimization Page ===
                updateMLDisplay(data.data);
                if (gaFitness > 0 || psoFitness > 0) {
                    addToComparisonChart(gaFitness, psoFitness);
                    addMLHistoryRow({
                        ga_fitness: gaFitness, ga_temp: gaTemp, ga_fan: gaFan,
                        pso_fitness: psoFitness, pso_brightness: psoBrightness
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
                        countEl.style.color = '#10b981';
                        countEl.style.textShadow = '0 0 10px rgba(16, 185, 129, 0.5)';
                    } else {
                        countEl.style.color = '#94a3b8';
                        countEl.style.textShadow = 'none';
                    }
                }
                if (countDisplayEl) {
                    countDisplayEl.textContent = personCount;
                    countDisplayEl.style.color = personCount > 0 ? '#10b981' : '#94a3b8';
                    if (personCount > 0) {
                        countDisplayEl.parentElement.style.background = 'rgba(16, 185, 129, 0.1)';
                        countDisplayEl.parentElement.style.borderColor = '#10b981';
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
                            el.style.color = '#10b981';
                        } else if (confidence > 50) {
                            el.style.color = '#f59e0b';
                        } else {
                            el.style.color = '#ef4444';
                        }
                    }
                });
                
                if (confDisplayEl && personCount > 0) {
                    confDisplayEl.parentElement.style.background = 'rgba(16, 185, 129, 0.1)';
                    confDisplayEl.parentElement.style.borderColor = '#10b981';
                } else if (confDisplayEl) {
                    confDisplayEl.parentElement.style.background = '';
                    confDisplayEl.parentElement.style.borderColor = 'var(--border)';
                }
                
                // Update person detected card
                const personEl = document.getElementById('cam-person');
                const personCard = document.getElementById('person-detected-card');
                if (personEl) {
                    personEl.textContent = personDetected ? 'Yes' : 'No';
                    personEl.style.color = personDetected ? '#10b981' : '#ef4444';
                }
                if (personCard) {
                    if (personDetected) {
                        personCard.style.background = 'rgba(16, 185, 129, 0.1)';
                        personCard.style.borderColor = '#10b981';
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
                        overlayBadge.innerHTML = '<i class="fas fa-user-check"></i> ' + personCount + ' Person(s) - ' + confidence + '%';
                    } else {
                        overlayBadge.className = 'person-badge not-detected';
                        overlayBadge.innerHTML = '<i class="fas fa-user-slash"></i> No Person';
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
                        lastSeenEl.style.color = '#10b981';
                        if (lastSeenLabel) lastSeenLabel.textContent = 'Person currently detected';
                    } else if (lastSeenAgo !== undefined && lastSeenAgo >= 0) {
                        const mins = Math.floor(lastSeenAgo / 60);
                        const secs = lastSeenAgo % 60;
                        lastSeenEl.textContent = mins > 0 ? (mins + 'm ' + secs + 's lalu') : (secs + 's lalu');
                        lastSeenEl.style.color = lastSeenAgo > 300 ? '#ef4444' : '#6366f1';
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
                        autoOffEl.textContent = 'MATI';
                        autoOffEl.style.color = '#ef4444';
                        if (autoOffLabel) autoOffLabel.textContent = 'AC already auto-OFF';
                    } else if (autoOffIn !== undefined && autoOffIn >= 0 && !personDetected) {
                        const offMins = Math.floor(autoOffIn / 60);
                        const offSecs = autoOffIn % 60;
                        autoOffEl.textContent = offMins + 'm ' + offSecs + 's';
                        autoOffEl.style.color = autoOffIn < 120 ? '#ef4444' : '#f59e0b';
                        if (autoOffLabel) autoOffLabel.textContent = 'Countdown auto-OFF AC';
                    } else {
                        autoOffEl.textContent = '--';
                        autoOffEl.style.color = '#10b981';
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
                // Disable run buttons while running
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                });
            } else if (status === 'completed') {
                showToast(algo + ' optimization completed! GA: ' + (data.ga_fitness || 0).toFixed(2) + ', PSO: ' + (data.pso_fitness || 0).toFixed(2), 'success');
                // Re-enable run buttons
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                });
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
                    statusElement.style.color = '#ef4444';
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
                statusElement.style.color = '#10b981';
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
                    var cs = window.getComputedStyle(el);
                    if (cs.position === 'fixed' && cs.display === 'none') {
                        el.style.pointerEvents = 'none';
                    }
                });
            } catch(e) {}
            try { initCharts(); } catch(e) { console.error('[ERROR] initCharts:', e); }
            try { ensureChartsReady(); } catch(e) { console.error('[ERROR] ensureChartsReady:', e); }
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
                            if (res) res.textContent = '[AUTO-DIAGNOSA] MQTT TIDAK TERHUBUNG!\\nBroker: ' + d.broker + '\\nError: ' + (d.error || 'Tidak diketahui') + '\\n\\nKlik tombol "Test Frontend" atau "Test MQTT Broker" di bawah.';
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
                } catch(e) {}
            }, 8000);
            
            console.log('[OK] Dashboard Ready!');
        };
    </script>

    <!-- Failsafe Navigation - independent script block -->
    <script>
        (function() {
            try {
                var ovBoot = document.getElementById('sidebar-overlay');
                if (ovBoot) {
                    ovBoot.classList.remove('active');
                    ovBoot.style.display = 'none';
                    ovBoot.style.pointerEvents = 'none';
                }
            } catch (e) {}

            if (typeof window.showPage === 'function') {
                console.log('[OK] showPage already defined');
                return;
            }
            console.warn('[FAILSAFE] Main script failed - activating failsafe navigation');
            window.showPage = function(pageId) {
                var pages = document.querySelectorAll('.page');
                var navs = document.querySelectorAll('.nav-item');
                for (var i = 0; i < pages.length; i++) pages[i].classList.remove('active');
                for (var i = 0; i < navs.length; i++) navs[i].classList.remove('active');
                var el = document.getElementById(pageId);
                if (el) el.classList.add('active');
                for (var i = 0; i < navs.length; i++) {
                    var oc = navs[i].getAttribute('onclick');
                    if (oc && oc.indexOf(pageId) !== -1) navs[i].classList.add('active');
                }
                localStorage.setItem('currentPage', pageId);
            };
            window.toggleSidebar = function() {
                var sb = document.getElementById('sidebar');
                var ov = document.getElementById('sidebar-overlay');
                if (sb) sb.classList.toggle('open');
                if (ov) ov.classList.toggle('active');
            };
            window.toggleTheme = function() {
                var cur = document.documentElement.getAttribute('data-theme');
                var nw = cur === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', nw);
                localStorage.setItem('theme', nw);
            };
            // Start basic data polling
            function basicUpdate() {
                try {
                    var x = new XMLHttpRequest();
                    x.open('GET', '/api/data');
                    x.onload = function() {
                        if (x.status === 200) {
                            var d = JSON.parse(x.responseText);
                            var ac = d && d.ac ? d.ac : {};
                            var energy = d && d.energy ? d.energy : {};
                            var n = function(v, fallback) {
                                var num = parseFloat(v);
                                return isNaN(num) ? (fallback || 0) : num;
                            };
                            var setText = function(id, val) {
                                var el = document.getElementById(id);
                                if (el) el.textContent = val;
                            };

                            // Dashboard AC cards
                            setText('dash-temp', n(ac.temperature, 0).toFixed(1));
                            setText('dash-hum', n(ac.humidity, 0).toFixed(1));
                            setText('dash-temp1', n(ac.temp1, 0).toFixed(1));
                            setText('dash-temp2', n(ac.temp2, 0).toFixed(1));
                            setText('dash-temp3', n(ac.temp3, 0).toFixed(1));
                            setText('dash-hum1', n(ac.hum1, 0).toFixed(1));
                            setText('dash-hum2', n(ac.hum2, 0).toFixed(1));
                            setText('dash-hum3', n(ac.hum3, 0).toFixed(1));
                            setText('dash-heat-index', n(ac.heat_index, 0).toFixed(1));
                            setText('dash-ac-state', ac.ac_state || 'OFF');
                            setText('dash-ac-temp', Math.round(n(ac.ac_temp, 24)));
                            setText('dash-ac-fan', Math.round(n(ac.fan_speed, 1)));
                            setText('dash-ac-mode', ac.ac_fan_mode || 'COOL');
                            setText('dash-rssi', Math.round(n(ac.rssi, 0)));
                            setText('dash-ac-room-temp', n(ac.temperature, 0).toFixed(1));
                            setText('dash-ac-room-hum', n(ac.humidity, 0).toFixed(1));

                            // Energy cards
                            setText('energy-voltage', n(energy.voltage, 0).toFixed(1));
                            setText('energy-current', n(energy.current, 0).toFixed(2));
                            setText('energy-power', n(energy.power, 0).toFixed(1));
                            setText('energy-kwh', n(energy.energy, 0).toFixed(3));
                            setText('energy-freq', n(energy.frequency, 0).toFixed(1));
                            setText('energy-pf', n(energy.pf, 0).toFixed(2));

                            // Legacy fallback ids (if any old card still exists)
                            var t = document.getElementById('room-temp');
                            var h = document.getElementById('room-hum');
                            if (t) t.textContent = n(ac.temperature, 0).toFixed(1) + String.fromCharCode(176) + 'C';
                            if (h) h.textContent = n(ac.humidity, 0).toFixed(1) + '%';
                        }
                    };
                    x.send();
                } catch(e) {}
            }

            function drawLineOnCanvas(canvasId, points, color, label) {
                try {
                    var canvas = document.getElementById(canvasId);
                    if (!canvas) return;
                    var rect = canvas.getBoundingClientRect();
                    var w = Math.max(320, Math.floor(rect.width || canvas.width || 640));
                    var h = Math.max(150, Math.floor(rect.height || canvas.height || 220));
                    canvas.width = w;
                    canvas.height = h;
                    var ctx = canvas.getContext('2d');
                    if (!ctx) return;

                    ctx.clearRect(0, 0, w, h);
                    ctx.fillStyle = 'rgba(15, 23, 42, 0.04)';
                    ctx.fillRect(0, 0, w, h);

                    // Grid
                    ctx.strokeStyle = 'rgba(148,163,184,0.25)';
                    ctx.lineWidth = 1;
                    for (var i = 1; i <= 4; i++) {
                        var gy = (h / 5) * i;
                        ctx.beginPath();
                        ctx.moveTo(0, gy);
                        ctx.lineTo(w, gy);
                        ctx.stroke();
                    }

                    if (!points || points.length === 0) {
                        ctx.fillStyle = '#94a3b8';
                        ctx.font = '13px sans-serif';
                        ctx.fillText('Waiting ' + label + ' data...', 14, Math.floor(h / 2));
                        return;
                    }

                    var vals = [];
                    for (var j = 0; j < points.length; j++) {
                        var v = parseFloat(points[j].value);
                        if (!isNaN(v)) vals.push(v);
                    }
                    if (vals.length === 0) return;

                    var minV = Math.min.apply(null, vals);
                    var maxV = Math.max.apply(null, vals);
                    if (minV === maxV) {
                        minV = minV - 1;
                        maxV = maxV + 1;
                    }

                    var padX = 26;
                    var padY = 20;
                    var plotW = w - (padX * 2);
                    var plotH = h - (padY * 2);

                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    var dotPoints = [];
                    for (var k = 0; k < vals.length; k++) {
                        var x = padX + (k / Math.max(1, vals.length - 1)) * plotW;
                        var y = padY + (1 - ((vals[k] - minV) / (maxV - minV))) * plotH;
                        dotPoints.push({x: x, y: y});
                        if (k === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }
                    ctx.stroke();

                    // Draw circular markers for each sample point.
                    for (var m = 0; m < dotPoints.length; m++) {
                        ctx.beginPath();
                        ctx.arc(dotPoints[m].x, dotPoints[m].y, 2.8, 0, Math.PI * 2);
                        ctx.fillStyle = color;
                        ctx.fill();
                        ctx.lineWidth = 1;
                        ctx.strokeStyle = '#ffffff';
                        ctx.stroke();
                    }

                    var latest = vals[vals.length - 1];
                    var minVal = Math.min.apply(null, vals);
                    var maxVal = Math.max.apply(null, vals);
                    var sumVal = 0;
                    for (var n = 0; n < vals.length; n++) sumVal += vals[n];
                    var avgVal = sumVal / Math.max(1, vals.length);
                    ctx.fillStyle = '#334155';
                    ctx.font = '12px sans-serif';
                    ctx.fillText(label + ': ' + latest.toFixed(2) + '  min:' + minVal.toFixed(2) + '  max:' + maxVal.toFixed(2) + '  avg:' + avgVal.toFixed(2), 12, 15);
                } catch (e) {}
            }

            function fetchEnergySeries(field, period, cb) {
                try {
                    var x = new XMLHttpRequest();
                    x.open('GET', '/api/energy/history?field=' + encodeURIComponent(field) + '&period=' + encodeURIComponent(period));
                    x.onload = function() {
                        if (x.status === 200) {
                            try {
                                var res = JSON.parse(x.responseText);
                                cb((res && res.data) ? res.data : []);
                            } catch (e) {
                                cb([]);
                            }
                        } else {
                            cb([]);
                        }
                    };
                    x.onerror = function() { cb([]); };
                    x.send();
                } catch (e) {
                    cb([]);
                }
            }

            function loadEnergyChartsFailsafe() {
                fetchEnergySeries('power', '1h', function(points) {
                    drawLineOnCanvas('energyPowerChart', points, '#ef4444', 'Power (W)');
                });
                fetchEnergySeries('voltage', '1h', function(points) {
                    drawLineOnCanvas('energyVoltageChart', points, '#3b82f6', 'Voltage (V)');
                });
                fetchEnergySeries('energy_kwh', '24h', function(points) {
                    drawLineOnCanvas('energyKwhChart', points, '#10b981', 'Energy (kWh)');
                });
            }

            basicUpdate();
            setInterval(basicUpdate, 2000);
            loadEnergyChartsFailsafe();
            setInterval(loadEnergyChartsFailsafe, 8000);
            // Restore saved page
            var saved = localStorage.getItem('currentPage');
            if (saved && document.getElementById(saved)) {
                window.showPage(saved);
            }
        })();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("=" * 60)
    print("  Smart Room Dashboard - YOLOv8n + Auto AC Control")
    print("=" * 60)
    
    # Load saved IR codes from file
    print("  [INIT] Loading saved IR codes...")
    try:
        import os
        ir_file = os.path.join(os.path.dirname(__file__), 'ir_codes.json')
        if os.path.exists(ir_file):
            with open(ir_file, 'r') as f:
                mqtt_data['ir_codes'] = json.load(f)
            print(f"  [OK] Loaded {len(mqtt_data['ir_codes'])} IR codes from file")
            # Verify each code's completeness
            for btn_name, code in mqtt_data['ir_codes'].items():
                if isinstance(code, str) and code.startswith('RAW:'):
                    raw_count = code[4:].count(',') + 1
                    status = '[OK]' if raw_count >= 100 else '[WARN]'
                    print(f"    {status} {btn_name}: RAW {raw_count} values, {len(code)} chars")
                else:
                    print(f"    [IR] {btn_name}: {len(code)} chars")
        else:
            print("  [INFO] No saved IR codes found")
    except Exception as e:
        print(f"  [WARN] Error loading IR codes: {e}")
    
    print("  [INIT] Loading YOLO model (please wait)...")
    
    # Load YOLO SYNCHRONOUSLY
    yolo_loaded = load_yolo_model()
    
    if yolo_loaded:
        print("  [OK] YOLO ready for person detection!")
    else:
        print("  [WARN] YOLO failed to load, running without detection")
    
    # Start background camera detection thread (runs 24/7 for auto ON/OFF)
    print("  [CAM] Starting background camera detection thread...")
    detection_thread = threading.Thread(target=camera_detection_loop, daemon=True)
    detection_thread.start()
    print("  [OK] Detection thread running — auto ON/OFF active")
    
    print("=" * 60)
    print("  [URL] Dashboard URL: http://172.20.0.65:5000")
    print("  [URL] Video Feed:    http://172.20.0.65:5000/video_feed")
    print("  Features:")
    print("     - YOLOv8n Person Detection (background thread)")
    print("     - Auto ON: 3 frames (~2s) person confirmed -> AC ON")
    print("     - Auto OFF: 10 min no person -> AC OFF")
    print("     - 4K Camera (fallback 1080p)")
    print("     - Real-time Person Count & Confidence")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
