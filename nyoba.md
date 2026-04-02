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
camera_enabled = True  # Camera ON/OFF toggle

# YOLO Configuration
yolo_model = None
yolo_lock = threading.Lock()

# Smart Person-Based Auto ON/OFF
_person_consecutive_frames = 0
_no_person_start_time = None
PERSON_CONFIRM_FRAMES = 3        # 3 frames with person → auto ON (~2 seconds)
PERSON_RECONFIRM_FRAMES = 15     # 15 frames (~10s) required to re-enable after auto-OFF
NO_PERSON_TIMEOUT_SECONDS = 600  # 10 minutes no person → auto OFF
AUTO_OFF_COOLDOWN = 120          # 2 min cooldown: block auto-ON after auto-OFF
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
    'lamp': {'lux': 0, 'motion': False, 'brightness': 0, 'mode': 'ADAPTIVE', 'rssi': 0, 'uptime': 0},
    'camera': {'person_detected': False, 'count': 0, 'confidence': 0, 'status': 'inactive'},
    'system': {'ga_fitness': 0, 'pso_fitness': 0, 'optimization_runs': 0, 'ga_temp': 0, 'ga_fan': 0, 'pso_brightness': 0, 'ga_history': [], 'pso_history': []},
    'ir_codes': {},
    'ir_states': {}  # Track toggle states for power buttons
}

log_messages = deque(maxlen=100)
ir_learning_mode = False
ir_learning_button = ""
ir_learning_device = ""  # Track device name

# ==================== INFLUXDB WRITE FUNCTIONS ====================
def write_to_influxdb(measurement, fields, tags=None):
    """Write data point to InfluxDB"""
    try:
        print(f"🔄 Attempting to write to InfluxDB: {measurement}")
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
        print(f"✅ Successfully written to InfluxDB: {measurement}")
        return True
    except Exception as e:
        print(f"❌ InfluxDB Write Error ({measurement}): {str(e)}")
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
            print(f"✅ Sensor data saved: {temperature}°C, {humidity}%")
    except Exception as e:
        print(f"❌ Error saving sensor data: {e}")
        import traceback
        traceback.print_exc()

def save_lamp_data(lux, brightness, motion):
    """Save lamp sensor data to InfluxDB"""
    try:
        result = write_to_influxdb(
            measurement="lamp_sensor",
            fields={
                "lux": float(lux),
                "brightness": float(brightness),
                "motion": bool(motion)
            },
            tags={"device": "esp32_lamp", "location": "room"}
        )
        if result:
            print(f"✅ Lamp data saved: {lux} lux, {brightness}%")
    except Exception as e:
        print(f"❌ Error saving lamp data: {e}")
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
            print(f"✅ Detection saved: {person_count} person(s), {confidence:.2f} confidence")
    except Exception as e:
        print(f"❌ Error saving detection data: {e}")

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
        print(f"✅ IR command saved: {device} - {command}")
    except Exception as e:
        print(f"❌ Error saving IR command: {e}")

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
        print(f"✅ AC control saved: {ac_temp}°C, Fan: {fan_speed}, State: {ac_state}")
    except Exception as e:
        print(f"❌ Error saving AC control: {e}")

# ==================== YOLO INITIALIZATION ====================
def load_yolo_model():
    global yolo_model
    try:
        import os
        
        yolo_dir = '/home/iotlab/smartroom/yolo'
        model_path = f'{yolo_dir}/yolov8n.pt'
        
        # Try local model first, fallback to auto-download
        if os.path.exists(model_path):
            print(f"🧠 Loading YOLOv8n from {model_path}...")
            yolo_model = YOLO(model_path)
        else:
            print("🧠 YOLOv8n not found locally, downloading...")
            os.makedirs(yolo_dir, exist_ok=True)
            yolo_model = YOLO('yolov8n.pt')
        
        # Warm up with dummy inference
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        yolo_model.predict(dummy, verbose=False)
        
        print("✅ YOLOv8n model loaded successfully!")
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 
                           'msg': 'YOLOv8n loaded successfully', 'level': 'success'})
        return True
        
    except Exception as e:
        print(f"❌ YOLO loading error: {str(e)}")
        import traceback
        traceback.print_exc()
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 
                           'msg': f'YOLO error: {str(e)}', 'level': 'error'})
        return False

def detect_persons(frame):
    """Detect persons using YOLOv8"""
    global yolo_model
    
    if yolo_model is None:
        return frame, 0, 0.0
    
    try:
        with yolo_lock:
            # YOLOv8 inference — classes=[0] filters for 'person' only
            results = yolo_model.predict(frame, conf=0.35, classes=[0], verbose=False)
            
            person_count = 0
            max_confidence = 0.0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"Person {confidence*100:.0f}%"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    person_count += 1
                    if confidence > max_confidence:
                        max_confidence = confidence
            
            if person_count > 0:
                print(f"✅ YOLOv8: {person_count} person(s) detected")
            
            return frame, person_count, max_confidence
            
    except Exception as e:
        print(f"❌ YOLO detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return frame, 0, 0.0

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
    # After auto-OFF → require 15 frames (~10s) to prevent false positives from turning AC on
    # Normal mode → require 3 frames (~2s)
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
            print(f"✅ Person RE-CONFIRMED after {_person_consecutive_frames} frames (cooldown cleared)")
        elif _person_consecutive_frames >= required_frames and in_cooldown:
            cooldown_left = AUTO_OFF_COOLDOWN - (time.time() - _auto_off_time)
            if int(_person_consecutive_frames) % 30 == 0:  # Print every ~20s
                print(f"⏳ Person detected but cooldown active ({int(cooldown_left)}s left) — need sustained presence")
        
        # Auto ON: person confirmed AND cooldown elapsed
        if _person_consecutive_frames >= required_frames and not in_cooldown and not _auto_off_triggered:
            if mqtt_data['ac'].get('ac_state') == 'OFF':
                print(f"🟢 Person confirmed ({_person_consecutive_frames} frames, req={required_frames}) → Auto turning ON AC")
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
                    print(f"❌ Auto ON error: {e}")
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
                print(f"🔴 No person for {int(elapsed)}s → Auto turning OFF AC (cooldown {AUTO_OFF_COOLDOWN}s starts)")
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
                    print(f"❌ Auto OFF error: {e}")

# ==================== CAMERA FUNCTIONS ====================
def get_camera():
    global camera
    if camera is None:
        # Auto-detect camera: try index 0-4
        for idx in range(5):
            print(f"📷 Trying camera index {idx}...")
            cam = cv2.VideoCapture(idx)
            if cam.isOpened():
                # Verify it can actually capture a frame
                ret, test_frame = cam.read()
                if ret and test_frame is not None:
                    camera = cam
                    print(f"✅ Camera found at index {idx}")
                    break
                else:
                    cam.release()
                    print(f"   Index {idx}: opened but can't read frames")
            else:
                cam.release()
                print(f"   Index {idx}: not available")
        
        if camera is not None and camera.isOpened():
            # Try 4K or best available
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            camera.set(cv2.CAP_PROP_FPS, 60)
            
            # Get actual values
            actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Fallback to 1080p if 4K not supported
            if actual_w < 1920:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                camera.set(cv2.CAP_PROP_FPS, 30)
                actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            actual_fps = int(camera.get(cv2.CAP_PROP_FPS))
            
            mqtt_data['camera']['status'] = 'active'
            print(f"✅ Camera initialized: {actual_w}x{actual_h} @ {actual_fps}fps")
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
    global _latest_frame_bytes, _detection_thread_running, camera
    _detection_thread_running = True
    last_save_time = time.time()
    save_interval = 5
    retry_delay = 2
    
    print("🎥 Camera detection background thread started (runs 24/7)")
    
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
                        print("⚠️ Camera read failed, will retry...")
            
            # If no frame captured, wait and retry (lock is released)
            if frame is None:
                time.sleep(retry_delay)
                continue
            
            mqtt_data['camera']['status'] = 'active'
            
            # YOLO person detection
            frame, person_count, confidence = detect_persons(frame)
            
            # Update shared data
            mqtt_data['camera']['person_detected'] = person_count > 0
            mqtt_data['camera']['count'] = person_count
            mqtt_data['camera']['confidence'] = int(confidence * 100)
            
            # ★ Smart auto ON/OFF — runs regardless of video viewer
            handle_person_based_control(person_count)
            
            # Save to InfluxDB every 5 seconds
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                save_person_detection(person_count, confidence)
                last_save_time = current_time
            
            # Emit to WebSocket for live dashboard updates
            socketio.emit('mqtt_update', {'type': 'camera', 'data': mqtt_data['camera']})
            
            # Draw overlays on frame
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, 'Smart Room Camera - YOLOv8 Detection', (20, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if person_count > 0:
                cv2.putText(frame, f'Persons Detected: {person_count}', (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Debug: print timer status every 60 seconds
            if _no_person_start_time is not None:
                elapsed = time.time() - _no_person_start_time
                if int(elapsed) % 60 < 1:
                    print(f"⏱️ No person timer: {int(elapsed)}s / {NO_PERSON_TIMEOUT_SECONDS}s | AC: {mqtt_data['ac'].get('ac_state')} | Mode: {mqtt_data['ac'].get('mode')}")
            
            # Encode frame and store for video_feed consumers
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ret:
                with _latest_frame_lock:
                    _latest_frame_bytes = buffer.tobytes()
            
            # Cap at ~20 FPS
            time.sleep(0.033)
            
        except Exception as e:
            print(f"❌ Detection loop error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(retry_delay)
    
    print("🛑 Camera detection background thread stopped")

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
mqtt_client = mqtt.Client()
mqtt_client.max_message_size = 0  # NO LIMIT! Default could truncate large RAW IR codes

def on_connect(client, userdata, flags, rc):
    print("\n" + "="*70)
    print("  🔵 FLASK MQTT CONNECTION EVENT")
    print("="*70)
    print(f"Return Code: {rc}")
    print(f"RC Meaning: {['Success', 'Protocol version', 'Client ID', 'Server unavailable', 'Bad credentials', 'Not authorized'][rc] if rc < 6 else 'Unknown'}")
    print(f"Flags: {flags}")
    print("="*70)
    
    if rc == 0:
        print("✅ MQTT CONNECTED SUCCESSFULLY!\n")
        
        # Subscribe to all smartroom topics
        result1, mid1 = client.subscribe("smartroom/#")
        print(f"📡 Subscribe smartroom/# - Result: {result1} (MID: {mid1})")
        
        # Also subscribe to alternative IR topics
        result2, mid2 = client.subscribe("ir/#")
        print(f"📡 Subscribe ir/# - Result: {result2} (MID: {mid2})")
        
        result3, mid3 = client.subscribe("IR/#")
        print(f"📡 Subscribe IR/# - Result: {result3} (MID: {mid3})")
        
        result4, mid4 = client.subscribe("+/ir/#")
        print(f"📡 Subscribe +/ir/# - Result: {result4} (MID: {mid4})")
        
        print("\n✅ All subscriptions sent!")
        print("Waiting for messages from ESP32...\n")
        print("="*70 + "\n")
        
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'MQTT Connected! RC: {rc}', 'level': 'success'})
    else:
        print(f"❌ MQTT CONNECTION FAILED! RC={rc}")
        print("="*70 + "\n")

def on_message(client, userdata, msg):
    global ir_learning_mode, ir_learning_button, ir_learning_device
    
    try:
        topic = msg.topic
        
        # Debug: Print ALL incoming MQTT messages with timestamp
        print("\n" + "─"*70)
        print(f"📨 [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] MQTT Message Received")
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
        except:
            # If not JSON, treat as plain text
            payload_text = msg.payload.decode()
            print(f"   Data (Text): {payload_text[:200]}..." if len(payload_text) > 200 else f"   Data (Text): {payload_text}")
            payload = {'raw': payload_text}
        
        # DEBUG: Check which condition will match
        print(f"\n🔍 Topic Routing Check:")
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
            print("🌡️  Processing AC Sensor Data:")
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
            print("   💾 Saving to InfluxDB...")
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
            print("   ✅ AC data updated in memory & InfluxDB")
            socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
            print("   📡 Sent to frontend via WebSocket")
            # Track device status & check alerts
            device_last_seen['esp32_ac']['last_seen'] = datetime.now()
            device_last_seen['esp32_ac']['status'] = 'online'
            check_alert_rules()
            
        elif 'lamp/sensors' in topic:
            mqtt_data['lamp'].update({
                'lux': payload.get('lux', 0),
                'motion': payload.get('motion', False),
                'brightness': payload.get('brightness', 0),
                'rssi': payload.get('rssi', 0),
                'uptime': payload.get('uptime', 0)
            })
            # Save lamp data to InfluxDB
            save_lamp_data(
                mqtt_data['lamp']['lux'],
                mqtt_data['lamp']['brightness'],
                mqtt_data['lamp']['motion']
            )
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
            # Track device status
            device_last_seen['esp32_lamp']['last_seen'] = datetime.now()
            device_last_seen['esp32_lamp']['status'] = 'online'
            
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
            # Update from optimization algorithm (GA→AC / PSO→Lamp)
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
            print(f"📊 Optimization Update: GA={mqtt_data['system']['ga_fitness']:.2f} (AC: {mqtt_data['system']['ga_temp']}°C Fan:{mqtt_data['system']['ga_fan']}), PSO={mqtt_data['system']['pso_fitness']:.2f} (Lamp: {mqtt_data['system']['pso_brightness']}%)")
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
                        print(f"🤖 ADAPTIVE → Applied to AC: {opt_temp}°C Fan:{opt_fan}")
                    else:
                        print(f"🤖 ADAPTIVE → AC apply debounced ({5 - (now - _last_adaptive_ac_apply):.1f}s remaining)")
            elif mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE':
                print(f"🤖 ADAPTIVE → BLOCKED: No person confirmed in last {NO_PERSON_TIMEOUT_SECONDS//60} min")
            # AUTO-APPLY: If Lamp is in ADAPTIVE mode
            if mqtt_data['lamp'].get('mode', 'MANUAL') == 'ADAPTIVE':
                opt_brightness = mqtt_data['system'].get('pso_brightness', 0)
                if opt_brightness > 0:
                    global _last_adaptive_lamp_apply
                    now = time.time()
                    if now - _last_adaptive_lamp_apply >= 5:
                        _last_adaptive_lamp_apply = now
                        client.publish('smartroom/lamp/control', json.dumps({'brightness': int(opt_brightness), 'source': 'adaptive'}))
                        print(f"🤖 ADAPTIVE → Applied to Lamp: {opt_brightness}%")
                    else:
                        print(f"🤖 ADAPTIVE → Lamp apply debounced")
        
        elif 'ml/status' in topic:
            # ML optimization status from main.py (running/completed/error/busy)
            print(f"🤖 ML Status: {payload.get('status', 'unknown')} ({payload.get('algorithm', '')})")
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
                    print(f"📊 ML Completed → Updated: GA={mqtt_data['system']['ga_fitness']:.2f}, PSO={mqtt_data['system']['pso_fitness']:.2f}")
                    socketio.emit('mqtt_update', {'type': 'system', 'data': mqtt_data['system']})
        
        elif 'ac/mode' in topic:
            mqtt_data['ac']['mode'] = payload.get('mode', 'ADAPTIVE')
            socketio.emit('mqtt_update', {'type': 'ac', 'data': mqtt_data['ac']})
            
        elif 'lamp/mode' in topic:
            mqtt_data['lamp']['mode'] = payload.get('mode', 'ADAPTIVE')
            socketio.emit('mqtt_update', {'type': 'lamp', 'data': mqtt_data['lamp']})
        
        elif 'ir/learned' in topic or 'IR/learned' in topic.lower():
            print("\n" + "="*70)
            print("🔴🔴🔴 IR SIGNAL RECEIVED FROM ESP32! 🔴🔴🔴")
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
                print(f"   ✅ RAW values count: {raw_value_count} (need 200+ for Mitsubishi AC)")
                if raw_value_count < 100:
                    print(f"   ⚠️  WARNING: Only {raw_value_count} values! Signal mungkin tidak lengkap!")
                    print(f"   ⚠️  Mitsubishi SRK AC butuh ~200+ values (2 frame)")
            
            # CHECK: Is Flask in learning mode?
            if not ir_learning_mode:
                print(f"⚠️  Flask NOT in learning mode — ignoring signal")
                print(f"   (Signal has {len(ir_code)} chars, {raw_value_count} RAW values)")
                print(f"   To capture: click Learn button on dashboard first")
                print("="*70)
            elif button_name and ir_code and len(ir_code) > 0:
                print(f"✅ Flask IS in learning mode — SAVING signal!")
                
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
                        print(f"⚠️ Power toggle detected for {device}: Same code for ON/OFF")
                
                # Save to memory
                mqtt_data['ir_codes'][button_name] = ir_code
                print(f"✅ IR code saved to memory: {button_name}")
                
                # Verify code integrity
                if ir_code.startswith('RAW:'):
                    raw_cnt = ir_code[4:].count(',') + 1
                    print(f"   ✅ Verified RAW signal: {raw_cnt} values stored for {button_name}")
                else:
                    print(f"   ✅ Non-RAW code stored: {len(ir_code)} chars")
                
                # Auto-save to InfluxDB immediately
                save_ir_command(device, button_name, len(ir_code))
                
                # Save to file for persistence
                try:
                    import os
                    ir_file = os.path.join(os.path.dirname(__file__), 'ir_codes.json')
                    with open(ir_file, 'w') as f:
                        json.dump(mqtt_data['ir_codes'], f, indent=2)
                    print(f"💾 IR codes saved to file: {ir_file}")
                    
                    # Verify file write
                    with open(ir_file, 'r') as f:
                        verify = json.load(f)
                    if button_name in verify:
                        saved_code = verify[button_name]
                        if saved_code == ir_code:
                            print(f"   ✅ File verified: {button_name} saved correctly ({len(saved_code)} chars)")
                        else:
                            print(f"   ⚠️  File mismatch! Memory={len(ir_code)} vs File={len(saved_code)}")
                    else:
                        print(f"   ⚠️  Button {button_name} NOT found in saved file!")
                except Exception as e:
                    print(f"❌ Error saving IR codes to file: {e}")
                
                log_messages.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'msg': f'IR Code learned: {button_name}{" (TOGGLE)" if is_power_toggle else ""}',
                    'level': 'success'
                })
                
                # Emit to frontend
                print("\n📡 Emitting 'ir_learned' event to frontend via WebSocket...")
                print(f"   Event data: button={button_name}, device={device}, is_toggle={is_power_toggle}")
                
                socketio.emit('ir_learned', {
                    'button': button_name, 
                    'code': ir_code[:50] + '...' if len(ir_code) > 50 else ir_code,  # Truncate for display
                    'device': device,
                    'is_toggle': is_power_toggle,
                    'status': 'success'
                })
                
                print("✅ WebSocket event emitted successfully!")
                print("   Frontend should update NOW!")
                print("="*70)
                
                print(f"✅ IR learning completed for: {button_name}")
                
                ir_learning_mode = False
                ir_learning_button = ""
                ir_learning_device = ""
            else:
                print(f"❌ IR learning failed: Missing button name or code")
                socketio.emit('ir_learned', {
                    'status': 'error',
                    'message': 'Invalid IR data received'
                })
        
        else:
            # No handler matched this topic
            print(f"⚠️  UNHANDLED TOPIC: {topic}")
            print(f"   No matching handler found for this topic!")
            print(f"   Available handlers: ac/sensors, lamp/sensors, camera/detection, ir/learned, etc.")
            
        print("─"*70 + "\n")
        
    except Exception as e:
        print(f"❌ MQTT Message Handler Error: {str(e)}")
        import traceback
        traceback.print_exc()
        log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'MQTT Error: {str(e)}', 'level': 'error'})

def on_disconnect(client, userdata, rc):
    print("\n" + "="*70)
    print("  ⚠️  FLASK MQTT DISCONNECTION EVENT")
    print("="*70)
    print(f"Disconnect Reason Code: {rc}")
    if rc != 0:
        print("❌ Unexpected disconnection!")
        print("Will attempt to reconnect...")
    else:
        print("✅ Clean disconnection")
    print("="*70 + "\n")
    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'MQTT Disconnected! RC: {rc}', 'level': 'warning'})

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

print("\n" + "="*70)
print("  🚀 INITIALIZING FLASK MQTT CLIENT")
print("="*70)
print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
print(f"Attempting connection...")
print("="*70 + "\n")

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    
    # Wait a bit for connection to establish
    import time
    time.sleep(2)
    
    if mqtt_client.is_connected():
        print("✅ MQTT Client connected and running!")
        print("📡 Loop thread started\n")
    else:
        print("⚠️  MQTT Client started but not yet connected")
        print("Waiting for on_connect callback...\n")
        
except Exception as e:
    print(f"❌ MQTT Connection Error: {e}")
    import traceback
    traceback.print_exc()

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
            fps = int(cam.get(cv2.CAP_PROP_FPS))
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
    return jsonify(mqtt_data)

@app.route('/api/occupancy/feedback', methods=['POST'])
def occupancy_feedback_submit():
    try:
        data = request.json or {}
        rating = int(data.get('rating', 0))
        comment = (data.get('comment') or '').strip()
        occupancy_count = int(data.get('occupancy_count', 0))
        google_form_url = (data.get('google_form_url') or GOOGLE_FORM_URL).strip()

        if rating < 1 or rating > 5:
            return jsonify({'status': 'error', 'message': 'Rating harus 1-5'}), 400

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

        return jsonify({'status': 'success', 'message': 'Feedback tersimpan', 'google_form_url': google_form_url})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/occupancy/feedback/list')
def occupancy_feedback_list():
    return jsonify({'feedback': list(occupancy_feedback)[:20]})

@app.route('/api/chart/<measurement>/<field>/<int:hours>')
def get_chart_data(measurement, field, hours):
    data = get_influx_data(measurement, field, hours)
    return jsonify(data)

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
    """Receive optimization data from main.py — GA→AC / PSO→Lamp"""
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
        
        print(f"📊 Optimization Update: GA={data.get('ga_fitness', 0):.2f} (AC:{mqtt_data['system']['ga_temp']}°C), PSO={data.get('pso_fitness', 0):.2f} (Lamp:{mqtt_data['system']['pso_brightness']}%)")
        
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
                    print(f"🤖 ADAPTIVE (HTTP) → Applied to AC: {opt_temp}°C Fan:{opt_fan}")
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Adaptive AC: {opt_temp}°C Fan:{opt_fan}', 'level': 'success'})
                else:
                    print(f"🤖 ADAPTIVE (HTTP) → AC apply debounced (MQTT already handled it)")
        elif mqtt_data['ac'].get('mode', 'MANUAL') == 'ADAPTIVE':
            print(f"🤖 ADAPTIVE (HTTP) → BLOCKED: No person confirmed in last {NO_PERSON_TIMEOUT_SECONDS//60} min")
        
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
                    print(f"🤖 ADAPTIVE (HTTP) → Applied to Lamp: {opt_brightness}%")
                    log_messages.append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Adaptive Lamp: {opt_brightness}%', 'level': 'success'})
                else:
                    print(f"🤖 ADAPTIVE (HTTP) → Lamp apply debounced")
        
        return jsonify({'status': 'success', 'message': 'Optimization data updated'})
    except Exception as e:
        print(f"❌ Optimization update error: {e}")
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
        print("🔴 FLASK: PUBLISHING IR LEARN COMMAND")
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
            print("✅ MQTT PUBLISH SUCCESS!")
        else:
            print(f"❌ MQTT PUBLISH FAILED! RC={result.rc}")
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
        print(f"❌ EXCEPTION in learn_ir(): {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ir/send', methods=['POST'])
def send_ir():
    try:
        data = request.json
        button_name = data.get('button', '')
        
        print("\n" + "="*70)
        print("📤 IR SEND REQUEST RECEIVED")
        print("="*70)
        print(f"Button requested: {button_name}")
        print(f"Available codes: {list(mqtt_data['ir_codes'].keys())}")
        
        if button_name not in mqtt_data['ir_codes']:
            print(f"❌ ERROR: Button '{button_name}' not found in learned codes!")
            print("="*70 + "\n")
            return jsonify({'status': 'error', 'message': 'IR code not learned yet'}), 400
        
        ir_code = mqtt_data['ir_codes'][button_name]
        
        # Verify code completeness before sending
        raw_value_count = 0
        if isinstance(ir_code, str) and ir_code.startswith('RAW:'):
            raw_part = ir_code[4:]
            raw_value_count = raw_part.count(',') + 1 if raw_part else 0
        
        print(f"✅ IR code found!")
        print(f"Code length: {len(ir_code)} chars")
        if raw_value_count > 0:
            print(f"RAW values: {raw_value_count} values")
            if raw_value_count < 100:
                print(f"⚠️  WARNING: Only {raw_value_count} RAW values - signal mungkin tidak lengkap!")
        print(f"Code preview: {ir_code[:100]}..." if len(ir_code) > 100 else f"Code: {ir_code}")
        
        # Handle toggle buttons (power ON/OFF with same code)
        action_suffix = ''
        if 'toggle' in button_name.lower() or button_name in mqtt_data['ir_states']:
            # Toggle the state
            current_state = mqtt_data['ir_states'].get(button_name, 'OFF')
            new_state = 'ON' if current_state == 'OFF' else 'OFF'
            mqtt_data['ir_states'][button_name] = new_state
            action_suffix = f' ({new_state})'
            print(f"🔄 Toggle button: {current_state} → {new_state}")
        
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
        
        print(f"\n📡 Publishing to MQTT...")
        print(f"Topic: smartroom/ir/send")
        print(f"Payload size: {len(mqtt_payload_str)} bytes")
        if verify_raw_count > 0:
            print(f"RAW values in payload: {verify_raw_count} (verified after JSON encode)")
        print(f"MQTT Connected: {mqtt_client.is_connected()}")
        
        result = mqtt_client.publish('smartroom/ir/send', mqtt_payload_str)
        
        print(f"Publish Result: {result.rc}")
        if result.rc == 0:
            print("✅ MQTT PUBLISH SUCCESS!")
            print("ESP32 should transmit IR signal NOW!")
        else:
            print(f"❌ MQTT PUBLISH FAILED! RC={result.rc}")
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
        print(f"❌ EXCEPTION in send_ir(): {e}")
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
                print(f"❌ Error updating IR codes file: {e}")
            
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
            print(f"❌ Error saving to file: {e}")
        
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
    
    # No person timeout → suggest turning off AC
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
            .sidebar-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1999; }
            .sidebar-overlay.active { display: block; }
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
                <p>Air Conditioning monitoring & status</p>
                <div id="device-status-bar" style="display: flex; gap: 15px; margin-top: 12px; flex-wrap: wrap;">
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
                        <span>Suhu terasa — Kombinasi temp &amp; humidity</span>
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
                                <i class="fas fa-thermometer-half"></i> Set Suhu
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
                                <i class="fas fa-users"></i> Jumlah Orang
                            </div>
                            <div style="font-size: 32px; font-weight: 800; color: #ef4444;" id="cam-count">0</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">person(s)</div>
                        </div>
                        <div style="text-align: center; padding: 14px 8px; background: rgba(245, 158, 11, 0.06); border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-crosshairs"></i> Confidence
                            </div>
                            <div style="font-size: 32px; font-weight: 800; color: #f59e0b;" id="cam-confidence">0%</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">akurasi deteksi</div>
                        </div>
                        <div style="text-align: center; padding: 14px 8px; background: rgba(16, 185, 129, 0.06); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.15);">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                <i class="fas fa-bolt"></i> Auto Control
                            </div>
                            <div style="font-size: 22px; font-weight: 800; color: #10b981;" id="cam-auto-status"><i class="fas fa-check-circle"></i></div>
                            <div style="font-size: 12px; color: var(--text-secondary);" id="cam-auto-label">Aktif (10 min timeout)</div>
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
                            <div style="font-size: 11px; color: var(--text-secondary);">Genetic Algorithm → AC | PSO → Lamp</div>
                        </div>
                    </div>
                    <div style="padding: 16px 20px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
                        <!-- GA Section -->
                        <div style="padding: 16px; background: rgba(16, 185, 129, 0.04); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.15);">
                            <div style="font-size: 12px; font-weight: 700; color: #10b981; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
                                <i class="fas fa-dna"></i> GA → AC Control
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
                                <i class="fas fa-lightbulb"></i> PSO → Lamp Control
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
                <p>Lighting monitoring & status</p>
                <div style="display: flex; gap: 15px; margin-top: 12px; flex-wrap: wrap;">
                    <div class="device-status-item" id="ds-esp32-lamp">
                        <span class="device-dot offline"></span>
                        <span>ESP32-Lamp</span>
                        <span class="device-time" id="ds-lamp-time">Never</span>
                    </div>
                </div>
            </div>

            <div class="stats-grid dashboard-grid">
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
                        <span class="stat-title">PSO → Lamp Control</span>
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
                <h1><i class="fas fa-video"></i> Live Camera Feed - YOLOv8 Detection</h1>
                <p>Real-time person detection menggunakan YOLOv8n</p>
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
                            Pastikan kamera USB terhubung dengan benar
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
                <h1>Energy Usage</h1>
                <p>Pemakaian energi AC dan Lamp dalam kWh</p>
            </div>

            <div class="power-grid">
                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">AC Energy / Hari</div>
                    <div class="power-value"><span id="ac-energy-kwh">0</span> kWh</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Power: <span id="ac-power">0</span> W</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Lamp Energy / Hari</div>
                    <div class="power-value"><span id="lamp-energy-kwh">0</span> kWh</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Power: <span id="lamp-power">0</span> W</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Total Energy / Hari</div>
                    <div class="power-value"><span id="total-energy-kwh">0</span> kWh</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Total Power: <span id="total-power">0</span> W</div>
                </div>

                <div class="power-card">
                    <div style="color: var(--text-secondary); margin-bottom: 10px;">Daily Cost</div>
                    <div class="power-value">Rp<span id="daily-cost">0</span></div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">@ Rp 1,500/kWh</div>
                </div>
            </div>

            <div class="chart-container" style="margin-top: 30px;">
                <div class="chart-header">
                    <div class="chart-title">Energy Trend (kWh/day equivalent)</div>
                </div>
                <canvas id="energyChart" height="80"></canvas>
            </div>
        </div>

        <!-- Control Panel Page -->
        <!-- AC Control Page -->
        <div id="control-ac" class="page">
            <div class="header">
                <h1><i class="fas fa-snowflake"></i> AC Control Panel</h1>
                <p>Kontrol AC menggunakan IRMitsubishiHeavy library</p>
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
                        <span style="font-size: 11px; font-weight: 400; opacity: 0.9;">AI mengatur AC otomatis</span>
                    </button>
                    <button id="btn-mode-manual" onclick="setACMode('MANUAL')" style="padding: 18px 16px; border-radius: 14px; border: 3px solid var(--border); background: var(--bg-card); color: var(--text-secondary); font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; align-items: center; gap: 6px; opacity: 0.6;">
                        <i class="fas fa-hand-paper" style="font-size: 28px;"></i>
                        <span>MANUAL</span>
                        <span style="font-size: 11px; font-weight: 400; opacity: 0.9;">Atur AC sendiri</span>
                    </button>
                </div>
                <!-- Current mode indicator -->
                <div id="ac-mode-indicator" style="margin-top: 14px; padding: 10px 16px; border-radius: 10px; text-align: center; font-size: 13px; font-weight: 600; background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);">
                    <i class="fas fa-robot"></i> Mode saat ini: <strong>ADAPTIVE</strong> — AC dikontrol otomatis oleh AI (GA optimization)
                </div>
            </div>

            <!-- ========== ADAPTIVE INFO BANNER (shown when ADAPTIVE) ========== -->
            <div id="adaptive-info-banner" style="margin-bottom: 20px; padding: 20px; border-radius: 14px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.12)); border: 2px solid rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, #10b981, #059669); display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-brain" style="color: white; font-size: 22px;"></i>
                    </div>
                    <div>
                        <div style="font-size: 16px; font-weight: 700; color: #10b981;">Mode Adaptive Aktif</div>
                        <div style="font-size: 12px; color: var(--text-secondary);">GA Optimization mengontrol suhu & fan speed secara otomatis</div>
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
                    <i class="fas fa-lock"></i> Kontrol manual dinonaktifkan saat mode Adaptive aktif. Beralih ke Manual untuk mengatur AC secara manual.
                </div>
            </div>

            <div class="control-panel" style="position: relative;">
                <!-- OVERLAY: blocks manual controls when ADAPTIVE -->
                <div id="ac-manual-overlay" style="display: block; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.45); backdrop-filter: blur(3px); border-radius: 12px; z-index: 10; display: flex; align-items: center; justify-content: center; cursor: not-allowed;">
                    <div style="text-align: center; color: white;">
                        <i class="fas fa-lock" style="font-size: 36px; margin-bottom: 10px; opacity: 0.8;"></i>
                        <div style="font-size: 14px; font-weight: 600;">Mode Adaptive Aktif</div>
                        <div style="font-size: 12px; opacity: 0.7;">Beralih ke Manual untuk menggunakan kontrol ini</div>
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
                    <i class="fas fa-info-circle"></i> Menggunakan IRMitsubishiHeavy library — tidak perlu learning manual.
                </div>
            </div>
        </div>

        <!-- Lamp Control Page -->
        <div id="control-lamp" class="page">
            <div class="header">
                <h1><i class="fas fa-lightbulb"></i> Lamp Control Panel</h1>
                <p>Manual lamp control and brightness settings</p>
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

        <!-- ML Optimization Page -->
        <div id="ml-optimization" class="page">
            <div class="header">
                <h1><i class="fas fa-brain"></i> Machine Learning Optimization</h1>
                <p>GA → Adaptive AC | PSO → Adaptive Lamp | Data from InfluxDB</p>
            </div>

            <!-- ML Summary Cards -->
            <div class="stats-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-title">GA → AC</span>
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
                        <span class="stat-title">PSO → Lamp</span>
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
                        🌡️ <span id="ml-cur-temp">--</span>°C &nbsp; 💧 <span id="ml-cur-hum">--</span>%<br>
                        💡 <span id="ml-cur-lux">--</span> lux &nbsp; 👤 <span id="ml-cur-person">--</span>
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
                            <input type="number" id="ml-ga-pop" value="20" min="5" max="100" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>Generations</label>
                            <input type="number" id="ml-ga-gen" value="50" min="10" max="500" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>Mutation Rate</label>
                            <input type="number" id="ml-ga-mut" value="0.1" min="0.01" max="0.5" step="0.01" class="ml-input">
                        </div>
                        <div class="ml-param-item">
                            <label>Crossover Rate</label>
                            <input type="number" id="ml-ga-cross" value="0.7" min="0.1" max="1.0" step="0.05" class="ml-input">
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
            </div>

            <div class="log-container" id="log-container">
            </div>
        </div>

        <!-- Occupancy Trend & Feedback Page -->
        <div id="occupancy-feedback" class="page">
            <div class="header">
                <h1><i class="fas fa-users"></i> Occupancy Trend & Feedback</h1>
                <p>Pantau tren okupansi dan kirim penilaian kenyamanan ruangan (1-5)</p>
            </div>

            <div class="occupancy-top">
                <div class="occupancy-kpi">
                    <div class="kpi-label">Occupancy Saat Ini</div>
                    <div class="kpi-value" id="occ-live-count">0</div>
                    <div class="occupancy-mini-note">Orang terdeteksi saat ini</div>
                    <div class="occupancy-mini-note">Confidence: <span id="occ-live-confidence">0%</span></div>
                </div>

                <div class="occupancy-chart-card">
                    <div class="chart-header" style="margin-bottom: 10px;">
                        <div class="chart-title">Occupancy Trend</div>
                        <div class="chart-options">
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
                        <div class="chart-title"><i class="fas fa-star"></i> Feedback Kenyamanan</div>
                    </div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;">
                        Skala: 1 = Tidak Puas, 5 = Sangat Puas
                    </div>
                    <div class="rating-row" id="rating-row" style="margin-top: 14px;">
                        <button class="rating-btn" onclick="selectFeedbackRating(1)" title="Tidak puas">1</button>
                        <button class="rating-btn" onclick="selectFeedbackRating(2)" title="Kurang puas">2</button>
                        <button class="rating-btn" onclick="selectFeedbackRating(3)" title="Cukup">3</button>
                        <button class="rating-btn" onclick="selectFeedbackRating(4)" title="Puas">4</button>
                        <button class="rating-btn" onclick="selectFeedbackRating(5)" title="Sangat puas">5</button>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:12px; color:var(--text-secondary);">
                        <span>Tidak Puas</span>
                        <span>Sangat Puas</span>
                    </div>
                    <textarea id="feedback-comment" class="feedback-input" rows="4" placeholder="Masukkan feedback pengguna ruangan..."></textarea>
                    <input id="google-form-url" class="feedback-input" placeholder="Masukkan URL Google Form Anda" />

                    <div style="display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap;">
                        <button class="btn btn-primary" onclick="saveGoogleFormUrl()">
                            <i class="fas fa-save"></i> Simpan Link Form
                        </button>
                        <button class="btn btn-success" onclick="openGoogleForm()">
                            <i class="fas fa-external-link-alt"></i> Buka Google Form
                        </button>
                        <button class="btn btn-warning" onclick="submitOccupancyFeedback()">
                            <i class="fas fa-paper-plane"></i> Kirim Feedback
                        </button>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title"><i class="fas fa-history"></i> Feedback Terbaru</div>
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
        const socket = io();
        
        let charts = {};
        let chartRanges = {
            temp: 1,
            hum: 1,
            acTemp: 1,
            lampLux: 1,
            lampBright: 1,
            occupancy: 1
        };

        let selectedFeedbackRating = 0;
        const DEFAULT_GOOGLE_FORM_URL = 'https://docs.google.com/forms/';

        let learnedCodes = {};

        // ==================== LOCALSTORAGE PERSISTENCE ====================
        function saveSettings() {
            const settings = {
                acTemp: document.getElementById('ac-temp-slider')?.value || 24,
                fanSpeed: document.getElementById('fan-speed-slider')?.value || 1,
                lampBrightness: document.getElementById('brightness-slider')?.value || 0
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
                    const brightnessSlider = document.getElementById('brightness-slider');
                    
                    if (acTempSlider) {
                        acTempSlider.value = settings.acTemp || 24;
                        document.getElementById('ac-temp-display').textContent = acTempSlider.value;
                    }
                    
                    if (fanSpeedSlider) {
                        fanSpeedSlider.value = settings.fanSpeed || 1;
                        document.getElementById('fan-speed-display').textContent = fanSpeedSlider.value;
                    }
                    
                    if (brightnessSlider) {
                        brightnessSlider.value = settings.lampBrightness || 0;
                        document.getElementById('brightness-display').textContent = brightnessSlider.value;
                    }
                } catch (e) {
                    console.error('Error loading settings:', e);
                }
            }
        }

        // ==================== CHARTS ====================
        function initCharts() {
            const chartConfig = {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: false, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } }
                    }
                }
            };

            charts.temp = new Chart(document.getElementById('tempChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'Temperature (°C)', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', tension: 0.4, fill: true }] }
            });

            charts.hum = new Chart(document.getElementById('humChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'Humidity (%)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.4, fill: true }] }
            });

            charts.acTemp = new Chart(document.getElementById('acTempChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'AC Target Temp (°C)', data: [], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)', tension: 0.4, fill: true }] }
            });

            charts.lampLux = new Chart(document.getElementById('lampLuxChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'Light Intensity (lux)', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.4, fill: true }] }
            });

            charts.lampBright = new Chart(document.getElementById('lampBrightChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'Brightness (%)', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', tension: 0.4, fill: true }] }
            });

            charts.energy = new Chart(document.getElementById('energyChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'Total Energy (kWh/day)', data: [], borderColor: '#a855f7', backgroundColor: 'rgba(168,85,247,0.1)', tension: 0.4, fill: true }] }
            });

            charts.occupancy = new Chart(document.getElementById('occupancyChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'Occupancy (person)', data: [], borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.15)', tension: 0.35, fill: true }] }
            });

            // ML Optimization Charts
            charts.gaFitness = new Chart(document.getElementById('gaFitnessChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'GA Best Fitness', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.15)', tension: 0.4, fill: true, pointRadius: 2 }] }
            });

            charts.psoFitness = new Chart(document.getElementById('psoFitnessChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'PSO Best Fitness', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.15)', tension: 0.4, fill: true, pointRadius: 2 }] }
            });

            charts.comparison = new Chart(document.getElementById('comparisonChart'), {
                ...chartConfig,
                options: {
                    ...chartConfig.options,
                    plugins: { legend: { display: true, labels: { color: '#94a3b8' } } }
                },
                data: {
                    labels: [],
                    datasets: [
                        { label: 'GA (AC)', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', tension: 0.4, fill: false, pointRadius: 3 },
                        { label: 'PSO (Lamp)', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.4, fill: false, pointRadius: 3 }
                    ]
                }
            });
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
            document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            
            document.getElementById(pageId).classList.add('active');
            event.target.closest('.nav-item').classList.add('active');
            
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
                checkCameraStatus();
            }
            if (pageId === 'ml-optimization') {
                refreshMLData();
            }
            if (pageId === 'occupancy-feedback') {
                updateChartData('occupancy', chartRanges.occupancy || 1);
                loadFeedbackHistory();
            }
        }

        // ==================== ML OPTIMIZATION ====================
        let mlHistory = [];
        let mlRunCount = 0;

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
                        if (tempEl) tempEl.textContent = d.ac?.temperature?.toFixed(1) || '--';
                        if (humEl) humEl.textContent = d.ac?.humidity?.toFixed(1) || '--';
                        if (luxEl) luxEl.textContent = d.lamp?.lux || '--';
                        if (personEl) personEl.textContent = d.camera?.person_detected ? 'Yes' : 'No';
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

            chart.data.labels = history.map((_, i) => algo === 'GA' ? `Gen ${i+1}` : `Iter ${i+1}`);
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
                return `<tr>
                    <td>${e.run}</td>
                    <td>${e.time}</td>
                    <td><span class="ml-badge ${getBadge(e.ga_fitness)}">${e.ga_fitness.toFixed(2)}</span></td>
                    <td>${e.ga_temp}\u00b0C</td>
                    <td>${e.ga_fan}</td>
                    <td><span class="ml-badge ${getBadge(e.pso_fitness)}">${e.pso_fitness.toFixed(2)}</span></td>
                    <td>${e.pso_brightness}%</td>
                    <td><span class="ml-badge ${getBadge(e.combined)}">${e.combined.toFixed(2)}</span></td>
                </tr>`;
            }).join('');
        }

        function refreshMLHistory() {
            refreshMLData();
            showToast('ML data refreshed', 'success');
        }

        function getMLParams(algo) {
            if (algo === 'ga') {
                return {
                    population_size: parseInt(document.getElementById('ml-ga-pop')?.value) || 20,
                    generations: parseInt(document.getElementById('ml-ga-gen')?.value) || 50,
                    mutation_rate: parseFloat(document.getElementById('ml-ga-mut')?.value) || 0.1,
                    crossover_rate: parseFloat(document.getElementById('ml-ga-cross')?.value) || 0.7
                };
            } else {
                return {
                    swarm_size: parseInt(document.getElementById('ml-pso-swarm')?.value) || 30,
                    iterations: parseInt(document.getElementById('ml-pso-iter')?.value) || 100,
                    w: parseFloat(document.getElementById('ml-pso-w')?.value) || 0.7,
                    c: parseFloat(document.getElementById('ml-pso-c')?.value) || 1.5
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
        let cameraEnabled = true;

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
                        error.querySelector('p').textContent = 'Kamera dimatikan. Klik tombol Camera ON untuk mengaktifkan.';
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
                    showToast('Mode Adaptive aktif! Beralih ke Manual untuk kontrol manual.', 'error');
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

        let selectedACMode = 'COOL';

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
                    showToast('Mode Adaptive aktif! Beralih ke Manual untuk kontrol manual.', 'error');
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
        function updateBrightness(value) {
            document.getElementById('brightness-display').textContent = value;
            saveSettings();
        }

        function applyLampSettings() {
            const brightness = document.getElementById('brightness-slider').value;
            
            fetch('/api/lamp/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ brightness: parseInt(brightness) })
            })
            .then(r => r.json())
            .then(result => showToast('Lamp brightness set!'))
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
                    indicator.innerHTML = '<i class="fas fa-robot"></i> Mode saat ini: <strong>ADAPTIVE</strong> — AC dikontrol otomatis oleh AI (GA optimization)';
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
                    indicator.innerHTML = '<i class="fas fa-hand-paper"></i> Mode saat ini: <strong>MANUAL</strong> — Atur AC secara manual menggunakan tombol di bawah';
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
        let currentLearningButton = null;
        let learningCheckInterval = null;
        
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
                    showToast('Mode Adaptive aktif! Beralih ke Manual untuk kontrol manual.', 'error');
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
                            statusElement.textContent = 'Learned ✓';
                            statusElement.style.color = '#10b981';
                        }
                    });
                    
                    if (codeCount > 0) {
                        showToast(`IR codes loaded: ${codeCount} button(s)`, 'success');
                    }
                })
                .catch(e => {
                    console.error('Error loading IR codes:', e);
                    showToast('Error loading IR codes', 'error');
                });
        }

        function resetAllIRCodes() {
            if (!confirm('⚠️ Reset all learned IR codes?\\n\\nThis will delete ALL saved remote buttons!')) {
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
                    
                    showToast(`${count} IR codes saved to server`, 'success');
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
                    
                    showToast(`Exported ${count} IR codes`, 'success');
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
                        debugInfo += '⚠️ No IR codes learned yet\\n\\n';
                        debugInfo += 'Steps to learn:\\n';
                        debugInfo += '1. Select protocol (RAW recommended for Mitsubishi)\\n';
                        debugInfo += '2. Click \"Learn\" button\\n';
                        debugInfo += '3. Press remote button (hold 2-3 seconds)\\n';
                        debugInfo += '4. Wait for \"Learned ✓\" status\\n';
                        debugInfo += '5. Click \"Send\" to test\\n';
                    } else {
                        Object.keys(codes).forEach(button => {
                            const codeStr = codes[button];
                            const codePreview = codeStr.substring(0, 60) + (codeStr.length > 60 ? '...' : '');
                            debugInfo += `📡 ${button}:\\n`;
                            debugInfo += `   Code: ${codePreview}\\n`;
                            debugInfo += `   Length: ${codeStr.length} chars\\n`;
                            
                            // Parse protocol
                            if (codeStr.includes(':')) {
                                const protocol = codeStr.split(':')[0];
                                debugInfo += `   Protocol: ${protocol}\\n`;
                            }
                            debugInfo += '\\n';
                        });
                        
                        debugInfo += '═══════════════════════════════════\\n';
                        debugInfo += `Total: ${Object.keys(codes).length} codes\\n`;
                    }
                    
                    debugInfo += '\\n💡 TROUBLESHOOTING:\\n';
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
                    
                    showToast(`Testing ${buttons.length} codes...`, 'info');
                    
                    let index = 0;
                    const testInterval = setInterval(() => {
                        if (index >= buttons.length) {
                            clearInterval(testInterval);
                            showToast('Test complete!', 'success');
                            return;
                        }
                        
                        const button = buttons[index];
                        console.log(`Testing ${button}...`);
                        sendIRCode(button);
                        showToast(`Testing: ${button}`, 'info');
                        
                        index++;
                    }, 2000); // 2 second delay between tests
                })
                .catch(e => showToast('Test error: ' + e, 'error'));
        }

        // ==================== DETECTION ALERT ====================
        let lastDetectionTime = 0;
        const DETECTION_COOLDOWN = 60000; // 60 seconds (1 minute) between alerts
        let detectionSoundEnabled = localStorage.getItem('detectionSound') !== 'false';

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
            
            console.log('🚨 PERSON DETECTED:', {
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
                showToast('🔊 Sound alerts ON — anda akan mendengar suara saat orang terdeteksi', 'success');
            } else {
                showToast('🔇 Sound alerts OFF — notifikasi suara dimatikan', 'info');
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
                showToast('Masukkan URL Google Form terlebih dahulu', 'error');
                return;
            }
            localStorage.setItem('googleFormUrl', url);
            showToast('URL Google Form tersimpan', 'success');
        }

        function openGoogleForm() {
            const url = (document.getElementById('google-form-url').value || '').trim() || localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;
            window.open(url, '_blank');
        }

        function submitOccupancyFeedback() {
            const comment = (document.getElementById('feedback-comment').value || '').trim();
            const formUrl = (document.getElementById('google-form-url').value || '').trim() || localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;

            if (!selectedFeedbackRating) {
                showToast('Pilih rating 1-5 terlebih dahulu', 'error');
                return;
            }

            fetch('/api/occupancy/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rating: selectedFeedbackRating,
                    comment: comment,
                    google_form_url: formUrl,
                    occupancy_count: parseInt(document.getElementById('cam-count')?.textContent || '0', 10)
                })
            })
            .then(r => r.json())
            .then(result => {
                if (result.status === 'success') {
                    showToast('Feedback berhasil dikirim', 'success');
                    document.getElementById('feedback-comment').value = '';
                    selectFeedbackRating(0);
                    loadFeedbackHistory();
                    if (formUrl && formUrl !== DEFAULT_GOOGLE_FORM_URL) {
                        window.open(formUrl, '_blank');
                    }
                } else {
                    showToast(result.message || 'Gagal kirim feedback', 'error');
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
                        container.innerHTML = '<div style="color: var(--text-secondary); font-size: 13px;">Belum ada feedback.</div>';
                        return;
                    }

                    container.innerHTML = rows.map(item => {
                        const safeComment = (item.comment || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        return `
                            <div class="feedback-history-item">
                                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                    <strong>Rating: ${item.rating}/5</strong>
                                    <span style="font-size:12px; color: var(--text-secondary);">${item.time}</span>
                                </div>
                                <div style="font-size: 13px; color: var(--text-secondary);">Occupancy: ${item.occupancy_count} person(s)</div>
                                <div style="margin-top:6px; font-size:13px;">${safeComment || '-'}</div>
                            </div>
                        `;
                    }).join('');
                })
                .catch(() => {});
        }

        // ==================== DATA UPDATES ====================
        function updateDashboard() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    const temperature = data.ac.temperature;
                    document.getElementById('dash-temp').textContent = temperature.toFixed(1);
                    document.getElementById('dash-hum').textContent = data.ac.humidity.toFixed(1);
                    // Individual sensor readings
                    const t1El = document.getElementById('dash-temp1');
                    const t2El = document.getElementById('dash-temp2');
                    const t3El = document.getElementById('dash-temp3');
                    if (t1El) t1El.textContent = (data.ac.temp1 || 0).toFixed(1);
                    if (t2El) t2El.textContent = (data.ac.temp2 || 0).toFixed(1);
                    if (t3El) t3El.textContent = (data.ac.temp3 || 0).toFixed(1);
                    // Individual humidity readings
                    const h1El = document.getElementById('dash-hum1');
                    const h2El = document.getElementById('dash-hum2');
                    const h3El = document.getElementById('dash-hum3');
                    if (h1El) h1El.textContent = (data.ac.hum1 || 0).toFixed(1);
                    if (h2El) h2El.textContent = (data.ac.hum2 || 0).toFixed(1);
                    if (h3El) h3El.textContent = (data.ac.hum3 || 0).toFixed(1);
                    
                    // Heat Index
                    const hiEl = document.getElementById('dash-heat-index');
                    if (hiEl) hiEl.textContent = (data.ac.heat_index || 0).toFixed(1);
                    
                    // ESP32 Signal & Uptime
                    const rssiEl = document.getElementById('dash-rssi');
                    if (rssiEl) rssiEl.textContent = data.ac.rssi || 0;
                    const uptEl = document.getElementById('dash-uptime');
                    if (uptEl) {
                        const secs = data.ac.uptime || 0;
                        const h = Math.floor(secs / 3600);
                        const m = Math.floor((secs % 3600) / 60);
                        uptEl.textContent = h > 0 ? h + 'h ' + m + 'm' : m + 'm ' + (secs % 60) + 's';
                    }
                    
                    // AC State - use REAL state from ESP32, not temperature guessing
                    const acStateEl = document.getElementById('dash-ac-state');
                    let acState = data.ac.ac_state || 'OFF';
                    
                    if (acState === 'ON') {
                        acStateEl.style.color = '#10b981'; // Green
                    } else {
                        acStateEl.style.color = '#ef4444'; // Red
                    }
                    
                    acStateEl.textContent = acState;
                    
                    // AC panel power dot
                    const panelDot = document.getElementById('ac-panel-dot');
                    if (panelDot) {
                        panelDot.style.background = acState === 'ON' ? '#10b981' : '#ef4444';
                        panelDot.style.boxShadow = acState === 'ON' ? '0 0 8px rgba(16,185,129,0.5)' : '0 0 8px rgba(239,68,68,0.5)';
                    }
                    
                    // Set Temperature
                    const acTemp = data.ac.ac_temp || 24;
                    document.getElementById('dash-ac-temp').textContent = acTemp;
                    
                    // Fan Speed with label
                    const fanSpeed = data.ac.fan_speed || 1;
                    document.getElementById('dash-ac-fan').textContent = fanSpeed;
                    const fanLabel = document.getElementById('dash-ac-fan-label');
                    if (fanLabel) {
                        const fanNames = {1: 'Low', 2: 'Medium', 3: 'High'};
                        fanLabel.textContent = fanNames[fanSpeed] || 'Level ' + fanSpeed;
                    }
                    
                    // AC Mode (COOL/HEAT/DRY/FAN/AUTO) with icon + color — uses ac_fan_mode from ESP32
                    const acMode = data.ac.ac_fan_mode || 'COOL';
                    document.getElementById('dash-ac-mode').textContent = acMode;
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
                    const ctrlMode = data.ac.mode || 'ADAPTIVE';
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
                    if (roomHum) roomHum.textContent = data.ac.humidity.toFixed(1);
                    // Individual sensors in AC panel footer
                    const acT1 = document.getElementById('dash-ac-temp1');
                    const acT2 = document.getElementById('dash-ac-temp2');
                    const acT3 = document.getElementById('dash-ac-temp3');
                    if (acT1) acT1.textContent = (data.ac.temp1 || 0).toFixed(1);
                    if (acT2) acT2.textContent = (data.ac.temp2 || 0).toFixed(1);
                    if (acT3) acT3.textContent = (data.ac.temp3 || 0).toFixed(1);
                    
                    // Update AC Live Status Bar in control panel
                    const liveDot = document.getElementById('ac-live-dot');
                    const liveState = document.getElementById('ac-live-state');
                    if (liveDot && liveState) {
                        liveState.textContent = acState;
                        liveDot.style.background = acState === 'ON' ? '#10b981' : '#ef4444';
                        document.getElementById('ac-live-temp').textContent = acTemp;
                        document.getElementById('ac-live-fan').textContent = fanSpeed;
                        document.getElementById('ac-live-mode').textContent = acMode;
                    }
                    document.getElementById('dash-lux').textContent = data.lamp.lux.toFixed(0);
                    document.getElementById('dash-brightness').textContent = Math.round(data.lamp.brightness / 255 * 100);
                    document.getElementById('dash-motion').textContent = data.lamp.motion ? 'MOTION DETECTED' : 'NO MOTION';
                    
                    const personDetected = data.camera.person_detected;
                    const personCount = data.camera.count || 0;
                    const confidence = data.camera.confidence || 0;
                    
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
                    const gaFitness = parseFloat(data.system.ga_fitness) || 0;
                    const psoFitness = parseFloat(data.system.pso_fitness) || 0;
                    
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
                        const psoBright = data.system.pso_brightness || 0;
                        psoBrightEl.textContent = Math.round(psoBright / 255 * 100);
                    }
                    
                    // Optimization Runs
                    const optRunsEl = document.getElementById('dash-opt-runs');
                    if (optRunsEl) {
                        optRunsEl.textContent = data.system.optimization_runs || 0;
                    }
                    
                    // Calculate AC power based on temperature logic
                    let acPower = 0;
                    const actualACState = temperature < 30 ? 'ON' : 'OFF';
                    
                    if (actualACState === 'ON') {
                        acPower = data.ac.fan_speed === 1 ? 100 : (data.ac.fan_speed === 2 ? 200 : 300);
                    }
                    let lampPower = (data.lamp.brightness / 255) * 10;
                    let totalPower = acPower + lampPower;
                    let acEnergyKwh = (acPower / 1000) * 24;
                    let lampEnergyKwh = (lampPower / 1000) * 24;
                    let totalEnergyKwh = acEnergyKwh + lampEnergyKwh;
                    let dailyCost = (totalPower / 1000) * 24 * 1500;
                    
                    document.getElementById('ac-power').textContent = acPower.toFixed(0);
                    document.getElementById('lamp-power').textContent = lampPower.toFixed(1);
                    document.getElementById('total-power').textContent = totalPower.toFixed(1);
                    document.getElementById('ac-energy-kwh').textContent = acEnergyKwh.toFixed(2);
                    document.getElementById('lamp-energy-kwh').textContent = lampEnergyKwh.toFixed(2);
                    document.getElementById('total-energy-kwh').textContent = totalEnergyKwh.toFixed(2);
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
                    
                    updateModeBadges();
                });
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
        function updateDeviceStatus() {
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
        let alertQueue = [];
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
            
            // Real-time optimization (GA→AC / PSO→Lamp) updates
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
                
                console.log('📊 Optimization Update:', {
                    ga: gaFitness.toFixed(2), ac_temp: gaTemp, ac_fan: gaFan,
                    pso: psoFitness.toFixed(2), lamp_brightness: psoBrightness,
                    runs: runs
                });
                
                // Show toast with specific results
                if (gaFitness > 0 && psoFitness > 0) {
                    showToast(`GA→AC: ${gaTemp}°C (${gaFitness.toFixed(1)}) | PSO→Lamp: ${psoBrightness}% (${psoFitness.toFixed(1)})`, 'success');
                } else if (gaFitness > 0) {
                    showToast(`GA→AC: ${gaTemp}°C Fan:${gaFan} (Fitness: ${gaFitness.toFixed(2)})`, 'success');
                } else if (psoFitness > 0) {
                    showToast(`PSO→Lamp: ${psoBrightness}% (Fitness: ${psoFitness.toFixed(2)})`, 'success');
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
                        overlayBadge.innerHTML = `<i class="fas fa-user-check"></i> ${personCount} Person(s) - ${confidence}%`;
                    } else {
                        overlayBadge.className = 'person-badge not-detected';
                        overlayBadge.innerHTML = '<i class="fas fa-user-slash"></i> No Person';
                    }
                }
                
                // Show alert when person detected
                if (personDetected && personCount > 0) {
                    showDetectionAlert(personCount, confidence);
                }
                
                console.log('📹 Camera Update:', {
                    count: personCount,
                    confidence: confidence,
                    detected: personDetected
                });
            }
        });

        // ML Optimization status from main.py
        socket.on('ml_status', function(data) {
            console.log('🤖 ML Status:', data);
            const status = data.status || '';
            const algo = (data.algorithm || '').toUpperCase();
            
            if (status === 'running') {
                showToast(`🚀 ${algo} optimization running...`, 'success');
                // Disable run buttons while running
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                });
            } else if (status === 'completed') {
                showToast(`✅ ${algo} optimization completed! GA: ${(data.ga_fitness || 0).toFixed(2)}, PSO: ${(data.pso_fitness || 0).toFixed(2)}`, 'success');
                // Re-enable run buttons
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                });
                // Refresh ML data
                refreshMLData();
            } else if (status === 'error') {
                showToast(`❌ ${algo} error: ${data.message || 'Unknown error'}`, 'error');
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                });
            } else if (status === 'busy') {
                showToast(`⏳ Optimization already in progress`, 'warning');
            }
        });

        socket.on('ir_learned', function(data) {
            console.log('🎉 IR Learned event received:', data);
            
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
                showToast('❌ IR Learning failed: ' + (data.message || 'Unknown error'), 'error');
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
                statusElement.textContent = 'Learned ✓';
                statusElement.style.color = '#10b981';
            }
            
            let message = '✅ IR Code learned: ' + buttonName;
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
                console.log('⚠️ IR-related MQTT message while in learning mode!');
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
                const navItem = document.querySelector('[onclick="showPage(\\''+savedPage+'\\')"]');
                if (navItem) navItem.classList.add('active');
            }
            
            const savedRanges = localStorage.getItem('chartRanges');
            if (savedRanges) chartRanges = JSON.parse(savedRanges);
        }

        window.onload = function() {
            console.log('🚀 Smart Room Dashboard Loading...');
            initCharts();
            loadSavedPreferences();
            loadSavedSettings();
            updateSoundToggleUI();
            updateCameraToggleUI();
            updateDashboard();
            updateDeviceStatus();
            updateLogs();
            checkCameraStatus();
            const googleFormInput = document.getElementById('google-form-url');
            if (googleFormInput) {
                googleFormInput.value = localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;
            }
            loadFeedbackHistory();
            
            // Initialize AC mode UI on page load
            updateModeBadges();
            
            Object.keys(chartRanges).forEach(chartName => {
                updateChartData(chartName, chartRanges[chartName]);
            });
            
            setInterval(updateDashboard, 1000);
            setInterval(updateDeviceStatus, 5000);
            setInterval(updateLogs, 5000);
            
            setInterval(() => {
                Object.keys(chartRanges).forEach(chartName => {
                    updateChartData(chartName, chartRanges[chartName]);
                });
            }, 30000);
            
            console.log('✅ Dashboard Ready!');
        };
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("=" * 60)
    print("  🏠 Smart Room Dashboard - YOLOv8n + Auto AC Control")
    print("=" * 60)
    
    # Load saved IR codes from file
    print("  📡 Loading saved IR codes...")
    try:
        import os
        ir_file = os.path.join(os.path.dirname(__file__), 'ir_codes.json')
        if os.path.exists(ir_file):
            with open(ir_file, 'r') as f:
                mqtt_data['ir_codes'] = json.load(f)
            print(f"  ✅ Loaded {len(mqtt_data['ir_codes'])} IR codes from file")
            # Verify each code's completeness
            for btn_name, code in mqtt_data['ir_codes'].items():
                if isinstance(code, str) and code.startswith('RAW:'):
                    raw_count = code[4:].count(',') + 1
                    status = '✅' if raw_count >= 100 else '⚠️ '
                    print(f"    {status} {btn_name}: RAW {raw_count} values, {len(code)} chars")
                else:
                    print(f"    📡 {btn_name}: {len(code)} chars")
        else:
            print("  ℹ️  No saved IR codes found")
    except Exception as e:
        print(f"  ⚠️  Error loading IR codes: {e}")
    
    print("  📥 Loading YOLO model (please wait)...")
    
    # Load YOLO SYNCHRONOUSLY
    yolo_loaded = load_yolo_model()
    
    if yolo_loaded:
        print("  ✅ YOLO ready for person detection!")
    else:
        print("  ⚠️  YOLO failed to load, running without detection")
    
    # Start background camera detection thread (runs 24/7 for auto ON/OFF)
    print("  🎥 Starting background camera detection thread...")
    detection_thread = threading.Thread(target=camera_detection_loop, daemon=True)
    detection_thread.start()
    print("  ✅ Detection thread running — auto ON/OFF active")
    
    print("=" * 60)
    print("  🌐 Dashboard URL: http://172.20.0.65:5000")
    print("  📹 Video Feed:    http://172.20.0.65:5000/video_feed")
    print("  ✨ Features:")
    print("     - YOLOv8n Person Detection (background thread)")
    print("     - Auto ON: 3 frames (~2s) person confirmed → AC ON")
    print("     - Auto OFF: 10 min no person → AC OFF")
    print("     - 4K Camera (fallback 1080p)")
    print("     - Real-time Person Count & Confidence")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
