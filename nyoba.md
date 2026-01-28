# mqtt_server.py - MQTT Server di Raspberry Pi

import json
import time
import paho.mqtt.client as mqtt
from datetime import datetime
from controller import ACController
from logger import DataLogger
from display import OLEDDisplay

# MQTT Settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_SENSOR = "smartac/sensor"      # Data dari ESP32
TOPIC_COMMAND = "smartac/command"    # Perintah ke ESP32
TOPIC_STATUS = "smartac/status"      # Status dari ESP32

# Global variables
controller = None
logger = None
display = None
mqtt_client = None
last_data = {
    'temp': 25.0,
    'humidity': 50.0,
    'presence': False,
    'ac_state': 'OFF'
}

def on_connect(client, userdata, flags, rc):
    """Callback saat terhubung ke MQTT broker"""
    if rc == 0:
        print("[MQTT] Connected to broker!")
        client.subscribe(TOPIC_SENSOR)
        client.subscribe(TOPIC_STATUS)
        print(f"[MQTT] Subscribed to: {TOPIC_SENSOR}, {TOPIC_STATUS}")
    else:
        print(f"[MQTT] Connection failed, code: {rc}")

def on_message(client, userdata, msg):
    """Callback saat menerima pesan"""
    global last_data
    
    topic = msg.topic
    payload = msg.payload.decode()
    
    try:
        data = json.loads(payload)
        
        if topic == TOPIC_SENSOR:
            # Data sensor dari ESP32
            temp = data.get('temp', 25.0)
            humidity = data.get('humidity', 50.0)
            presence = data.get('presence', False)
            
            last_data['temp'] = temp
            last_data['humidity'] = humidity
            last_data['presence'] = presence
            
            now = datetime.now().strftime("%H:%M:%S")
            pir_status = "ADA" if presence else "TIDAK ADA"
            print(f"[{now}] T:{temp:.1f}C H:{humidity:.0f}% PIR:{pir_status}")
            
            # Proses dengan controller
            if controller:
                presence_score = 100 if presence else 0
                action, target_temp, reason = controller.decide(presence_score, temp, humidity)
                
                # Kirim perintah ke ESP32
                if action in ["ON", "OFF", "ECO"]:
                    command = {
                        'action': action,
                        'target_temp': target_temp,
                        'reason': reason
                    }
                    client.publish(TOPIC_COMMAND, json.dumps(command))
                    print(f"[COMMAND] {action} - {reason}")
                
                # Log data
                if logger:
                    comfort = "OK"
                    logger.log(temp, humidity, presence_score, 
                              controller.get_state(), target_temp, comfort)
            
            # Update OLED
            if display:
                display.show_status(temp, humidity, pir_status, 
                                   controller.get_state() if controller else "OFF", 
                                   target_temp if controller else 25)
        
        elif topic == TOPIC_STATUS:
            # Status dari ESP32
            print(f"[STATUS] {data}")
            last_data['ac_state'] = data.get('ac_state', 'OFF')
    
    except Exception as e:
        print(f"[ERROR] {e}")

def main():
    global controller, logger, display, mqtt_client
    
    print("=" * 50)
    print("   SMART AC - MQTT SERVER")
    print("   Raspberry Pi")
    print("=" * 50)
    
    # Inisialisasi komponen
    try:
        display = OLEDDisplay()
        display.show_startup()
        time.sleep(2)
        print("  OK OLED")
    except Exception as e:
        print(f"  X OLED: {e}")
        display = None
    
    try:
        controller = ACController("config.json")
        print("  OK Controller")
    except Exception as e:
        print(f"  X Controller: {e}")
    
    logger = DataLogger("ac_log.csv")
    print("  OK Logger")
    
    # Setup MQTT Client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print("  OK MQTT Client")
    except Exception as e:
        print(f"  X MQTT: {e}")
        return
    
    print("\n[SERVER] Running! Waiting for ESP32 data...")
    print("[SERVER] Press Ctrl+C to stop.\n")
    
    if display:
        display.show_message("MQTT Server", "Waiting for", "ESP32...")
    
    # Loop
    try:
        mqtt_client.loop_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
    
    mqtt_client.disconnect()
    if display:
        display.cleanup()
    print("[SERVER] Done!")

if __name__ == "__main__":
    main()
