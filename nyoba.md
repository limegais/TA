#!/usr/bin/env python3
"""
Smart AC Controller dengan GA Optimization
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from threading import Thread

from shared.mqtt_client import MQTTHandler
from shared.database import DatabaseHandler
from shared.utils import setup_logger
from ga_optimizer import GAOptimizer
import ac_config as config

# ===================== SETUP =====================
logger = setup_logger('AC_Controller', '../logs/ac.log')

sensor_data = {
    "temperature": 0,
    "humidity": 0,
    "pir": 0,
    "ac_state": 0,
    "ac_mode": 0,
    "ac_temp": 24,
    "last_update": None
}

ga_optimizer = GAOptimizer()
db = DatabaseHandler(data_dir="../data")
mqtt_handler = None

auto_control_enabled = True
running = True

# ===================== CALLBACKS =====================
def on_sensor_data(topic, data):
    """Handle data sensor dari ESP1"""
    global sensor_data
    
    sensor_data["temperature"] = data.get("temperature", 0)
    sensor_data["humidity"] = data.get("humidity", 0)
    sensor_data["pir"] = data.get("pir", 0)
    sensor_data["ac_state"] = data.get("ac_state", 0)
    sensor_data["ac_mode"] = data.get("ac_mode", 0)
    sensor_data["ac_temp"] = data.get("ac_temp", 24)
    sensor_data["last_update"] = datetime.now()
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n[{timestamp}] 📨 ESP1 Sensor Data:")
    print(f"   🌡️  Temp: {sensor_data['temperature']}°C")
    print(f"   💧 Humidity: {sensor_data['humidity']}%")
    print(f"   🚶 PIR: {'YES' if sensor_data['pir'] else 'NO'}")
    print(f"   ❄️  AC: {'ON' if sensor_data['ac_state'] else 'OFF'}")
    
    db.save_sensor_data("esp1_ac", data)


def on_status(topic, data):
    """Handle status message"""
    print(f"[STATUS] {data}")


def on_ir_learn(topic, data):
    """Handle captured IR code"""
    print(f"\n[IR] Code Captured!")
    print(f"   Protocol: {data.get('protocol')}")
    print(f"   Address: 0x{data.get('address', 0):04X}")
    print(f"   Command: 0x{data.get('command', 0):04X}")
    db.save_event("esp1", "ir_capture", json.dumps(data))


# ===================== CONTROL FUNCTIONS =====================
def send_ac_command(command, temp=24):
    """Kirim command ke ESP1"""
    global mqtt_handler
    
    payload = {"cmd": command, "temp": temp}
    mqtt_handler.publish(config.TOPIC_COMMAND, payload)
    print(f"📤 Sent: {payload}")
    db.save_event("esp1", "command", command)


def run_optimization():
    """Jalankan GA optimization"""
    global sensor_data, ga_optimizer, mqtt_handler
    
    temp = sensor_data["temperature"]
    humidity = sensor_data["humidity"]
    occupancy = sensor_data["pir"] == 1
    
    if temp == 0:
        print("[GA] No sensor data, skipping...")
        return
    
    print("\n" + "=" * 50)
    print("🧬 Running GA Optimization...")
    print(f"   Input: Temp={temp}°C, Hum={humidity}%, Occ={occupancy}")
    
    rec = ga_optimizer.get_recommendation(temp, humidity, occupancy)
    
    print(f"   📊 Result:")
    print(f"      Mode: {rec['recommended_mode']}")
    print(f"      Temp: {rec['recommended_temp']}°C")
    print(f"      Confidence: {rec['confidence']:.1%}")
    print(f"      Reason: {rec['reason']}")
    print("=" * 50)
    
    mqtt_handler.publish(config.TOPIC_RECOMMENDATION, rec)
    
    if auto_control_enabled:
        apply_recommendation(rec)


def apply_recommendation(rec):
    """Apply GA recommendation"""
    global sensor_data
    
    mode = rec['recommended_mode']
    temp = rec['recommended_temp']
    current_ac = sensor_data['ac_state']
    
    if mode == 'off' and current_ac == 1:
        print("[AUTO] AC OFF")
        send_ac_command("AC_OFF")
    
    elif mode in ['cool', 'eco'] and current_ac == 0:
        print(f"[AUTO] AC {mode.upper()}")
        if mode == 'eco':
            send_ac_command("AC_ECO")
        else:
            send_ac_command("AC_ON", temp)


def optimization_loop():
    """Background optimization loop"""
    global running
    
    while running:
        time.sleep(config.OPTIMIZATION_INTERVAL)
        if auto_control_enabled and running:
            try:
                run_optimization()
            except Exception as e:
                print(f"[ERROR] Optimization: {e}")


# ===================== MENU =====================
def print_menu():
    print("\n" + "=" * 50)
    print("🌀 SMART AC CONTROLLER")
    print("=" * 50)
    print("1. AC ON")
    print("2. AC OFF")
    print("3. AC ECO")
    print("4. Run GA Now")
    print("5. Show Data")
    print("6. Toggle Auto")
    print("0. Exit")
    print("=" * 50)


def interactive_menu():
    """Menu interaktif"""
    global auto_control_enabled, running
    
    while running:
        print_menu()
        try:
            choice = input("Pilih: ").strip()
            
            if choice == "1":
                send_ac_command("AC_ON")
            elif choice == "2":
                send_ac_command("AC_OFF")
            elif choice == "3":
                send_ac_command("AC_ECO")
            elif choice == "4":
                run_optimization()
            elif choice == "5":
                print(f"\n📊 Data:\n{json.dumps(sensor_data, indent=2, default=str)}")
            elif choice == "6":
                auto_control_enabled = not auto_control_enabled
                print(f"Auto: {'ON' if auto_control_enabled else 'OFF'}")
            elif choice == "0":
                print("Exiting...")
                running = False
                break
        except KeyboardInterrupt:
            running = False
            break
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(0.3)


# ===================== MAIN =====================
def main():
    global mqtt_handler, running
    
    print("\n" + "=" * 50)
    print("🌀 SMART AC CONTROLLER + GA")
    print("=" * 50)
    
    # Setup MQTT
    mqtt_handler = MQTTHandler(
        config.MQTT_BROKER, 
        config.MQTT_PORT, 
        "AC_Controller"
    )
    mqtt_handler.connect()
    
    # Subscribe
    mqtt_handler.subscribe(config.TOPIC_SENSOR, on_sensor_data)
    mqtt_handler.subscribe(config.TOPIC_STATUS, on_status)
    mqtt_handler.subscribe(config.TOPIC_IR_LEARN, on_ir_learn)
    
    # Start MQTT loop
    mqtt_handler.loop_start()
    
    # Start optimization thread
    opt_thread = Thread(target=optimization_loop, daemon=True)
    opt_thread.start()
    
    print("[OK] AC Controller started!")
    
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        running = False
        mqtt_handler.disconnect()


if __name__ == "__main__":
    main()
