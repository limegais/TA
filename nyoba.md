#!/usr/bin/env python3
"""
Smart Light Controller dengan PSO Optimization
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from threading import Thread

from shared.mqtt_client import MQTTHandler
from shared.database import DatabaseHandler
from shared.utils import setup_logger
from pso_optimizer import PSOOptimizer
import light_config as config

# ===================== SETUP =====================
logger = setup_logger('Light_Controller', '../logs/light.log')

sensor_data = {
    "lux": 0,
    "lamp_pwm": 0,
    "lamp_state": 0,
    "auto_mode": 1,
    "last_update": None
}

pso_optimizer = PSOOptimizer()
db = DatabaseHandler(data_dir="../data")
mqtt_handler = None

auto_control_enabled = True
current_activity = "working"
running = True

# ===================== CALLBACKS =====================
def on_sensor_data(topic, data):
    """Handle data dari ESP2"""
    global sensor_data
    
    sensor_data["lux"] = data.get("lux_smooth", data.get("lux", 0))
    sensor_data["lamp_pwm"] = data.get("lamp_pwm", 0)
    sensor_data["lamp_state"] = data.get("lamp_state", 0)
    sensor_data["auto_mode"] = data.get("auto_mode", 1)
    sensor_data["last_update"] = datetime.now()
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    percent = sensor_data["lamp_pwm"] * 100 // 255
    
    print(f"\n[{timestamp}] 📨 ESP2 Sensor Data:")
    print(f"   💡 Lux: {sensor_data['lux']:.1f}")
    print(f"   🔆 Lamp: {sensor_data['lamp_pwm']} ({percent}%)")
    
    db.save_sensor_data("esp2_light", data)


def on_status(topic, data):
    """Handle status"""
    print(f"[STATUS] {data}")


# ===================== CONTROL =====================
def send_lamp_command(command, pwm=128):
    """Kirim command ke ESP2"""
    global mqtt_handler
    
    payload = {"cmd": command, "pwm": pwm}
    mqtt_handler.publish(config.TOPIC_COMMAND, payload)
    print(f"📤 Sent: {payload}")
    db.save_event("esp2", "command", command)


def run_optimization():
    """Jalankan PSO"""
    global sensor_data, pso_optimizer, mqtt_handler, current_activity
    
    lux = sensor_data["lux"]
    
    print("\n" + "=" * 50)
    print("🔮 Running PSO Optimization...")
    print(f"   Input: Lux={lux:.1f}, Activity={current_activity}")
    
    # Untuk versi tanpa PIR, assume occupied during working hours
    from shared.utils import is_working_hours
    occupancy = is_working_hours()
    
    rec = pso_optimizer.get_recommendation(lux, occupancy, current_activity)
    
    print(f"   📊 Result:")
    print(f"      PWM: {rec['recommended_pwm']} ({rec['recommended_percent']}%)")
    print(f"      Confidence: {rec['confidence']:.1%}")
    print(f"      Reason: {rec['reason']}")
    print("=" * 50)
    
    mqtt_handler.publish(config.TOPIC_RECOMMENDATION, rec)
    
    if auto_control_enabled:
        apply_recommendation(rec)


def apply_recommendation(rec):
    """Apply PSO recommendation"""
    pwm = rec['recommended_pwm']
    
    if abs(pwm - sensor_data['lamp_pwm']) > 10:
        print(f"[AUTO] Set PWM: {pwm}")
        send_lamp_command("LAMP_SET", pwm)


def optimization_loop():
    """Background loop"""
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
    print("💡 SMART LIGHT CONTROLLER")
    print("=" * 50)
    print("1. Lamp ON")
    print("2. Lamp OFF")
    print("3. Set Brightness")
    print("4. Run PSO Now")
    print("5. Show Data")
    print("6. Set Activity")
    print("7. Toggle Auto")
    print("0. Exit")
    print("=" * 50)


def interactive_menu():
    """Menu interaktif"""
    global auto_control_enabled, current_activity, running
    
    while running:
        print_menu()
        try:
            choice = input("Pilih: ").strip()
            
            if choice == "1":
                send_lamp_command("LAMP_ON")
            elif choice == "2":
                send_lamp_command("LAMP_OFF")
            elif choice == "3":
                pwm = int(input("PWM (0-255): "))
                send_lamp_command("LAMP_SET", pwm)
            elif choice == "4":
                run_optimization()
            elif choice == "5":
                print(f"\n📊 Data:\n{json.dumps(sensor_data, indent=2, default=str)}")
            elif choice == "6":
                print("Options: reading, working, relaxing, sleeping, away")
                current_activity = input("Activity: ").strip()
                print(f"Activity: {current_activity}")
            elif choice == "7":
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
    print("💡 SMART LIGHT CONTROLLER + PSO")
    print("=" * 50)
    
    mqtt_handler = MQTTHandler(
        config.MQTT_BROKER,
        config.MQTT_PORT,
        "Light_Controller"
    )
    mqtt_handler.connect()
    
    mqtt_handler.subscribe(config.TOPIC_SENSOR, on_sensor_data)
    mqtt_handler.subscribe(config.TOPIC_STATUS, on_status)
    
    mqtt_handler.loop_start()
    
    opt_thread = Thread(target=optimization_loop, daemon=True)
    opt_thread.start()
    
    print("[OK] Light Controller started!")
    
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        running = False
        mqtt_handler.disconnect()


if __name__ == "__main__":
    main()
