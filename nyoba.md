#!/usr/bin/env python3
"""
Smart AC - Raspberry Pi MQTT Server
Menerima data dari ESP32 dan mengirim perintah kontrol AC

Jalankan: python3 mqtt_server.py
"""

import json
import time
from datetime import datetime
import paho.mqtt.client as mqtt

# ===================== KONFIGURASI =====================
MQTT_BROKER = "localhost"  # atau "127.0.0.1"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "RaspberryPi_Server"

# Topics
TOPIC_ESP1_SENSOR = "smartac/esp1/sensor"
TOPIC_ESP1_COMMAND = "smartac/esp1/command"
TOPIC_ESP1_STATUS = "smartac/esp1/status"
TOPIC_ESP1_IR_LEARN = "smartac/esp1/ir_learn"

TOPIC_ESP2_SENSOR = "smartac/esp2/sensor"
TOPIC_ESP2_COMMAND = "smartac/esp2/command"

# ===================== DATA STORAGE =====================
sensor_data = {
    "esp1": {
        "temperature": 0,
        "humidity": 0,
        "pir": 0,
        "ac_state": 0,
        "ac_mode": 0,
        "ac_temp": 24,
        "last_update": None
    },
    "esp2": {
        "lux": 0,
        "pir": 0,
        "lamp_pwm": 0,
        "last_update": None
    }
}

# ===================== MQTT CALLBACKS =====================
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback saat terhubung ke broker"""
    if rc == 0:
        print("=" * 50)
        print("✅ Connected to MQTT Broker!")
        print("=" * 50)
        
        # Subscribe ke semua topic
        client.subscribe(TOPIC_ESP1_SENSOR)
        client.subscribe(TOPIC_ESP1_STATUS)
        client.subscribe(TOPIC_ESP1_IR_LEARN)
        client.subscribe(TOPIC_ESP2_SENSOR)
        
        print(f"📡 Subscribed to:")
        print(f"   - {TOPIC_ESP1_SENSOR}")
        print(f"   - {TOPIC_ESP1_STATUS}")
        print(f"   - {TOPIC_ESP1_IR_LEARN}")
        print(f"   - {TOPIC_ESP2_SENSOR}")
        print("=" * 50)
        print("Waiting for data from ESP32...\n")
    else:
        print(f"❌ Connection failed with code {rc}")


def on_disconnect(client, userdata, rc, properties=None):
    """Callback saat terputus dari broker"""
    print(f"⚠️ Disconnected from broker (rc={rc})")
    if rc != 0:
        print("Attempting to reconnect...")


def on_message(client, userdata, msg):
    """Callback saat menerima pesan"""
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n[{timestamp}] 📨 Received from: {topic}")
    
    # Parse JSON
    try:
        data = json.loads(payload)
        
        # Handle ESP1 Sensor Data (AC Node)
        if topic == TOPIC_ESP1_SENSOR:
            handle_esp1_sensor(data)
        
        # Handle ESP1 Status
        elif topic == TOPIC_ESP1_STATUS:
            print(f"   📢 ESP1 Status: {payload}")
        
        # Handle ESP1 IR Learning
        elif topic == TOPIC_ESP1_IR_LEARN:
            handle_ir_learn(data)
        
        # Handle ESP2 Sensor Data (Light Node)
        elif topic == TOPIC_ESP2_SENSOR:
            handle_esp2_sensor(data)
            
    except json.JSONDecodeError:
        print(f"   📄 Raw message: {payload}")


def handle_esp1_sensor(data):
    """Handle data sensor dari ESP1 (AC Node)"""
    global sensor_data
    
    sensor_data["esp1"]["temperature"] = data.get("temperature", 0)
    sensor_data["esp1"]["humidity"] = data.get("humidity", 0)
    sensor_data["esp1"]["pir"] = data.get("pir", 0)
    sensor_data["esp1"]["ac_state"] = data.get("ac_state", 0)
    sensor_data["esp1"]["ac_mode"] = data.get("ac_mode", 0)
    sensor_data["esp1"]["ac_temp"] = data.get("ac_temp", 24)
    sensor_data["esp1"]["last_update"] = datetime.now()
    
    temp = sensor_data["esp1"]["temperature"]
    hum = sensor_data["esp1"]["humidity"]
    pir = "YES" if sensor_data["esp1"]["pir"] else "NO"
    ac = "ON" if sensor_data["esp1"]["ac_state"] else "OFF"
    
    print(f"   🌡️  Temperature: {temp}°C")
    print(f"   💧 Humidity: {hum}%")
    print(f"   🚶 Motion (PIR): {pir}")
    print(f"   ❄️  AC State: {ac}")
    
    # Auto control logic (simple example)
    check_ac_auto_control()


def handle_esp2_sensor(data):
    """Handle data sensor dari ESP2 (Light Node)"""
    global sensor_data
    
    sensor_data["esp2"]["lux"] = data.get("lux", 0)
    sensor_data["esp2"]["pir"] = data.get("pir", 0)
    sensor_data["esp2"]["lamp_pwm"] = data.get("lamp_pwm", 0)
    sensor_data["esp2"]["last_update"] = datetime.now()
    
    lux = sensor_data["esp2"]["lux"]
    pir = "YES" if sensor_data["esp2"]["pir"] else "NO"
    pwm = sensor_data["esp2"]["lamp_pwm"]
    
    print(f"   💡 Light Level: {lux} lux")
    print(f"   🚶 Motion (PIR): {pir}")
    print(f"   🔆 Lamp PWM: {pwm}")


def handle_ir_learn(data):
    """Handle IR code yang di-capture"""
    print(f"   📡 IR Code Captured!")
    print(f"      Protocol: {data.get('protocol', 'Unknown')}")
    print(f"      Address: 0x{data.get('address', 0):04X}")
    print(f"      Command: 0x{data.get('command', 0):04X}")
    print(f"      Bits: {data.get('bits', 0)}")
    
    # Save to file
    with open("ir_codes_log.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {json.dumps(data)}\n")
    print(f"      💾 Saved to ir_codes_log.txt")


# ===================== AUTO CONTROL LOGIC =====================
def check_ac_auto_control():
    """Simple auto control untuk AC"""
    global sensor_data
    
    temp = sensor_data["esp1"]["temperature"]
    pir = sensor_data["esp1"]["pir"]
    ac_state = sensor_data["esp1"]["ac_state"]
    
    # Contoh logic sederhana:
    # - Jika temp > 28°C dan ada orang -> AC ON
    # - Jika temp < 24°C atau tidak ada orang 10 menit -> AC OFF
    
    if temp > 28 and pir == 1 and ac_state == 0:
        print("\n🔥 Suhu tinggi & ada orang -> Menyalakan AC...")
        send_ac_command("AC_ON")
    
    elif temp < 24 and ac_state == 1:
        print("\n❄️ Suhu sudah dingin -> Matikan AC...")
        send_ac_command("AC_OFF")


# ===================== SEND COMMANDS =====================
def send_ac_command(command, temp=24):
    """Kirim perintah ke ESP1 (AC Node)"""
    global client
    
    payload = json.dumps({
        "cmd": command,
        "temp": temp
    })
    
    client.publish(TOPIC_ESP1_COMMAND, payload)
    print(f"📤 Sent to ESP1: {payload}")


def send_lamp_command(command, pwm=128):
    """Kirim perintah ke ESP2 (Light Node)"""
    global client
    
    payload = json.dumps({
        "cmd": command,
        "pwm": pwm
    })
    
    client.publish(TOPIC_ESP2_COMMAND, payload)
    print(f"📤 Sent to ESP2: {payload}")


# ===================== INTERACTIVE MENU =====================
def print_menu():
    """Print menu interaktif"""
    print("\n" + "=" * 50)
    print("📋 SMART AC CONTROL MENU")
    print("=" * 50)
    print("1. AC ON (Cool)")
    print("2. AC OFF")
    print("3. AC ECO Mode")
    print("4. Show Current Data")
    print("5. Lamp ON")
    print("6. Lamp OFF")
    print("7. Set Lamp Brightness")
    print("0. Exit")
    print("=" * 50)


def show_current_data():
    """Tampilkan data sensor terkini"""
    print("\n" + "=" * 50)
    print("📊 CURRENT SENSOR DATA")
    print("=" * 50)
    
    print("\n🏠 ESP1 (AC Node):")
    print(f"   Temperature: {sensor_data['esp1']['temperature']}°C")
    print(f"   Humidity: {sensor_data['esp1']['humidity']}%")
    print(f"   PIR: {'Motion' if sensor_data['esp1']['pir'] else 'No motion'}")
    print(f"   AC: {'ON' if sensor_data['esp1']['ac_state'] else 'OFF'}")
    print(f"   Last update: {sensor_data['esp1']['last_update']}")
    
    print("\n💡 ESP2 (Light Node):")
    print(f"   Lux: {sensor_data['esp2']['lux']}")
    print(f"   PIR: {'Motion' if sensor_data['esp2']['pir'] else 'No motion'}")
    print(f"   Lamp PWM: {sensor_data['esp2']['lamp_pwm']}")
    print(f"   Last update: {sensor_data['esp2']['last_update']}")
    print("=" * 50)


def interactive_mode():
    """Mode interaktif untuk kontrol manual"""
    import threading
    import sys
    
    def menu_thread():
        while True:
            print_menu()
            try:
                choice = input("Pilih (0-7): ").strip()
                
                if choice == "1":
                    send_ac_command("AC_ON")
                elif choice == "2":
                    send_ac_command("AC_OFF")
                elif choice == "3":
                    send_ac_command("AC_ECO")
                elif choice == "4":
                    show_current_data()
                elif choice == "5":
                    send_lamp_command("LAMP_ON")
                elif choice == "6":
                    send_lamp_command("LAMP_OFF")
                elif choice == "7":
                    pwm = input("Brightness (0-255): ")
                    send_lamp_command("LAMP_SET", int(pwm))
                elif choice == "0":
                    print("Exiting...")
                    client.disconnect()
                    sys.exit(0)
                else:
                    print("Invalid choice!")
                    
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(0.5)
    
    # Start menu in separate thread
    thread = threading.Thread(target=menu_thread, daemon=True)
    thread.start()


# ===================== MAIN =====================
if __name__ == "__main__":
    print("\n")
    print("=" * 50)
    print("🏠 SMART AC - RASPBERRY PI SERVER")
    print("=" * 50)
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print("=" * 50)
    
    # Create MQTT client
    # Untuk paho-mqtt versi 2.x
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
    except:
        # Untuk paho-mqtt versi 1.x
        client = mqtt.Client(MQTT_CLIENT_ID)
    
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    # Connect to broker
    try:
        print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # Start interactive mode
        interactive_mode()
        
        # Start MQTT loop
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        client.disconnect()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Mosquitto is running:")
        print("  sudo systemctl start mosquitto")
