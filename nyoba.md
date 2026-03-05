"""
Smart Room IoT - Fitness Function
GA fitness for AC control, PSO fitness for Lamp control
Data sourced from InfluxDB sensor history
"""

from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta

# ==================== INFLUXDB CONFIG ====================
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "rfi_HvWdjwaG8jB3Rqx6g0y5kMWRfSfq_HmLLUvkom1yaHKvwonU9Qfj6nlZjTqb_I0leIREUnMhvQQXtgETfg=="
INFLUX_ORG = "IOTLAB"
INFLUX_BUCKET = "SENSORDATA"

# ==================== OPTIMIZATION BOUNDS ====================
TEMP_MIN, TEMP_MAX = 16, 30        # AC temperature range (°C)
FAN_MIN, FAN_MAX = 1, 3            # Fan speed levels
BRIGHTNESS_MIN, BRIGHTNESS_MAX = 0, 100  # Lamp brightness (%)

# ==================== SENSOR DATA (live + DB) ====================
current_data = {
    'temperature': 28.0,
    'humidity': 55.0,
    'person_detected': False,
    'lux': 200,
    'avg_temperature': 28.0,    # rata-rata dari DB
    'avg_humidity': 55.0,       # rata-rata dari DB
    'avg_lux': 200.0,          # rata-rata dari DB
    'data_source': 'default'   # 'default', 'mqtt', 'influxdb'
}

# ==================== INFLUXDB DATA FETCH ====================
def fetch_sensor_data_from_db(time_range_minutes=30):
    """
    Ambil data sensor dari InfluxDB untuk digunakan oleh GA/PSO
    
    Parameters:
    - time_range_minutes: Berapa menit terakhir data diambil
    
    Returns:
    - dict: Data sensor rata-rata dari database
    """
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        # Query rata-rata suhu & humidity dari ac_sensor
        ac_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: -{time_range_minutes}m)
            |> filter(fn: (r) => r._measurement == "ac_sensor")
            |> filter(fn: (r) => r._field == "temperature" or r._field == "humidity")
            |> mean()
        '''
        
        # Query rata-rata lux dari lamp_sensor
        lamp_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: -{time_range_minutes}m)
            |> filter(fn: (r) => r._measurement == "lamp_sensor")
            |> filter(fn: (r) => r._field == "lux")
            |> mean()
        '''
        
        # Query deteksi orang terakhir
        camera_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: -{time_range_minutes}m)
            |> filter(fn: (r) => r._measurement == "camera_detection")
            |> filter(fn: (r) => r._field == "person_count")
            |> last()
        '''
        
        result_data = {
            'temperature': current_data['temperature'],
            'humidity': current_data['humidity'],
            'lux': current_data['lux'],
            'person_detected': current_data['person_detected'],
            'data_points': 0
        }
        
        # Parse AC sensor data
        ac_tables = query_api.query(ac_query)
        for table in ac_tables:
            for record in table.records:
                field = record.get_field()
                value = record.get_value()
                if field == 'temperature' and value is not None:
                    result_data['temperature'] = round(float(value), 1)
                elif field == 'humidity' and value is not None:
                    result_data['humidity'] = round(float(value), 1)
                result_data['data_points'] += 1
        
        # Parse Lamp sensor data
        lamp_tables = query_api.query(lamp_query)
        for table in lamp_tables:
            for record in table.records:
                field = record.get_field()
                value = record.get_value()
                if field == 'lux' and value is not None:
                    result_data['lux'] = round(float(value), 1)
                result_data['data_points'] += 1
        
        # Parse Camera data
        try:
            cam_tables = query_api.query(camera_query)
            for table in cam_tables:
                for record in table.records:
                    value = record.get_value()
                    if value is not None:
                        result_data['person_detected'] = int(value) > 0
                    result_data['data_points'] += 1
        except:
            pass  # Camera data might not exist
        
        client.close()
        
        # Update current_data with DB values
        current_data['avg_temperature'] = result_data['temperature']
        current_data['avg_humidity'] = result_data['humidity']
        current_data['avg_lux'] = result_data['lux']
        current_data['temperature'] = result_data['temperature']
        current_data['humidity'] = result_data['humidity']
        current_data['lux'] = result_data['lux']
        current_data['person_detected'] = result_data['person_detected']
        current_data['data_source'] = 'influxdb'
        
        print(f"✅ InfluxDB: {result_data['data_points']} data points fetched ({time_range_minutes}m range)")
        print(f"   Temp: {result_data['temperature']}°C | Humidity: {result_data['humidity']}%")
        print(f"   Lux: {result_data['lux']} | Person: {result_data['person_detected']}")
        
        return result_data
        
    except Exception as e:
        print(f"⚠️  InfluxDB query failed: {e}")
        print(f"   Using last known sensor data instead")
        current_data['data_source'] = 'mqtt_fallback'
        return {
            'temperature': current_data['temperature'],
            'humidity': current_data['humidity'],
            'lux': current_data['lux'],
            'person_detected': current_data['person_detected'],
            'data_points': 0
        }

# ==================== GA FITNESS: AC OPTIMIZATION ====================
def calculate_ac_fitness(temp_set, fan_speed):
    """
    Fitness function untuk GA - Optimasi AC (suhu + fan speed)
    
    Tujuan: Temukan setting AC terbaik berdasarkan kondisi ruangan
    - Jika ada orang → prioritas kenyamanan (target 24-26°C)
    - Jika kosong → prioritas hemat energi (target 27-29°C)
    
    Parameters:
    - temp_set: Setting suhu AC (16-30°C)
    - fan_speed: Kecepatan fan (1-3)
    
    Returns:
    - fitness: float score (higher = better, max ~100)
    """
    temp_room = current_data['temperature']
    humidity = current_data['humidity']
    person_detected = current_data['person_detected']
    
    fitness = 0
    
    # ===== 1. COMFORT SCORE (max 50) =====
    if person_detected:
        target_temp = 25
        ideal_range = (24, 26)
    else:
        target_temp = 28
        ideal_range = (27, 29)
    
    temp_diff = abs(temp_set - target_temp)
    if ideal_range[0] <= temp_set <= ideal_range[1]:
        fitness += 50  # Perfect comfort
    elif temp_diff <= 2:
        fitness += 40
    elif temp_diff <= 4:
        fitness += 25
    else:
        fitness += max(0, 50 - (temp_diff * 7))
    
    # ===== 2. HUMIDITY RESPONSE (max 15) =====
    if humidity > 70:
        # High humidity → lower temp + higher fan helps dehumidify
        if temp_set <= 24 and fan_speed >= 2:
            fitness += 15
        elif temp_set <= 26:
            fitness += 8
    elif 40 <= humidity <= 60:
        fitness += 15  # Humidity already ideal
    elif humidity < 40:
        # Dry air → don't overcool
        if temp_set >= 25:
            fitness += 10
    
    # ===== 3. FAN SPEED APPROPRIATENESS (max 15) =====
    temp_gap = abs(temp_room - temp_set)
    if temp_gap > 5:
        # Big temperature gap → high fan is good
        if fan_speed == 3:
            fitness += 15
        elif fan_speed == 2:
            fitness += 10
    elif temp_gap <= 2:
        # Small gap → low fan is efficient
        if fan_speed == 1:
            fitness += 15
        elif fan_speed == 2:
            fitness += 10
    else:
        # Medium gap → medium fan
        if fan_speed == 2:
            fitness += 15
        else:
            fitness += 8
    
    # ===== 4. ENERGY EFFICIENCY (max 20) =====
    # Higher temp + lower fan = less energy
    ac_power = (30 - temp_set) * 50 + fan_speed * 30  # Watts estimation
    max_power = (30 - 16) * 50 + 3 * 30  # Maximum possible
    energy_ratio = 1 - (ac_power / max_power)
    
    if person_detected:
        # When occupied: 30% energy weight
        fitness += energy_ratio * 6
    else:
        # When empty: 100% energy weight
        fitness += energy_ratio * 20
    
    # ===== 5. PENALTY: Overcooling empty room =====
    if not person_detected and temp_set < 24:
        fitness -= 15  # Waste of energy
    
    # ===== 6. PENALTY: Too cold + strong fan =====
    if temp_set < 20 and fan_speed == 3:
        fitness -= 10  
    
    return max(0, round(fitness, 2))

# ==================== PSO FITNESS: LAMP OPTIMIZATION ====================
def calculate_lamp_fitness(brightness):
    """
    Fitness function untuk PSO - Optimasi Lamp (brightness)
    
    Tujuan: Temukan brightness terbaik berdasarkan lux & ketersediaan orang
    - Jika ada orang → target 300-500 lux (cukup terang untuk bekerja)
    - Jika kosong → matikan atau minimal
    
    Parameters:
    - brightness: Brightness lamp (0-100%)
    
    Returns:
    - fitness: float score (higher = better, max ~100)
    """
    ambient_lux = current_data['lux']
    person_detected = current_data['person_detected']
    
    fitness = 0
    
    # Estimasi total lux = ambient + lamp contribution
    lamp_lux_contribution = brightness * 5  # 1% brightness ≈ 5 lux
    total_lux = ambient_lux + lamp_lux_contribution
    
    # ===== 1. LIGHTING COMFORT (max 50) =====
    if person_detected:
        target_lux = 400  # Office/study lighting standard
        lux_range = (300, 500)
        
        if lux_range[0] <= total_lux <= lux_range[1]:
            fitness += 50  # Perfect lighting
        elif 200 <= total_lux <= 600:
            fitness += 35
        elif total_lux < 200:
            # Too dark when someone is present
            fitness += max(0, 50 - (200 - total_lux) * 0.2)
        else:
            # Too bright
            fitness += max(0, 50 - (total_lux - 600) * 0.1)
    else:
        # No one → darkness is fine → reward low brightness
        if brightness <= 5:
            fitness += 50  # Perfect: lamp off when empty
        elif brightness <= 20:
            fitness += 35
        else:
            fitness += max(0, 50 - brightness * 0.5)
    
    # ===== 2. ENERGY EFFICIENCY (max 30) =====
    lamp_power = brightness * 0.5  # Watts (assumption: 50W max)
    max_power = 100 * 0.5
    energy_ratio = 1 - (lamp_power / max_power)
    
    if person_detected:
        fitness += energy_ratio * 10  # Less weight on energy when occupied
    else:
        fitness += energy_ratio * 30  # Full weight on energy when empty
    
    # ===== 3. AMBIENT LIGHT ADAPTATION (max 20) =====
    if person_detected:
        if ambient_lux >= 400:
            # Plenty of natural light → lamp should be low
            if brightness <= 20:
                fitness += 20
            elif brightness <= 40:
                fitness += 12
            else:
                fitness += 0  # Wasting energy
        elif ambient_lux >= 200:
            # Some natural light → moderate lamp
            if 20 <= brightness <= 60:
                fitness += 20
            else:
                fitness += 8
        else:
            # Dark → lamp should be high
            if brightness >= 60:
                fitness += 20
            elif brightness >= 40:
                fitness += 12
            else:
                fitness += 5  # Too dark for comfort
    else:
        # Empty room: any ambient level doesn't matter, lamp should be off
        if brightness <= 5:
            fitness += 20
    
    return max(0, round(fitness, 2))

# ==================== COMBINED FITNESS (backward compatibility) ====================
def calculate_fitness(temp_set, fan_speed, brightness):
    """Combined fitness for both AC and Lamp (backward compatible)"""
    ac_score = calculate_ac_fitness(temp_set, fan_speed)
    lamp_score = calculate_lamp_fitness(brightness)
    return round((ac_score + lamp_score) / 2, 2)

# ==================== SENSOR DATA UPDATE ====================
def update_sensor_data(temperature=None, humidity=None, person_detected=None, lux=None):
    """Update current sensor readings from MQTT"""
    if temperature is not None:
        current_data['temperature'] = temperature
        current_data['data_source'] = 'mqtt'
    if humidity is not None:
        current_data['humidity'] = humidity
    if person_detected is not None:
        current_data['person_detected'] = person_detected
    if lux is not None:
        current_data['lux'] = lux

def get_current_conditions():
    """Return current sensor data"""
    return current_data.copy()
