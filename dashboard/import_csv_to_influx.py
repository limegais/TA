import csv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone
import pytz

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "rfi_HvWdjwaG8jB3Rqx6g0y5kMWRfSfq_HmLLUvkom1yaHKvwonU9Qfj6nlZjTqb_I0leIREUnMhvQQXtgETfg=="
INFLUX_ORG = "IOTLAB"
INFLUX_BUCKET = "SENSORDATA"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

jkt_tz = pytz.timezone('Asia/Jakarta')

device_map = {
    '1': {'tag': 'mysql_ac', 'phase': 'R'},
    '2': {'tag': 'mysql_outlet', 'phase': 'outlet'},
    '3': {'tag': 'mysql_lamp', 'phase': 'lamp'}
}

csv_file = r"c:\Users\ramad\Downloads\energy_kwh (3).csv"

points = []
count = 0
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        id_kwh = row.get('id_kwh')
        if not id_kwh or id_kwh not in device_map:
            continue
            
        dev_info = device_map[id_kwh]
        try:
            total_energy_wh = float(row.get('total_energy', 0))
        except ValueError:
            total_energy_wh = 0.0
            
        energy_kwh = total_energy_wh / 1000.0
        
        dt_str = row.get('created_at')
        if not dt_str:
            continue
            
        try:
            dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            dt_aware = jkt_tz.localize(dt_naive).astimezone(timezone.utc)
        except Exception as e:
            continue
            
        pt = Point("energy_monitor") \
            .tag("device", dev_info['tag']) \
            .tag("phase", dev_info['phase']) \
            .field("energy_kwh", energy_kwh) \
            .time(dt_aware)
            
        points.append(pt)
        count += 1
        
        if len(points) >= 5000:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
            print(f"Inserted {count} points...")
            points = []

if points:
    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
    
print(f"Successfully migrated {count} historical data points from CSV to InfluxDB!")
