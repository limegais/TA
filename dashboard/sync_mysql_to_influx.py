import pymysql
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

conn = pymysql.connect(host='iotlab-uns.com', user='iotlabun_aslab', password='Pass2myDB', database='iotlabun_sbms')

devices = {
    1: {'tag': 'mysql_ac', 'phase': 'R'}, 
    2: {'tag': 'mysql_outlet', 'phase': 'outlet'},
    3: {'tag': 'mysql_lamp', 'phase': 'lamp'}
}

jkt_tz = pytz.timezone('Asia/Jakarta')

with conn.cursor(pymysql.cursors.DictCursor) as cur:
    for id_kwh, dev_info in devices.items():
        print(f"Migrating id_kwh={id_kwh} ({dev_info['tag']})...")
        cur.execute("""
            SELECT id, tegangan, arus, frekuensi, active_power, reactive_power, apparent_power, created_at
            FROM energies
            WHERE id_kwh = %s
            ORDER BY created_at ASC
        """, (id_kwh,))
        
        rows = cur.fetchall()
        print(f"Found {len(rows)} rows for id_kwh={id_kwh}.")
        
        prev_ts = None
        cumul_wh = 0.0
        
        points = []
        for r in rows:
            # created_at is naive datetime from MySQL, we localize to Jakarta time
            if r['created_at'] is None: continue
            
            created_at_dt = jkt_tz.localize(r['created_at'])
            cur_ts = created_at_dt.timestamp()
            
            p = float(r['active_power'] or 0)
            ap = float(r['apparent_power'] or 0)
            pf = round(p / ap, 3) if ap > 0.001 else 0.0
            
            if prev_ts is not None and cur_ts > prev_ts:
                dt_h = (cur_ts - prev_ts) / 3600.0
                if dt_h < 24: # ignore jumps larger than a day
                    cumul_wh += p * dt_h
            prev_ts = cur_ts
            
            energy_kwh = cumul_wh / 1000.0
            
            pt = Point("energy_monitor") \
                .tag("device", dev_info['tag']) \
                .tag("phase", dev_info['phase']) \
                .field("voltage", float(r['tegangan'] or 0)) \
                .field("current", float(r['arus'] or 0)) \
                .field("power", p) \
                .field("energy_kwh", energy_kwh) \
                .field("frequency", float(r['frekuensi'] or 0)) \
                .field("power_factor", pf) \
                .time(created_at_dt.astimezone(timezone.utc))
                
            points.append(pt)
            
            if len(points) >= 1000:
                write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
                points = []
                
        if points:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
        print(f"Finished id_kwh={id_kwh}.")

print("Migration completed!")
