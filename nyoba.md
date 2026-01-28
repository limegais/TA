# ir_mitsubishi.py - Khusus untuk AC Mitsubishi

import time
import lgpio
import json
import os

class MitsubishiAC:
    def __init__(self, tx_pin=17, rx_pin=18):
        self.tx_pin = tx_pin
        self.rx_pin = rx_pin
        self.handle = None
        self.codes_file = "mitsubishi_codes.json"
        self.freq = 38000
        
        try:
            self.handle = lgpio.gpiochip_open(4)
            lgpio.gpio_claim_output(self.handle, self.tx_pin, 0)
            lgpio.gpio_claim_input(self.handle, self.rx_pin)
            print("[MITSUBISHI] IR Ready!")
        except Exception as e:
            print(f"[MITSUBISHI] Error: {e}")
            self.handle = None
        
        self.codes = self.load_codes()
    
    def load_codes(self):
        if os.path.exists(self.codes_file):
            try:
                with open(self.codes_file, 'r') as f:
                    codes = json.load(f)
                    print(f"[MITSUBISHI] Loaded {len(codes)} commands")
                    return codes
            except:
                pass
        return {}
    
    def save_codes(self):
        with open(self.codes_file, 'w') as f:
            json.dump(self.codes, f, indent=2)
    
    def send_raw(self, pulses):
        if self.handle is None:
            return False
        
        period = 1.0 / self.freq
        half_period = period / 2
        
        for i, duration in enumerate(pulses):
            duration_sec = duration / 1000000.0
            end_time = time.time() + duration_sec
            
            if i % 2 == 0:
                while time.time() < end_time:
                    lgpio.gpio_write(self.handle, self.tx_pin, 1)
                    time.sleep(half_period)
                    lgpio.gpio_write(self.handle, self.tx_pin, 0)
                    time.sleep(half_period)
            else:
                lgpio.gpio_write(self.handle, self.tx_pin, 0)
                time.sleep(duration_sec)
        
        lgpio.gpio_write(self.handle, self.tx_pin, 0)
        return True
    
    def capture(self, timeout=10):
        if self.handle is None:
            print("[MITSUBISHI] Not initialized!")
            return None
        
        print(f"[CAPTURE] Tekan tombol remote ({timeout} detik)...")
        
        pulses = []
        start_time = time.time()
        last_value = lgpio.gpio_read(self.handle, self.rx_pin)
        last_time = time.time()
        capturing = False
        
        while (time.time() - start_time) < timeout:
            value = lgpio.gpio_read(self.handle, self.rx_pin)
            
            if value != last_value:
                now = time.time()
                duration = int((now - last_time) * 1000000)
                
                if not capturing:
                    capturing = True
                    print("[CAPTURE] Sinyal terdeteksi!")
                
                if duration < 100000:
                    pulses.append(duration)
                
                last_time = now
                last_value = value
            
            if capturing and len(pulses) > 20:
                if (time.time() - last_time) > 0.1:
                    break
            
            time.sleep(0.00005)
        
        if len(pulses) > 20:
            print(f"[CAPTURE] Berhasil! {len(pulses)} pulses")
            return pulses
        else:
            print("[CAPTURE] Tidak ada sinyal")
            return None
    
    def learn_command(self, name):
        print(f"\n[LEARN] '{name}'")
        pulses = self.capture()
        if pulses:
            self.codes[name] = pulses
            self.save_codes()
            print(f"[LEARN] Tersimpan!")
            return True
        return False
    
    def send_command(self, name):
        if name in self.codes:
            print(f"[SEND] {name}")
            self.send_raw(self.codes[name])
            return True
        else:
            print(f"[SEND] '{name}' tidak ditemukan!")
            return False
    
    def power_on(self):
        return self.send_command("power_on")
    
    def power_off(self):
        return self.send_command("power_off")
    
    def list_commands(self):
        return list(self.codes.keys())
    
    def cleanup(self):
        if self.handle is not None:
            lgpio.gpio_write(self.handle, self.tx_pin, 0)
            lgpio.gpiochip_close(self.handle)


if __name__ == "__main__":
    print("=" * 50)
    print("  MITSUBISHI AC - IR LEARNER")
    print("=" * 50)
    
    ac = MitsubishiAC(tx_pin=17, rx_pin=18)
    
    while True:
        print("\n" + "-" * 50)
        print("Menu:")
        print("  1. Learn tombol baru")
        print("  2. Test kirim command")
        print("  3. Lihat commands tersimpan")
        print("  4. Hapus command")
        print("  5. Keluar")
        print("-" * 50)
        
        choice = input("Pilih (1-5): ").strip()
        
        if choice == "1":
            print("\nContoh nama: power_on, power_off, temp_24, temp_25")
            name = input("Nama command: ").strip()
            if name:
                input("Tekan ENTER, lalu tekan tombol remote...")
                ac.learn_command(name)
        
        elif choice == "2":
            cmds = ac.list_commands()
            if cmds:
                print("\nCommands:", cmds)
                name = input("Nama command: ").strip()
                if name:
                    ac.send_command(name)
                    print("Cek apakah AC merespon!")
            else:
                print("Belum ada command!")
        
        elif choice == "3":
            cmds = ac.list_commands()
            print("\nCommands:", cmds if cmds else "Kosong")
        
        elif choice == "4":
            name = input("Nama command: ").strip()
            if name in ac.codes:
                del ac.codes[name]
                ac.save_codes()
                print("Dihapus!")
        
        elif choice == "5":
            break
    
    ac.cleanup()
    print("Selesai!")
