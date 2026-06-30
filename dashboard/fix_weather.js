"""Fix the mangled weather section in dashboard.js"""
import re

filepath = r'c:\Users\ramad\Downloads\smartroom-20260526T081719Z-3-001\smartroom\dashboard\static\js\dashboard.js'

with open(filepath, 'rb') as f:
    raw = f.read()

content = raw.decode('utf-8')

# Find the broken section: from "var absHum" to the end of file
# We need to replace from after "var absHum = Math.abs(deltaHum).toFixed(0);" 
# all the way to end of file

# Find the anchor point
anchor = "var absHum = Math.abs(deltaHum).toFixed(0);"
anchor_pos = content.find(anchor)
if anchor_pos == -1:
    print("[ERROR] Could not find anchor point")
    exit(1)

# Find the end of that line
end_of_anchor = content.find('\n', anchor_pos)
if end_of_anchor == -1:
    print("[ERROR] Could not find end of anchor line")
    exit(1)

# Keep everything up to and including the anchor line
prefix = content[:end_of_anchor + 1]

# Determine original line ending
if '\r\r\n' in content[:1000]:
    nl = '\r\r\n'
elif '\r\n' in content[:1000]:
    nl = '\r\n'
else:
    nl = '\n'

print(f"[INFO] Detected line ending: {repr(nl)}")
print(f"[INFO] Anchor found at position {anchor_pos}")

# New code to append
new_code = """                if (deltaHum > 5) {
                    humBadge.textContent = 'Lebih lembab +' + absHum + '%';
                    humBadge.style.background = 'rgba(14,165,233,0.18)';
                    humBadge.style.color = '#0ea5e9';
                } else if (deltaHum < -5) {
                    humBadge.textContent = 'Lebih lembab +' + absHum + '%';
                    humBadge.style.background = 'rgba(239,68,68,0.15)';
                    humBadge.style.color = '#ef4444';
                } else {
                    humBadge.textContent = '= Sama';
                    humBadge.style.background = 'rgba(16,185,129,0.15)';
                    humBadge.style.color = '#10b981';
                }
            }

            // --- Status badge ---
            var statusBadge = document.getElementById('weather-status-badge');
            if (statusBadge) {
                if (out.fetch_ok) {
                    statusBadge.textContent = 'Online';
                    statusBadge.style.background = 'rgba(16,185,129,0.12)';
                    statusBadge.style.color = '#10b981';
                } else {
                    statusBadge.textContent = 'Offline';
                    statusBadge.style.background = 'rgba(239,68,68,0.12)';
                    statusBadge.style.color = '#ef4444';
                }
            }

            // --- Smart Insight ---
            var insightEl = document.getElementById('weather-insight-text');
            if (insightEl && out.temperature !== null) {
                var insights = [];
                var tempO = out.temperature;
                var humO  = out.humidity;
                var uvO   = out.uv_index;
                var rainO = out.precipitation;
                var windO = out.wind_speed;
                var desc  = out.weather_desc || '';

                if (rainO > 0) {
                    insights.push('Hujan ' + rainO + 'mm terdeteksi — pastikan ventilasi tertutup untuk menjaga kelembaban ruangan.');
                } else if (tempO > 33) {
                    insights.push('Suhu luar sangat panas (' + tempO + '°C) — AC bekerja lebih keras. GA akan merekomendasikan setpoint lebih rendah.');
                } else if (tempO > 28 && deltaTemp < 0) {
                    insights.push('Cuaca panas di luar (' + tempO + '°C) — ruangan Anda lebih dingin. AC berjalan optimal.');
                } else if (tempO < 25) {
                    insights.push('Cuaca sejuk di luar (' + tempO + '°C) — pertimbangkan membuka ventilasi untuk hemat energi.');
                }
                if (uvO >= 8) {
                    insights.push('Indeks UV sangat tinggi (' + uvO + ') — hindari paparan sinar matahari langsung.');
                } else if (uvO >= 5) {
                    insights.push('Indeks UV tinggi (' + uvO + ') — gunakan perlindungan jika berada di luar.');
                }
                if (humO > 85) {
                    insights.push('Kelembaban luar sangat tinggi (' + humO + '%) — potensi kondensasi pada kaca jendela.');
                }
                if (windO > 30) {
                    insights.push('Angin kencang (' + windO + ' km/h) terdeteksi di area UNS.');
                }

                insightEl.textContent = insights.length > 0
                    ? insights[0]
                    : 'Kondisi cuaca ' + (desc || 'normal') + ' di UNS Surakarta. Suhu luar: ' + tempO + '°C, Kelembaban: ' + humO + '%.';
            }
        }

        function fetchOutdoorWeather(retryCount) {
            retryCount = retryCount || 0;
            var MAX_RETRIES = 3;
            fetch('/api/outdoor-weather')
                .then(function(r) {
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    return r.json();
                })
                .then(function(data) {
                    try {
                        updateWeatherUI(data);
                        var out = data.outdoor || {};
                        if (!out.fetch_ok && out.temperature === null) {
                            var descEl = document.getElementById('weather-desc');
                            if (descEl) descEl.textContent = 'Menunggu data...';
                            var insightEl = document.getElementById('weather-insight-text');
                            if (insightEl) insightEl.textContent = 'Server belum bisa mengakses Open-Meteo API. Periksa koneksi internet server.';
                            setTimeout(function() { fetchOutdoorWeather(0); }, 15000);
                        }
                    } catch(e) {
                        console.warn('[WEATHER] UI update error:', e);
                    }
                })
                .catch(function(e) {
                    console.warn('[WEATHER] Fetch failed (attempt ' + (retryCount+1) + '):', e.message);
                    if (retryCount < MAX_RETRIES) {
                        var delay = (retryCount + 1) * 3000;
                        setTimeout(function() { fetchOutdoorWeather(retryCount + 1); }, delay);
                        return;
                    }
                    var statusBadge = document.getElementById('weather-status-badge');
                    if (statusBadge) {
                        statusBadge.textContent = 'Error';
                        statusBadge.style.background = 'rgba(239,68,68,0.12)';
                        statusBadge.style.color = '#ef4444';
                    }
                    var descEl = document.getElementById('weather-desc');
                    if (descEl) descEl.textContent = 'Tidak tersedia';
                    var insightEl = document.getElementById('weather-insight-text');
                    if (insightEl) insightEl.textContent = 'Gagal memuat cuaca setelah 3x percobaan. Periksa koneksi internet server atau klik tombol refresh.';
                    setTimeout(function() { fetchOutdoorWeather(0); }, 30000);
                });
        }

        function refreshOutdoorWeather() {
            var btn = document.querySelector('[onclick="refreshOutdoorWeather()"]');
            if (btn) { btn.style.opacity = '0.4'; btn.style.pointerEvents = 'none'; }
            var statusBadge = document.getElementById('weather-status-badge');
            if (statusBadge) {
                statusBadge.textContent = 'Memuat';
                statusBadge.style.background = 'rgba(148,163,184,0.12)';
                statusBadge.style.color = '#94a3b8';
            }
            fetchOutdoorWeather(0);
            setTimeout(function() {
                if (btn) { btn.style.opacity = '1'; btn.style.pointerEvents = 'auto'; }
            }, 2000);
        }

        // Fetch pertama saat halaman load
        setTimeout(fetchOutdoorWeather, 800);
        // Auto-refresh setiap 10 menit
        _weatherRefreshTimer = setInterval(fetchOutdoorWeather, 600000);
"""

# Convert new code to use the same line ending as original
new_lines = new_code.split('\n')
new_code_fixed = nl.join(new_lines)

result = prefix + new_code_fixed

# Write back preserving encoding
with open(filepath, 'wb') as f:
    f.write(result.encode('utf-8'))

print(f"[OK] File fixed successfully! Total size: {len(result)} bytes")
print(f"[OK] Replaced from position {end_of_anchor + 1} to end")
