"""
verify_extraction.py
Verifikasi bahwa konten yang diekstrak dari HTML_TEMPLATE identik
dengan yang ada di app.py.bak (versi asli).

Strategi:
1. Parse HTML_TEMPLATE dari app.py.bak → css_orig, js_orig, page_blocks_orig
2. Baca file hasil ekstraksi → css_new, js_new, page_blocks_new
3. Bandingkan normalisasi (strip trailing whitespace tiap baris, ignore \r)
4. Laporkan perbedaan jika ada
"""

import os, re, difflib, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAK_FILE = os.path.join(BASE_DIR, 'app.py.bak')
CSS_FILE = os.path.join(BASE_DIR, 'static', 'css', 'dashboard.css')
JS_FILE  = os.path.join(BASE_DIR, 'static', 'js', 'dashboard.js')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

def norm(text):
    """Normalize: strip \r, strip trailing whitespace each line."""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    return '\n'.join(line.rstrip() for line in lines).strip()

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ── 1. Read original HTML_TEMPLATE from backup ─────────────────────────────────
print(f"Reading backup: {BAK_FILE}")
bak_src = read_file(BAK_FILE)
bak_lines = bak_src.splitlines(keepends=True)
print(f"  {len(bak_lines):,} lines")

# Find HTML_TEMPLATE block
html_start = None
for i, ln in enumerate(bak_lines):
    if ln.strip().startswith("HTML_TEMPLATE = '''"):
        html_start = i
        break

html_end = None
for i in range(html_start + 1, len(bak_lines)):
    if bak_lines[i].strip() == "'''":
        html_end = i
        break

orig_html = ''.join(bak_lines[html_start + 1 : html_end])
print(f"  HTML_TEMPLATE: lines {html_start+1}–{html_end+1}, {len(orig_html):,} chars")

# Find LOGIN_TEMPLATE block
login_start = None
for i, ln in enumerate(bak_lines):
    if ln.strip().startswith("LOGIN_TEMPLATE = '''"):
        login_start = i
        break
login_end = None
for i in range(login_start + 1, len(bak_lines)):
    if bak_lines[i].strip() == "'''":
        login_end = i
        break
orig_login = ''.join(bak_lines[login_start + 1 : login_end])

# ── 2. Extract CSS and JS from original ────────────────────────────────────────
css_orig_m = re.search(r'<style>(.*?)</style>', orig_html, re.DOTALL)
css_orig = css_orig_m.group(1) if css_orig_m else ''

js_orig_m = re.search(r'<script(?![^>]*src=)>(.*?)</script>', orig_html, re.DOTALL)
js_orig = js_orig_m.group(1) if js_orig_m else ''

# ── 3. Read extracted files ────────────────────────────────────────────────────
css_new = read_file(CSS_FILE)
js_new  = read_file(JS_FILE)
login_new = read_file(os.path.join(TEMPLATES_DIR, 'login.html'))

PAGE_FILES = [
    ('dashboard-ac',       'pages/dashboard_ac.html'),
    ('dashboard-lamp',     'pages/dashboard_lamp.html'),
    ('ac-analytics',       'pages/ac_analytics.html'),
    ('lamp-analytics',     'pages/lamp_analytics.html'),
    ('camera',             'pages/camera.html'),
    ('energy',             'pages/energy.html'),
    ('control-ac',         'pages/control_ac.html'),
    ('control-lamp',       'pages/control_lamp.html'),
    ('ml-optimization',    'pages/ml_optimization.html'),
    ('logs',               'pages/logs.html'),
    ('occupancy-feedback', 'pages/occupancy_feedback.html'),
    ('control-outlet',     'pages/control_outlet.html'),
    ('outlet-analysis',    'pages/outlet_analytics.html'),
]

def extract_div_by_id(html, div_id):
    pat = rf'(<div\b[^>]*\bid="{re.escape(div_id)}"[^>]*>)'
    m = re.search(pat, html, re.DOTALL)
    if not m:
        return None
    tag_end = m.end()
    depth = 1
    pos = tag_end
    while pos < len(html) and depth > 0:
        open_m  = re.search(r'<div\b', html[pos:])
        close_m = re.search(r'</div\s*>', html[pos:])
        if close_m is None:
            break
        if open_m and open_m.start() < close_m.start():
            depth += 1
            pos += open_m.end()
        else:
            depth -= 1
            end_of_close = pos + close_m.end()
            if depth == 0:
                return html[m.start():end_of_close]
            pos = end_of_close
    return None

# ── 4. Compare CSS ─────────────────────────────────────────────────────────────
errors = []

def compare(name, orig, new):
    o = norm(orig)
    n = norm(new)
    if o == n:
        print(f"  ✅ {name}: IDENTICAL")
        return True
    else:
        diff = list(difflib.unified_diff(
            o.splitlines(), n.splitlines(),
            fromfile=f'{name} (original)',
            tofile=f'{name} (new)',
            lineterm='', n=3
        ))
        print(f"  ❌ {name}: DIFFERENT ({len(diff)} diff lines)")
        # Show first 40 lines of diff
        for dl in diff[:40]:
            print(f"     {dl}")
        if len(diff) > 40:
            print(f"     ... ({len(diff)-40} more lines)")
        errors.append(name)
        return False

print("\n── CSS ────────────────────────────────────────────────────────────────")
compare('dashboard.css', css_orig, css_new)

print("\n── JS ─────────────────────────────────────────────────────────────────")
compare('dashboard.js', js_orig, js_new)

print("\n── Login Template ─────────────────────────────────────────────────────")
compare('login.html', orig_login, login_new)

print("\n── Page Sections ──────────────────────────────────────────────────────")
for div_id, rel_path in PAGE_FILES:
    orig_block = extract_div_by_id(orig_html, div_id)
    new_block  = read_file(os.path.join(TEMPLATES_DIR, rel_path))
    if orig_block is None:
        print(f"  ⚠️  {div_id}: NOT FOUND IN ORIGINAL")
        errors.append(div_id)
        continue
    compare(rel_path, orig_block, new_block)

print("\n── Sidebar ─────────────────────────────────────────────────────────────")
sidebar_orig = extract_div_by_id(orig_html, 'sidebar')
sidebar_new  = read_file(os.path.join(TEMPLATES_DIR, 'partials', 'sidebar.html'))
compare('sidebar.html', sidebar_orig or '', sidebar_new)

print("\n── Sensor Health Bar ────────────────────────────────────────────────────")
shb_orig = extract_div_by_id(orig_html, 'sensor-health-bar')
shb_new  = read_file(os.path.join(TEMPLATES_DIR, 'partials', 'sensor_health_bar.html'))
compare('sensor_health_bar.html', shb_orig or '', shb_new)

print("\n── Energy Bubble ────────────────────────────────────────────────────────")
eb_orig = extract_div_by_id(orig_html, 'energy-bubble')
eb_new  = read_file(os.path.join(TEMPLATES_DIR, 'partials', 'modals.html'))
compare('modals.html (energy-bubble)', eb_orig or '', eb_new)

print("\n── app.py routes unchanged ──────────────────────────────────────────────")
new_app = read_file(os.path.join(BASE_DIR, 'app.py'))
# Verify critical route functions are still present
for needle in [
    "def login()",
    "def index()",
    "def video_feed()",
    "def control_ac()",
    "def control_lamp()",
    "def ml_run()",
    "def energy_history()",
    "render_template('login.html'",
    "render_template('dashboard.html')",
    "from flask import Flask, render_template,",
]:
    if needle in new_app:
        print(f"  ✅ Found: {needle!r}")
    else:
        print(f"  ❌ MISSING: {needle!r}")
        errors.append(f"missing:{needle}")

# Verify render_template_string is gone
if 'render_template_string' in new_app:
    print("  ❌ render_template_string still present in app.py!")
    errors.append("render_template_string still present")
else:
    print("  ✅ render_template_string removed from app.py")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f"❌ VERIFICATION FAILED — {len(errors)} issue(s):")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("✅ ALL CHECKS PASSED — Tampilan & fungsionalitas IDENTIK dengan original")
