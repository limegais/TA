"""
extract_templates.py  — PRECISE version
Splits app.py monolithic HTML_TEMPLATE into separate files:
  - static/css/dashboard.css
  - static/js/dashboard.js
  - templates/login.html
  - templates/partials/sidebar.html
  - templates/partials/sensor_health_bar.html
  - templates/partials/modals.html
  - templates/pages/dashboard_ac.html
  - templates/pages/dashboard_lamp.html
  - templates/pages/ac_analytics.html
  - templates/pages/lamp_analytics.html
  - templates/pages/camera.html
  - templates/pages/energy.html
  - templates/pages/control_ac.html
  - templates/pages/control_lamp.html
  - templates/pages/ml_optimization.html
  - templates/pages/logs.html
  - templates/pages/occupancy_feedback.html
  - templates/pages/control_outlet.html
  - templates/pages/outlet_analytics.html

Then rewrites app.py to remove the two template variables and use render_template.
"""

import os, re, sys

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
APP_PY        = os.path.join(BASE_DIR, 'app.py')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
PAGES_DIR     = os.path.join(TEMPLATES_DIR, 'pages')
PARTIALS_DIR  = os.path.join(TEMPLATES_DIR, 'partials')
CSS_FILE      = os.path.join(BASE_DIR, 'static', 'css', 'dashboard.css')
JS_FILE       = os.path.join(BASE_DIR, 'static', 'js', 'dashboard.js')

for d in [PAGES_DIR, PARTIALS_DIR,
          os.path.dirname(CSS_FILE), os.path.dirname(JS_FILE)]:
    os.makedirs(d, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────
def write_file(path, content):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(content)
    print(f"  wrote  {os.path.relpath(path, BASE_DIR)}  ({len(content):,} chars)")

def read_lines_0idx(lines, start_0, end_0):
    """Return joined string of lines[start_0 .. end_0] (end_0 EXCLUSIVE, 0-indexed)."""
    return ''.join(lines[start_0:end_0])

# ── read source ────────────────────────────────────────────────────────────────
print(f"Reading {APP_PY} …")
with open(APP_PY, 'r', encoding='utf-8') as fh:
    raw = fh.read()
lines = raw.splitlines(keepends=True)   # keeps \r\n or \n
print(f"  {len(lines):,} lines, {len(raw):,} bytes")

# ── locate template boundaries (1-indexed line numbers from grep output) ───────
#
# LOGIN_TEMPLATE = ''' starts at line 5131
# HTML_TEMPLATE  = ''' starts at line 5182
# In Python, line N (1-indexed)  ==  lines[N-1]  (0-indexed)

LOGIN_TPL_START = 5131 - 1   # 0-indexed: opening line  LOGIN_TEMPLATE = '''
LOGIN_TPL_END   = 5179 - 1   # 0-indexed: closing line  '''   (inclusive)

HTML_TPL_START  = 5182 - 1   # 0-indexed: opening line  HTML_TEMPLATE = '''
# End of file after '''  — we'll scan for the closing '''
def find_closing_triplequote(lines_list, start_0idx):
    for i in range(start_0idx + 1, len(lines_list)):
        s = lines_list[i].rstrip('\r\n')
        if s.strip() == "'''":
            return i
    raise RuntimeError("Could not find closing ''' for HTML_TEMPLATE")

HTML_TPL_END = find_closing_triplequote(lines, HTML_TPL_START)   # 0-indexed inclusive
print(f"  LOGIN_TEMPLATE: lines {LOGIN_TPL_START+1}–{LOGIN_TPL_END+1}")
print(f"  HTML_TEMPLATE:  lines {HTML_TPL_START+1}–{HTML_TPL_END+1}")

# Extract raw HTML strings (content between the triple-quote delimiters)
login_html = ''.join(lines[LOGIN_TPL_START + 1 : LOGIN_TPL_END])
main_html  = ''.join(lines[HTML_TPL_START  + 1 : HTML_TPL_END ])

# ── CSS (lines 5195–5920 in app.py, i.e. inside <style>…</style> of HTML_TEMPLATE) ──
css_m = re.search(r'<style>(.*?)</style>', main_html, re.DOTALL)
if not css_m:
    sys.exit("ERROR: Cannot find <style>…</style> in HTML_TEMPLATE")
css_raw = css_m.group(1).strip('\n') + '\n'

# ── JS (the one big inline <script> block, no src=) ──────────────────────────
# All CDN scripts have src=; our big JS has no src= attribute
js_m = re.search(r'<script(?![^>]*src=)>(.*?)</script>', main_html, re.DOTALL)
if not js_m:
    sys.exit("ERROR: Cannot find inline <script>…</script> in HTML_TEMPLATE")
js_raw = js_m.group(1).strip('\n') + '\n'

print(f"  CSS: {len(css_raw):,} chars")
print(f"  JS:  {len(js_raw):,} chars")

# ── Write CSS and JS ──────────────────────────────────────────────────────────
write_file(CSS_FILE, css_raw)
write_file(JS_FILE,  js_raw)

# ── Write Login template ──────────────────────────────────────────────────────
write_file(os.path.join(TEMPLATES_DIR, 'login.html'), login_html)

# ── Extract div blocks by id ──────────────────────────────────────────────────
def extract_div_by_id(html, div_id):
    """Return the full <div id="div_id" ...>...</div> block as a string."""
    pat = rf'(<div\b[^>]*\bid="{re.escape(div_id)}"[^>]*>)'
    m = re.search(pat, html, re.DOTALL)
    if not m:
        return None
    tag_end = m.end()
    depth = 1
    pos = tag_end
    while pos < len(html) and depth > 0:
        # Find next <div or </div (whichever comes first)
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

# page section ids
PAGE_IDS = [
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

for div_id, rel_path in PAGE_IDS:
    block = extract_div_by_id(main_html, div_id)
    if block:
        write_file(os.path.join(TEMPLATES_DIR, rel_path), block + '\n')
    else:
        print(f"  [WARN] Could not extract: {div_id}")

# ── Extract sidebar ────────────────────────────────────────────────────────────
sidebar = extract_div_by_id(main_html, 'sidebar')
if sidebar:
    write_file(os.path.join(PARTIALS_DIR, 'sidebar.html'), sidebar + '\n')

# ── Extract sensor-health-bar ──────────────────────────────────────────────────
shb = extract_div_by_id(main_html, 'sensor-health-bar')
if shb:
    write_file(os.path.join(PARTIALS_DIR, 'sensor_health_bar.html'), shb + '\n')

# ── Extract energy-bubble + all modals → partials/modals.html ─────────────────
modals_parts = []
eb = extract_div_by_id(main_html, 'energy-bubble')
if eb:
    modals_parts.append(eb)
# Collect any div class="modal ..." blocks
for m in re.finditer(r'<div\b[^>]*\bclass="modal\b[^"]*"[^>]*>', main_html):
    block = extract_div_by_id(main_html[m.start():], 
                               re.search(r'\bid="([^"]+)"', m.group(0), re.DOTALL) and
                               re.search(r'\bid="([^"]+)"', m.group(0)).group(1) or '')
    if block:
        modals_parts.append(block)

if modals_parts:
    write_file(os.path.join(PARTIALS_DIR, 'modals.html'), '\n'.join(modals_parts) + '\n')

# ── Build new base.html ────────────────────────────────────────────────────────
# Check if it already references the external CSS/JS
base_html_path = os.path.join(TEMPLATES_DIR, 'base.html')
with open(base_html_path, 'r', encoding='utf-8') as fh:
    base_content = fh.read()
print(f"  base.html already present, not overwritten:\n{base_content[:200]}")

# ── Rewrite app.py ────────────────────────────────────────────────────────────
print("\nPatching app.py …")

new_lines = list(lines)  # copy

# 1. Replace "from flask import Flask, render_template_string, ..."
#    → add render_template, remove render_template_string
for i, ln in enumerate(new_lines):
    if 'render_template_string' in ln and 'from flask import' in ln:
        new_lines[i] = ln.replace('render_template_string, ', '').replace(
            'render_template_string,', '').replace(
            ', render_template_string', '')
        # ensure render_template is present
        if 'render_template' not in new_lines[i]:
            new_lines[i] = new_lines[i].replace(
                'from flask import Flask,', 'from flask import Flask, render_template,')
        break

# 2. Replace render_template_string(LOGIN_TEMPLATE, ...) → render_template('login.html', ...)
for i, ln in enumerate(new_lines):
    if "render_template_string(LOGIN_TEMPLATE," in ln:
        new_lines[i] = ln.replace(
            "render_template_string(LOGIN_TEMPLATE, error='Invalid username or password')",
            "render_template('login.html', error='Invalid username or password')"
        ).replace(
            "render_template_string(LOGIN_TEMPLATE, error=None)",
            "render_template('login.html', error=None)"
        )
    elif "render_template_string(HTML_TEMPLATE)" in ln:
        new_lines[i] = ln.replace(
            "render_template_string(HTML_TEMPLATE)",
            "render_template('dashboard.html')"
        )

# 3. Remove LOGIN_TEMPLATE and HTML_TEMPLATE variable blocks
#    Replace lines LOGIN_TPL_START..LOGIN_TPL_END and HTML_TPL_START..HTML_TPL_END
#    with a short comment

login_replacement = [
    "# LOGIN_TEMPLATE moved to templates/login.html\n",
]
html_replacement = [
    "# HTML_TEMPLATE (CSS + HTML + JS) moved to:\n",
    "#   static/css/dashboard.css\n",
    "#   static/js/dashboard.js\n",
    "#   templates/ (dashboard.html, pages/, partials/)\n",
]

# Build the final line list:
# Keep lines before LOGIN_TPL_START, insert replacement, skip to LOGIN_TPL_END+1,
# keep lines up to HTML_TPL_START, insert replacement, skip to HTML_TPL_END+1,
# keep the rest.
result = (
    new_lines[:LOGIN_TPL_START] +
    login_replacement +
    new_lines[LOGIN_TPL_END + 1 : HTML_TPL_START] +
    html_replacement +
    new_lines[HTML_TPL_END + 1 :]
)

new_app_py = ''.join(result)

# Backup original
import shutil
bak = APP_PY + '.bak'
if not os.path.exists(bak):
    shutil.copy2(APP_PY, bak)
    print(f"  Backed up original to {bak}")

write_file(APP_PY, new_app_py)
print(f"\nDone! Lines in new app.py: {new_app_py.count(chr(10)):,}")
print("Check that 'render_template' is imported (not render_template_string).")
