        window.onerror = function(msg, url, line, col, error) {
            console.error('[JS ERROR] ' + msg + ' at line ' + line + ':' + col);
            alert('[JS ERROR] ' + msg + '\nLine: ' + line + ':' + col);
            return false;
        };

        var socket = null;
        try {
            socket = io();
            console.log('[OK] Socket.IO connected');
        } catch(e) {
            console.warn('[WARN] Socket.IO not available:', e.message);
            socket = { on: function(){}, emit: function(){} };
        }
        
        var charts = {};

        // ==================== ROLE-BASED ACCESS CONTROL ====================
        var userRole = 'admin'; // default until /api/auth/role responds
        var ADMIN_PAGES = ['ac-analytics','lamp-analytics','camera','energy','ml-optimization','logs','occupancy-feedback','outlet-analysis'];

        function applyRoleRestrictions(role, username) {
            userRole = role;
            var uEl = document.getElementById('role-username');
            var lEl = document.getElementById('role-label');
            if (uEl) uEl.textContent = username || role;
            if (lEl) {
                lEl.textContent = role === 'admin' ? 'ADMIN' : 'USER';
                lEl.style.background = role === 'admin' ? '#2563eb' : '#3b82f6';
            }
            if (role !== 'admin') {
                document.body.classList.add('role-user');
            } else {
                document.body.classList.remove('role-user');
            }
        }
        var chartRanges = {
            temp: 1,
            hum: 1,
            acTemp: 1,
            acPower: 1,
            lampLux: 1,
            lampBright: 1,
            lampPower: 1,
            occupancy: 1
        };

        var selectedFeedbackRating = 0;
        var DEFAULT_GOOGLE_FORM_URL = 'https://docs.google.com/forms/';

        var learnedCodes = {};

        function formatChartValue(v, unit) {
            const n = parseFloat(v || 0);
            if (!Number.isFinite(n)) return '--';
            return (unit ? n.toFixed(2) + ' ' + unit : n.toFixed(2));
        }

        // styleLineChart removed - all styling moved into makeOpts/makeEnergyOpts at chart creation
        // Mutating chart.options post-creation causes _scriptable recursion in Chart.js 4.x

        // Gradients removed - CanvasGradient objects cause _scriptable recursion in Chart.js 4.x
        // Static rgba() backgroundColor set at chart creation provides the fill effect

        // ==================== LOCALSTORAGE PERSISTENCE ====================
        function saveSettings() {
            const getVal = (id, fallback) => {
                const el = document.getElementById(id);
                return el ? el.value : fallback;
            };
            const settings = {
                acTemp: getVal('ac-temp-slider', 24) || 24,
                fanSpeed: getVal('fan-speed-slider', 1) || 1,
                lampBrightness1: getVal('brightness-slider-1', 0) || 0,
                lampBrightness2: getVal('brightness-slider-2', 0) || 0,
            };
            localStorage.setItem('smartroom_settings', JSON.stringify(settings));
        }

        function loadSavedSettings() {
            const saved = localStorage.getItem('smartroom_settings');
            if (saved) {
                try {
                    const settings = JSON.parse(saved);
                    
                    const acTempSlider = document.getElementById('ac-temp-slider');
                    const fanSpeedSlider = document.getElementById('fan-speed-slider');
                    
                    if (acTempSlider) {
                        acTempSlider.value = settings.acTemp || 24;
                        const acTempDisplay = document.getElementById('ac-temp-display');
                        if (acTempDisplay) acTempDisplay.textContent = acTempSlider.value;
                    }
                    
                    if (fanSpeedSlider) {
                        fanSpeedSlider.value = settings.fanSpeed || 1;
                        const fanSpeedDisplay = document.getElementById('fan-speed-display');
                        if (fanSpeedDisplay) fanSpeedDisplay.textContent = fanSpeedSlider.value;
                    }
                    
                    for (let i = 1; i <= 2; i++) {
                        const slider = document.getElementById('brightness-slider-' + i);
                        if (slider) {
                            slider.value = settings['lampBrightness' + i] || 0;
                            const brightDisplay = document.getElementById('brightness-display-' + i);
                            if (brightDisplay) brightDisplay.textContent = slider.value;
                        }
                    }
                } catch (e) {
                    console.error('Error loading settings:', e);
                }
            }
        }

        // Energy-specific options with tooltip formatting and styled axes.
        // This is used both during initial chart creation and later when Energy Usage
        // charts are recreated between line and bar types.
        function makeEnergyOpts(unit) {
            return {
                responsive: true,
                maintainAspectRatio: true,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#f0f7ff',
                        bodyColor: '#bfdbfe',
                        borderColor: 'rgba(148,163,184,0.35)',
                        borderWidth: 1,
                        displayColors: true,
                        callbacks: {
                            label: function(ctx) {
                                var dsLabel = (ctx.dataset && ctx.dataset.label) ? ctx.dataset.label : 'Value';
                                var delta = ctx.parsed.y;
                                // If kWh chart and cumulative arrays exist, show calculation
                                if (unit === 'kWh' && ctx.chart._kwhCumAc) {
                                    var idx = ctx.dataIndex;
                                    var cumArr = [ctx.chart._kwhCumAc, ctx.chart._kwhCumOutlet, ctx.chart._kwhCumLamp, null][ctx.datasetIndex];
                                    if (cumArr && cumArr.length > idx + 1) {
                                        var prev = cumArr[idx].toFixed(4);
                                        var curr = cumArr[idx + 1].toFixed(4);
                                        return dsLabel + ': ' + curr + ' \u2212 ' + prev + ' = ' + delta.toFixed(5) + ' kWh';
                                    }
                                    return dsLabel + ': ' + delta.toFixed(5) + ' kWh';
                                }
                                return dsLabel + ': ' + formatChartValue(delta, unit);
                            },
                            title: function(items) {
                                if (!items || !items[0]) return '';
                                var lbl = items[0].label || '';
                                if (unit === 'kWh') {
                                    return lbl.indexOf('\u2192') >= 0 ? lbl.replace('\u2192', ' \u2192 ') : lbl;
                                }
                                return 'Time: ' + lbl;
                            }
                        }
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(148,163,184,0.12)' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
                    y: { beginAtZero: true, grid: { color: 'rgba(148,163,184,0.14)' }, ticks: { color: '#94a3b8' } }
                }
            };
        }

        // ==================== CHARTS ====================
        function initCharts() {
            function makeOpts(showLegend) {
                var opts = {
                    responsive: true,
                    maintainAspectRatio: true,
                    animation: false,
                    plugins: { legend: { display: !!showLegend } },
                    scales: {
                        y: { beginAtZero: false, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } }
                    }
                };
                if (showLegend) {
                    opts.plugins.legend.labels = { color: '#94a3b8', font: { size: 12 } };
                }
                return opts;
            }

            charts.temp = new Chart(document.getElementById('tempChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Temperature (\u00b0C)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: true }] }
            });

            charts.hum = new Chart(document.getElementById('humChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Humidity (%)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.4, fill: true }] }
            });

            charts.acTemp = new Chart(document.getElementById('acTempChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Set Temperature (\u00b0C)', data: [], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.4, fill: true }] }
            });

            charts.acHum = new Chart(document.getElementById('acHumChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Set RH (%)', data: [], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.4, fill: true }] }
            });

            charts.lampLux = new Chart(document.getElementById('lampLuxChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Light Intensity (lux)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.1)', tension: 0.4, fill: true }] }
            });

            charts.lampBright = new Chart(document.getElementById('lampBrightChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Brightness (%)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.4, fill: true }] }
            });

            charts.acPower = new Chart(document.getElementById('acPowerChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'AC Power (W)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: true, pointRadius: 3, pointHoverRadius: 6 }] }
            });

            charts.lampPower = new Chart(document.getElementById('lampPowerChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Lamp Power (W)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.1)', tension: 0.4, fill: true, pointRadius: 3, pointHoverRadius: 6 }] }
            });

            // AC & Lamp individual energy kWh charts (analytics pages)
            charts.acEnergyKwh = new Chart(document.getElementById('acEnergyKwhChart'), {
                type: 'line',
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f0f7ff',
                            bodyColor: '#bfdbfe', borderColor: 'rgba(59,130,246,0.4)', borderWidth: 1,
                            displayColors: false,
                            callbacks: {
                                label: function(ctx) { return 'Delta: ' + ctx.parsed.y.toFixed(5) + ' kWh'; }
                            }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.10)' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } } },
                        y: { beginAtZero: true, grid: { color: 'rgba(59,130,246,0.10)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'kWh', color: '#64748b', font: { size: 10 } } }
                    }
                },
                data: { labels: [], datasets: [{
                    label: 'AC Energy (kWh)', data: [],
                    borderColor: '#3b82f6', borderWidth: 2, tension: 0.3,
                    backgroundColor: function(ctx) {
                        var c = ctx.chart.ctx; var a = ctx.chart.chartArea; if (!a) return 'rgba(59,130,246,0.1)';
                        var g = c.createLinearGradient(0, a.top, 0, a.bottom);
                        g.addColorStop(0, 'rgba(59,130,246,0.25)'); g.addColorStop(1, 'rgba(59,130,246,0.02)'); return g;
                    },
                    fill: true, pointRadius: 2, pointHoverRadius: 6,
                    pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 2
                }] }
            });

            charts.lampEnergyKwh = new Chart(document.getElementById('lampEnergyKwhChart'), {
                type: 'line',
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f0f7ff',
                            bodyColor: '#bfdbfe', borderColor: 'rgba(37,99,235,0.4)', borderWidth: 1,
                            displayColors: false,
                            callbacks: {
                                label: function(ctx) { return 'Delta: ' + ctx.parsed.y.toFixed(5) + ' kWh'; }
                            }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.10)' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } } },
                        y: { beginAtZero: true, grid: { color: 'rgba(37,99,235,0.10)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'kWh', color: '#64748b', font: { size: 10 } } }
                    }
                },
                data: { labels: [], datasets: [{
                    label: 'Lamp Energy (kWh)', data: [],
                    borderColor: '#2563eb', borderWidth: 2, tension: 0.3,
                    backgroundColor: function(ctx) {
                        var c = ctx.chart.ctx; var a = ctx.chart.chartArea; if (!a) return 'rgba(37,99,235,0.1)';
                        var g = c.createLinearGradient(0, a.top, 0, a.bottom);
                        g.addColorStop(0, 'rgba(37,99,235,0.25)'); g.addColorStop(1, 'rgba(37,99,235,0.02)'); return g;
                    },
                    fill: true, pointRadius: 2, pointHoverRadius: 6,
                    pointBackgroundColor: '#2563eb', pointBorderColor: '#fff', pointBorderWidth: 2
                }] }
            });

            charts.outletEnergyKwhMySQL = new Chart(document.getElementById('outletEnergyKwhMySQLChart'), {
                type: 'line',
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f0f7ff',
                            bodyColor: '#a7f3d0', borderColor: 'rgba(5,150,105,0.4)', borderWidth: 1,
                            displayColors: false,
                            callbacks: {
                                label: function(ctx) { return 'Delta: ' + ctx.parsed.y.toFixed(5) + ' kWh'; }
                            }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.10)' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } } },
                        y: { beginAtZero: true, grid: { color: 'rgba(5,150,105,0.10)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'kWh', color: '#64748b', font: { size: 10 } } }
                    }
                },
                data: { labels: [], datasets: [{
                    label: 'Outlet Energy (kWh)', data: [],
                    borderColor: '#059669', borderWidth: 2, tension: 0.3,
                    backgroundColor: function(ctx) {
                        var c = ctx.chart.ctx; var a = ctx.chart.chartArea; if (!a) return 'rgba(5,150,105,0.1)';
                        var g = c.createLinearGradient(0, a.top, 0, a.bottom);
                        g.addColorStop(0, 'rgba(5,150,105,0.25)'); g.addColorStop(1, 'rgba(5,150,105,0.02)'); return g;
                    },
                    fill: true, pointRadius: 2, pointHoverRadius: 6,
                    pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2
                }] }
            });


            charts.energyPower = new Chart(document.getElementById('energyPowerChart'), {
                type: 'line', options: makeEnergyOpts('W'),
                data: { labels: [], datasets: [
                    { label: 'AC Power (W)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Outlet Power (W)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Lamp Power (W)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Total Power (W)', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.08)', tension: 0.4, fill: false, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#7c3aed', pointBorderColor: '#fff', pointBorderWidth: 2, borderDash: [5,3] }
                ]}
            });

            charts.energyVoltage = new Chart(document.getElementById('energyVoltageChart'), {
                type: 'line', options: makeEnergyOpts('V'),
                data: { labels: [], datasets: [
                    { label: 'AC Voltage (V)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Outlet Voltage (V)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Lamp Voltage (V)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2 }
                ]}
            });

            charts.energyCurrent = new Chart(document.getElementById('energyCurrentChart'), {
                type: 'line', options: makeEnergyOpts('A'),
                data: { labels: [], datasets: [
                    { label: 'AC Current (A)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Outlet Current (A)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Lamp Current (A)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', tension: 0.4, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2 },
                    { label: 'Total Current (A)', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.08)', tension: 0.4, fill: false, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#7c3aed', pointBorderColor: '#fff', pointBorderWidth: 2, borderDash: [5,3] }
                ]}
            });

            charts.energyKwh = new Chart(document.getElementById('energyKwhChart'), {
                type: 'line', options: makeEnergyOpts('kWh'),
                data: { labels: [], datasets: [
                    { label: 'AC Energy (kWh)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.08)', tension: 0.3, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                    { label: 'Outlet Energy (kWh)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.08)', tension: 0.3, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                    { label: 'Lamp Energy (kWh)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.08)', tension: 0.3, fill: true, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                    { label: 'Total Energy (kWh)', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.06)', tension: 0.3, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#7c3aed', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2, borderDash: [5,3] }
                ]}
            });

            // MySQL Real-time Charts (ring buffer, 30 points)
            charts.mysqlVoltFreq = new Chart(document.getElementById('mysqlVoltFreqChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Frequency (Hz)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.12)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.08)' }, ticks: { color: '#94a3b8', maxTicksLimit: 8, maxRotation: 0, font: { size: 11 } } },
                        y: { title: { display: true, text: 'Hz', color: '#94a3b8', font: { size: 11 } }, suggestedMin: 49, suggestedMax: 51, ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(148,163,184,0.08)' } }
                    }
                }
            });

            charts.mysqlCurrent = new Chart(document.getElementById('mysqlCurrentChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'AC (A)', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true },
                        { label: 'Outlet (A)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true },
                        { label: 'Lamp (A)', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    plugins: { legend: { display: true, labels: { boxWidth: 12, font: { size: 11 }, color: '#94a3b8', padding: 16 } } },
                    scales: {
                        x: { grid: { color: 'rgba(148,163,184,0.08)' }, ticks: { color: '#94a3b8', maxTicksLimit: 8, maxRotation: 0, font: { size: 11 } } },
                        y: { beginAtZero: true, title: { display: true, text: 'A', color: '#94a3b8', font: { size: 11 } }, ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(148,163,184,0.08)' } }
                    }
                }
            });

            // Energy Comparison: 4 separate charts (Before/After x Power/kWh) for side-by-side view
            var compareLineOpts = function(unit) {
                return {
                    responsive: true,
                    maintainAspectRatio: true,
                    animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 }, maxRotation: 45, maxTicksLimit: 12 } },
                        y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { color: '#94a3b8', font: { size: 10 } }, title: { display: true, text: unit, color: '#94a3b8', font: { size: 11 } } }
                    }
                };
            };

            charts.energyCompareBefore = new Chart(document.getElementById('energyCompareBeforeChart'), {
                type: 'line', options: compareLineOpts('W'),
                data: { labels: [], datasets: [{ label: 'Before - Power', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.energyCompareAfter = new Chart(document.getElementById('energyCompareAfterChart'), {
                type: 'line', options: compareLineOpts('W'),
                data: { labels: [], datasets: [{ label: 'After - Power', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.energyCompareKwhBefore = new Chart(document.getElementById('energyCompareKwhBeforeChart'), {
                type: 'line', options: compareLineOpts('kWh'),
                data: { labels: [], datasets: [{ label: 'Before - Energy', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.energyCompareKwhAfter = new Chart(document.getElementById('energyCompareKwhAfterChart'), {
                type: 'line', options: compareLineOpts('kWh'),
                data: { labels: [], datasets: [{ label: 'After - Energy', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });

            // Lamp Comparison Charts
            charts.lampCompareBefore = new Chart(document.getElementById('lampCompareBeforeChart'), {
                type: 'line', options: compareLineOpts('W'),
                data: { labels: [], datasets: [{ label: 'Before - Lamp Power', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.lampCompareAfter = new Chart(document.getElementById('lampCompareAfterChart'), {
                type: 'line', options: compareLineOpts('W'),
                data: { labels: [], datasets: [{ label: 'After - Lamp Power', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.lampCompareKwhBefore = new Chart(document.getElementById('lampCompareKwhBeforeChart'), {
                type: 'line', options: compareLineOpts('kWh'),
                data: { labels: [], datasets: [{ label: 'Before - Lamp Energy', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });
            charts.lampCompareKwhAfter = new Chart(document.getElementById('lampCompareKwhAfterChart'), {
                type: 'line', options: compareLineOpts('kWh'),
                data: { labels: [], datasets: [{ label: 'After - Lamp Energy', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2, pointHoverRadius: 5, pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });

            charts.occupancy = new Chart(document.getElementById('occupancyChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'Occupancy (person)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.35, fill: true, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#0ea5e9', pointBorderColor: '#fff', pointBorderWidth: 1 }] }
            });

            // Energy chart styling is set at creation via makeEnergyOpts - no post-init mutation needed
            // ML charts (gaFitness, psoFitness, comparison) are initialized separately via initMLCharts()
        }

        function initMLCharts() {
            function makeOpts(showLegend) {
                var opts = {
                    responsive: true, maintainAspectRatio: true, animation: false,
                    plugins: { legend: { display: !!showLegend } },
                    scales: {
                        y: { beginAtZero: false, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } }
                    }
                };
                if (showLegend) {
                    opts.plugins.legend.labels = { color: '#94a3b8', font: { size: 12 } };
                }
                return opts;
            }
            if (charts.gaFitness) { try { charts.gaFitness.destroy(); } catch(e){} charts.gaFitness = null; }
            if (charts.psoFitness) { try { charts.psoFitness.destroy(); } catch(e){} charts.psoFitness = null; }
            if (charts.comparison) { try { charts.comparison.destroy(); } catch(e){} charts.comparison = null; }

            charts.gaFitness = new Chart(document.getElementById('gaFitnessChart'), {
                type: 'line', options: makeOpts(false),
                data: { labels: [], datasets: [{ label: 'GA Best Fitness', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)', tension: 0.4, fill: true, pointRadius: 2 }] }
            });

            charts.psoFitness = new Chart(document.getElementById('psoFitnessChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'PWM 1 (0-255)', data: [], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.3, fill: false, pointRadius: 4, yAxisID: 'yPWM' },
                        { label: 'PWM 2 (0-255)', data: [], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.3, fill: false, pointRadius: 4, yAxisID: 'yPWM' },
                        { label: 'Lux Avg (lux)', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', tension: 0.3, fill: true, pointRadius: 4, yAxisID: 'yLux', borderDash: [4,2] },
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: true, animation: { duration: 300 },
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: true, labels: { color: '#94a3b8', font: { size: 11 } } },
                        tooltip: { callbacks: {
                            title: items => `Iteration ${items[0].label}`,
                            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}`
                        }}
                    },
                    scales: {
                        x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        yPWM: { type: 'linear', position: 'left', min: 0, max: 255, ticks: { color: '#2563eb', font: { size: 10 } }, title: { display: true, text: 'PWM', color: '#2563eb', font: { size: 10 } }, grid: { color: 'rgba(37,99,235,0.08)' } },
                        yLux: { type: 'linear', position: 'right', min: 0, ticks: { color: '#0ea5e9', font: { size: 10 } }, title: { display: true, text: 'Lux', color: '#0ea5e9', font: { size: 10 } }, grid: { drawOnChartArea: false } },
                    }
                }
            });



            // Load any pending ML data that was stored while the ML page was hidden
            if (window.__pendingMLData) {
                var pending = window.__pendingMLData;
                window.__pendingMLData = null;
                Object.keys(pending).forEach(function(chartName) {
                    var pd = pending[chartName];
                    if (pd && pd.history && pd.history.length > 0) {
                        try { updateMLChart(chartName, pd.history, pd.algo); } catch(e) { console.warn('[CHART] pending ML data load failed:', e); }
                    }
                });
            }
        }

        // Ensure charts are available even if init runs before library/page is fully ready.
        function ensureChartsReady() {
            if (typeof Chart === 'undefined') {
                console.warn('[CHART] Chart.js not loaded yet');
                if (!window.__chartFallbackRequested) {
                    window.__chartFallbackRequested = true;
                    try {
                        var s = document.createElement('script');
                        s.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js';
                        s.onload = function() {
                            console.log('[CHART] Fallback Chart.js loaded');
                            try { initCharts(); } catch(e) { console.error('[CHART] init after fallback failed:', e); }
                            try { initMLCharts(); } catch(e) { console.error('[CHART] initML after fallback failed:', e); }
                            try { loadAllEnergyCharts(); } catch(e) { console.error('[CHART] load after fallback failed:', e); }
                        };
                        s.onerror = function() {
                            console.error('[CHART] Fallback Chart.js failed to load');
                        };
                        document.head.appendChild(s);
                    } catch (e) {
                        console.error('[CHART] Fallback loader error:', e);
                    }
                }
                return false;
            }

            // Only call initCharts() if core charts (temp, hum) are also missing - meaning initCharts never ran.
            // If charts.temp already exists, initCharts() already ran and calling it again will crash on duplicate canvases.
            if ((!charts.energyPower || !charts.energyVoltage || !charts.energyKwh || !charts.energyCompareBefore || !charts.energyCompareKwhBefore) && !charts.temp) {
                try {
                    initCharts();
                } catch (e) {
                    console.error('[CHART] initCharts retry failed:', e);
                    return false;
                }
            }

            // ML charts are independent - init separately to avoid destroying already-created canvases
            // Guard: only init when canvas is visible (has positive width), to avoid 0px Chart.js failures
            if (!charts.gaFitness || !charts.psoFitness || !charts.comparison) {
                var mlCanvas = document.getElementById('gaFitnessChart');
                if (mlCanvas && mlCanvas.offsetWidth > 0) {
                    try {
                        initMLCharts();
                    } catch (e) {
                        console.error('[CHART] initMLCharts failed:', e);
                        return false;
                    }
                }
            }

            return true;
        }

        function updateChartData(chartName, hours) {
            let endpoint = '';
            switch(chartName) {
                case 'temp': endpoint = '/api/chart/ac_sensor/temperature/' + hours; break;
                case 'hum': endpoint = '/api/chart/ac_sensor/humidity/' + hours; break;
                case 'acTemp': endpoint = '/api/chart/ac_sensor/ac_temp/' + hours; break;
                case 'acHum': endpoint = '/api/chart/ac_sensor/set_rh/' + hours; break;
                case 'lampLux': endpoint = '/api/chart/lamp_sensor/lux/' + hours; break;
                case 'lampBright': endpoint = '/api/chart/lamp_sensor/brightness/' + hours; break;
                case 'occupancy': endpoint = '/api/chart/camera_detection/person_count/' + hours; break;
                case 'acPower': endpoint = '/api/chart/ac_power/' + hours; break;
                case 'lampPower': endpoint = '/api/chart/lamp_power/' + hours; break;
            }

            fetch(endpoint)
                .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
                .then(data => {
                    if (data && data.length > 0) {
                        charts[chartName].data.labels = data.map(d => d.time);
                        charts[chartName].data.datasets[0].data = data.map(d => d.value);
                        charts[chartName].update('none');
                    }
                })
                .catch(e => console.warn('Chart data unavailable (' + chartName + '):', e.message));
        }

        function changeChartRange(chartName, hours) {
            chartRanges[chartName] = hours;
            localStorage.setItem('chartRanges', JSON.stringify(chartRanges));
            
            const buttons = event.target.parentElement.querySelectorAll('.chart-option-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            updateChartData(chartName, hours);
        }

        // ==================== ENERGY HISTORY (PZEM-016 from InfluxDB) ====================
        var energyChartMap = {
            'power': 'energyPower',
            'voltage': 'energyVoltage',
            'current': 'energyCurrent',
            'energy_kwh': 'energyKwh'
        };

        var energyCanvasMap = {
            'power': 'energyPowerChart',
            'voltage': 'energyVoltageChart',
            'current': 'energyCurrentChart',
            'energy_kwh': 'energyKwhChart'
        };

        var energyColorMap = {
            'power': '#1e40af',
            'voltage': '#3b82f6',
            'current': '#0ea5e9',
            'energy_kwh': '#3b82f6'
        };

        function updateEnergyStats(field, values) {
            if (!values || values.length === 0) return;
            var prefixMap = {'power': 'energy-power', 'voltage': 'energy-voltage', 'current': 'energy-current', 'energy_kwh': 'energy-kwh'};
            const prefix = prefixMap[field] || ('energy-' + field);
            const minV = Math.min(...values);
            const maxV = Math.max(...values);
            const avgV = values.reduce((a, b) => a + b, 0) / values.length;
            const lastV = values[values.length - 1];
            const isKwh = field === 'energy_kwh';
            const totalV = values.reduce((a, b) => a + b, 0);
            const dp = isKwh ? 4 : 2;
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set(prefix + '-latest', lastV.toFixed(dp));
            set(prefix + '-min', minV.toFixed(dp));
            set(prefix + '-max', maxV.toFixed(dp));
            // kWh: show total sum of deltas
            set(prefix + '-avg', isKwh ? totalV.toFixed(4) : avgV.toFixed(2));
        }

        function drawEnergyFallback(field, points) {
            const canvasId = energyCanvasMap[field];
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;

            const rect = canvas.getBoundingClientRect();
            const width = Math.max(320, Math.floor(rect.width || canvas.width || 640));
            const height = Math.max(160, Math.floor(rect.height || canvas.height || 220));
            canvas.width = width;
            canvas.height = height;

            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = 'rgba(148, 163, 184, 0.12)';
            ctx.fillRect(0, 0, width, height);

            ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 4; i++) {
                const y = (height / 5) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            if (!points || points.length === 0) {
                ctx.fillStyle = '#94a3b8';
                ctx.font = '13px sans-serif';
                ctx.fillText('Waiting energy data...', 16, Math.floor(height / 2));
                return;
            }

            const values = points.map(p => parseFloat(p.value || 0)).filter(v => Number.isFinite(v));
            if (values.length === 0) return;

            let minV = Math.min(...values);
            let maxV = Math.max(...values);
            if (minV === maxV) {
                minV -= 1;
                maxV += 1;
            }

            const padX = 26;
            const padY = 18;
            const plotW = width - padX * 2;
            const plotH = height - padY * 2;

            ctx.strokeStyle = energyColorMap[field] || '#2563eb';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const dotColor = energyColorMap[field] || '#2563eb';
            values.forEach((v, i) => {
                const x = padX + (i / Math.max(1, values.length - 1)) * plotW;
                const y = padY + (1 - ((v - minV) / (maxV - minV))) * plotH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Point markers for each incoming sample
            values.forEach((v, i) => {
                const x = padX + (i / Math.max(1, values.length - 1)) * plotW;
                const y = padY + (1 - ((v - minV) / (maxV - minV))) * plotH;
                ctx.beginPath();
                ctx.arc(x, y, 2.8, 0, Math.PI * 2);
                ctx.fillStyle = dotColor;
                ctx.fill();
                ctx.lineWidth = 1;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();
            });

            const last = values[values.length - 1];
            ctx.fillStyle = '#cbd5e1';
            ctx.font = '12px sans-serif';
            ctx.fillText('Latest: ' + last.toFixed(2), 12, 14);
            updateEnergyStats(field, values);
        }

        // Track current period per field so auto-refresh doesn't reset user selection
        var _activePeriod = { power: '24h', voltage: '24h', current: '24h', energy_kwh: '24h' };

        // Energy is computed by integrating high-resolution Power data 
        // to avoid the 1kWh low-resolution stepping from the raw cumulative sensor.
        var _periodMins = {
            '1h': 1, '6h': 5, '24h': 60, '7d': 1440, '30d': 1440, '12mo': 43200, '5y': 525600
        };

        function loadEnergyHistory(field, period, btnElement) {
            // Save current period for this field
            _activePeriod[field] = period;
            // Update button active state
            if (btnElement) {
                const buttons = btnElement.parentElement.querySelectorAll('.chart-option-btn');
                buttons.forEach(btn => btn.classList.remove('active'));
                btnElement.classList.add('active');
            }

            const chartName = energyChartMap[field];
            if (!chartName) return;

            var fetchField = field;
            // Fetch AC, Outlet, and Lamp in parallel
            Promise.all([
                fetch('/api/energy/history?field=' + fetchField + '&period=' + period + '&device=ac').then(r => r.json()),
                fetch('/api/energy/history?field=' + fetchField + '&period=' + period + '&device=outlet').then(r => r.json()),
                fetch('/api/energy/history?field=' + fetchField + '&period=' + period + '&device=lamp').then(r => r.json())
            ])
            .then(function(results) {
                var acResult = results[0];
                var outletResult = results[1];
                var lampResult = results[2];
                var acData = acResult.data || [];
                var outletData = outletResult.data || [];
                var lampData = lampResult.data || [];
                var chart = charts[chartName];

                if (chart && chart.data && chart.data.datasets) {
                    var acValues = acData.map(function(d){ return parseFloat(d.value || 0); });
                    var outletValues = outletData.map(function(d){ return parseFloat(d.value || 0); });
                    var lampValues = lampData.map(function(d){ return parseFloat(d.value || 0); });
                    var acLabels = acData.map(function(d){ return d.time; });
                    var outletLabels = outletData.map(function(d){ return d.time; });
                    var lampLabels = lampData.map(function(d){ return d.time; });
                    
                    // energy_kwh is cumulative - compute delta per interval
                    if (field === 'energy_kwh') {
                        function computeDeltas(cumValues) {
                            var deltas = [];
                            if (cumValues.length <= 1) return cumValues.slice();
                            for (var i = 1; i < cumValues.length; i++) {
                                var diff = cumValues[i] - cumValues[i - 1];
                                deltas.push(parseFloat(Math.max(0, diff).toFixed(5)));
                            }
                            return deltas;
                        }
                        function buildRangeLabels(labels) {
                            var rangeLabels = [];
                            if (labels.length <= 1) return labels.slice();
                            for (var i = 1; i < labels.length; i++) {
                                rangeLabels.push(labels[i - 1] + '\u2192' + labels[i]);
                            }
                            return rangeLabels;
                        }

                        var acCumRaw = acValues.slice();
                        var outletCumRaw = outletValues.slice();
                        var lampCumRaw = lampValues.slice();
                        acValues = computeDeltas(acValues);
                        outletValues = computeDeltas(outletValues);
                        lampValues = computeDeltas(lampValues);

                        // Daily stays as a line; every longer Energy Usage period is a bar chart.
                        var useBarChart = (period !== '24h');
                        if (useBarChart) {
                            if (acLabels.length > 1) acLabels = acLabels.slice(1);
                            if (outletLabels.length > 1) outletLabels = outletLabels.slice(1);
                            if (lampLabels.length > 1) lampLabels = lampLabels.slice(1);
                        } else {
                            acLabels = buildRangeLabels(acLabels);
                            outletLabels = buildRangeLabels(outletLabels);
                            lampLabels = buildRangeLabels(lampLabels);
                        }
                    }

                    // -- Determine chart type: Daily (24h) = line, others = bar --
                    var useBarForAll = (period !== '24h');
                    var targetType = useBarForAll ? 'bar' : 'line';
                    var unitMap = { power: 'kW', voltage: 'V', current: 'A', energy_kwh: 'kWh' };
                    var unit = unitMap[field] || '';
                    var hasTotal = (field !== 'voltage');  // Voltage has no Total dataset

                    // Recreate chart if type needs to change (line <-> bar)
                    if (chart.config.type !== targetType) {
                        chart.destroy();
                        var canvasId = energyCanvasMap[field];
                        var canvas = document.getElementById(canvasId);
                        var baseOpts = makeEnergyOpts(unit);
                        var datasetsToUse;

                        if (useBarForAll) {
                            baseOpts.plugins.legend = { display: true, labels: { boxWidth: 14, font: { size: 11 }, color: '#94a3b8', padding: 14 } };
                            baseOpts.scales.x.grid = { display: false };
                            baseOpts.scales.x.ticks.maxRotation = 45;
                            baseOpts.scales.y.title = { display: true, text: unit, color: '#94a3b8', font: { size: 11 } };
                            
                            datasetsToUse = [
                                { label: 'AC (' + unit + ')', data: [], backgroundColor: 'rgba(30,64,175,0.6)', borderColor: '#1e40af', borderWidth: 1, borderRadius: 4, barPercentage: 0.6, categoryPercentage: 0.85 },
                                { label: 'Outlet (' + unit + ')', data: [], backgroundColor: 'rgba(5,150,105,0.6)', borderColor: '#059669', borderWidth: 1, borderRadius: 4, barPercentage: 0.6, categoryPercentage: 0.85 },
                                { label: 'Lamp (' + unit + ')', data: [], backgroundColor: 'rgba(234,179,8,0.6)', borderColor: '#eab308', borderWidth: 1, borderRadius: 4, barPercentage: 0.6, categoryPercentage: 0.85 }
                            ];
                            if (hasTotal) {
                                datasetsToUse.push({ label: 'Total (' + unit + ')', data: [], backgroundColor: 'rgba(124,58,237,0.4)', borderColor: '#7c3aed', borderWidth: 1, borderRadius: 4, barPercentage: 0.6, categoryPercentage: 0.85 });
                            }
                        } else {
                            // Line chart (Daily / 24h)
                            datasetsToUse = [
                                { label: 'AC (' + unit + ')', data: [], borderColor: '#1e40af', backgroundColor: 'rgba(30,64,175,0.08)', tension: 0.3, fill: (field === 'energy_kwh'), pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#1e40af', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                                { label: 'Outlet (' + unit + ')', data: [], borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.08)', tension: 0.3, fill: (field === 'energy_kwh'), pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#059669', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 },
                                { label: 'Lamp (' + unit + ')', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.08)', tension: 0.3, fill: (field === 'energy_kwh'), pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: '#eab308', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2 }
                            ];
                            if (hasTotal) {
                                datasetsToUse.push({ label: 'Total (' + unit + ')', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.06)', tension: 0.3, fill: false, pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: '#7c3aed', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2, borderDash: [5,3] });
                            }
                        }
                        
                        try {
                            chart = new Chart(canvas, {
                                type: targetType,
                                options: baseOpts,
                                data: { labels: [], datasets: datasetsToUse }
                            });
                            charts[chartName] = chart;
                        } catch (err) {
                            console.error('Chart.js recreation error:', err);
                        }
                    }

                    if (field === 'energy_kwh') {
                        chart._kwhCumAc = acCumRaw;
                        chart._kwhCumOutlet = outletCumRaw;
                        chart._kwhCumLamp = lampCumRaw;
                    }

                    // Compute Total values (AC + Outlet + Lamp)
                    var maxLen = Math.max(acValues.length, outletValues.length, lampValues.length);
                    var totalValues = [];
                    for (var ti = 0; ti < maxLen; ti++) {
                        totalValues.push((acValues[ti] || 0) + (outletValues[ti] || 0) + (lampValues[ti] || 0));
                    }
                    
                    chart.data.labels = acLabels.length > 0 ? acLabels : (outletLabels.length > 0 ? outletLabels : lampLabels);
                    chart.data.datasets[0].data = acValues;

                    // Voltage chart: datasets = [AC, Outlet, Lamp] (no Total)
                    if (field === 'voltage') {
                        if (chart.data.datasets[1]) chart.data.datasets[1].data = outletValues;
                        if (chart.data.datasets[2]) chart.data.datasets[2].data = lampValues;
                    } else {
                        // Power, Current, kWh: datasets = [AC, Outlet, Lamp, Total]
                        if (chart.data.datasets[1]) chart.data.datasets[1].data = outletValues;
                        if (chart.data.datasets[2]) chart.data.datasets[2].data = lampValues;
                        if (chart.data.datasets[3]) chart.data.datasets[3].data = totalValues;
                    }
                    chart.update('none');
                    updateEnergyStats(field, acValues.filter(function(v){ return Number.isFinite(v); }));
                    updateOutletEnergyStats(field, outletValues.filter(function(v){ return Number.isFinite(v); }));
                    updateLampEnergyStats(field, lampValues.filter(function(v){ return Number.isFinite(v); }));
                } else {
                    drawEnergyFallback(field, acData);
                }

                // Show "No Data" overlay on canvas if empty
                var canvasId = energyCanvasMap[field];
                var noDataId = 'nodata-' + canvasId;
                var existingOverlay = document.getElementById(noDataId);
                if (existingOverlay) existingOverlay.remove();
                if (acData.length === 0 && outletData.length === 0 && lampData.length === 0 && canvasId) {
                    var canvas = document.getElementById(canvasId);
                    if (canvas && canvas.parentElement) {
                        var overlay = document.createElement('div');
                        overlay.id = noDataId;
                        overlay.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:var(--text-secondary);font-size:14px;font-weight:600;pointer-events:none;text-align:center;';
                        overlay.textContent = 'Waiting for MySQL data (polling every 5 seconds)...';
                        canvas.parentElement.style.position = 'relative';
                        canvas.parentElement.appendChild(overlay);
                    }
                }
            })
            .catch(function(e){ console.error('Energy history error:', e); });
        }

        // -- toggleDataset: show/hide chart datasets via visibility checkboxes --
        function toggleDataset(chartName, datasetIndex, checkbox) {
            var chart = charts[chartName];
            if (!chart || !chart.data || !chart.data.datasets[datasetIndex]) return;
            chart.data.datasets[datasetIndex].hidden = !checkbox.checked;
            chart.update('none');
        }

        function updateOutletEnergyStats(field, values) {
            if (!values || values.length === 0) return;
            var avg = values.reduce(function(a,b){ return a+b; }, 0) / values.length;
            var latest = values[values.length - 1];
            var unit = {power:'kW', current:'A', energy_kwh:'kWh', voltage:'V'}[field] || '';
            var fmt = function(v) {
                return field === 'energy_kwh'
                    ? v.toFixed(4).replace(/\\.0+$/, '').replace(/(\\.\\d*?)0+$/, '$1')
                    : v.toFixed(2);
            };
            var setEl = function(id, v) { var el = document.getElementById(id); if (el) el.textContent = fmt(v) + ' ' + unit; };
            if (field === 'power') { setEl('outlet-power-latest', latest); setEl('outlet-power-avg', avg); }
            else if (field === 'current') { setEl('outlet-current-latest', latest); setEl('outlet-current-avg', avg); }
            else if (field === 'energy_kwh') {
                var total = values.reduce(function(a,b){ return a+b; }, 0);
                setEl('outlet-kwh-latest', latest);
                setEl('outlet-kwh-avg', total);
            }
        }

        function updateLampEnergyStats(field, values) {
            if (!values || values.length === 0) return;
            var avg = values.reduce(function(a,b){ return a+b; }, 0) / values.length;
            var latest = values[values.length - 1];
            var unit = {power:'kW', current:'A', energy_kwh:'kWh', voltage:'V'}[field] || '';
            var fmt = function(v) {
                return field === 'energy_kwh'
                    ? v.toFixed(4).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1')
                    : v.toFixed(2);
            };
            var setEl = function(id, v) { var el = document.getElementById(id); if (el) el.textContent = fmt(v) + ' ' + unit; };
            if (field === 'power') { setEl('lamp-power-latest', latest); setEl('lamp-power-avg', avg); }
            else if (field === 'current') { setEl('lamp-current-latest', latest); setEl('lamp-current-avg', avg); }
            else if (field === 'energy_kwh') {
                var total = values.reduce(function(a,b){ return a+b; }, 0);
                setEl('lamp-kwh-latest', latest);
                setEl('lamp-kwh-avg', total);
            }
        }

        // Analytics page individual energy chart loader
        var _analyticsEnergyPeriod = { ac: '24h', lamp: '24h', outlet: '24h' };
        function loadAnalyticsEnergy(device, period, btnEl) {
            _analyticsEnergyPeriod[device] = period;
            if (btnEl) {
                var btns = btnEl.parentElement.querySelectorAll('.chart-option-btn');
                btns.forEach(function(b){ b.classList.remove('active'); });
                btnEl.classList.add('active');
            }
            fetch('/api/energy/history?field=energy_kwh&period=' + period + '&device=' + device)
                .then(function(r){ return r.json(); })
                .then(function(result) {
                    var raw = result.data || [];
                    var cumValues = raw.map(function(d){ return parseFloat(d.value || 0); });
                    var labels = raw.map(function(d){ return d.time; });
                    // Compute deltas from cumulative values
                    var deltas = [];
                    var rangeLabels = [];
                    for (var i = 1; i < cumValues.length; i++) {
                        var diff = cumValues[i] - cumValues[i - 1];
                        deltas.push(parseFloat(Math.max(0, diff).toFixed(5)));
                        rangeLabels.push(labels[i - 1] + '\u2192' + labels[i]);
                    }
                    var chartKey = device === 'ac' ? 'acEnergyKwh' : (device === 'lamp' ? 'lampEnergyKwh' : 'outletEnergyKwhMySQL');
                    var chart = charts[chartKey];
                    if (chart && chart.data) {
                        chart.data.labels = rangeLabels;
                        chart.data.datasets[0].data = deltas;
                        chart.update('none');
                    }
                    // Update stat cards
                    var prefix = device; // device is 'ac', 'lamp', or 'outlet'
                    var lastEl = document.getElementById(prefix + '-analytics-kwh-last');
                    var totalEl = document.getElementById(prefix + '-analytics-kwh-total');
                    var avgEl = document.getElementById(prefix + '-analytics-kwh-avg');
                    if (deltas.length > 0) {
                        var last = deltas[deltas.length - 1];
                        var total = deltas.reduce(function(a,b){ return a+b; }, 0);
                        var avg = total / deltas.length;
                        if (lastEl) lastEl.textContent = last.toFixed(4) + ' kWh';
                        if (totalEl) totalEl.textContent = total.toFixed(4) + ' kWh';
                        if (avgEl) avgEl.textContent = avg.toFixed(4) + ' kWh';
                    } else {
                        if (lastEl) lastEl.textContent = '--';
                        if (totalEl) totalEl.textContent = '--';
                        if (avgEl) avgEl.textContent = '--';
                    }
                })
                .catch(function(e){ console.error('Analytics energy error (' + device + '):', e); });
        }

        function loadAllEnergyCharts() {
            loadEnergyHistory('power',      _activePeriod.power,      null);
            loadEnergyHistory('voltage',    _activePeriod.voltage,    null);
            loadEnergyHistory('current',    _activePeriod.current,    null);
            loadEnergyHistory('energy_kwh', _activePeriod.energy_kwh, null);
            loadEnergyCompare('power');
            loadEnergyCompare('energy_kwh');
            loadLampEnergyCompare('power');
            loadLampEnergyCompare('energy_kwh');
            loadRecordingState();
            loadLampRecordingState();
            loadDailySummary();
        }

        // ==================== DAILY SUMMARY ====================
        var _dailySummaryTimer = null;
        function loadDailySummary() {
            fetch('/api/energy/daily-summary')
                .then(r => r.json())
                .then(d => {
                    if (d.error) return;
                    var setText = function(id, v) { var el = document.getElementById(id); if (el) el.textContent = v; };
                    var fmtKwh = function(v) {
                        var num = parseFloat(v || 0);
                        if (!Number.isFinite(num)) return '--';
                        return num.toFixed(4).replace(/\\.0+$/, '').replace(/(\\.\\d*?)0+$/, '$1');
                    };
                    setText('daily-summary-date', d.date || '--');
                    setText('ds-ac-kwh',       d.ac ? fmtKwh(d.ac.kwh) : '--');
                    setText('ds-lamp-kwh',     d.lamp ? fmtKwh(d.lamp.kwh) : '--');
                    setText('ds-total-kwh',    d.total_kwh !== undefined ? fmtKwh(d.total_kwh) : '--');
                    setText('ds-cost',         d.cost_rp !== undefined ? d.cost_rp.toLocaleString() : '--');
                    setText('ds-ac-peak',      d.ac ? d.ac.power_peak_w.toFixed(1) : '--');
                    setText('ds-lamp-peak',    d.lamp ? d.lamp.power_peak_w.toFixed(1) : '--');
                    setText('ds-ac-runtime',   d.ac ? d.ac.runtime_h.toFixed(2) : '--');
                    setText('ds-lamp-runtime', d.lamp ? d.lamp.runtime_h.toFixed(2) : '--');
                    if (d.outlet) {
                        setText('outlet-today-kwh', d.outlet.kwh !== undefined ? fmtKwh(d.outlet.kwh) : '--');
                        setText('outlet-peak-power', d.outlet.power_peak_w !== undefined ? d.outlet.power_peak_w.toFixed(1) : '--');
                    }
                })
                .catch(function(e) { console.error('Daily summary error:', e); });
        }
        // Auto-refresh daily summary every 10 minutes
        if (_dailySummaryTimer) clearInterval(_dailySummaryTimer);
        _dailySummaryTimer = setInterval(loadDailySummary, 600000);

        // ==================== TIMEZONE WARNING ====================
        function checkTimezoneWarning() {
            fetch('/api/system/tz-status')
                .then(r => r.json())
                .then(d => {
                    if (!d.ok && d.warning) {
                        var banner = document.getElementById('tz-warning-banner');
                        if (!banner) {
                            banner = document.createElement('div');
                            banner.id = 'tz-warning-banner';
                            banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#0ea5e9;color:#1e293b;padding:8px 20px;font-size:13px;font-weight:600;text-align:center;display:flex;align-items:center;justify-content:center;gap:12px;';
                            banner.innerHTML = '<span>  Timezone Warning: ' + d.warning + '</span><button onclick="this.parentElement.remove()" style="background:rgba(0,0,0,0.15);border:none;padding:3px 10px;border-radius:5px;cursor:pointer;color:#1e293b;"> </button>';
                            document.body.prepend(banner);
                        }
                    }
                })
                .catch(function() {});
        }
        checkTimezoneWarning();

        // ==================== BEFORE vs AFTER RECORDING ====================
        var energyRecording = {
            before: {active: false, start: null, end: null},
            after: {active: false, start: null, end: null}
        };

        function loadRecordingState() {
            fetch('/api/energy/record')
                .then(r => r.json())
                .then(data => {
                    updateRecordingUI(data.recording);
                })
                .catch(e => console.error('Load recording state error:', e));
        }

        function toggleRecording(phase) {
            var rec = energyRecording[phase];
            var action = rec.active ? 'stop' : 'start';
            fetch('/api/energy/record', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({phase: phase, action: action})
            })
            .then(r => r.json())
            .then(data => {
                updateRecordingUI(data.recording);
                loadEnergyCompare('power');
                loadEnergyCompare('energy_kwh');
                showToast(action === 'start'
                    ? 'Recording ' + phase.toUpperCase() + ' started'
                    : 'Recording ' + phase.toUpperCase() + ' stopped', 'success');
            })
            .catch(e => console.error('Record error:', e));
        }

        function updateRecordingUI(rec) {
            if (!rec) return;
            energyRecording = rec;
            ['before', 'after'].forEach(function(phase) {
                var r = rec[phase];
                var statusEl = document.getElementById('status-' + phase);
                var btnEl = document.getElementById('btn-record-' + phase);
                if (!statusEl || !btnEl) return;

                var baseColor = phase === 'before' ? '#0ea5e9' : '#3b82f6';
                var darkColor = phase === 'before' ? '#0284c7' : '#2563eb';

                if (r.active) {
                    var start = new Date(r.start + 'Z');
                    var elapsed = Math.round((Date.now() - start.getTime()) / 3600000);
                    var days = Math.floor(elapsed / 24);
                    var hours = elapsed % 24;
                    statusEl.innerHTML = '<i class="fas fa-circle" style="color: #1e40af; font-size: 8px; animation: blink 1s infinite;"></i> <strong>Recording</strong> since ' + start.toLocaleDateString() + ' ' + start.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) + '<br>Duration: ' + days + 'd ' + hours + 'h';
                    btnEl.innerHTML = 'STOP RECORDING';
                    btnEl.style.background = 'linear-gradient(135deg, #1e40af, #1d4ed8)';
                } else if (r.start && r.end) {
                    var start = new Date(r.start + 'Z');
                    var end = new Date(r.end + 'Z');
                    var dur = Math.round((end - start) / 3600000);
                    var days = Math.floor(dur / 24);
                    var hours = dur % 24;
                    statusEl.innerHTML = '<strong>Completed</strong><br>' + start.toLocaleDateString() + ' &rarr; ' + end.toLocaleDateString() + '<br>Duration: ' + days + 'd ' + hours + 'h';
                    btnEl.innerHTML = 'RESET & RE-RECORD';
                    btnEl.style.background = 'linear-gradient(135deg, ' + baseColor + ', ' + darkColor + ')';
                } else {
                    statusEl.innerHTML = '&#9675; Not started';
                    btnEl.innerHTML = 'START RECORDING';
                    btnEl.style.background = 'linear-gradient(135deg, ' + baseColor + ', ' + darkColor + ')';
                }
            });
        }

        var compareRange = 'all';  // current compare range selection
        function loadEnergyCompare(field, range) {
            if (range) compareRange = range;
            var beforeChart, afterChart;
            if (field === 'energy_kwh') {
                beforeChart = charts.energyCompareKwhBefore;
                afterChart = charts.energyCompareKwhAfter;
            } else {
                beforeChart = charts.energyCompareBefore;
                afterChart = charts.energyCompareAfter;
            }
            if (!beforeChart || !afterChart) return;

            fetch('/api/energy/compare?field=' + field + '&range=' + compareRange)
                .then(r => r.json())
                .then(result => {
                    var beforeData = result.before || [];
                    var afterData = result.after || [];
                    var summary = result.summary || {};

                    // BEFORE chart - its own labels & data
                    beforeChart.data.labels = beforeData.map(function(d) { return d.label; });
                    beforeChart.data.datasets[0].data = beforeData.map(function(d) { return d.value; });
                    beforeChart.update('none');

                    // AFTER chart - its own labels & data
                    afterChart.data.labels = afterData.map(function(d) { return d.label; });
                    afterChart.data.datasets[0].data = afterData.map(function(d) { return d.value; });
                    afterChart.update('none');

                    // Sync Y-axis scale so both charts have same max for fair visual comparison
                    var allVals = beforeData.map(function(d){return d.value;}).concat(afterData.map(function(d){return d.value;}));
                    if (allVals.length > 0) {
                        var maxVal = Math.max.apply(null, allVals) * 1.1;
                        beforeChart.options.scales.y.max = maxVal;
                        afterChart.options.scales.y.max = maxVal;
                        beforeChart.update('none');
                        afterChart.update('none');
                    }

                    if (field === 'power') {
                        document.getElementById('compare-avg-before').textContent = summary.avg_before || '--';
                        document.getElementById('compare-avg-after').textContent = summary.avg_after || '--';
                        var savingsEl = document.getElementById('compare-savings');
                        savingsEl.textContent = summary.savings_percent || '--';
                        if (summary.savings_percent > 0) {
                            savingsEl.style.color = '#3b82f6';
                        } else if (summary.savings_percent < 0) {
                            savingsEl.style.color = '#1e40af';
                        }
                    }

                    if (result.recording) updateRecordingUI(result.recording);
                })
                .catch(e => console.error('Energy compare error:', e));
        }

        function setCompareRange(range, btn) {
            compareRange = range;
            // Toggle active class on range buttons
            document.querySelectorAll('#compare-range-7d,#compare-range-30d,#compare-range-all').forEach(function(b){ b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            loadEnergyCompare('power');
            loadEnergyCompare('energy_kwh');
        }

        // Listen for recording state changes from server
        socket.on('energy_recording', function(data) {
            updateRecordingUI(data.recording);
        });

        // ==================== LAMP BEFORE/AFTER RECORDING ====================
        var lampRecording = { before: {active:false, start:null, end:null}, after: {active:false, start:null, end:null} };
        var compareLampRange = 'all';

        function loadLampRecordingState() {
            fetch('/api/lamp/energy/record')
                .then(function(r){ return r.json(); })
                .then(function(data){ updateLampRecordingUI(data.recording); })
                .catch(function(e){ console.error('Load lamp recording state error:', e); });
        }

        function toggleLampRecording(phase) {
            var rec = lampRecording[phase];
            var action = rec.active ? 'stop' : 'start';
            fetch('/api/lamp/energy/record', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({phase: phase, action: action})
            })
            .then(function(r){ return r.json(); })
            .then(function(data){
                updateLampRecordingUI(data.recording);
                loadLampEnergyCompare('power');
                loadLampEnergyCompare('energy_kwh');
                showToast(action === 'start'
                    ? 'Lamp recording ' + phase.toUpperCase() + ' started'
                    : 'Lamp recording ' + phase.toUpperCase() + ' stopped', 'success');
            })
            .catch(function(e){ console.error('Lamp record error:', e); });
        }

        function updateLampRecordingUI(rec) {
            if (!rec) return;
            lampRecording = rec;
            ['before', 'after'].forEach(function(phase) {
                var r = rec[phase];
                var statusEl = document.getElementById('lamp-status-' + phase);
                var btnEl = document.getElementById('lamp-btn-record-' + phase);
                if (!statusEl || !btnEl) return;

                var baseColor = phase === 'before' ? '#0ea5e9' : '#3b82f6';
                var darkColor = phase === 'before' ? '#0284c7' : '#2563eb';

                if (r.active) {
                    var start = new Date(r.start + 'Z');
                    var elapsed = Math.round((Date.now() - start.getTime()) / 3600000);
                    statusEl.innerHTML = '<i class="fas fa-circle" style="color:#1e40af;font-size:8px;animation:blink 1s infinite;"></i> <strong>Recording</strong> since ' + start.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) + '<br>Duration: ' + Math.floor(elapsed/24) + 'd ' + (elapsed%24) + 'h';
                    btnEl.textContent = 'STOP RECORDING';
                    btnEl.style.background = 'linear-gradient(135deg,#1e40af,#1d4ed8)';
                } else if (r.start && r.end) {
                    var start = new Date(r.start + 'Z');
                    var end = new Date(r.end + 'Z');
                    var dur = Math.round((end - start) / 3600000);
                    statusEl.innerHTML = '<strong>Completed</strong><br>' + start.toLocaleDateString() + ' &rarr; ' + end.toLocaleDateString() + '<br>Duration: ' + Math.floor(dur/24) + 'd ' + (dur%24) + 'h';
                    btnEl.textContent = 'RESET & RE-RECORD';
                    btnEl.style.background = 'linear-gradient(135deg,' + baseColor + ',' + darkColor + ')';
                } else {
                    statusEl.innerHTML = '&#9675; Not started';
                    btnEl.textContent = 'START RECORDING';
                    btnEl.style.background = 'linear-gradient(135deg,' + baseColor + ',' + darkColor + ')';
                }
            });
        }

        function loadLampEnergyCompare(field, range) {
            if (range) compareLampRange = range;
            var beforeChart, afterChart;
            if (field === 'energy_kwh') {
                beforeChart = charts.lampCompareKwhBefore;
                afterChart = charts.lampCompareKwhAfter;
            } else {
                beforeChart = charts.lampCompareBefore;
                afterChart = charts.lampCompareAfter;
            }
            if (!beforeChart || !afterChart) return;

            fetch('/api/lamp/energy/compare?field=' + field + '&range=' + compareLampRange)
                .then(function(r){ return r.json(); })
                .then(function(result){
                    var beforeData = result.before || [];
                    var afterData = result.after || [];
                    var summary = result.summary || {};

                    beforeChart.data.labels = beforeData.map(function(d){ return d.label; });
                    beforeChart.data.datasets[0].data = beforeData.map(function(d){ return d.value; });
                    beforeChart.update('none');

                    afterChart.data.labels = afterData.map(function(d){ return d.label; });
                    afterChart.data.datasets[0].data = afterData.map(function(d){ return d.value; });
                    afterChart.update('none');

                    var allVals = beforeData.map(function(d){return d.value;}).concat(afterData.map(function(d){return d.value;}));
                    if (allVals.length > 0) {
                        var maxVal = Math.max.apply(null, allVals) * 1.1;
                        beforeChart.options.scales.y.max = maxVal;
                        afterChart.options.scales.y.max = maxVal;
                        beforeChart.update('none');
                        afterChart.update('none');
                    }

                    if (field === 'power') {
                        var el = document.getElementById('lamp-compare-avg-before'); if (el) el.textContent = summary.avg_before || '--';
                        el = document.getElementById('lamp-compare-avg-after'); if (el) el.textContent = summary.avg_after || '--';
                        el = document.getElementById('lamp-compare-savings');
                        if (el) {
                            el.textContent = summary.savings_percent || '--';
                            el.style.color = summary.savings_percent > 0 ? '#3b82f6' : (summary.savings_percent < 0 ? '#1e40af' : '');
                        }
                    }
                    if (result.recording) updateLampRecordingUI(result.recording);
                })
                .catch(function(e){ console.error('Lamp compare error:', e); });
        }

        function setLampCompareRange(range, btn) {
            compareLampRange = range;
            document.querySelectorAll('#lamp-compare-range-7d,#lamp-compare-range-30d,#lamp-compare-range-all').forEach(function(b){ b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            loadLampEnergyCompare('power');
            loadLampEnergyCompare('energy_kwh');
        }

        socket.on('lamp_recording', function(data) {
            updateLampRecordingUI(data.recording);
        });

        // ==================== THEME TOGGLE ====================
        function initTheme() {
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeUI(savedTheme);
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme');
            const newTheme = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeUI(newTheme);
        }

        function updateThemeUI(theme) {
            const icon = document.getElementById('theme-icon');
            const label = document.getElementById('theme-label');
            if (icon && label) {
                if (theme === 'dark') {
                    icon.className = 'fas fa-sun';
                    label.textContent = 'Light Mode';
                } else {
                    icon.className = 'fas fa-moon';
                    label.textContent = 'Dark Mode';
                }
            }
        }

        // Initialize theme on load
        initTheme();

        // ==================== NAVIGATION ====================
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebar-overlay');
            const icon = document.getElementById('hamburger-icon');
            sidebar.classList.toggle('open');
            overlay.classList.toggle('active');
            icon.className = sidebar.classList.contains('open') ? 'fas fa-times' : 'fas fa-bars';
        }

        function showPage(pageId) {
            console.log('[NAV] showPage called:', pageId);
            // Block user role from accessing admin-only pages
            if (userRole !== 'admin' && ADMIN_PAGES.indexOf(pageId) !== -1) {
                console.warn('[NAV] Access denied for role:', userRole, '  page:', pageId);
                pageId = 'dashboard-ac';
            }
            document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            
            const pageEl = document.getElementById(pageId);
            if (pageEl) {
                pageEl.classList.add('active');
                console.log('[NAV] Page activated:', pageId);
            } else {
                console.error('[NAV] Page not found:', pageId);
            }
            
            // Mark the correct nav-item as active
            document.querySelectorAll('.nav-item').forEach(item => {
                if (item.getAttribute('onclick') && item.getAttribute('onclick').includes("'" + pageId + "'")) {
                    item.classList.add('active');
                }
            });
            
            localStorage.setItem('currentPage', pageId);

            // Close mobile sidebar
            if (window.innerWidth <= 768) {
                const sidebar = document.getElementById('sidebar');
                const overlay = document.getElementById('sidebar-overlay');
                sidebar.classList.remove('open');
                overlay.classList.remove('active');
                document.getElementById('hamburger-icon').className = 'fas fa-bars';
            }

            if (pageId === 'camera') {
                try { checkCameraStatus(); } catch(e) { console.error('[NAV] camera init error:', e); }
            }
            if (pageId === 'ml-optimization') {
                // Use requestAnimationFrame: fires AFTER browser paints the page with display:block
                requestAnimationFrame(function() {
                    // Only create charts if they don't exist yet; otherwise just resize
                    if (!charts.gaFitness || !charts.psoFitness || !charts.comparison) {
                        try { initMLCharts(); } catch(e) { console.error('[NAV] initMLCharts rAF error:', e); }
                    } else {
                        // Charts exist - just resize them for correct dimensions
                        try {
                            ['gaFitness', 'psoFitness', 'comparison'].forEach(function(k) {
                                if (charts[k]) { charts[k].resize(); charts[k].update('none'); }
                            });
                        } catch(e) { console.warn('[NAV] chart resize error:', e); }
                    }
                    try { refreshMLData(); } catch(e) { console.error('[NAV] refreshMLData error:', e); }
                });
            }
            if (pageId === 'ac-analytics') {
                ['temp', 'hum', 'acTemp', 'acHum', 'acPower'].forEach(function(cn) {
                    try { updateChartData(cn, chartRanges[cn] || 1); } catch(e) {}
                });
                try { loadAnalyticsEnergy('ac', _analyticsEnergyPeriod.ac || '24h', null); } catch(e) {}
            }
            if (pageId === 'lamp-analytics') {
                ['lampLux', 'lampBright', 'lampPower'].forEach(function(cn) {
                    try { updateChartData(cn, chartRanges[cn] || 1); } catch(e) {}
                });
                try { loadAnalyticsEnergy('lamp', _analyticsEnergyPeriod.lamp || '24h', null); } catch(e) {}
            }
            if (pageId === 'occupancy-feedback') {
                try { updateChartData('occupancy', chartRanges.occupancy || 1); } catch(e) {}
                try { loadFeedbackHistory(); } catch(e) {}
            }
            if (pageId === 'energy') {
                try { ensureChartsReady(); } catch(e) { console.error('[NAV] ensureChartsReady error:', e); }
                try { loadAllEnergyCharts(); } catch(e) { console.error('[NAV] energy init error:', e); }
                // Chart.js often needs a resize pass when canvas was initialized in a hidden page.
                setTimeout(function() {
                    try {

                        Object.keys(charts).forEach(function(k) {
                            if (charts[k] && typeof charts[k].resize === 'function') {
                                charts[k].resize();
                                charts[k].update('none');
                            }
                        });
                    } catch(e) {
                        console.error('[NAV] chart resize error:', e);
                    }
                }, 120);
            }
            if (pageId === 'control-outlet') {
                try { refreshOutletStatus(); } catch(e) { console.error('[NAV] outlet status error:', e); }
                if (_sbmsConnected) { try { sbmsRefreshStatus(); _sbmsStartAutoRefresh(); } catch(e) {} }
            }
            if (pageId === 'outlet-analysis') {
                requestAnimationFrame(function() {
                    try { loadAnalyticsEnergy('outlet', _analyticsEnergyPeriod.outlet || '24h', null); } catch(e) { console.error(e); }
                    try { initOutletCharts(); } catch(e) { console.error('[NAV] outlet chart init error:', e); }
                    try { loadOutletAnalytics('1h', document.getElementById('outlet-range-1h')); } catch(e) {}
                    try { loadOutletKwhChart('24h', document.getElementById('outlet-kwh-24h')); } catch(e) {}
                    try { refreshOutletStatus(); } catch(e) {}
                });
            }
        }

        // ==================== OUTLET CONTROL ====================
        var outletStates = {1:'OFF', 2:'OFF', 3:'OFF', 4:'OFF'};
        var outletCharts = {};

        function outletControl(num, state) {
            var payload = {
                "id": num === 1 ? 8 : num,
                "nama": "Outlet",
                "group_key": "master-room",
                "status": state === 'ON' ? 1 : 0,
                "is_master": false
            };
            fetch('/api/outlet/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'ok' || data.success) {
                    updateOutletUI(num, state);
                    showToast('Outlet ' + num + ' turned ' + state, state === 'ON' ? 'success' : 'info');
                } else {
                    showToast('Outlet ' + num + ' command failed: ' + (data.message || 'unknown error'), 'error');
                }
            })
            .catch(e => showToast('Outlet control error: ' + e.message, 'error'));
        }

        function outletAllOff() {
            [1,2,3,4].forEach(function(n) { outletControl(n, 'OFF'); });
        }

        function outletAllOn() {
            [1,2,3,4].forEach(function(n) { outletControl(n, 'ON'); });
        }

        function updateOutletUI(num, state) {
            outletStates[num] = state;
            var dot = document.getElementById('outlet'+num+'-dot');
            var badge = document.getElementById('outlet'+num+'-status-badge');
            var isOn = state === 'ON';
            if (dot) { dot.style.background = isOn ? '#2563eb' : '#1e40af'; }
            if (badge) {
                badge.textContent = state;
                badge.style.background = isOn ? 'rgba(37,99,235,0.18)' : 'rgba(30,64,175,0.15)';
                badge.style.color = isOn ? '#2563eb' : '#1e40af';
            }
            // Update active count
            var activeCount = Object.values(outletStates).filter(function(s){ return s==='ON'; }).length;
            var el = document.getElementById('outlet-active-count');
            if (el) el.textContent = activeCount;
        }

        function refreshOutletStatus() {
            fetch('/api/outlet/status')
            .then(r => r.json())
            .then(data => {
                if (!data || !data.outlets) return;
                data.outlets.forEach(function(o) {
                    updateOutletUI(o.id, o.state);
                    var pw = document.getElementById('outlet'+o.id+'-power');
                    if (pw) pw.textContent = (o.power !== undefined ? o.power : '--') + 'W';
                    var oaPw = document.getElementById('oa-o'+o.id+'-power');
                    if (oaPw) oaPw.textContent = (o.power !== undefined ? o.power : '--');
                    var oaKwh = document.getElementById('oa-o'+o.id+'-kwh');
                    if (oaKwh) oaKwh.textContent = (o.energy !== undefined ? o.energy.toFixed(3) : '--');
                });
                if (data.total_power !== undefined) {
                    var tp = document.getElementById('outlet-total-power');
                    if (tp) tp.textContent = data.total_power;
                }
                if (data.today_kwh !== undefined) {
                    var tk = document.getElementById('outlet-today-kwh');
                    if (tk) tk.textContent = data.today_kwh.toFixed(3);
                }
                if (data.peak_power !== undefined) {
                    var pk = document.getElementById('outlet-peak-power');
                    if (pk) pk.textContent = data.peak_power;
                }
                // Update progress bars
                var maxPower = Math.max.apply(null, (data.outlets || []).map(function(o){ return o.power || 0; })) || 1;
                (data.outlets || []).forEach(function(o) {
                    var fill = document.getElementById('oa-o'+o.id+'-bar-fill');
                    if (fill) fill.style.width = (Math.min(100, ((o.power||0)/maxPower)*100)).toFixed(1) + '%';
                });
            })
            .catch(function(){});
        }

        function loadOutletAnalytics(range, btn) {
            document.querySelectorAll('[id^="outlet-range-"]').forEach(function(b){ b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            fetch('/api/outlet/history?range=' + range)
            .then(r => r.json())
            .then(data => {
                if (!outletCharts.power) { initOutletCharts(); }
                var ch = outletCharts.power;
                if (!ch || !data.labels) return;
                ch.data.labels = data.labels;
                ['outlet1','outlet2','outlet3','outlet4'].forEach(function(k, i) {
                    if (ch.data.datasets[i] && data[k]) ch.data.datasets[i].data = data[k];
                });
                ch.update();
            })
            .catch(function(){});
        }

        function loadOutletKwhChart(range, btn) {
            document.querySelectorAll('[id^="outlet-kwh-"]').forEach(function(b){ b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            fetch('/api/outlet/energy?range=' + range)
            .then(r => r.json())
            .then(data => {
                if (!outletCharts.kwh) { initOutletCharts(); }
                var ch = outletCharts.kwh;
                if (!ch || !data.labels) return;
                ch.data.labels = data.labels;
                ['outlet1','outlet2','outlet3','outlet4'].forEach(function(k, i) {
                    if (ch.data.datasets[i] && data[k]) ch.data.datasets[i].data = data[k];
                });
                ch.update();
            })
            .catch(function(){});
        }

        function initOutletCharts() {
            var baseColors = ['#2563eb','#0ea5e9','#3b82f6','#1e40af'];
            var canvasPower = document.getElementById('outletPowerChart');
            var canvasKwh = document.getElementById('outletKwhChart');
            var makeDatasets = function(label_prefix, alpha) {
                return [1,2,3,4].map(function(i) {
                    return {
                        label: 'Outlet ' + i,
                        data: [],
                        borderColor: baseColors[i-1],
                        backgroundColor: baseColors[i-1].replace(')', ', ' + alpha + ')').replace('rgb', 'rgba').replace('#', 'rgba(') || baseColors[i-1],
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3,
                        pointRadius: 2
                    };
                });
            };
            var chartOpts = {
                responsive: true, maintainAspectRatio: true,
                plugins: { legend: { labels: { color: 'var(--text-primary)', font: { size: 11 } } } },
                scales: {
                    x: { ticks: { color: 'var(--text-secondary)', maxRotation: 0, font: { size: 10 } }, grid: { color: 'rgba(37,99,235,0.08)' } },
                    y: { ticks: { color: 'var(--text-secondary)', font: { size: 10 } }, grid: { color: 'rgba(37,99,235,0.08)' } }
                }
            };
            try {
                if (canvasPower) {
                    if (outletCharts.power) outletCharts.power.destroy();
                    outletCharts.power = new Chart(canvasPower.getContext('2d'), {
                        type: 'line', data: { labels: [], datasets: makeDatasets('Outlet', 0.15) }, options: JSON.parse(JSON.stringify(chartOpts))
                    });
                }
                if (canvasKwh) {
                    if (outletCharts.kwh) outletCharts.kwh.destroy();
                    outletCharts.kwh = new Chart(canvasKwh.getContext('2d'), {
                        type: 'bar', data: { labels: [], datasets: makeDatasets('Outlet', 0.7) }, options: JSON.parse(JSON.stringify(chartOpts))
                    });
                }
            } catch(e) { console.error('[Outlet] Chart init error:', e); }
        }

        function exportOutletCSV() {
            window.open('/api/outlet/export/csv', '_blank');
        }

        // ==================== SBMS API CONTROL ====================
        var _sbmsRefreshTimer = null;
        var _sbmsConnected = false;

        function sbmsConnect() {
            var url = document.getElementById('sbms-server-url').value.trim();
            var email = document.getElementById('sbms-email').value.trim();
            var password = document.getElementById('sbms-password').value;
            if (!url) { showToast('Masukkan URL server SBMS', 'error'); return; }
            if (!email || !password) { showToast('Masukkan email dan password', 'error'); return; }
            var btn = document.getElementById('sbms-connect-btn');
            btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
            // Step 1: Set server URL
            fetch('/api/sbms/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({server_url: url})
            })
            .then(function(r) { return r.json(); })
            .then(function() {
                // Step 2: Login
                return fetch('/api/sbms/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email: email, password: password})
                });
            })
            .then(function(r) { return r.json().then(function(d) { return {ok: r.ok, data: d}; }); })
            .then(function(res) {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-plug"></i> Connect';
                if (res.ok && res.data.status === 'ok') {
                    _sbmsConnected = true;
                    _sbmsUpdateConnUI(true, res.data.user, url);
                    showToast('Connected to SBMS! (' + res.data.user + ')', 'success');
                    sbmsRefreshStatus();
                    _sbmsStartAutoRefresh();
                } else {
                    showToast('SBMS Login gagal: ' + (res.data.message || 'Unknown error'), 'error');
                }
            })
            .catch(function(e) {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-plug"></i> Connect';
                showToast('Koneksi error: ' + e.message, 'error');
            });
        }

        function sbmsDisconnect() {
            fetch('/api/sbms/logout', {method: 'POST'})
            .then(function() {
                _sbmsConnected = false;
                _sbmsUpdateConnUI(false);
                _sbmsStopAutoRefresh();
                showToast('Disconnected from SBMS', 'info');
            })
            .catch(function(e) { showToast('Logout error: ' + e.message, 'error'); });
        }

        function sbmsControl(deviceId, status) {
            if (!_sbmsConnected) { showToast('Connect ke SBMS dulu!', 'error'); return; }
            var names = {1:'Master', 2:'AC', 3:'Lampu', 4:'Outlet'};
            var btns = document.querySelectorAll('#sbms-card-' + deviceId + ' .sbms-btn');
            btns.forEach(function(b) { b.disabled = true; });
            fetch('/api/sbms/control/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'set', status: status})
            })
            .then(function(r) { return r.json().then(function(d) { return {ok: r.ok, data: d}; }); })
            .then(function(res) {
                btns.forEach(function(b) { b.disabled = false; });
                if (res.ok && res.data.status === 'ok') {
                    var dev = res.data.device || {};
                    showToast((names[deviceId]||'Device') + '   ' + (status ? 'ON' : 'OFF'), 'success');
                    // Update all devices from response
                    if (dev.devices && Array.isArray(dev.devices)) {
                        _sbmsUpdateAllDeviceCards(dev.devices);
                    } else {
                        _sbmsUpdateDeviceCard(deviceId, status);
                    }
                } else {
                    showToast('Kontrol gagal: ' + (res.data.message || 'Error'), 'error');
                    if (res.data.message && res.data.message.indexOf('not logged') >= 0) {
                        _sbmsConnected = false;
                        _sbmsUpdateConnUI(false);
                    }
                }
            })
            .catch(function(e) {
                btns.forEach(function(b) { b.disabled = false; });
                showToast('Control error: ' + e.message, 'error');
            });
        }

        function sbmsToggle(deviceId) {
            if (!_sbmsConnected) { showToast('Connect ke SBMS dulu!', 'error'); return; }
            var names = {5:'Master', 6:'AC', 7:'Lampu', 8:'Outlet'};
            var btns = document.querySelectorAll('#sbms-card-' + deviceId + ' .sbms-btn');
            btns.forEach(function(b) { b.disabled = true; });
            fetch('/api/sbms/control/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'toggle'})
            })
            .then(function(r) { return r.json().then(function(d) { return {ok: r.ok, data: d}; }); })
            .then(function(res) {
                btns.forEach(function(b) { b.disabled = false; });
                if (res.ok && res.data.status === 'ok') {
                    var dev = res.data.device || {};
                    showToast((names[deviceId]||'Device') + ' toggled', 'success');
                    if (dev.devices && Array.isArray(dev.devices)) {
                        _sbmsUpdateAllDeviceCards(dev.devices);
                    } else {
                        sbmsRefreshStatus();
                    }
                } else {
                    showToast('Toggle gagal: ' + (res.data.message || 'Error'), 'error');
                }
            })
            .catch(function(e) {
                btns.forEach(function(b) { b.disabled = false; });
                showToast('Toggle error: ' + e.message, 'error');
            });
        }

        function sbmsRefreshStatus() {
            fetch('/api/sbms/devices')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.status === 'ok' && data.devices) {
                    var devices = Array.isArray(data.devices) ? data.devices : [];
                    _sbmsUpdateAllDeviceCards(devices);
                    _sbmsUpdateTable(devices);
                }
            })
            .catch(function(e) { console.error('[SBMS] Refresh error:', e); });
        }

        function _sbmsUpdateDeviceCard(id, statusVal) {
            var card = document.getElementById('sbms-card-' + id);
            var badge = document.getElementById('sbms-status-' + id);
            var text = document.getElementById('sbms-status-text-' + id);
            var isOn = (statusVal === 1 || statusVal === '1' || statusVal === true);
            if (card) {
                if (isOn) card.classList.add('is-on'); else card.classList.remove('is-on');
            }
            if (badge) {
                badge.className = 'sbms-device-status ' + (isOn ? 'on' : 'off');
            }
            if (text) text.textContent = isOn ? 'ON' : 'OFF';
        }

        function _sbmsUpdateAllDeviceCards(devices) {
            devices.forEach(function(d) {
                _sbmsUpdateDeviceCard(d.id, d.status);
            });
        }

        function _sbmsUpdateTable(devices) {
            var tbody = document.getElementById('sbms-devices-tbody');
            if (!tbody) return;
            var names = {5:'Master', 6:'AC', 7:'Lampu', 8:'Outlet'};
            var icons = {5:' ', 6:'  ', 7:' ', 8:' '};
            if (!devices || devices.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="color:var(--text-secondary);font-style:italic;">No devices found</td></tr>';
                return;
            }
            var html = '';
            devices.forEach(function(d) {
                var isOn = (d.status === 1 || d.status === '1' || d.status === true);
                var statusBadge = isOn
                    ? '<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 10px;border-radius:12px;background:rgba(5,150,105,0.12);color:#059669;font-weight:700;font-size:11px;"><span style="width:6px;height:6px;border-radius:50%;background:#059669;"></span> ON</span>'
                    : '<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 10px;border-radius:12px;background:rgba(30,64,175,0.12);color:#1e40af;font-weight:700;font-size:11px;"><span style="width:6px;height:6px;border-radius:50%;background:#1e40af;"></span> OFF</span>';
                html += '<tr>'
                    + '<td style="font-weight:700;">' + d.id + '</td>'
                    + '<td>' + (icons[d.id]||'') + ' ' + (d.nama || names[d.id] || 'Device ' + d.id) + '</td>'
                    + '<td>' + statusBadge + '</td>'
                    + '<td>' + (d.is_master ? ' ' : '-') + '</td>'
                    + '<td><code style="font-size:11px;background:rgba(37,99,235,0.06);padding:2px 6px;border-radius:4px;">' + (d.group_key || '-') + '</code></td>'
                    + '</tr>';
            });
            tbody.innerHTML = html;
        }

        function _sbmsUpdateConnUI(connected, userEmail, serverUrl) {
            var panel = document.getElementById('sbms-conn-panel');
            var badge = document.getElementById('sbms-conn-status');
            var text = document.getElementById('sbms-conn-text');
            var loginForm = document.getElementById('sbms-login-form');
            var connInfo = document.getElementById('sbms-connected-info');
            if (connected) {
                panel.classList.add('connected');
                badge.className = 'sbms-conn-badge online';
                text.textContent = 'Connected';
                loginForm.style.display = 'none';
                connInfo.style.display = 'block';
                document.getElementById('sbms-user-email').textContent = userEmail || '-';
                document.getElementById('sbms-server-display').textContent = serverUrl || '-';
            } else {
                panel.classList.remove('connected');
                badge.className = 'sbms-conn-badge offline';
                text.textContent = 'Disconnected';
                loginForm.style.display = 'block';
                connInfo.style.display = 'none';
                // Reset device cards
                [1,2,3,4].forEach(function(id) { _sbmsUpdateDeviceCard(id, 0); });
                _sbmsUpdateTable([]);
            }
        }

        function _sbmsStartAutoRefresh() {
            _sbmsStopAutoRefresh();
            _sbmsRefreshTimer = setInterval(function() {
                var page = document.getElementById('control-outlet');
                if (page && page.classList.contains('active') && _sbmsConnected) {
                    sbmsRefreshStatus();
                }
            }, 5000);
        }

        function _sbmsStopAutoRefresh() {
            if (_sbmsRefreshTimer) { clearInterval(_sbmsRefreshTimer); _sbmsRefreshTimer = null; }
        }

        // Check SBMS connection state on page load
        (function() {
            fetch('/api/sbms/config')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.connected && data.server_url) {
                    _sbmsConnected = true;
                    _sbmsUpdateConnUI(true, data.user, data.server_url);
                    _sbmsStartAutoRefresh();
                }
            })
            .catch(function() {});
        })();

        // ==================== ML OPTIMIZATION ====================
        var mlHistory = [];
        var mlRunCount = 0;

        function refreshMLData() {
            fetch('/api/ml/status')
                .then(r => r.json())
                .then(data => {
                    updateMLDisplay(data);
                    // Update current sensor conditions
                    fetch('/api/data').then(r => r.json()).then(d => {
                        const tempEl = document.getElementById('ml-cur-temp');
                        const humEl = document.getElementById('ml-cur-hum');
                        const luxEl = document.getElementById('ml-cur-lux');
                        const personEl = document.getElementById('ml-cur-person');
                        if (tempEl) {
                            const t = d && d.ac ? parseFloat(d.ac.temperature || 0) : NaN;
                            tempEl.textContent = Number.isFinite(t) ? t.toFixed(1) : '--';
                        }
                        if (humEl) {
                            const h = d && d.ac ? parseFloat(d.ac.humidity || 0) : NaN;
                            humEl.textContent = Number.isFinite(h) ? h.toFixed(1) : '--';
                        }
                        if (luxEl) {
                            const luxRaw = d && d.lamp ? (d.lamp.lux_avg != null ? d.lamp.lux_avg : (d.lamp.lux1 != null ? d.lamp.lux1 : null)) : null;
                            luxEl.textContent = (typeof luxRaw === 'number') ? luxRaw.toFixed(1) : '--';
                        }
                        if (personEl) personEl.textContent = (d && d.camera && d.camera.person_detected) ? 'Yes' : 'No';
                    }).catch(() => {});
                })
                .catch(err => console.error('ML refresh error:', err));
        }

        function updateMLDisplay(data) {
            const setEl = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };

            setEl('ml-ga-fitness', (data.ga_fitness || 0).toFixed(2));
            // PSO fitness is already in percent from server
            setEl('ml-pso-fitness', (data.pso_fitness != null ? parseFloat(data.pso_fitness) : 0).toFixed(1));
            setEl('dash-pso-fitness-main', (data.pso_fitness != null ? parseFloat(data.pso_fitness) : 0).toFixed(2));
            setEl('dash-pso-brightness-main', data.pso_brightness != null ? Math.round(parseFloat(data.pso_brightness)) : '--');
            setEl('ml-ga-temp', data.ga_temp || '--');
            setEl('ml-ga-fan', data.ga_fan || '--');
            setEl('ml-ga-mode', data.ga_mode || 'COOL');
            setEl('ml-ga-rh', data.ga_set_rh || '--');
            var p1 = data.pso_pwm1 != null && data.pso_pwm1 > 0 ? data.pso_pwm1 : '--';
            var p2 = data.pso_pwm2 != null && data.pso_pwm2 > 0 ? data.pso_pwm2 : '--';
            setEl('ml-pso-pwm1', p1);
            setEl('ml-pso-pwm2', p2);
            // Sinkron Lamp Dashboard brightness dari PSO result
            if (data.pso_brightness1 != null) setText('dash-bright1', Math.round(parseFloat(data.pso_brightness1)));
            if (data.pso_brightness2 != null) setText('dash-bright2', Math.round(parseFloat(data.pso_brightness2)));
            setEl('ml-opt-runs', data.optimization_runs || 0);

            // GA chart - use server data, or fallback to localStorage
            var gaHistory = (data.ga_history && data.ga_history.length > 0) ? data.ga_history : null;
            if (!gaHistory) {
                try { gaHistory = JSON.parse(localStorage.getItem('ml_ga_history')); } catch(e) {}
            }
            if (gaHistory && gaHistory.length > 0) {
                updateMLChart('gaFitness', gaHistory, 'GA');
            }

            // PSO chart - use server data, or fallback to localStorage
            if (data.pso_iteration_log && data.pso_iteration_log.length > 0) {
                updatePSOIterChart(data.pso_iteration_log);
            } else {
                var savedIterLog = null;
                try { savedIterLog = JSON.parse(localStorage.getItem('ml_pso_iterlog')); } catch(e) {}
                if (savedIterLog && savedIterLog.length > 0) {
                    updatePSOIterChart(savedIterLog);
                } else if (data.pso_history && data.pso_history.length > 0) {
                    // Fallback: if no iteration_log yet, show pso_history as fitness line
                    const chart = charts.psoFitness;
                    if (chart) {
                        chart.data.labels = data.pso_history.map((_, i) => i + 1);
                        chart.data.datasets[0].data = [];
                        chart.data.datasets[1].data = [];
                        chart.data.datasets[2].data = data.pso_history.map(f => Math.max(0, 350 - Math.sqrt(f)));
                        chart.update('none');
                    }
                }
            }
        }

        function updateMLChart(chartName, history, algo) {
            // Init chart if not yet created (e.g. ML tab never opened)
            if (!charts[chartName]) {
                var mlCanvas = document.getElementById('gaFitnessChart');
                if (mlCanvas && mlCanvas.offsetWidth > 0) {
                    try { initMLCharts(); } catch(e) { console.warn('[CHART] initMLCharts failed:', e); return; }
                } else {
                    // ML tab not yet active - store data to render when tab opens
                    window.__pendingMLData = window.__pendingMLData || {};
                    window.__pendingMLData[chartName] = { history, algo };
                    // Also save to localStorage so data survives refresh
                    if (algo === 'GA') {
                        try { localStorage.setItem('ml_ga_history', JSON.stringify(history)); } catch(e) {}
                    }
                    return;
                }
            }
            const chart = charts[chartName];
            if (!chart || !history || history.length === 0) return;

            if (algo === 'GA') {
                chart.data.labels           = history.map((_, i) => 'Gen ' + (i + 1));
                chart.data.datasets[0].data = history;
                chart.update('none');
                // Cache to localStorage for persistence across refresh
                try { localStorage.setItem('ml_ga_history', JSON.stringify(history)); } catch(e) {}
            } else if (algo === 'PSO') {
                // PSO fitness history sebagai fallback jika belum ada iteration_log
                // Tampilkan sebagai Lux Avg estimasi (dataset index 2)
                chart.data.labels               = history.map((_, i) => i + 1);
                chart.data.datasets[0].data     = [];
                chart.data.datasets[1].data     = [];
                chart.data.datasets[2].data     = history.map(f => Math.max(0, 350 - Math.sqrt(Math.max(0, f))));
                chart.update('none');
            }
        }

        function updatePSOIterChart(iterLog) {
            if (!iterLog || iterLog.length === 0) return;
            const chart = charts.psoFitness;
            if (!chart) return;
            chart.data.labels           = iterLog.map(d => d.iter);
            chart.data.datasets[0].data = iterLog.map(d => d.pwm1);
            chart.data.datasets[1].data = iterLog.map(d => d.pwm2);
            chart.data.datasets[2].data = iterLog.map(d => d.lux_avg);
            chart.update('none');
            // Cache to localStorage for persistence across refresh
            try { localStorage.setItem('ml_pso_iterlog', JSON.stringify(iterLog)); } catch(e) {}
            // Update tabel
            const tbody = document.getElementById('pso-iter-tbody');
            const wrap  = document.getElementById('pso-iter-table-wrap');
            if (!tbody) return;
            wrap.style.display = 'block';
            tbody.innerHTML = iterLog.map(d => {
                const inRange     = d.lux_avg >= 315 && d.lux_avg <= 385;
                const statusColor = inRange ? '#3b82f6' : '#0ea5e9';
                const statusText  = inRange ? '  Target' : '  Not yet';
                const fitPct      = Math.max(0, 100.0 - (d.fitness / 122500.0) * 100.0).toFixed(1);
                return `<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:4px 8px;text-align:center;font-weight:600;">${d.iter}</td>
                    <td style="padding:4px 8px;text-align:center;color:#2563eb;">${d.pwm1}</td>
                    <td style="padding:4px 8px;text-align:center;color:#2563eb;">${d.pwm2}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux1 != null ? d.lux1 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux2 != null ? d.lux2 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux3 != null ? d.lux3 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#0ea5e9;font-weight:600;">${d.lux_avg}</td>
                    <td style="padding:4px 8px;text-align:center;color:#3b82f6;font-weight:600;">${fitPct}%</td>
                    <td style="padding:4px 8px;text-align:center;color:${statusColor};font-weight:600;">${statusText}</td>
                </tr>`;
            }).join('');
        }

        socket.on('pso_iter_progress', function(d) {
            // Local helper - does not depend on outer scope
            const _s = function(id, val) { var e = document.getElementById(id); if (e) e.textContent = val; };

            if (d.status === 'done') {
                // Sinkron Lamp Dashboard brightness
                _s('dash-bright1', Math.round(parseFloat(d.b1 || 0)));
                _s('dash-bright2', Math.round(parseFloat(d.b2 || 0)));
                // Update stat card ML Optimization
                _s('ml-pso-pwm1',  d.pwm1 != null ? d.pwm1 : '--');
                _s('ml-pso-pwm2',  d.pwm2 != null ? d.pwm2 : '--');
                _s('ml-pso-lux1',  d.lux1 != null ? d.lux1 : '--');
                _s('ml-pso-lux2',  d.lux2 != null ? d.lux2 : '--');
                _s('ml-pso-lux3',  d.lux3 != null ? d.lux3 : '--');
                // Fitness %
                const fitPct = Math.max(0, 100.0 - ((d.fitness || 0) / 122500.0) * 100.0).toFixed(1);
                _s('ml-pso-fitness', fitPct);

                // Iteration table - replace waiting row with full data
                const tbody = document.getElementById('pso-iter-tbody');
                const wrap  = document.getElementById('pso-iter-table-wrap');
                if (!tbody) return;
                wrap.style.display = 'block';
                const inRange     = (d.lux_avg || 0) >= 315 && (d.lux_avg || 0) <= 385;
                const statusColor = inRange ? '#3b82f6' : '#0ea5e9';
                const statusText  = inRange ? '  Target' : '  Not yet';
                const newRow = `<tr data-iter="${d.iter}" style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:4px 8px;text-align:center;font-weight:600;">${d.iter}</td>
                    <td style="padding:4px 8px;text-align:center;color:#2563eb;">${d.pwm1 != null ? d.pwm1 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#2563eb;">${d.pwm2 != null ? d.pwm2 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux1 != null ? d.lux1 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux2 != null ? d.lux2 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#22d3ee;">${d.lux3 != null ? d.lux3 : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#0ea5e9;font-weight:600;">${d.lux_avg != null ? d.lux_avg : '--'}</td>
                    <td style="padding:4px 8px;text-align:center;color:#3b82f6;font-weight:600;">${fitPct}%</td>
                    <td style="padding:4px 8px;text-align:center;color:${statusColor};font-weight:600;">${statusText}</td>
                </tr>`;
                const existingRow = tbody.querySelector('tr[data-iter="' + d.iter + '"]');
                if (existingRow) existingRow.outerHTML = newRow;
                else tbody.insertAdjacentHTML('beforeend', newRow);

                // Update chart live
                const chart = charts.psoFitness;
                if (chart) {
                    const idx = chart.data.labels.indexOf(d.iter);
                    if (idx === -1) {
                        chart.data.labels.push(d.iter);
                        chart.data.datasets[0].data.push(d.pwm1);
                        chart.data.datasets[1].data.push(d.pwm2);
                        chart.data.datasets[2].data.push(d.lux_avg);
                    } else {
                        chart.data.datasets[0].data[idx] = d.pwm1;
                        chart.data.datasets[1].data[idx] = d.pwm2;
                        chart.data.datasets[2].data[idx] = d.lux_avg;
                    }
                    chart.update('none');
                }

            } else if (d.status === 'waiting') {
                const tbody = document.getElementById('pso-iter-tbody');
                const wrap  = document.getElementById('pso-iter-table-wrap');
                if (!tbody) return;
                wrap.style.display = 'block';
                // Delete baris lama jika ada, insert baru
                const old = tbody.querySelector('tr[data-iter="' + d.iter + '"]');
                if (old) old.remove();
                tbody.insertAdjacentHTML('beforeend',
                    '<tr data-iter="' + d.iter + '" style="border-bottom:1px solid rgba(255,255,255,0.05);opacity:0.6;">' +
                        '<td style="padding:4px 8px;text-align:center;font-weight:600;">' + d.iter + '</td>' +
                        '<td style="padding:4px 8px;text-align:center;color:#2563eb;">' + (d.pwm1 != null ? d.pwm1 : '--') + '</td>' +
                        '<td style="padding:4px 8px;text-align:center;color:#2563eb;">' + (d.pwm2 != null ? d.pwm2 : '--') + '</td>' +
                        '<td colspan="6" style="padding:4px 8px;text-align:center;color:var(--text-secondary);">  Reading sensors...</td>' +
                    '</tr>');

            } else if (d.status === 'new_cycle') {
                const tbody = document.getElementById('pso-iter-tbody');
                if (tbody) tbody.innerHTML = '';
                const chart = charts.psoFitness;
                if (chart) {
                    chart.data.labels = [];
                    chart.data.datasets.forEach(function(ds) { ds.data = []; });
                    chart.update('none');
                }
            }
        });



        function addMLHistoryRow(data) {
            mlRunCount++;
            const entry = {
                run: mlRunCount,
                time: new Date().toLocaleTimeString(),
                ga_fitness: data.ga_fitness || 0,
                ga_temp: data.ga_temp || '--',
                ga_fan: data.ga_fan || '--',
                pso_fitness: data.pso_fitness || 0,
                pso_brightness: data.pso_brightness || '--',
                // GA maximize (score) vs PSO minimize (error) - normalize PSO before combining
                combined: (data.ga_fitness || 0) * 0.5 + Math.max(0, 100 - (data.pso_fitness || 0) / 100) * 0.5
            };
            mlHistory.unshift(entry);
            if (mlHistory.length > 50) mlHistory.pop();

            renderMLHistory();
        }

        function renderMLHistory() {
            const tbody = document.getElementById('ml-history-body');
            if (!tbody) return;

            if (mlHistory.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #94a3b8;">No optimization data yet. Run GA or PSO to start.</td></tr>';
                return;
            }

            tbody.innerHTML = mlHistory.map(e => {
                const getBadgeGA  = (f) => f >= 80 ? 'good' : f >= 50 ? 'mid' : 'low';
                const getBadgePSO = (err) => err <= 100 ? 'good' : err <= 500 ? 'mid' : 'low'; // PSO: lower error = better
                return '<tr>' +
                    '<td>' + e.run + '</td>' +
                    '<td>' + e.time + '</td>' +
                    '<td><span class="ml-badge ' + getBadgeGA(e.ga_fitness) + '">' + e.ga_fitness.toFixed(2) + '</span></td>' +
                    '<td>' + e.ga_temp + '\u00b0C</td>' +
                    '<td>' + e.ga_fan + '</td>' +
                    '<td><span class="ml-badge ' + getBadgePSO(e.pso_fitness) + '">' + e.pso_fitness.toFixed(2) + '</span></td>' +
                    '<td>' + e.pso_brightness + '%</td>' +
                    '<td><span class="ml-badge ' + getBadgeGA(e.combined) + '">' + e.combined.toFixed(2) + '</span></td>' +
                '</tr>';
            }).join('');
        }

        function refreshMLHistory() {
            refreshMLData();
            showToast('ML data refreshed', 'success');
        }

        // ==================== EXPORT FUNCTIONS ====================
        function downloadCSV(filename, csvContent) {
            csvContent = "sep=,\\n" + csvContent;
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
            URL.revokeObjectURL(link.href);
        }

        function exportMLHistory() {
            if (!mlHistory || mlHistory.length === 0) {
                showToast('No optimization data to export', 'error');
                return;
            }
            let csv = 'Run,Time,GA Fitness,AC Temp (C),Fan Speed,PSO Fitness,Brightness (%),Combined\\n';
            mlHistory.forEach(e => {
                csv += e.run + ',"' + e.time + '",' + e.ga_fitness.toFixed(2) + ',' + e.ga_temp + ',' + e.ga_fan + ',' + e.pso_fitness.toFixed(2) + ',' + e.pso_brightness + ',' + e.combined.toFixed(2) + '\\n';
            });
            downloadCSV('ml_optimization_history.csv', csv);
            showToast('ML history exported', 'success');
        }

        function exportLogs() {
            const container = document.getElementById('log-container');
            if (!container || !container.children.length) {
                showToast('No logs to export', 'error');
                return;
            }
            let csv = 'Time,Level,Message\\n';
            Array.from(container.children).forEach(entry => {
                const text = entry.textContent || '';
                const timeMatch = text.match(/\\[(.+?)\\]/);
                const time = timeMatch ? timeMatch[1] : '';
                const msg = text.replace(/\\[.+?\\]\\s*/, '').replace(/"/g, '""');
                const level = entry.className.replace('log-entry ', '').trim();
                csv += '"' + time + '","' + level + '","' + msg + '"\\n';
            });
            downloadCSV('system_logs.csv', csv);
            showToast('Logs exported', 'success');
        }

        function exportGAReport() {
            fetch('/api/ga/export-csv')
                .then(function(r) {
                    if (!r.ok) {
                        return r.json().then(function(d) { throw new Error(d.error || 'Export failed'); });
                    }
                    return r.blob();
                })
                .then(function(blob) {
                    var ts = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '');
                    var link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = 'ga_report_' + ts + '.csv';
                    link.click();
                    URL.revokeObjectURL(link.href);
                    showToast('GA report exported successfully!', 'success');
                })
                .catch(function(e) { showToast('GA export failed: ' + e.message, 'error'); });
        }

        function exportPSOReport() {
            fetch('/api/pso/export-csv')
                .then(function(r) {
                    if (!r.ok) {
                        return r.json().then(function(d) { throw new Error(d.error || 'Export failed'); });
                    }
                    return r.blob();
                })
                .then(function(blob) {
                    var ts = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '');
                    var link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = 'pso_report_' + ts + '.csv';
                    link.click();
                    URL.revokeObjectURL(link.href);
                    showToast('PSO report exported successfully!', 'success');
                })
                .catch(function(e) { showToast('PSO export failed: ' + e.message, 'error'); });
        }

        function exportFeedback() {
            fetch('/api/occupancy/feedback/list')
                .then(r => r.json())
                .then(data => {
                    const rows = data.feedback || [];
                    if (rows.length === 0) {
                        showToast('No feedback data to export', 'error');
                        return;
                    }
                    let csv = 'Time,Rating,Occupancy Count,Comment\\n';
                    rows.forEach(item => {
                        const comment = (item.comment || '').replace(/"/g, '""');
                        csv += '"' + item.time + '",' + item.rating + ',' + item.occupancy_count + ',"' + comment + '"\\n';
                    });
                    downloadCSV('occupancy_feedback.csv', csv);
                    showToast('Feedback exported', 'success');
                })
                .catch(() => showToast('Failed to fetch feedback data', 'error'));
        }

        function exportEnergyHistory() {
            const period = '24h';
            fetch('/api/energy/history?period=' + period)
                .then(r => r.json())
                .then(data => {
                    if (!data.power || data.power.length === 0) {
                        showToast('No energy data to export', 'error');
                        return;
                    }
                    let csv = 'Time,Power (W),Voltage (V),Energy (kWh)\\n';
                    const times = data.power.map(p => p.time);
                    const powers = data.power || [];
                    const voltages = data.voltage || [];
                    const energies = data.energy_kwh || [];
                    times.forEach((t, i) => {
                        const pw = powers[i] ? powers[i].value : '';
                        const vl = voltages[i] ? voltages[i].value : '';
                        const en = energies[i] ? energies[i].value : '';
                        csv += '"' + t + '",' + pw + ',' + vl + ',' + en + '\\n';
                    });
                    downloadCSV('energy_history_' + period + '.csv', csv);
                    showToast('Energy data exported (' + period + ')', 'success');
                })
                .catch(() => showToast('Failed to fetch energy history', 'error'));
        }

        // Generic chart data export (single dataset charts)
        function exportChartData(chartName, valueLabel) {
            const chart = charts[chartName];
            if (!chart || !chart.data.labels || chart.data.labels.length === 0) {
                showToast('No data to export for ' + chartName, 'error');
                return;
            }
            let csv = 'Time,' + valueLabel + '\\n';
            chart.data.labels.forEach((label, i) => {
                const val = chart.data.datasets[0].data[i];
                csv += '"' + label + '",' + (val !== null && val !== undefined ? val : '') + '\\n';
            });
            downloadCSV(chartName + '_data.csv', csv);
            showToast(chartName + ' data exported', 'success');
        }

        // Export for multi-dataset comparison charts (Before/After, GA vs PSO)
        function exportCompareChart(chartName, valueLabel) {
            const chart = charts[chartName];
            if (!chart || !chart.data.labels || chart.data.labels.length === 0) {
                showToast('No data to export for ' + chartName, 'error');
                return;
            }
            const dsNames = chart.data.datasets.map(ds => ds.label || valueLabel);
            let csv = 'Time,' + dsNames.join(',') + '\\n';
            chart.data.labels.forEach((label, i) => {
                let row = '"' + label + '"';
                chart.data.datasets.forEach(ds => {
                    const val = ds.data[i];
                    row += ',' + (val !== null && val !== undefined ? val : '');
                });
                csv += row + '\\n';
            });
            downloadCSV(chartName + '_compare.csv', csv);
            showToast(chartName + ' comparison exported', 'success');
        }

        function getMLParams(algo) {
            const getInput = (id, fallback) => {
                const el = document.getElementById(id);
                return el ? el.value : fallback;
            };
            if (algo === 'ga') {
                return {
                    population_size: parseInt(getInput('ml-ga-pop', 15), 10) || 15,
                    generations: parseInt(getInput('ml-ga-gen', 30), 10) || 30,
                    mutation_rate: parseFloat(getInput('ml-ga-mut', 0.25)) || 0.25,
                    crossover_rate: parseFloat(getInput('ml-ga-cross', 0.8)) || 0.8,
                    elitism_ratio: 0.2
                };
            } else {
                return {
                    swarm_size: parseInt(getInput('ml-pso-swarm', 30), 10) || 30,
                    iterations: parseInt(getInput('ml-pso-iter', 100), 10) || 100,
                    w: parseFloat(getInput('ml-pso-w', 0.7)) || 0.7,
                    c: parseFloat(getInput('ml-pso-c', 1.5)) || 1.5
                };
            }
        }

        function runGAOptimization() {
            showToast('Starting GA \u2192 AC optimization...', 'success');
            fetch('/api/ml/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ algorithm: 'ga', params: getMLParams('ga') })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success') {
                    showToast('GA optimization triggered!', 'success');
                } else {
                    showToast('GA error: ' + d.message, 'error');
                }
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function runPSOOptimization() {
            showToast('Starting PSO \u2192 Lamp optimization...', 'success');
            fetch('/api/ml/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ algorithm: 'pso', params: getMLParams('pso') })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success') {
                    showToast('PSO optimization triggered!', 'success');
                } else {
                    showToast('PSO error: ' + d.message, 'error');
                }
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function runBothOptimization() {
            showToast('Starting GA + PSO optimization...', 'success');
            fetch('/api/ml/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    algorithm: 'both',
                    params: { ga: getMLParams('ga'), pso: getMLParams('pso') }
                })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success') {
                    showToast('GA + PSO optimization triggered!', 'success');
                } else {
                    showToast('Error: ' + d.message, 'error');
                }
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function clearMLCharts() {
            ['gaFitness', 'psoFitness', 'comparison'].forEach(name => {
                if (charts[name]) {
                    charts[name].data.labels = [];
                    charts[name].data.datasets.forEach(ds => ds.data = []);
                    charts[name].update('none');
                }
            });
            showToast('Charts cleared', 'success');
        }

        // ==================== CAMERA ====================
        var cameraEnabled = true;

        function toggleCamera() {
            fetch('/api/camera/toggle', { method: 'POST' })
                .then(r => r.json())
                .then(result => {
                    cameraEnabled = result.enabled;
                    updateCameraToggleUI();
                    const img = document.getElementById('camera-img');
                    const error = document.getElementById('camera-error');
                    if (cameraEnabled) {
                        img.src = '/video_feed?' + new Date().getTime();
                        img.style.display = 'block';
                        error.style.display = 'none';
                        showToast('Camera ON', 'success');
                    } else {
                        img.style.display = 'none';
                        error.style.display = 'flex';
                        error.querySelector('h3').textContent = 'Camera OFF';
                        error.querySelector('p').textContent = 'Camera is off. Click Camera ON button to activate.';
                        showToast('Camera OFF', 'info');
                    }
                    checkCameraStatus();
                })
                .catch(e => showToast('Error: ' + e, 'error'));
        }

        function updateCameraToggleUI() {
            const btn = document.getElementById('camera-toggle-btn');
            const text = document.getElementById('camera-toggle-text');
            if (!btn) return;
            if (cameraEnabled) {
                btn.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
                text.textContent = 'Camera ON';
            } else {
                btn.style.background = 'linear-gradient(135deg, #1e40af, #1d4ed8)';
                text.textContent = 'Camera OFF';
            }
        }

        function retryCamera() {
            const img = document.getElementById('camera-img');
            const error = document.getElementById('camera-error');
            
            fetch('/api/camera/restart', { method: 'POST' })
                .then(r => r.json())
                .then(result => {
                    if (result.status === 'success') {
                        img.src = '/video_feed?' + new Date().getTime();
                        img.style.display = 'block';
                        error.style.display = 'none';
                        showToast('Camera restarted successfully!');
                        checkCameraStatus();
                    } else {
                        showToast('Camera restart failed', 'error');
                    }
                })
                .catch(e => {
                    showToast('Camera restart error', 'error');
                });
        }

        function checkCameraStatus() {
            fetch('/api/camera/status')
                .then(r => r.json())
                .then(data => {
                    const statusEl = document.getElementById('cam-status');
                    if (data.status === 'active') {
                        statusEl.textContent = 'Active';
                        statusEl.style.color = '#3b82f6';
                        document.getElementById('cam-resolution').textContent = data.width + ' x ' + data.height;
                        document.getElementById('cam-fps').textContent = data.fps + ' FPS';
                    } else {
                        statusEl.textContent = 'Inactive';
                        statusEl.style.color = '#1e40af';
                    }
                })
                .catch(e => {
                    document.getElementById('cam-status').textContent = 'Error';
                    document.getElementById('cam-status').style.color = '#1e40af';
                });
        }

        function updateCameraTime() {
            const el = document.getElementById('camera-time');
            if (el) el.textContent = new Date().toLocaleTimeString();
        }
        setInterval(updateCameraTime, 1000);

        // ==================== AC CONTROLS ====================
        function updateACTemp(value) {
            document.getElementById('ac-temp-display').textContent = value;
            saveSettings();
        }

        function updateACRH(value) {
            document.getElementById('ac-rh-display').textContent = value;
            saveSettings();
        }

        function updateFanSpeed(value) {
            document.getElementById('fan-speed-display').textContent = value;
            saveSettings();
        }

        function applyACSnapshot(ac) {
            if (!ac) return;
            if (ac.ac_temp !== undefined && ac.ac_temp !== null) {
                var tempValue = parseInt(ac.ac_temp, 10);
                var tempSlider = document.getElementById('ac-temp-slider');
                var tempDisplay = document.getElementById('ac-temp-display');
                if (tempSlider && Number.isFinite(tempValue)) tempSlider.value = tempValue;
                if (tempDisplay && Number.isFinite(tempValue)) tempDisplay.textContent = tempValue;
            }
            if (ac.fan_speed !== undefined && ac.fan_speed !== null) {
                var fanValue = parseInt(ac.fan_speed, 10);
                var fanSlider = document.getElementById('fan-speed-slider');
                var fanDisplay = document.getElementById('fan-speed-display');
                if (fanSlider && Number.isFinite(fanValue)) fanSlider.value = fanValue;
                if (fanDisplay && Number.isFinite(fanValue)) fanDisplay.textContent = fanValue;
            }
            if (ac.set_rh !== undefined && ac.set_rh !== null) {
                var rhValue = parseInt(ac.set_rh, 10);
                var rhSlider = document.getElementById('ac-rh-slider');
                var rhDisplay = document.getElementById('ac-rh-display');
                if (rhSlider && Number.isFinite(rhValue)) rhSlider.value = rhValue;
                if (rhDisplay && Number.isFinite(rhValue)) rhDisplay.textContent = rhValue;
            }
            if (ac.ac_fan_mode) {
                selectedACMode = String(ac.ac_fan_mode).toUpperCase();
                document.querySelectorAll('.ac-set-mode-btn').forEach(function(btn) {
                    var active = btn.getAttribute('data-mode') === selectedACMode;
                    btn.style.background = active ? 'var(--primary)' : 'var(--card-bg)';
                    btn.style.border = active ? '2px solid var(--primary)' : '2px solid var(--border)';
                    btn.style.color = active ? 'white' : 'var(--text)';
                });
            }
            if (typeof updateDashboard === 'function') {
                try { updateDashboard(); } catch (e) {}
            }
            updateExtraFeatureButtons(ac);
        }

        function updateExtraFeatureButtons(ac) {
            if (!ac) return;
            var features = [
                {id: 'btn-swing', statusId: 'swing-status', key: 'swing', color: '#0ea5e9'},
                {id: 'btn-turbo', statusId: 'turbo-status', key: 'turbo', color: '#1e40af'},
                {id: 'btn-econo', statusId: 'econo-status', key: 'econo', color: '#3b82f6'}
            ];
            features.forEach(function(f) {
                var btn = document.getElementById(f.id);
                var status = document.getElementById(f.statusId);
                var isOn = !!ac[f.key];
                if (btn) {
                    btn.style.background = isOn ? f.color : 'var(--bg-card)';
                    btn.style.color = isOn ? 'white' : 'var(--text-secondary)';
                    btn.style.borderColor = isOn ? f.color : 'var(--border)';
                }
                if (status) status.textContent = isOn ? 'ON' : 'OFF';
            });
        }

        function sendACCommand(command) {
            // Block manual commands when in ADAPTIVE mode
            fetch('/api/data').then(r => r.json()).then(data => {
                if (data.ac.mode === 'ADAPTIVE') {
                    showToast('Adaptive mode is active! Switch to Manual for manual control.', 'error');
                    return;
                }
                fetch('/api/ac/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: command })
                })
                .then(r => r.json())
                .then(result => {
                    if (result && result.ac) applyACSnapshot(result.ac);
                    const label = command.replace('_', ' ');
                    showToast('AC: ' + label, result && result.status === 'success' ? 'success' : 'info');
                })
                .catch(e => showToast('Error: ' + (e.message || e), 'error'));
            });
        }

        var selectedACMode = 'COOL';

        function selectACSetMode(mode, btn) {
            selectedACMode = mode;
            document.querySelectorAll('.ac-set-mode-btn').forEach(b => {
                b.style.background = 'var(--card-bg)';
                b.style.border = '2px solid var(--border)';
                b.style.color = 'var(--text)';
            });
            btn.style.background = 'var(--primary)';
            btn.style.border = '2px solid var(--primary)';
            btn.style.color = 'white';
        }

        function applyACSettings() {
            // Block manual commands when in ADAPTIVE mode
            fetch('/api/data').then(r => r.json()).then(data => {
                if (data.ac.mode === 'ADAPTIVE') {
                    showToast('Adaptive mode is active! Switch to Manual for manual control.', 'error');
                    return;
                }
                const temp = document.getElementById('ac-temp-slider').value;
                const rh = document.getElementById('ac-rh-slider').value;
                const fan = document.getElementById('fan-speed-slider').value;

                fetch('/api/ac/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        command: 'SET',
                        temperature: parseInt(temp),
                        set_rh: parseInt(rh),
                        fan_speed: parseInt(fan),
                        mode: selectedACMode
                    })
                })
                .then(r => r.json())
                .then(result => {
                    if (result && result.ac) applyACSnapshot(result.ac);
                    showToast('AC: ' + temp + '\xB0C, RH ' + rh + '%, Fan ' + fan + ', ' + selectedACMode);
                })
                .catch(e => showToast('Error: ' + e, 'error'));
            });
        }

        // ==================== LAMP CONTROLS ====================
        function updateBrightness(lampNum, value) {
            document.getElementById('brightness-display-' + lampNum).textContent = value;
            saveSettings();
        }

        function syncAllSliders() {
            const val = document.getElementById('brightness-slider-1').value;
            document.getElementById('brightness-slider-2').value = val;
            document.getElementById('brightness-display-2').textContent = val;
            saveSettings();
        }

        function applyLampSettings() {
            const b1 = parseInt(document.getElementById('brightness-slider-1').value);
            const b2 = parseInt(document.getElementById('brightness-slider-2').value);
            
            // Switch to MANUAL mode FIRST, then send brightness only after mode confirmed
            fetch('/api/lamp/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'MANUAL' })
            })
            .then(() => {
                applyLampModeUI('MANUAL');
                return fetch('/api/lamp/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ brightness1: b1, brightness2: b2, source: 'dashboard' })
                });
            })
            .then(r => r.json())
            .then(result => showToast('Lamp   MANUAL | L1=' + b1 + '% L2=' + b2 + '%'))
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        // ==================== MODE TOGGLES ====================
        function setACMode(mode) {
            fetch('/api/ac/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            })
            .then(r => r.json())
            .then(result => {
                applyACModeUI(mode);
                showToast('AC Mode: ' + mode, 'success');
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function applyACModeUI(mode) {
            const overlay = document.getElementById('ac-manual-overlay');
            const banner = document.getElementById('adaptive-info-banner');
            const indicator = document.getElementById('ac-mode-indicator');
            const btnAdaptive = document.getElementById('btn-mode-adaptive');
            const btnManual = document.getElementById('btn-mode-manual');

            if (mode === 'ADAPTIVE') {
                // Show overlay on manual controls
                if (overlay) overlay.style.display = 'flex';
                if (banner) banner.style.display = 'block';
                // Indicator
                if (indicator) {
                    indicator.style.background = 'rgba(59, 130, 246, 0.1)';
                    indicator.style.color = '#3b82f6';
                    indicator.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                    indicator.innerHTML = 'Current mode: <strong>ADAPTIVE</strong> - AC controlled automatically by GA optimization';
                }
                // Button styles
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
                    btnAdaptive.style.borderColor = '#3b82f6';
                    btnAdaptive.style.color = 'white';
                    btnAdaptive.style.opacity = '1';
                }
                if (btnManual) {
                    btnManual.style.background = 'var(--bg-card)';
                    btnManual.style.borderColor = 'var(--border)';
                    btnManual.style.color = 'var(--text-secondary)';
                    btnManual.style.opacity = '0.6';
                }
            } else {
                // Hide overlay - allow manual controls
                if (overlay) overlay.style.display = 'none';
                if (banner) banner.style.display = 'none';
                // Indicator
                if (indicator) {
                    indicator.style.background = 'rgba(14, 165, 233, 0.1)';
                    indicator.style.color = '#0ea5e9';
                    indicator.style.borderColor = 'rgba(14, 165, 233, 0.3)';
                    indicator.innerHTML = 'Current mode: <strong>MANUAL</strong> - Control AC manually using buttons below';
                }
                // Button styles
                if (btnManual) {
                    btnManual.style.background = 'linear-gradient(135deg, #0ea5e9, #0284c7)';
                    btnManual.style.borderColor = '#0ea5e9';
                    btnManual.style.color = 'white';
                    btnManual.style.opacity = '1';
                }
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'var(--bg-card)';
                    btnAdaptive.style.borderColor = 'var(--border)';
                    btnAdaptive.style.color = 'var(--text-secondary)';
                    btnAdaptive.style.opacity = '0.6';
                }
            }
            // Update adaptive info cards
            updateAdaptiveInfo();
        }

        function updateAdaptiveInfo() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    const gaTemp = document.getElementById('adaptive-ga-temp');
                    const gaFan = document.getElementById('adaptive-ga-fan');
                    const gaFitness = document.getElementById('adaptive-ga-fitness');
                    if (gaTemp) gaTemp.textContent = (data.system.ga_temp || 0) > 0 ? data.system.ga_temp + '\xB0C' : '--';
                    if (gaFan) gaFan.textContent = (data.system.ga_fan || 0) > 0 ? 'Lv ' + data.system.ga_fan : '--';
                    if (gaFitness) gaFitness.textContent = (data.system.ga_fitness || 0) > 0 ? parseFloat(data.system.ga_fitness).toFixed(2) : '--';
                });
        }

        function updateModeBadges() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    applyACModeUI(data.ac.mode);
                    applyLampModeUI(data.lamp.mode);
                });
        }

        function setLampMode(mode) {
            fetch('/api/lamp/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            })
            .then(r => r.json())
            .then(result => {
                applyLampModeUI(mode);
                showToast('Lamp Mode: ' + mode, 'success');
            })
            .catch(e => showToast('Error: ' + e, 'error'));
        }

        function applyLampModeUI(mode) {
            const overlay = document.getElementById('lamp-manual-overlay');
            const indicator = document.getElementById('lamp-mode-indicator');
            const btnAdaptive = document.getElementById('btn-lamp-adaptive');
            const btnManual = document.getElementById('btn-lamp-manual');
            if (mode === 'ADAPTIVE') {
                if (overlay) overlay.style.display = 'flex';
                if (indicator) {
                    indicator.style.background = 'rgba(14, 165, 233, 0.1)';
                    indicator.style.color = '#0ea5e9';
                    indicator.style.borderColor = 'rgba(14, 165, 233, 0.3)';
                    indicator.innerHTML = 'Current mode: <strong>ADAPTIVE</strong> - Lamps controlled automatically by PSO optimization';
                }
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'linear-gradient(135deg, #0ea5e9, #0284c7)';
                    btnAdaptive.style.borderColor = '#0ea5e9';
                    btnAdaptive.style.color = 'white';
                    btnAdaptive.style.opacity = '1';
                }
                if (btnManual) {
                    btnManual.style.background = 'var(--bg-card)';
                    btnManual.style.borderColor = 'var(--border)';
                    btnManual.style.color = 'var(--text-secondary)';
                    btnManual.style.opacity = '0.6';
                }
            } else {
                if (overlay) overlay.style.display = 'none';
                if (indicator) {
                    indicator.style.background = 'rgba(37, 99, 235, 0.1)';
                    indicator.style.color = '#2563eb';
                    indicator.style.borderColor = 'rgba(37, 99, 235, 0.3)';
                    indicator.innerHTML = 'Current mode: <strong>MANUAL</strong> - Control lamps manually using sliders below';
                }
                if (btnManual) {
                    btnManual.style.background = 'linear-gradient(135deg, #2563eb, #1d4ed8)';
                    btnManual.style.borderColor = '#2563eb';
                    btnManual.style.color = 'white';
                    btnManual.style.opacity = '1';
                }
                if (btnAdaptive) {
                    btnAdaptive.style.background = 'var(--bg-card)';
                    btnAdaptive.style.borderColor = 'var(--border)';
                    btnAdaptive.style.color = 'var(--text-secondary)';
                    btnAdaptive.style.opacity = '0.6';
                }
            }
        }

        // ==================== IR REMOTE ====================
        var currentLearningButton = null;
        var learningCheckInterval = null;
        
        function learnIRCode(buttonName, deviceName = 'AC') {
            // Clear any previous learning interval
            if (learningCheckInterval) {
                clearInterval(learningCheckInterval);
                learningCheckInterval = null;
            }
            
            const buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
            const statusElement = document.getElementById('status-' + buttonName);
            
            currentLearningButton = buttonName;
            
            if (buttonElement) {
                buttonElement.classList.add('learning');
            }
            
            if (statusElement) {
                statusElement.textContent = 'Ready! Press remote...';
                statusElement.style.color = '#0ea5e9';
            }
            
            fetch('/api/ir/learn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    button: buttonName,
                    device: deviceName 
                })
            })
            .then(r => r.json())
            .then(result => {
                console.log('Learning mode activated:', result);
                showToast('Press remote button for: ' + buttonName, 'info');
                
                // Auto-check status every 500ms
                let checkCount = 0;
                learningCheckInterval = setInterval(() => {
                    checkCount++;
                    
                    fetch('/api/ir/codes')
                        .then(r => r.json())
                        .then(data => {
                            if (data.codes && data.codes[buttonName]) {
                                // Code learned!
                                clearInterval(learningCheckInterval);
                                
                                if (buttonElement) {
                                    buttonElement.classList.remove('learning');
                                    buttonElement.classList.add('learned');
                                }
                                
                                if (statusElement) {
                                    statusElement.textContent = 'Learned OK';
                                    statusElement.style.color = '#3b82f6';
                                }
                                
                                showToast('IR Code learned: ' + buttonName, 'success');
                                currentLearningButton = null;
                            }
                            
                            // Stop after 120 checks (60 seconds)
                            if (checkCount > 120) {
                                clearInterval(learningCheckInterval);
                                if (buttonElement) buttonElement.classList.remove('learning');
                                if (statusElement) {
                                    statusElement.textContent = 'Timeout';
                                    statusElement.style.color = '#1e40af';
                                }
                                showToast('Learning timeout - no signal', 'error');
                                currentLearningButton = null;
                            }
                        })
                        .catch(e => console.error('Status check error:', e));
                }, 500);
            })
            .catch(e => { 
                if (buttonElement) buttonElement.classList.remove('learning'); 
                if (statusElement) {
                    statusElement.textContent = 'Error';
                    statusElement.style.color = '#1e40af';
                }
                showToast('Error: ' + e, 'error'); 
            });
        }

        function sendIRCode(buttonName) {
            // For IR Learning panel Send buttons - still uses learned codes
            fetch('/api/ir/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ button: buttonName })
            })
            .then(r => r.json())
            .then(result => {
                if (result.status === 'success') {
                    showToast('IR Code sent: ' + buttonName, 'success');
                } else {
                    showToast(result.message || 'Failed to send', 'error');
                }
            })
            .catch(e => showToast('Error: ' + (e.message || e), 'error'));
        }

        function sendACMode(modeName, btnElement) {
            // Block manual commands when in ADAPTIVE mode
            fetch('/api/data').then(r => r.json()).then(data => {
                if (data.ac.mode === 'ADAPTIVE') {
                    showToast('Adaptive mode is active! Switch to Manual for manual control.', 'error');
                    return;
                }
                // Visual feedback
                document.querySelectorAll('.ac-mode-btn').forEach(btn => {
                    btn.style.opacity = '0.6';
                    btn.style.transform = 'scale(0.95)';
                });
                if (btnElement) {
                    btnElement.style.opacity = '1';
                    btnElement.style.transform = 'scale(1.05)';
                }

                // Single endpoint handles state tracking + IR code transmission
                fetch('/api/ac/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: modeName })
                })
                .then(r => r.json())
                .then(result => {
                    if (result && result.ac) applyACSnapshot(result.ac);
                    const modeLabel = modeName.replace('MODE_', '');
                    showToast('AC Mode: ' + modeLabel, result && result.status === 'success' ? 'success' : 'info');
                    setTimeout(() => {
                        document.querySelectorAll('.ac-mode-btn').forEach(btn => {
                            btn.style.opacity = '1';
                            btn.style.transform = 'scale(1)';
                        });
                        if (btnElement) {
                            btnElement.style.transform = 'scale(1.05)';
                            btnElement.style.boxShadow = '0 0 15px rgba(37, 99, 235, 0.5)';
                        }
                    }, 300);
                })
                .catch(e => {
                    showToast('Error: ' + (e.message || e), 'error');
                    document.querySelectorAll('.ac-mode-btn').forEach(btn => {
                        btn.style.opacity = '1';
                        btn.style.transform = 'scale(1)';
                    });
                });
            }); // end fetch /api/data
        }

        function loadIRCodes() {
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    learnedCodes = data.codes || data;
                    const codeCount = Object.keys(learnedCodes).length;
                    
                    // Update summary
                    const summaryEl = document.getElementById('ir-total-learned');
                    if (summaryEl) {
                        summaryEl.textContent = codeCount;
                        summaryEl.style.color = codeCount > 0 ? '#3b82f6' : '#1e40af';
                    }
                    
                    // Update button status
                    Object.keys(learnedCodes).forEach(buttonName => {
                        const buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
                        const statusElement = document.getElementById('status-' + buttonName);
                        if (buttonElement && statusElement) {
                            buttonElement.classList.add('learned');
                            statusElement.textContent = 'Learned [OK]';
                            statusElement.style.color = '#3b82f6';
                        }
                    });
                    
                    if (codeCount > 0) {
                        showToast('IR codes loaded: ' + codeCount + ' button(s)', 'success');
                    }
                })
                .catch(e => {
                    console.error('Error loading IR codes:', e);
                    showToast('Error loading IR codes', 'error');
                });
        }

        function resetAllIRCodes() {
            if (!confirm('[WARNING] Reset all learned IR codes?\\n\\nThis will delete ALL saved remote buttons!')) {
                return;
            }
            
            // Clear UI
            const buttons = ['POWER_ON', 'POWER_OFF', 'TEMP_UP', 'TEMP_DOWN', 'MODE_AUTO', 'MODE_COOL', 'MODE_FAN', 'MODE_DRY'];
            buttons.forEach(buttonName => {
                const buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
                const statusElement = document.getElementById('status-' + buttonName);
                
                if (buttonElement) {
                    buttonElement.classList.remove('learned');
                }
                if (statusElement) {
                    statusElement.textContent = 'Not learned';
                    statusElement.style.color = '#94a3b8';
                }
                
                // Delete from server
                fetch('/api/ir/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ button: buttonName })
                }).catch(e => console.log('Delete error:', e));
            });
            
            // Update summary
            const summaryEl = document.getElementById('ir-total-learned');
            if (summaryEl) {
                summaryEl.textContent = '0';
                summaryEl.style.color = '#1e40af';
            }
            
            learnedCodes = {};
            showToast('All IR codes reset!', 'success');
        }

        function saveAllIRCodes() {
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    const codes = data.codes || data;
                    const count = Object.keys(codes).length;
                    
                    if (count === 0) {
                        showToast('No IR codes to save yet', 'warning');
                        return;
                    }
                    
                    showToast(count + ' IR codes saved to server', 'success');
                    console.log('IR Codes saved:', codes);
                })
                .catch(e => showToast('Save error: ' + e, 'error'));
        }

        function exportIRCodes() {
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    const codes = data.codes || data;
                    const count = Object.keys(codes).length;
                    
                    if (count === 0) {
                        showToast('No IR codes to export', 'warning');
                        return;
                    }
                    
                    // Create JSON file and download
                    const jsonStr = JSON.stringify(codes, null, 2);
                    const blob = new Blob([jsonStr], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'ir_codes_' + new Date().toISOString().slice(0, 10) + '.json';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    showToast('Exported ' + count + ' IR codes', 'success');
                })
                .catch(e => showToast('Export error: ' + e, 'error'));
        }

        function showIRDebugInfo() {
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    const codes = data.codes || data;
                    
                    let debugInfo = '                                   \\n';
                    debugInfo += '  IR CODES DEBUG INFO\\n';
                    debugInfo += '                                   \\n\\n';
                    
                    if (Object.keys(codes).length === 0) {
                        debugInfo += '[WARN] No IR codes learned yet\\n\\n';
                        debugInfo += 'Steps to learn:\\n';
                        debugInfo += '1. Select protocol (RAW recommended for Mitsubishi)\\n';
                        debugInfo += '2. Click \"Learn\" button\\n';
                        debugInfo += '3. Press remote button (hold 2-3 seconds)\\n';
                        debugInfo += '4. Wait for \"Learned [OK]\" status\\n';
                        debugInfo += '5. Click \"Send\" to test\\n';
                    } else {
                        Object.keys(codes).forEach(button => {
                            const codeStr = codes[button];
                            const codePreview = codeStr.substring(0, 60) + (codeStr.length > 60 ? '...' : '');
                            debugInfo += '[IR] ' + button + ':\\n';
                            debugInfo += '   Code: ' + codePreview + '\\n';
                            debugInfo += '   Length: ' + codeStr.length + ' chars\\n';
                            
                            // Parse protocol
                            if (codeStr.includes(':')) {
                                const protocol = codeStr.split(':')[0];
                                debugInfo += '   Protocol: ' + protocol + '\\n';
                            }
                            debugInfo += '\\n';
                        });
                        
                        debugInfo += '                                   \\n';
                        debugInfo += 'Total: ' + Object.keys(codes).length + ' codes\\n';
                    }
                    
                    debugInfo += '\\n[TIP] TROUBLESHOOTING:\\n';
                    debugInfo += '- If AC not responding: Use RAW protocol\\n';
                    debugInfo += '- If IR LED not blinking: Check ESP32 connection\\n';
                    debugInfo += '- If code too short: Press remote longer (2-3s)\\n';
                    debugInfo += '- Distance to AC: 1-3 meters, direct line\\n';
                    
                    alert(debugInfo);
                    console.log('IR Debug Info:', codes);
                })
                .catch(e => showToast('Debug error: ' + e, 'error'));
        }

        function testAllIRCodes() {
            if (!confirm('Test all learned IR codes?\\n\\nThis will send all codes one by one with 2 second delay.')) {
                return;
            }
            
            fetch('/api/ir/codes')
                .then(r => r.json())
                .then(data => {
                    const codes = data.codes || data;
                    const buttons = Object.keys(codes);
                    
                    if (buttons.length === 0) {
                        showToast('No codes to test', 'warning');
                        return;
                    }
                    
                    showToast('Testing ' + buttons.length + ' codes...', 'info');
                    
                    let index = 0;
                    const testInterval = setInterval(() => {
                        if (index >= buttons.length) {
                            clearInterval(testInterval);
                            showToast('Test complete!', 'success');
                            return;
                        }
                        
                        const button = buttons[index];
                        console.log('Testing ' + button + '...');
                        sendIRCode(button);
                        showToast('Testing: ' + button, 'info');
                        
                        index++;
                    }, 2000); // 2 second delay between tests
                })
                .catch(e => showToast('Test error: ' + e, 'error'));
        }

        // ==================== DETECTION ALERT ====================
        var lastDetectionTime = 0;
        var DETECTION_COOLDOWN = 60000; // 60 seconds (1 minute) between alerts
        var detectionSoundEnabled = localStorage.getItem('detectionSound') !== 'false';

        // Initialize sound toggle button visual on load
        function updateSoundToggleUI() {
            const btn = document.getElementById('sound-toggle-btn');
            const text = document.getElementById('sound-toggle-text');
            if (!btn) return;
            if (detectionSoundEnabled) {
                btn.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
                if (text) text.textContent = 'Sound ON';
            } else {
                btn.style.background = 'linear-gradient(135deg, #1e40af, #1d4ed8)';
                if (text) text.textContent = 'Sound OFF';
            }
        }

        function playDetectionSound() {
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const now = audioContext.currentTime;

                // Beep 1 - alert tone
                const osc1 = audioContext.createOscillator();
                const gain1 = audioContext.createGain();
                osc1.connect(gain1);
                gain1.connect(audioContext.destination);
                osc1.frequency.value = 880;
                osc1.type = 'sine';
                gain1.gain.setValueAtTime(0.4, now);
                gain1.gain.exponentialRampToValueAtTime(0.01, now + 0.25);
                osc1.start(now);
                osc1.stop(now + 0.25);

                // Beep 2 - higher pitched
                const osc2 = audioContext.createOscillator();
                const gain2 = audioContext.createGain();
                osc2.connect(gain2);
                gain2.connect(audioContext.destination);
                osc2.frequency.value = 1100;
                osc2.type = 'sine';
                gain2.gain.setValueAtTime(0.4, now + 0.3);
                gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.55);
                osc2.start(now + 0.3);
                osc2.stop(now + 0.55);

                // Beep 3 - highest
                const osc3 = audioContext.createOscillator();
                const gain3 = audioContext.createGain();
                osc3.connect(gain3);
                gain3.connect(audioContext.destination);
                osc3.frequency.value = 1320;
                osc3.type = 'sine';
                gain3.gain.setValueAtTime(0.35, now + 0.6);
                gain3.gain.exponentialRampToValueAtTime(0.01, now + 1.0);
                osc3.start(now + 0.6);
                osc3.stop(now + 1.0);
            } catch(e) {
                console.log('Audio notification not available:', e);
            }
        }

        function showDetectionAlert(count, confidence) {
            const now = Date.now();
            if (now - lastDetectionTime < DETECTION_COOLDOWN) return;
            
            lastDetectionTime = now;
            
            document.getElementById('alert-person-count').textContent = count;
            document.getElementById('alert-person-confidence').textContent = confidence + '%';
            document.getElementById('alert-time').textContent = new Date().toLocaleTimeString();
            
            const alertBox = document.getElementById('detection-alert');
            alertBox.classList.add('show');
            
            // Play 3-beep alert sound if enabled
            if (detectionSoundEnabled) {
                playDetectionSound();
            }
            
            // Auto hide after 5 seconds
            setTimeout(() => {
                alertBox.classList.remove('show');
            }, 5000);
            
            console.log('[ALERT] PERSON DETECTED:', {
                count: count,
                confidence: confidence + '%',
                time: new Date().toLocaleTimeString()
            });
        }

        function closeDetectionAlert() {
            document.getElementById('detection-alert').classList.remove('show');
        }

        function toggleDetectionSound() {
            detectionSoundEnabled = !detectionSoundEnabled;
            localStorage.setItem('detectionSound', detectionSoundEnabled);
            updateSoundToggleUI();
            if (detectionSoundEnabled) {
                playDetectionSound();
                showToast('Sound alerts ON - you will hear a sound when a person is detected', 'success');
            } else {
                showToast('Sound alerts OFF - sound notifications disabled', 'info');
            }
        }

        // ==================== TOAST ====================
        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            const toastMessage = document.getElementById('toast-message');
            const icon = type === 'success' ? 'check' : (type === 'info' ? 'info' : 'exclamation');
            
            toastMessage.innerHTML = message;
            toast.classList.add('show');
            setTimeout(() => { toast.classList.remove('show'); }, 3000);
        }

        var energyBubbleTimer = null;
        function showEnergyBubble(power, voltage, current) {
            const bubble = document.getElementById('energy-bubble');
            const text = document.getElementById('energy-bubble-text');
            if (!bubble || !text) return;
            text.textContent = power.toFixed(3) + ' kW | ' + voltage.toFixed(1) + ' V | ' + current.toFixed(2) + ' A';
            bubble.classList.add('show');
            if (energyBubbleTimer) clearTimeout(energyBubbleTimer);
            energyBubbleTimer = setTimeout(() => { bubble.classList.remove('show'); }, 4000);
        }

        // ==================== OCCUPANCY FEEDBACK ====================
        function selectFeedbackRating(value) {
            selectedFeedbackRating = value;
            document.querySelectorAll('#rating-row .rating-btn').forEach((btn, idx) => {
                btn.classList.toggle('active', idx + 1 === value);
            });
        }

        function saveGoogleFormUrl() {
            const url = (document.getElementById('google-form-url').value || '').trim();
            if (!url) {
                showToast('Please enter Google Form URL first', 'error');
                return;
            }
            localStorage.setItem('googleFormUrl', url);
            showToast('Google Form URL saved', 'success');
        }

        function openGoogleForm() {
            const url = (document.getElementById('google-form-url').value || '').trim() || localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;
            window.open(url, '_blank');
        }

        function submitOccupancyFeedback() {
            const comment = (document.getElementById('feedback-comment').value || '').trim();
            const formUrl = (document.getElementById('google-form-url').value || '').trim() || localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;

            if (!selectedFeedbackRating) {
                showToast('Please select a rating 1-5 first', 'error');
                return;
            }

            fetch('/api/occupancy/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rating: selectedFeedbackRating,
                    comment: comment,
                    google_form_url: formUrl,
                    occupancy_count: parseInt((document.getElementById('cam-count') ? document.getElementById('cam-count').textContent : '0') || '0', 10)
                })
            })
            .then(r => r.json())
            .then(result => {
                if (result.status === 'success') {
                    showToast('Feedback submitted successfully', 'success');
                    document.getElementById('feedback-comment').value = '';
                    selectFeedbackRating(0);
                    loadFeedbackHistory();
                    if (formUrl && formUrl !== DEFAULT_GOOGLE_FORM_URL) {
                        window.open(formUrl, '_blank');
                    }
                } else {
                    showToast(result.message || 'Failed to submit feedback', 'error');
                }
            })
            .catch(e => showToast('Error: ' + (e.message || e), 'error'));
        }

        function loadFeedbackHistory() {
            fetch('/api/occupancy/feedback/list')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('feedback-history');
                    if (!container) return;

                    const rows = data.feedback || [];
                    if (rows.length === 0) {
                        container.innerHTML = '<div style="color: var(--text-secondary); font-size: 13px;">No feedback yet.</div>';
                        return;
                    }

                    container.innerHTML = rows.map(item => {
                        const safeComment = (item.comment || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        return '<div class="feedback-history-item">' +
                            '<div style="display:flex; justify-content:space-between; margin-bottom:6px;">' +
                                '<strong>Rating: ' + item.rating + '/5</strong>' +
                                '<span style="font-size:12px; color: var(--text-secondary);">' + item.time + '</span>' +
                            '</div>' +
                            '<div style="font-size: 13px; color: var(--text-secondary);">Occupancy: ' + item.occupancy_count + ' person(s)</div>' +
                            '<div style="margin-top:6px; font-size:13px;">' + (safeComment || '-') + '</div>' +
                        '</div>';
                    }).join('');
                })
                .catch(() => {});
        }

        // ==================== DATA UPDATES ====================
        function updateDashboard() {
            fetch('/api/data', { cache: 'no-store' })
                .then(r => r.json())
                .then(data => { try {
                    data = data || {};
                    const ac = data.ac || {};
                    const lamp = data.lamp || {};
                    const camera = data.camera || {};
                    const system = data.system || {};
                    const energy = data.energy || {};
                    const num = (v, d = 0) => {
                        const n = parseFloat(v);
                        return Number.isFinite(n) ? n : d;
                    };
                    const setText = (id, val) => {
                        const el = document.getElementById(id);
                        if (el) el.textContent = val;
                    };

                    const temperature = num(ac.temperature);
                    setText('dash-temp', temperature.toFixed(1));
                    setText('dash-hum', num(ac.humidity).toFixed(1));
                    // Individual sensor readings - sorted coldest hottest (S1 S3)
                    var rawTemps = [
                        { t: num(ac.temp1), h: num(ac.hum1) },
                        { t: num(ac.temp2), h: num(ac.hum2) },
                        { t: num(ac.temp3), h: num(ac.hum3) }
                    ];
                    // Sort by temperature ascending (coldest first = S1)
                    rawTemps.sort(function(a, b) { return a.t - b.t; });
                    const t1El = document.getElementById('dash-temp1');
                    const t2El = document.getElementById('dash-temp2');
                    const t3El = document.getElementById('dash-temp3');
                    if (t1El) t1El.textContent = rawTemps[0].t.toFixed(1);
                    if (t2El) t2El.textContent = rawTemps[1].t.toFixed(1);
                    if (t3El) t3El.textContent = rawTemps[2].t.toFixed(1);
                    // Individual humidity readings (sorted same order as temp)
                    const h1El = document.getElementById('dash-hum1');
                    const h2El = document.getElementById('dash-hum2');
                    const h3El = document.getElementById('dash-hum3');
                    if (h1El) h1El.textContent = rawTemps[0].h.toFixed(1);
                    if (h2El) h2El.textContent = rawTemps[1].h.toFixed(1);
                    if (h3El) h3El.textContent = rawTemps[2].h.toFixed(1);
                    
                    // Heat Index
                    const hiEl = document.getElementById('dash-heat-index');
                    if (hiEl) hiEl.textContent = num(ac.heat_index).toFixed(1);
                    
                    // ESP32 Signal & Uptime
                    const rssiEl = document.getElementById('dash-rssi');
                    if (rssiEl) rssiEl.textContent = ac.rssi || 0;
                    const uptEl = document.getElementById('dash-uptime');
                    if (uptEl) {
                        const secs = num(ac.uptime);
                        const h = Math.floor(secs / 3600);
                        const m = Math.floor((secs % 3600) / 60);
                        uptEl.textContent = h > 0 ? h + 'h ' + m + 'm' : m + 'm ' + (secs % 60) + 's';
                    }
                    
                    // AC State - use REAL state from ESP32, not temperature guessing
                    const acStateEl = document.getElementById('dash-ac-state');
                    let acState = ac.ac_state || 'OFF';
                    
                    if (acStateEl) {
                        if (acState === 'ON') {
                            acStateEl.style.color = '#3b82f6'; // Green
                        } else {
                            acStateEl.style.color = '#1e40af'; // Red
                        }
                        acStateEl.textContent = acState;
                    }
                    
                    // AC panel power dot
                    const panelDot = document.getElementById('ac-panel-dot');
                    if (panelDot) {
                        panelDot.style.background = acState === 'ON' ? '#3b82f6' : '#1e40af';
                        panelDot.style.boxShadow = acState === 'ON' ? '0 0 8px rgba(59,130,246,0.5)' : '0 0 8px rgba(30,64,175,0.5)';
                    }
                    
                    // Set Temperature
                    const acTemp = num(ac.ac_temp, 24);
                    setText('dash-ac-temp', acTemp);
                    
                    // Fan Speed with label
                    const fanSpeed = num(ac.fan_speed, 1);
                    setText('dash-ac-fan', fanSpeed);
                    const fanLabel = document.getElementById('dash-ac-fan-label');
                    if (fanLabel) {
                        const fanNames = {1: 'Low', 2: 'Medium', 3: 'High'};
                        fanLabel.textContent = fanNames[fanSpeed] || 'Level ' + fanSpeed;
                    }
                    
                    // AC Mode (COOL/HEAT/DRY/FAN/AUTO) with icon + color - uses ac_fan_mode from ESP32
                    const acMode = ac.ac_fan_mode || 'COOL';
                    setText('dash-ac-mode', acMode);
                    const modeIconEl = document.getElementById('dash-ac-mode-icon');
                    const modeTextEl = document.getElementById('dash-ac-mode');
                    const modeIcons = {
                        'COOL': {text: 'COOL', color: '#0ea5e9'},
                        'HEAT': {text: 'HEAT', color: '#0ea5e9'},
                        'DRY':  {text: 'DRY', color: '#2563eb'},
                        'FAN':  {text: 'FAN', color: '#3b82f6'},
                        'AUTO': {text: 'AUTO', color: '#2563eb'}
                    };
                    const modeInfo = modeIcons[acMode] || modeIcons['COOL'];
                    if (modeIconEl) {
                        modeIconEl.textContent = modeInfo.text;
                        modeIconEl.style.color = modeInfo.color;
                    }
                    if (modeTextEl) modeTextEl.style.color = modeInfo.color;
                    
                    // Operating Mode (ADAPTIVE / MANUAL)
                    const ctrlMode = ac.mode || 'ADAPTIVE';
                    const ctrlIcon = document.getElementById('dash-ac-ctrl-icon');
                    const ctrlText = document.getElementById('dash-ac-ctrl-mode');
                    if (ctrlIcon && ctrlText) {
                        if (ctrlMode === 'ADAPTIVE') {
                            ctrlIcon.textContent = 'A';
                            ctrlIcon.style.color = '#3b82f6';
                            ctrlText.textContent = 'ADAPTIVE';
                            ctrlText.style.color = '#3b82f6';
                        } else {
                            ctrlIcon.textContent = 'M';
                            ctrlIcon.style.color = '#0ea5e9';
                            ctrlText.textContent = 'MANUAL';
                            ctrlText.style.color = '#0ea5e9';
                        }
                    }
                    
                    // Source badge
                    const srcBadge = document.getElementById('dash-ac-source');
                    if (srcBadge) {
                        if (ctrlMode === 'ADAPTIVE') {
                            srcBadge.textContent = 'AI Controlled';
                            srcBadge.style.background = 'rgba(59, 130, 246, 0.12)';
                            srcBadge.style.color = '#3b82f6';
                            srcBadge.style.borderColor = 'rgba(59, 130, 246, 0.25)';
                        } else {
                            srcBadge.textContent = 'Manual Control';
                            srcBadge.style.background = 'rgba(14, 165, 233, 0.12)';
                            srcBadge.style.color = '#0ea5e9';
                            srcBadge.style.borderColor = 'rgba(14, 165, 233, 0.25)';
                        }
                    }
                    
                    // Room environment in AC panel footer
                    const roomTemp = document.getElementById('dash-ac-room-temp');
                    const roomHum = document.getElementById('dash-ac-room-hum');
                    if (roomTemp) roomTemp.textContent = temperature.toFixed(1);
                    if (roomHum) roomHum.textContent = num(ac.humidity).toFixed(1);
                    // Individual sensors in AC panel footer (use sorted rawTemps from above)
                    const acT1 = document.getElementById('dash-ac-temp1');
                    const acT2 = document.getElementById('dash-ac-temp2');
                    const acT3 = document.getElementById('dash-ac-temp3');
                    if (acT1 && rawTemps) acT1.textContent = rawTemps[0].t.toFixed(1);
                    if (acT2 && rawTemps) acT2.textContent = rawTemps[1].t.toFixed(1);
                    if (acT3 && rawTemps) acT3.textContent = rawTemps[2].t.toFixed(1);
                    
                    // Update AC Live Status Bar in control panel
                    const liveDot = document.getElementById('ac-live-dot');
                    const liveState = document.getElementById('ac-live-state');
                    if (liveDot && liveState) {
                        liveState.textContent = acState;
                        liveDot.style.background = acState === 'ON' ? '#3b82f6' : '#1e40af';
                        setText('ac-live-temp', acTemp);
                        setText('ac-live-fan', fanSpeed);
                        setText('ac-live-mode', acMode);
                    }
                    updateExtraFeatureButtons(ac);
                    setText('dash-lux1', num(lamp.lux1).toFixed(0));
                    setText('dash-lux2', num(lamp.lux2).toFixed(0));
                    setText('dash-lux3', num(lamp.lux3).toFixed(0));
                    setText('dash-lux-avg', num(lamp.lux_avg).toFixed(1));
                    setText('dash-bright1', Math.round(num(lamp.brightness1)));
                    setText('dash-bright2', Math.round(num(lamp.brightness2)));
                    setText('dash-motion', lamp.motion ? 'MOTION DETECTED' : 'NO MOTION');
                    // Update lamp control page live readings
                    setText('ctrl-lux1', num(lamp.lux1).toFixed(0));
                    setText('ctrl-lux2', num(lamp.lux2).toFixed(0));
                    setText('ctrl-lux3', num(lamp.lux3).toFixed(0));
                    const ctrlMotion = document.getElementById('ctrl-motion-status');
                    if (ctrlMotion) ctrlMotion.innerHTML = lamp.motion ? '<span style="color:#3b82f6">MOTION</span>' : '<span style="color:#1e40af">IDLE</span>';
                    // Sync lamp mode buttons
                    applyLampModeUI(lamp.mode || 'ADAPTIVE');
                    
                    const personDetected = !!camera.person_detected;
                    const personCount = num(camera.count);
                    const confidence = num(camera.confidence);
                    
                    const camPersonEl = document.getElementById('cam-person');
                    if (camPersonEl) {
                        camPersonEl.textContent = personDetected ? 'Yes' : 'No';
                        camPersonEl.style.color = personDetected ? '#3b82f6' : '#1e40af';
                    }
                    
                    const camCountEl = document.getElementById('cam-count');
                    if (camCountEl) {
                        camCountEl.textContent = personCount;
                        camCountEl.style.color = personCount > 0 ? '#3b82f6' : '#94a3b8';
                    }
                    
                    const camConfEl = document.getElementById('cam-confidence');
                    if (camConfEl) {
                        camConfEl.textContent = confidence + '%';
                        camConfEl.style.color = confidence > 70 ? '#3b82f6' : (confidence > 50 ? '#0ea5e9' : '#1e40af');
                    }
                    
                    // Person detection status badge
                    const camBadge = document.getElementById('cam-status-badge');
                    if (camBadge) {
                        if (personCount > 0) {
                            camBadge.innerHTML = '<i class="fas fa-circle" style="font-size: 7px; vertical-align: middle;"></i> ' + personCount + ' Person Detected';
                            camBadge.style.background = 'rgba(59, 130, 246, 0.12)';
                            camBadge.style.color = '#3b82f6';
                            camBadge.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                        } else {
                            camBadge.innerHTML = '<i class="fas fa-circle" style="font-size: 7px; vertical-align: middle;"></i> No Person';
                            camBadge.style.background = 'rgba(30, 64, 175, 0.12)';
                            camBadge.style.color = '#1e40af';
                            camBadge.style.borderColor = 'rgba(30, 64, 175, 0.3)';
                        }
                    }

                    const occCountEl = document.getElementById('occ-live-count');
                    const occConfEl = document.getElementById('occ-live-confidence');
                    if (occCountEl) occCountEl.textContent = personCount;
                    if (occConfEl) occConfEl.textContent = confidence + '%';
                    
                    // Show detection alert if person detected
                    if (personDetected && personCount > 0) {
                        showDetectionAlert(personCount, confidence);
                    }
                    
                    // Update GA/PSO Fitness (from optimization algorithm)
                    const gaFitness = num(system.ga_fitness);
                    const psoFitness = num(system.pso_fitness);
                    
                    const gaEl = document.getElementById('ga-fitness');
                    const psoEl = document.getElementById('pso-fitness');
                    
                    if (gaEl) {
                        gaEl.textContent = gaFitness.toFixed(2);
                        gaEl.style.color = gaFitness > 0 ? '#3b82f6' : '#94a3b8';
                    }
                    if (psoEl) {
                        psoEl.textContent = psoFitness.toFixed(2);
                        psoEl.style.color = psoFitness > 0 ? '#3b82f6' : '#94a3b8';
                    }
                    
                    // PSO Brightness - pso-brightness element not on main page (only on ML page)
                    const psoBrightEl = document.getElementById('pso-brightness');
                    if (psoBrightEl) {
                        psoBrightEl.textContent = Math.round(num(system.pso_brightness));
                    }
                    
                    // Optimization Runs
                    const optRunsEl = document.getElementById('dash-opt-runs');
                    if (optRunsEl) {
                        optRunsEl.textContent = num(system.optimization_runs);
                    }
                    
                    // Energy data dihandle oleh mysql_energy_update event
                    // (ac-power, lamp-power, total-power, total-current, total-energy-kwh,
                    //  daily-cost diisi langsung dari MySQL via socket)

                    updateModeBadges();
                } catch(e) { var p = document.getElementById('diag-panel'); if (p) { p.style.display='block'; var d = document.getElementById('diag-result'); if (d) d.textContent += '\\n[DASHBOARD ERROR] ' + e.message; } console.error(e); } });
        }

        function updateLogs() {
            fetch('/api/logs')
                .then(r => r.json())
                .then(logs => {
                    const container = document.getElementById('log-container');
                    container.innerHTML = '';
                    logs.reverse().forEach(log => {
                        const entry = document.createElement('div');
                        entry.className = 'log-entry ' + log.level;
                        entry.innerHTML = '<strong>[' + log.time + ']</strong> ' + log.msg;
                        container.appendChild(entry);
                    });
                });
        }

        // ==================== DEVICE STATUS ====================
        function diagLog(msg) {
            var el = document.getElementById('diag-result');
            if (el) el.textContent += msg + '\\n';
        }
        function diagClear(title) {
            var el = document.getElementById('diag-result');
            if (el) el.textContent = '[' + new Date().toLocaleTimeString() + '] ' + title + '\\n';
        }

        function checkMqttStatus(showDetail) {
            fetch('/api/mqtt/status')
                .then(r => r.json())
                .then(data => {
                    var dot = document.getElementById('mqtt-dot');
                    var txt = document.getElementById('mqtt-status-text');
                    if (dot && txt) {
                        if (data.connected) {
                            dot.className = 'device-dot online';
                            txt.textContent = 'Connected' + (data.message_count > 0 ? ' (' + data.message_count + ' msgs)' : '');
                        } else {
                            dot.className = 'device-dot offline';
                            txt.textContent = data.error ? data.error.substring(0, 30) : 'Disconnected';
                        }
                    }
                    if (showDetail) {
                        diagClear('=== MQTT STATUS ===');
                        diagLog('Broker: ' + data.broker);
                        diagLog('Connected: ' + (data.connected ? 'YES' : 'NO'));
                        diagLog('Messages received: ' + data.message_count);
                        diagLog('Last connect: ' + (data.last_connect || 'Never'));
                        diagLog('Last message: ' + (data.last_message || 'Never'));
                        diagLog('Error: ' + (data.error || 'None'));
                        diagLog('Subscriptions: ' + (data.subscriptions || []).join(', '));
                        if (!data.connected) {
                            diagLog('');
                            diagLog('SOLUSI: Pastikan MQTT broker (Mosquitto) berjalan!');
                            diagLog('Windows: net start mosquitto');
                            diagLog('Linux/Pi: sudo systemctl start mosquitto');
                        }
                    }
                })
                .catch(function() {
                    var dot = document.getElementById('mqtt-dot');
                    var txt = document.getElementById('mqtt-status-text');
                    if (dot) dot.className = 'device-dot offline';
                    if (txt) txt.textContent = 'API Error';
                    if (showDetail) diagLog('ERROR: Cannot fetch /api/mqtt/status');
                });
        }

        function runSimulate() {
            diagClear('=== TEST FRONTEND (INJECT DATA DUMMY) ===');
            diagLog('Sending dummy data to server...');
            fetch('/api/simulate', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'ok') {
                        diagLog('SUCCESS! Dummy data injected:');
                        diagLog('  AC Temperature: ' + data.ac_temp + ' C');
                        diagLog('  Lamp Lux1: ' + data.lamp_lux + ' lux');
                        diagLog('');
                        diagLog('If values on dashboard change from 0 -> FRONTEND OK');
                        diagLog('If still 0 -> Problem in JavaScript/DOM');
                        updateDashboard();
                    } else {
                        diagLog('ERROR: ' + data.message);
                    }
                })
                .catch(function(e) { diagLog('FETCH ERROR: ' + e); });
        }

        function runMqttSelftest() {
            diagClear('=== TEST MQTT BROKER (SELF-TEST) ===');
            diagLog('Server will publish to smartroom/ac/sensors...');
            fetch('/api/mqtt/selftest', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'ok') {
                        diagLog('SUCCESS! Test message published to broker.');
                        diagLog(data.message);
                        diagLog('');
                        diagLog('Wait 2 seconds... then check if data appears on dashboard.');
                        diagLog('If visible -> MQTT Broker OK, problem is in ESP32');
                        diagLog('If not visible -> Problem with subscribe/routing topic');
                        setTimeout(function() { updateDashboard(); diagLog('Dashboard refreshed.'); }, 2000);
                    } else {
                        diagLog('FAILED: ' + data.message);
                        diagLog('');
                        diagLog('This means: MQTT Broker is not running or cannot connect!');
                        diagLog('Jalankan Mosquitto terlebih dahulu.');
                    }
                })
                .catch(function(e) { diagLog('FETCH ERROR: ' + e); });
        }

        function runMqttReconnect() {
            diagClear('=== RECONNECT MQTT ===');
            diagLog('Trying to reconnect to MQTT broker...');
            fetch('/api/mqtt/reconnect', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    diagLog('Response: ' + data.message);
                    setTimeout(function() {
                        checkMqttStatus(true);
                        diagLog('Status setelah reconnect dicek.');
                    }, 2000);
                })
                .catch(function(e) { diagLog('FETCH ERROR: ' + e); });
        }

        function updateDeviceStatus() {
            checkMqttStatus(false);
            fetch('/api/device/status')
                .then(r => r.json())
                .then(data => {
                    const mapping = {'esp32_ac': 'ds-esp32-ac', 'esp32_lamp': 'ds-esp32-lamp', 'camera': 'ds-camera'};
                    const timeMapping = {'esp32_ac': 'ds-ac-time', 'esp32_lamp': 'ds-lamp-time', 'camera': 'ds-cam-time'};
                    for (const [devId, info] of Object.entries(data)) {
                        const el = document.getElementById(mapping[devId]);
                        const timeEl = document.getElementById(timeMapping[devId]);
                        if (el) {
                            const dot = el.querySelector('.device-dot');
                            if (dot) {
                                dot.className = 'device-dot ' + info.status;
                            }
                        }
                        if (timeEl) timeEl.textContent = info.last_seen;
                    }
                })
                .catch(() => {});
        }

        // ==================== ALERT SYSTEM ====================
        var alertQueue = [];
        socket.on('alert', function(alert) {
            showAlertBanner(alert);
        });

        function showAlertBanner(alert) {
            const banner = document.createElement('div');
            banner.className = 'alert-banner ' + alert.level;
            banner.innerHTML = '<span>' + alert.message + '</span>' +
                '<button class="alert-close" onclick="this.parentElement.remove()">&times;</button>';
            document.body.appendChild(banner);
            setTimeout(() => { if (banner.parentElement) banner.remove(); }, 8000);
        }

        // ==================== SENSOR HEALTH ====================
        var _sensorHealthState = {};

        socket.on('sensor_health', function(health) {
            _sensorHealthState = health;
            updateSensorHealthBar(health);
        });

        socket.on('sensor_fault', function(data) {
            // Update individual badge immediately on fault/recovery
            var id = data.device;
            if (_sensorHealthState[id]) _sensorHealthState[id].status = data.status;
            else _sensorHealthState[id] = data;
            updateSensorHealthBar(_sensorHealthState);
        });

        function updateSensorHealthBar(health) {
            var bar = document.getElementById('sensor-health-bar');
            if (!bar) return;
            var html = '';
            var statusIcon = {ok: '&#x25CF;', warn: '&#x25CF;', fault: '&#x25CF;'};
            var statusColor = {ok: '#60a5fa', warn: '#0ea5e9', fault: '#1e40af'};
            var statusText  = {ok: 'Online', warn: 'Stale', fault: 'Offline'};
            for (var dev in health) {
                var h = health[dev];
                var col  = statusColor[h.status]  || '#6b7280';
                var icon = statusIcon[h.status]   || '&#x25CF;';
                var txt  = statusText[h.status]   || h.status;
                html += '<span class="sensor-health-badge" style="color:' + col + ';" title="' + h.label + ': ' + h.age + '">'
                      + icon + ' ' + h.label + ' <small style="opacity:.7;">' + txt + '</small></span>';
            }
            bar.innerHTML = html;
        }

        // Fetch initial health on load
        fetch('/api/sensor/health').then(function(r){ return r.json(); }).then(function(h){
            _sensorHealthState = h;
            updateSensorHealthBar(h);
        }).catch(function(){});

        // ==================== SOCKET.IO EVENTS ====================
        // Ring buffer untuk chart MySQL realtime
        var _mysqlBuf = { labels: [], acV: [], outletV: [], lampV: [], freq: [], acP: [], outletP: [], lampP: [], acA: [], outletA: [], lampA: [] };
        var _MYSQL_MAX = 30;

        // -- Fungsi update DOM energi (dipakai socket DAN direct-poll) --
        var _lastEnergyUpdate = 0;
        function _applyEnergyData(ac, outlet, lamp) {
            outlet = outlet || {};
            lamp = lamp || {};
            _lastEnergyUpdate = Date.now();
            var s = function(id, val) { var el = document.getElementById(id); if (el) el.textContent = val; };
            var f = function(v, d) { return parseFloat(v || 0).toFixed(d !== undefined ? d : 1); };
            var fkwh = function(v) {
                var num = parseFloat(v || 0);
                if (!Number.isFinite(num)) return '0';
                return num.toFixed(4).replace(/\\.0+$/, '').replace(/(\\.\\d*?)0+$/, '$1');
            };

            // -- Overview page badge + mini cards --
            var badge = document.getElementById('mysql-energy-badge');
            if (badge) {
                badge.innerHTML = '<i class="fas fa-circle" style="font-size:6px;vertical-align:middle;margin-right:4px;"></i> Online';
                badge.style.cssText = 'padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(37,99,235,0.15);color:#2563eb;border:1px solid rgba(37,99,235,0.3);';
            }
            s('mysql-ac-voltage', f(ac.voltage));       s('mysql-ac-current', f(ac.current, 2));
            s('mysql-ac-power',   f(ac.power));         s('mysql-ac-kwh',     fkwh(ac.energy));
            s('mysql-ac-freq',    f(ac.frequency, 2));  s('mysql-ac-pf',      f(ac.pf, 2));
            s('mysql-ac-ts',      ac.updated_at || '--');
            s('mysql-lamp-voltage', f(lamp.voltage));     s('mysql-lamp-current', f(lamp.current, 2));
            s('mysql-lamp-power',   f(lamp.power));       s('mysql-lamp-kwh',     fkwh(lamp.energy));
            s('mysql-lamp-freq',    f(lamp.frequency, 2)); s('mysql-lamp-pf',     f(lamp.pf, 2));
            s('mysql-lamp-ts',      lamp.updated_at || '--');

            // -- Energy Usage page live values --
            var liveBadge = document.getElementById('mysql-live-badge');
            if (liveBadge) {
                liveBadge.innerHTML = '<i class="fas fa-circle" style="font-size:6px;vertical-align:middle;margin-right:4px;"></i> Live';
                liveBadge.style.cssText = 'padding:4px 14px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(37,99,235,0.15);color:#2563eb;border:1px solid rgba(37,99,235,0.3);';
            }
            s('eu-ac-voltage', f(ac.voltage));       s('eu-ac-current', f(ac.current, 2));
            s('eu-ac-power',   f(ac.power));         s('eu-ac-kwh',     fkwh(ac.energy));
            s('eu-ac-freq',    f(ac.frequency, 2));  s('eu-ac-pf',      f(ac.pf, 2));
            s('eu-ac-ts',      ac.updated_at || '--');
            s('outlet-power',      f(outlet.power));
            s('eu-outlet-voltage', f(outlet.voltage));       s('eu-outlet-current', f(outlet.current, 2));
            s('eu-outlet-kwh',     fkwh(outlet.energy));     s('eu-outlet-freq',    f(outlet.frequency, 2));
            s('eu-outlet-pf',      f(outlet.pf, 2));          s('eu-outlet-ts',      outlet.updated_at || '--');
            
            // Populate Outlet Analysis Live Params
            s('oa-live-voltage', f(outlet.voltage));       s('oa-live-current', f(outlet.current, 2));
            s('oa-live-apparent', f(outlet.apparent_power, 1)); s('oa-live-reactive', f(outlet.reactive_power, 1));
            s('oa-live-freq', f(outlet.frequency, 2));      s('oa-live-pf', f(outlet.pf, 2));
            
            s('eu-lamp-voltage', f(lamp.voltage));     s('eu-lamp-current', f(lamp.current, 2));
            s('eu-lamp-power',   f(lamp.power));       s('eu-lamp-kwh',     fkwh(lamp.energy));
            s('eu-lamp-freq',    f(lamp.frequency, 2)); s('eu-lamp-pf',     f(lamp.pf, 2));
            s('eu-lamp-ts',      lamp.updated_at || '--');

            // -- Reactive / Apparent / PF quality / Power bars --
            s('eu-ac-reactive',   f(ac.reactive_power,  1));
            s('eu-ac-apparent',   f(ac.apparent_power,  1));
            s('eu-outlet-reactive', f(outlet.reactive_power, 1));
            s('eu-outlet-apparent', f(outlet.apparent_power, 1));
            s('eu-lamp-reactive', f(lamp.reactive_power, 1));
            s('eu-lamp-apparent', f(lamp.apparent_power, 1));
            var _pfBadge = function(pf) {
                var v = parseFloat(pf||0);
                if (v >= 0.95) return {t:'Baik',  bg:'rgba(59,130,246,0.15)', c:'#3b82f6'};
                if (v >= 0.85) return {t:'Cukup', bg:'rgba(14,165,233,0.15)', c:'#0ea5e9'};
                if (v >  0)    return {t:'Buruk', bg:'rgba(30,64,175,0.15)',  c:'#1e40af'};
                return {t:'--', bg:'rgba(107,114,128,0.15)', c:'#6b7280'};
            };
            var _applyBadge = function(id, pf) {
                var q = _pfBadge(pf); var el = document.getElementById(id);
                if (el) { el.textContent = q.t; el.style.background = q.bg; el.style.color = q.c; }
            };
            _applyBadge('eu-ac-pf-quality',   ac.pf);
            _applyBadge('eu-outlet-pf-quality', outlet.pf);
            _applyBadge('oa-live-pf-quality', outlet.pf);
            _applyBadge('eu-lamp-pf-quality', lamp.pf);
            var acBar = document.getElementById('eu-ac-pbar');
            var outletBar = document.getElementById('eu-outlet-pbar');
            var lpBar = document.getElementById('eu-lamp-pbar');
            if (acBar)  acBar.style.width  = Math.min(100, (parseFloat(ac.power||0)   / 3.0 * 100)).toFixed(1) + '%';
            if (outletBar) outletBar.style.width = Math.min(100, (parseFloat(outlet.power||0) / 1.0 * 100)).toFixed(1) + '%';
            if (lpBar)  lpBar.style.width  = Math.min(100, (parseFloat(lamp.power||0) / 0.5 * 100)).toFixed(1) + '%';

            // -- Update power-grid summary cards --
            var acPow  = parseFloat(ac.power||0);
            var outletPow = parseFloat(outlet.power||0);
            var lampPow = parseFloat(lamp.power||0);
            var acKwh  = parseFloat(ac.energy||0);
            var outletKwh = parseFloat(outlet.energy||0);
            var lampKwh = parseFloat(lamp.energy||0);
            var totPow = acPow + outletPow + lampPow;
            var totKwh = acKwh + outletKwh + lampKwh;
            s('ac-power',   acPow.toFixed(1));
            s('outlet-power', outletPow.toFixed(1));
            s('lamp-power', lampPow.toFixed(1));
            s('total-power',   totPow.toFixed(1));
            s('total-current', (parseFloat(ac.current||0) + parseFloat(outlet.current||0) + parseFloat(lamp.current||0)).toFixed(2));
            s('ac-energy-kwh',   fkwh(acKwh));
            s('outlet-energy-kwh', fkwh(outletKwh));
            s('outlet-analytics-total-energy', fkwh(outletKwh));
            s('outlet-total-power', outletPow.toFixed(1));
            s('lamp-energy-kwh', fkwh(lampKwh));
            s('total-energy-kwh', fkwh(totKwh));
            s('ac-voltage-card',  f(ac.voltage));
            s('ac-current-card',  f(ac.current, 2));
            s('ac-pf-card',       f(ac.pf, 2));
            s('outlet-voltage-card', f(outlet.voltage));
            s('outlet-current-card', f(outlet.current, 2));
            s('outlet-pf-card',      f(outlet.pf, 2));
            s('lamp-voltage-card', f(lamp.voltage));
            s('lamp-current-card', f(lamp.current, 2));
            s('lamp-pf-card',      f(lamp.pf, 2));
            s('total-freq-card',   f(ac.frequency, 2));
            s('energy-last-update', ac.updated_at ? ac.updated_at.substring(11,19) : '--');
            var costEst = Math.round(totKwh * 1500);
            s('daily-cost', costEst > 0 ? costEst.toLocaleString() : '0');

            // -- Ring buffer & chart --
            var now2 = new Date();
            var ts2  = now2.getHours() + ':' + String(now2.getMinutes()).padStart(2,'0') + ':' + String(now2.getSeconds()).padStart(2,'0');
            _mysqlBuf.labels.push(ts2);
            _mysqlBuf.acV.push(parseFloat(ac.voltage||0));
            _mysqlBuf.outletV.push(parseFloat(outlet.voltage||0));
            _mysqlBuf.lampV.push(parseFloat(lamp.voltage||0));
            _mysqlBuf.freq.push(parseFloat(ac.frequency||0));
            _mysqlBuf.acP.push(parseFloat(ac.power||0));
            _mysqlBuf.outletP.push(parseFloat(outlet.power||0));
            _mysqlBuf.lampP.push(parseFloat(lamp.power||0));
            _mysqlBuf.acA.push(parseFloat(ac.current||0));
            _mysqlBuf.outletA.push(parseFloat(outlet.current||0));
            _mysqlBuf.lampA.push(parseFloat(lamp.current||0));
            if (_mysqlBuf.labels.length > _MYSQL_MAX) {
                ['labels','acV','outletV','lampV','freq','acP','outletP','lampP','acA','outletA','lampA'].forEach(function(k){ _mysqlBuf[k].shift(); });
            }
            if (charts.mysqlVoltFreq) {
                charts.mysqlVoltFreq.data.labels = _mysqlBuf.labels.slice();
                charts.mysqlVoltFreq.data.datasets[0].data = _mysqlBuf.freq.slice();
                charts.mysqlVoltFreq.update('none');
            }
            if (charts.mysqlCurrent) {
                charts.mysqlCurrent.data.labels = _mysqlBuf.labels.slice();
                charts.mysqlCurrent.data.datasets[0].data = _mysqlBuf.acA.slice();
                charts.mysqlCurrent.data.datasets[1].data = _mysqlBuf.outletA.slice();
                charts.mysqlCurrent.data.datasets[2].data = _mysqlBuf.lampA.slice();
                charts.mysqlCurrent.update('none');
            }
        }

        // -- Socket handler --
        socket.on('mysql_energy_update', function(data) {
            var ac = data.ac || {};
            var outlet = data.outlet || {};
            var lamp = data.lamp || {};
            
            // PROTEKSI GANDA: Pastikan menjadi kWh, jika masih di atas 1000 (Wh), paksa bagi 1000
            if (ac.energy > 1000) ac.energy = ac.energy / 1000.0;
            if (outlet.energy > 1000) outlet.energy = outlet.energy / 1000.0;
            if (lamp.energy > 1000) lamp.energy = lamp.energy / 1000.0;

            _applyEnergyData(ac, outlet, lamp);
        });

        // -- Direct PHP polling fallback --
        // If socket sends no data for 10 seconds, browser fetches directly from PHP
        var _PHP_ENERGY_URL = 'https://iotlab-uns.com/api_energy.php?key=iotlab_smartroom_2024';
        var _phpPollErr = 0;
        function _phpDirectPoll() {
            if (Date.now() - _lastEnergyUpdate < 10000) return; // socket still active, skip
            fetch(_PHP_ENERGY_URL)
                .then(function(r) {
                    if (!r.ok) { console.error('[PHP-poll] HTTP ' + r.status); return null; }
                    return r.json();
                })
                .then(function(data) {
                    if (!data) { _phpPollErr++; return; }
                    if (data.error) {
                        _phpPollErr++;
                        console.error('[PHP-poll] Server error:', data.error, data.file + ':' + data.line);
                        return;
                    }
                    _phpPollErr = 0;
                    var ac     = data.ac     || {};
                    var outlet = data.outlet || {};
                    var lamp   = data.lamp   || {};
                    // Calculate pf because PHP does not send pf directly
                    var acAp = parseFloat(ac.apparent_power||0);
                    var outletAp = parseFloat(outlet.apparent_power||0);
                    var lpAp = parseFloat(lamp.apparent_power||0);
                    ac.pf   = acAp > 0 ? parseFloat((parseFloat(ac.active_power||0) / acAp).toFixed(3)) : 0;
                    outlet.pf = outletAp > 0 ? parseFloat((parseFloat(outlet.active_power||0) / outletAp).toFixed(3)) : 0;
                    lamp.pf = lpAp > 0 ? parseFloat((parseFloat(lamp.active_power||0) / lpAp).toFixed(3)) : 0;
                    // Adjust field names: active_power   power, frequency   frequency
                    ac.power       = ac.active_power;
                    // total_energy from DB is actually in Wh, divide by 1000 for kWh
                    ac.energy      = parseFloat(ac.total_energy || 0) / 1000.0;
                    ac.frequency   = ac.frequency;
                    ac.voltage     = ac.voltage;
                    ac.current     = ac.arus;
                    outlet.power     = outlet.active_power;
                    outlet.energy    = parseFloat(outlet.total_energy || 0) / 1000.0;
                    outlet.frequency = outlet.frequency;
                    outlet.voltage   = outlet.voltage;
                    outlet.current   = outlet.arus;
                    lamp.power     = lamp.active_power;
                    lamp.energy    = parseFloat(lamp.total_energy || 0) / 1000.0;
                    lamp.frequency = lamp.frequency;
                    lamp.voltage   = lamp.voltage;
                    lamp.current   = lamp.arus;
                    _applyEnergyData(ac, outlet, lamp);
                })
                .catch(function(e) { _phpPollErr++; console.warn('[PHP-poll] error:', e); });
        }
        setInterval(_phpDirectPoll, 6000);
        // Run immediately 3 seconds after load (no wait for socket)
        setTimeout(_phpDirectPoll, 3000);

        // ==================== SESSION STORAGE + SERVER STATE PERSISTENCE ====================
        function _ssSave(key, val) { try { sessionStorage.setItem(key, JSON.stringify(val)); } catch(e) {} }
        function _ssLoad(key, def) { try { var v = sessionStorage.getItem(key); return v !== null ? JSON.parse(v) : def; } catch(e) { return def; } }
        // Push recording state to server so other devices see badges
        function _recStatePush(obj) {
            fetch('/api/rec/state', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(obj) }).catch(function(){});
        }
        // Poll server state every 5 s to update badges on all devices
        function _recStatePoll() {
            fetch('/api/rec/state').then(function(r){ return r.json(); }).then(function(st) {
                // Only update badge if this device is not the one recording (to avoid overwriting active timer)
                if (!_recActive) {
                    var badge = document.getElementById('rec-status-badge');
                    if (badge) {
                        if (st.energy) {
                            badge.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;animation:blink 1s infinite;"></i> Recording (other device)';
                            badge.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(14,165,233,0.15);color:#0ea5e9;border:1px solid rgba(14,165,233,0.4);';
                        } else { _recUpdateBadge(); }
                    }
                }
                if (!_tempActive) {
                    var tb = document.getElementById('temp-rec-badge');
                    if (tb && st.temp) {
                        tb.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;animation:blink 1s infinite;"></i> Recording (other device)';
                        tb.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(14,165,233,0.15);color:#0ea5e9;border:1px solid rgba(14,165,233,0.4);';
                    } else if (tb && !st.temp) { _tempUpdateBadge(); }
                }
                if (!_luxActive) {
                    var lb = document.getElementById('lux-rec-badge');
                    if (lb && st.lux) {
                        lb.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;animation:blink 1s infinite;"></i> Recording (other device)';
                        lb.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(14,165,233,0.15);color:#0ea5e9;border:1px solid rgba(14,165,233,0.4);';
                    } else if (lb && !st.lux) { _luxUpdateBadge(); }
                }
            }).catch(function(){});
        }
        setInterval(_recStatePoll, 5000);

        // ==================== ENERGY RECORDING & CSV EXPORT ====================
        var _recRows    = _ssLoad('_recRows', []);
        var _recActive  = false;  // never auto-resume active state; user must press start
        var _recStartTs = null;
        var _recTimer   = null;
        var _recFilter  = 'all';  // 'all' | 'AC' | 'Lamp'

        function _recFmt(v, d) { return (parseFloat(v)||0).toFixed(d !== undefined ? d : 1); }

        // -- Server-side data sync --
        function _recServerPush() {
            fetch('/api/rec/data?type=energy', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rows: _recRows})
            }).then(function(r){ return r.json(); }).then(function(d){
                var b = document.getElementById('rec-sync-badge');
                if (b) { b.innerHTML = '<i class="fas fa-cloud-upload-alt" style="margin-right:4px;font-size:9px;"></i>' + d.count + ' baris'; b.style.color='#3b82f6'; b.style.borderColor='rgba(59,130,246,0.3)'; }
            }).catch(function(){});
        }
        function _recServerFetch() {
            fetch('/api/rec/data?type=energy').then(function(r){ return r.json(); }).then(function(d){
                if (!d.rows) return;
                // Always sync from server when not recording - server is source of truth
                // This fixes stale sessionStorage showing wrong data on other devices
                if (d.rows.length !== _recRows.length ||
                    JSON.stringify(d.rows[d.rows.length-1]) !== JSON.stringify(_recRows[_recRows.length-1])) {
                    _recRows = d.rows; _ssSave('_recRows', _recRows); _recUpdatePreview();
                }
                var b = document.getElementById('rec-sync-badge');
                if (b) { b.innerHTML = '<i class="fas fa-cloud" style="margin-right:4px;font-size:9px;"></i>' + d.count + ' baris'; b.style.color='#2563eb'; }
            }).catch(function(){});
        }
        function _recServerDelete() { fetch('/api/rec/data?type=energy', {method:'DELETE'}).catch(function(){}); }
        // Poll server for rows every 5 s on passive devices (matches badge poll speed)
        setInterval(function(){ if (!_recActive) _recServerFetch(); }, 5000);

        function recSetFilter(f) {
            _recFilter = f;
            var map = {all:'rec-tab-all', AC:'rec-tab-ac', Lamp:'rec-tab-lamp'};
            Object.keys(map).forEach(function(k) {
                var el = document.getElementById(map[k]); if (!el) return;
                if (k === f) { el.style.background='#fff'; el.style.fontWeight='700'; el.style.color=k==='AC'?'#1e40af':k==='Lamp'?'#eab308':'#2563eb'; el.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)'; }
                else { el.style.background='transparent'; el.style.fontWeight='600'; el.style.color='var(--text-secondary)'; el.style.boxShadow='none'; }
            });
            _recUpdatePreview();
        }

        function _recUpdateBadge() {
            var badge = document.getElementById('rec-status-badge');
            if (!badge) return;
            if (_recActive) {
                badge.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;animation:blink 1s infinite;"></i> Recording';
                badge.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(30,64,175,0.15);color:#1e40af;border:1px solid rgba(30,64,175,0.4);';
            } else {
                badge.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;"></i> Idle';
                badge.style.cssText = 'padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;background:rgba(107,114,128,0.15);color:#6b7280;border:1px solid rgba(107,114,128,0.3);';
            }
        }

        function _recUpdatePreview() {
            var tbody = document.getElementById('rec-preview-body');
            var countEl = document.getElementById('rec-count');
            if (countEl) countEl.textContent = _recRows.length;
            if (!tbody) return;
            var filtered = _recFilter === 'all' ? _recRows : _recRows.filter(function(r){ return r.device === _recFilter; });
            if (filtered.length === 0) {
                var msg = _recRows.length > 0 ? 'No data for filter <strong>' + _recFilter + '</strong>.' : 'No data yet. Click <strong>Start Record</strong> to begin.';
                tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;padding:16px;color:var(--text-secondary);font-size:12px;">' + msg + '</td></tr>';
                var rngHide = document.getElementById('rec-date-range'); if (rngHide) rngHide.style.display = 'none';
                return;
            }
            // Tampilkan rentang tanggal
            var rng = document.getElementById('rec-date-range'); var rngTxt = document.getElementById('rec-range-text');
            if (rng && rngTxt) { rng.style.display = ''; rngTxt.textContent = filtered[0].ts + ' \u2014 ' + filtered[filtered.length-1].ts; }
            // Tampilkan 100 baris terbaru (scroll)
            var shown = filtered.slice(-100).reverse();
            var shownEl = document.getElementById('rec-shown-count'); if (shownEl) shownEl.textContent = Math.min(filtered.length, 100);
            tbody.innerHTML = shown.map(function(r) {
                return '<tr style="border-bottom:1px solid var(--border);">' +
                    '<td style="padding:6px 10px;white-space:nowrap;color:var(--text-secondary);">' + r.ts + '</td>' +
                    '<td style="padding:6px 10px;text-align:center;font-weight:600;color:' + (r.device==='AC'?'#1e40af':'#eab308') + ';">' + r.device + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:#2563eb;font-weight:600;">' + r.voltage + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:#0ea5e9;font-weight:600;">' + r.arus + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:#1e40af;font-weight:600;">' + r.daya + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:#3b82f6;font-weight:600;">' + r.energi + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text-secondary);">' + r.reaktif + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text-secondary);">' + r.semu + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text-secondary);">' + r.freq + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text-secondary);">' + r.pf + '</td>' +
                '</tr>';
            }).join('');
        }

        function _recTick() {
            if (!_recStartTs) return;
            var sec = Math.floor((Date.now() - _recStartTs) / 1000);
            var m = Math.floor(sec / 60), s = sec % 60;
            var el = document.getElementById('rec-duration');
            if (el) el.textContent = m + ':' + String(s).padStart(2,'0');
        }

        function _recAddRow(ac, outlet, lamp) {
            if (!_recActive) return;
            var now = new Date();
            var ts  = now.toISOString().replace('T',' ').substring(0,19);
            // AC row
            _recRows.push({
                ts: ts, device: 'AC',
                voltage: _recFmt(ac.voltage || ac.voltage),
                arus:     _recFmt(ac.current || ac.arus, 3),
                daya:     _recFmt(ac.power   || ac.active_power, 2),
                energi:   _recFmt(ac.energy  || ac.total_energy, 4),
                reaktif:  _recFmt(ac.reactive_power, 2),
                semu:     _recFmt(ac.apparent_power, 2),
                freq:     _recFmt(ac.frequency || ac.frequency, 2),
                pf:       _recFmt(ac.pf, 3),
            });
            // Outlet row (MySQL id_kwh=2)
            _recRows.push({
                ts: ts, device: 'Outlet',
                voltage: _recFmt(outlet.voltage || outlet.voltage),
                arus:     _recFmt(outlet.current || outlet.arus, 3),
                daya:     _recFmt(outlet.power   || outlet.active_power, 2),
                energi:   _recFmt(outlet.energy  || outlet.total_energy, 4),
                reaktif:  _recFmt(outlet.reactive_power, 2),
                semu:     _recFmt(outlet.apparent_power, 2),
                freq:     _recFmt(outlet.frequency || outlet.frequency, 2),
                pf:       _recFmt(outlet.pf, 3),
            });
            // Lamp row
            _recRows.push({
                ts: ts, device: 'Lamp',
                voltage: _recFmt(lamp.voltage || lamp.voltage),
                arus:     _recFmt(lamp.current || lamp.arus, 3),
                daya:     _recFmt(lamp.power   || lamp.active_power, 2),
                energi:   _recFmt(lamp.energy  || lamp.total_energy, 4),
                reaktif:  _recFmt(lamp.reactive_power, 2),
                semu:     _recFmt(lamp.apparent_power, 2),
                freq:     _recFmt(lamp.frequency || lamp.frequency, 2),
                pf:       _recFmt(lamp.pf, 3),
            });
            _recUpdatePreview();
            _ssSave('_recRows', _recRows);
            _recServerPush();
        }

        function energyRecStart() {
            _recActive  = true;
            _recStartTs = Date.now();
            var startEl = document.getElementById('rec-start-time');
            if (startEl) startEl.textContent = new Date().toLocaleTimeString('id-ID');
            var btnS = document.getElementById('btn-rec-start');
            var btnE = document.getElementById('btn-rec-stop');
            if (btnS) { btnS.disabled = true; btnS.style.opacity = '0.5'; }
            if (btnE) { btnE.disabled = false; btnE.style.opacity = '1'; }
            _recTimer = setInterval(_recTick, 1000);
            _recUpdateBadge();
            _recStatePush({energy: true});
            showToast('Recording started', 'success');
        }

        function energyRecStop() {
            _recActive = false;
            clearInterval(_recTimer);
            var btnS = document.getElementById('btn-rec-start');
            var btnE = document.getElementById('btn-rec-stop');
            if (btnS) { btnS.disabled = false; btnS.style.opacity = '1'; }
            if (btnE) { btnE.disabled = true; btnE.style.opacity = '0.5'; }
            _recUpdateBadge();
            _recStatePush({energy: false});
            _recServerPush();
            showToast('Recording stopped - ' + _recRows.length + ' rows saved', 'info');
        }

        function energyRecClear() {
            _recRows = [];
            _recActive = false;
            clearInterval(_recTimer);
            sessionStorage.removeItem('_recRows');
            _recServerDelete();
            var countEl = document.getElementById('rec-count');
            var durEl   = document.getElementById('rec-duration');
            var startEl = document.getElementById('rec-start-time');
            if (countEl)  countEl.textContent  = '0';
            if (durEl)    durEl.textContent    = '0:00';
            if (startEl)  startEl.textContent  = '--';
            var btnS2 = document.getElementById('btn-rec-start');
            var btnE2 = document.getElementById('btn-rec-stop');
            if (btnS2) { btnS2.disabled = false; btnS2.style.opacity = '1'; }
            if (btnE2) { btnE2.disabled = true; btnE2.style.opacity = '0.5'; }
            _recUpdateBadge();
            _recUpdatePreview();
            var sb = document.getElementById('rec-sync-badge'); if (sb) { sb.innerHTML = '<i class="fas fa-cloud" style="margin-right:4px;font-size:9px;"></i>Sync'; sb.style.color='#2563eb'; sb.style.borderColor='rgba(37,99,235,0.2)'; }
            showToast('Recording data cleared', 'info');
        }

        function energyExportCSV(device) {
            if (_recRows.length === 0) { showToast('No recorded data yet', 'error'); return; }
            var rows = (device && device !== 'all') ? _recRows.filter(function(r){ return r.device === device; }) : _recRows;
            if (rows.length === 0) { showToast('No data for ' + device, 'error'); return; }
            var header = 'Time,Perangkat,Voltage (V),Current (A),Power Active (W),Energy (kWh),Power Reaktif (VAR),Power Semu (VA),Frequency (Hz),Power Factor\\n';
            var body = rows.map(function(r) {
                return '"' + r.ts + '",' + r.device + ',' + r.voltage + ',' + r.arus + ',' +
                       r.daya + ',' + r.energi + ',' + r.reaktif + ',' + r.semu + ',' + r.freq + ',' + r.pf;
            }).join('\\n');
            var now = new Date();
            var suffix = (device && device !== 'all') ? '_' + device.toLowerCase() : '';
            var fname = 'energy' + suffix + '_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
                        String(now.getDate()).padStart(2,'0') + '_' +
                        String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.csv';
            downloadCSV(fname, header + body);
            showToast(rows.length + ' rows exported: ' + fname, 'success');
        }

        // Hook to _applyEnergyData - record every 6 minutes (360 seconds)
        var _lastRecTime = 0;
        var _REC_INTERVAL_MS = 300000; // 5 minutes
        var _origApply = _applyEnergyData;
        _applyEnergyData = function(ac, outlet, lamp) {
            _origApply(ac, outlet, lamp);
            if (Date.now() - _lastRecTime >= _REC_INTERVAL_MS) {
                _lastRecTime = Date.now();
                _recAddRow(ac, outlet, lamp);
            }
        };

        // ==================== RECORD TEMP & HUMIDITY ====================
        function _tempServerPush() {
            fetch('/api/rec/data?type=temp', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rows: _tempRows})
            }).catch(function(){});
        }
        function _tempServerFetch() {
            fetch('/api/rec/data?type=temp').then(function(r){ return r.json(); }).then(function(d){
                if (!d.rows) return;
                if (d.rows.length !== _tempRows.length ||
                    JSON.stringify(d.rows[d.rows.length-1]) !== JSON.stringify(_tempRows[_tempRows.length-1])) {
                    _tempRows = d.rows; _ssSave('_tempRows', _tempRows); _tempUpdatePreview();
                    var c = document.getElementById('temp-rec-count'); if (c) c.textContent = _tempRows.length;
                }
            }).catch(function(){});
        }
        function _tempServerDelete() { fetch('/api/rec/data?type=temp', {method:'DELETE'}).catch(function(){}); }
        setInterval(function(){ if (!_tempActive) _tempServerFetch(); }, 5000);

        var _tempRows = _ssLoad('_tempRows', []), _tempActive = false, _tempStartTs = 0;
        var _tempTimer = null, _lastTempRecTime = 0;
        var _TEMP_REC_MS = 300000; // 5 minutes

        function _tempFmt(v, d) { var n = parseFloat(v); return Number.isFinite(n) ? n.toFixed(d != null ? d : 1) : '--'; }

        function _tempAddRow() {
            var ts = new Date().toLocaleString('id-ID');
            var temp = document.getElementById('dash-ac-room-temp') ? document.getElementById('dash-ac-room-temp').textContent : '--';
            var hum  = document.getElementById('dash-ac-room-hum')  ? document.getElementById('dash-ac-room-hum').textContent  : '--';
            var t1   = document.getElementById('dash-ac-temp1')      ? document.getElementById('dash-ac-temp1').textContent      : '--';
            var t2   = document.getElementById('dash-ac-temp2')      ? document.getElementById('dash-ac-temp2').textContent      : '--';
            var t3   = document.getElementById('dash-ac-temp3')      ? document.getElementById('dash-ac-temp3').textContent      : '--';
            var setT = document.getElementById('dash-ac-temp')       ? document.getElementById('dash-ac-temp').textContent       : '--';
            var fan  = document.getElementById('dash-ac-fan-label')  ? document.getElementById('dash-ac-fan-label').textContent  : '--';
            var mode = document.getElementById('dash-ac-mode')       ? document.getElementById('dash-ac-mode').textContent       : '--';
            var ctrl = document.getElementById('dash-ac-ctrl-mode')  ? document.getElementById('dash-ac-ctrl-mode').textContent  : '--';
            _tempRows.push({ ts: ts, temp: temp, hum: hum, t1: t1, t2: t2, t3: t3, setT: setT, fan: fan, mode: mode, ctrl: ctrl });
            _tempUpdatePreview();
            var c = document.getElementById('temp-rec-count');
            if (c) c.textContent = _tempRows.length;
            _ssSave('_tempRows', _tempRows);
            _tempServerPush();
        }

        function _tempTick() {
            if (!_tempActive) return;
            var sec = Math.floor((Date.now() - _tempStartTs) / 1000);
            var m = Math.floor(sec / 60), s = sec % 60;
            var d = document.getElementById('temp-rec-duration');
            if (d) d.textContent = m + ':' + String(s).padStart(2, '0');
            if (Date.now() - _lastTempRecTime >= _TEMP_REC_MS) {
                _lastTempRecTime = Date.now();
                _tempAddRow();
            }
        }

        function _tempUpdateBadge() {
            var b = document.getElementById('temp-rec-badge');
            if (!b) return;
            if (_tempActive) {
                b.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;color:#1e40af;animation:pulse 1s infinite;"></i> Recording...';
                b.style.background = 'rgba(30,64,175,0.12)'; b.style.color = '#1e40af'; b.style.borderColor = 'rgba(30,64,175,0.3)';
            } else {
                b.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;"></i> Idle';
                b.style.background = 'rgba(107,114,128,0.15)'; b.style.color = '#6b7280'; b.style.borderColor = 'rgba(107,114,128,0.3)';
            }
        }

        function _tempUpdatePreview() {
            var tbody = document.getElementById('temp-rec-preview');
            if (!tbody) return;
            if (_tempRows.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:16px;color:var(--text-secondary);">No data yet. Click <strong>Start Record</strong> to begin.</td></tr>';
                return;
            }
            // Tampilkan 100 baris terbaru (scroll)
            var shownT = document.getElementById('temp-shown-count'); if (shownT) shownT.textContent = Math.min(_tempRows.length, 100);
            var last100 = _tempRows.slice(-100).reverse();
            tbody.innerHTML = last100.map(function(r, i) {
                var bg = i % 2 === 0 ? 'background:rgba(59,130,246,0.03);' : '';
                return '<tr style="' + bg + '"><td style="padding:6px 10px;color:var(--text-secondary);white-space:nowrap;">' + r.ts + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#3b82f6;">' + r.temp + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#0ea5e9;">' + r.hum + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text);">' + r.t1 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text);">' + r.t2 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;color:var(--text);">' + r.t3 + '</td></tr>';
            }).join('');
        }

        function tempRecStart() {
            _tempActive = true; _tempStartTs = Date.now(); _lastTempRecTime = 0;
            var s = document.getElementById('temp-rec-start'); if (s) s.textContent = new Date().toLocaleTimeString('id-ID');
            var tS = document.getElementById('btn-temp-start'), tE = document.getElementById('btn-temp-stop');
            if (tS) { tS.disabled = true; tS.style.opacity = '0.5'; }
            if (tE) { tE.disabled = false; tE.style.opacity = '1'; }
            _tempTimer = setInterval(_tempTick, 1000);
            _tempUpdateBadge();
            _recStatePush({temp: true});
            showToast('Temperature & humidity recording started', 'success');
        }

        function tempRecStop() {
            _tempActive = false; clearInterval(_tempTimer);
            var tS = document.getElementById('btn-temp-start'), tE = document.getElementById('btn-temp-stop');
            if (tS) { tS.disabled = false; tS.style.opacity = '1'; }
            if (tE) { tE.disabled = true; tE.style.opacity = '0.5'; }
            _tempUpdateBadge();
            _recStatePush({temp: false});
            _tempServerPush();
            showToast('Recording stopped - ' + _tempRows.length + ' rows saved', 'info');
        }

        function tempRecClear() {
            _tempRows = []; _tempActive = false; clearInterval(_tempTimer);
            sessionStorage.removeItem('_tempRows');
            _tempServerDelete();
            ['temp-rec-count','temp-rec-duration','temp-rec-start'].forEach(function(id) { var e = document.getElementById(id); if (e) e.textContent = id.includes('count') ? '0' : id.includes('duration') ? '0:00' : '--'; });
            var tS2 = document.getElementById('btn-temp-start'), tE2 = document.getElementById('btn-temp-stop');
            if (tS2) { tS2.disabled = false; tS2.style.opacity = '1'; }
            if (tE2) { tE2.disabled = true; tE2.style.opacity = '0.5'; }
            _tempUpdateBadge(); _tempUpdatePreview();
            showToast('Temperature data cleared', 'info');
        }

        function tempExportCSV() {
            if (_tempRows.length === 0) { showToast('No recorded data yet', 'error'); return; }
            var header = 'Time,Temp Rata2 (C),Humidity (%),Sensor T1 (C),Sensor T2 (C),Sensor T3 (C),Set Temp (C),Fan Speed,Mode AC,Control\\n';
            var body = _tempRows.map(function(r) {
                return '"' + r.ts + '",' + r.temp + ',' + r.hum + ',' + r.t1 + ',' + r.t2 + ',' + r.t3 + ',' + r.setT + ',"' + r.fan + '",' + r.mode + ',' + r.ctrl;
            }).join('\\n');
            var now = new Date();
            var fname = 'temp_humidity_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
                        String(now.getDate()).padStart(2,'0') + '_' +
                        String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.csv';
            downloadCSV(fname, header + body);
            showToast(_tempRows.length + ' rows exported: ' + fname, 'success');
        }

        // ==================== REKAM LUX & BRIGHTNESS ====================
        function _luxServerPush() {
            fetch('/api/rec/data?type=lux', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rows: _luxRows})
            }).catch(function(){});
        }
        function _luxServerFetch() {
            fetch('/api/rec/data?type=lux').then(function(r){ return r.json(); }).then(function(d){
                if (!d.rows) return;
                if (d.rows.length !== _luxRows.length ||
                    JSON.stringify(d.rows[d.rows.length-1]) !== JSON.stringify(_luxRows[_luxRows.length-1])) {
                    _luxRows = d.rows; _ssSave('_luxRows', _luxRows); _luxUpdatePreview();
                    var c = document.getElementById('lux-rec-count'); if (c) c.textContent = _luxRows.length;
                }
            }).catch(function(){});
        }
        function _luxServerDelete() { fetch('/api/rec/data?type=lux', {method:'DELETE'}).catch(function(){}); }
        setInterval(function(){ if (!_luxActive) _luxServerFetch(); }, 5000);

        var _luxRows = _ssLoad('_luxRows', []), _luxActive = false, _luxStartTs = 0;
        var _luxTimer = null, _lastLuxRecTime = 0;
        var _LUX_REC_MS = 300000; // 5 minutes

        function _luxAddRow() {
            var ts = new Date().toLocaleString('id-ID');
            var get = function(id) { var e = document.getElementById(id); return e ? e.textContent : '--'; };
            _luxRows.push({
                ts: ts,
                lux1: get('dash-lux1'), lux2: get('dash-lux2'), lux3: get('dash-lux3'),
                avg: get('dash-lux-avg'), b1: get('dash-bright1'), b2: get('dash-bright2')
            });
            _luxUpdatePreview();
            var c = document.getElementById('lux-rec-count');
            if (c) c.textContent = _luxRows.length;
            _ssSave('_luxRows', _luxRows);
            _luxServerPush();
        }

        function _luxTick() {
            if (!_luxActive) return;
            var sec = Math.floor((Date.now() - _luxStartTs) / 1000);
            var m = Math.floor(sec / 60), s = sec % 60;
            var d = document.getElementById('lux-rec-duration');
            if (d) d.textContent = m + ':' + String(s).padStart(2, '0');
            if (Date.now() - _lastLuxRecTime >= _LUX_REC_MS) {
                _lastLuxRecTime = Date.now();
                _luxAddRow();
            }
        }

        function _luxUpdateBadge() {
            var b = document.getElementById('lux-rec-badge');
            if (!b) return;
            if (_luxActive) {
                b.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;color:#1e40af;animation:pulse 1s infinite;"></i> Recording...';
                b.style.background = 'rgba(30,64,175,0.12)'; b.style.color = '#1e40af'; b.style.borderColor = 'rgba(30,64,175,0.3)';
            } else {
                b.innerHTML = '<i class="fas fa-circle" style="font-size:7px;vertical-align:middle;margin-right:5px;"></i> Idle';
                b.style.background = 'rgba(107,114,128,0.15)'; b.style.color = '#6b7280'; b.style.borderColor = 'rgba(107,114,128,0.3)';
            }
        }

        function _luxUpdatePreview() {
            var tbody = document.getElementById('lux-rec-preview');
            if (!tbody) return;
            if (_luxRows.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:16px;color:var(--text-secondary);">No data yet. Click <strong>Start Record</strong> to begin.</td></tr>';
                return;
            }
            // Tampilkan 100 baris terbaru (scroll)
            var shownL = document.getElementById('lux-shown-count'); if (shownL) shownL.textContent = Math.min(_luxRows.length, 100);
            var last100 = _luxRows.slice(-100).reverse();
            tbody.innerHTML = last100.map(function(r, i) {
                var bg = i % 2 === 0 ? 'background:rgba(234,179,8,0.03);' : '';
                return '<tr style="' + bg + '"><td style="padding:6px 10px;color:var(--text-secondary);white-space:nowrap;">' + r.ts + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#eab308;">' + r.lux1 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#eab308;">' + r.lux2 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#eab308;">' + r.lux3 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#0ea5e9;">' + r.avg + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#fbbf24;">' + r.b1 + '</td>' +
                    '<td style="padding:6px 10px;text-align:right;font-weight:700;color:#fbbf24;">' + r.b2 + '</td></tr>';
            }).join('');
        }

        function luxRecStart() {
            _luxActive = true; _luxStartTs = Date.now(); _lastLuxRecTime = 0;
            var s = document.getElementById('lux-rec-start'); if (s) s.textContent = new Date().toLocaleTimeString('id-ID');
            var lS = document.getElementById('btn-lux-start'), lE = document.getElementById('btn-lux-stop');
            if (lS) { lS.disabled = true; lS.style.opacity = '0.5'; }
            if (lE) { lE.disabled = false; lE.style.opacity = '1'; }
            _luxTimer = setInterval(_luxTick, 1000);
            _luxUpdateBadge();
            _recStatePush({lux: true});
            showToast('Lux & brightness recording started', 'success');
        }

        function luxRecStop() {
            _luxActive = false; clearInterval(_luxTimer);
            var lS = document.getElementById('btn-lux-start'), lE = document.getElementById('btn-lux-stop');
            if (lS) { lS.disabled = false; lS.style.opacity = '1'; }
            if (lE) { lE.disabled = true; lE.style.opacity = '0.5'; }
            _luxUpdateBadge();
            _recStatePush({lux: false});
            _luxServerPush();
            showToast('Recording stopped - ' + _luxRows.length + ' rows saved', 'info');
        }

        function luxRecClear() {
            _luxRows = []; _luxActive = false; clearInterval(_luxTimer);
            sessionStorage.removeItem('_luxRows');
            _luxServerDelete();
            ['lux-rec-count','lux-rec-duration','lux-rec-start'].forEach(function(id) { var e = document.getElementById(id); if (e) e.textContent = id.includes('count') ? '0' : id.includes('duration') ? '0:00' : '--'; });
            var lS2 = document.getElementById('btn-lux-start'), lE2 = document.getElementById('btn-lux-stop');
            if (lS2) { lS2.disabled = false; lS2.style.opacity = '1'; }
            if (lE2) { lE2.disabled = true; lE2.style.opacity = '0.5'; }
            _luxUpdateBadge(); _luxUpdatePreview();
            showToast('Lux data cleared', 'info');
        }

        function luxExportCSV() {
            if (_luxRows.length === 0) { showToast('No recorded data yet', 'error'); return; }
            var header = 'Time,Lux 1 (lx),Lux 2 (lx),Lux 3 (lx),Avg Lux (lx),Brightness 1 (%),Brightness 2 (%)\\n';
            var body = _luxRows.map(function(r) {
                return '"' + r.ts + '",' + r.lux1 + ',' + r.lux2 + ',' + r.lux3 + ',' + r.avg + ',' + r.b1 + ',' + r.b2;
            }).join('\\n');
            var now = new Date();
            var fname = 'lux_brightness_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
                        String(now.getDate()).padStart(2,'0') + '_' +
                        String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.csv';
            downloadCSV(fname, header + body);
            showToast(_luxRows.length + ' rows exported: ' + fname, 'success');
        }

        // ==================== OCCUPANCY DAILY RECORDING ====================
        var _occRows    = _ssLoad('_occRows', []);   // {ts, hour, count, conf}
        var _occLastHour = -1;                        // last recorded hour (-1 = none yet)
        var _occRangeFilter = 'today';                // range: 'today'|'7d'|'30d'|'all'
        var _OCC_REC_LABEL = Array.from({length:24}, function(_,i){ return String(i).padStart(2,'0') + ':00'; });
        var _occChart   = null;

        // Restore last recorded hour from session
        if (_occRows.length > 0) {
            _occLastHour = _occRows[_occRows.length - 1].hour;
        }

        function _occInitChart() {
            var ctx = document.getElementById('occChart');
            if (!ctx || _occChart) return;
            _occChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: _OCC_REC_LABEL,
                    datasets: [{
                        label: 'Person Count',
                        data: new Array(24).fill(null),
                        backgroundColor: 'rgba(37,99,235,0.5)',
                        borderColor: '#2563eb',
                        borderWidth: 1,
                        borderRadius: 5,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: {
                        callbacks: { label: function(c) { return c.raw !== null ? c.raw + ' persons' : 'Not recorded'; } }
                    }},
                    scales: {
                        x: { ticks: { font: {size:10}, color: 'rgba(100,116,139,0.9)', maxRotation: 0 } },
                        y: { beginAtZero: true, ticks: { font: {size:10}, color: 'rgba(100,116,139,0.9)', stepSize: 1 },
                             title: { display: true, text: 'Person Count', font: {size:10}, color: '#2563eb' } }
                    }
                }
            });
            _occRefreshChart();
        }

        function _occGetFilteredRows() {
            var now = new Date();
            if (_occRangeFilter === 'today') {
                var today = now.toISOString().substring(0,10);
                return _occRows.filter(function(r){ return r.ts.substring(0,10) === today; });
            } else if (_occRangeFilter === '7d') {
                var cutoff = new Date(now - 7*24*60*60*1000).toISOString().substring(0,10);
                return _occRows.filter(function(r){ return r.ts.substring(0,10) >= cutoff; });
            } else if (_occRangeFilter === '30d') {
                var cutoff = new Date(now - 30*24*60*60*1000).toISOString().substring(0,10);
                return _occRows.filter(function(r){ return r.ts.substring(0,10) >= cutoff; });
            }
            return _occRows.slice();
        }

        function occSetRange(r) {
            _occRangeFilter = r;
            ['today','7d','30d','all'].forEach(function(t) {
                var btn = document.getElementById('occ-tab-' + t);
                if (!btn) return;
                if (t === r) {
                    btn.style.background = '#2563eb'; btn.style.color = '#fff'; btn.style.border = 'none';
                    btn.style.boxShadow = '0 1px 4px rgba(37,99,235,0.3)'; btn.style.fontWeight = '700';
                } else {
                    btn.style.background = 'rgba(37,99,235,0.12)'; btn.style.color = '#2563eb';
                    btn.style.border = '1px solid rgba(37,99,235,0.25)'; btn.style.boxShadow = ''; btn.style.fontWeight = '600';
                }
            });
            _occRefreshChart();
        }

        function _occRefreshChart() {
            if (!_occChart) return;
            var filtered = _occGetFilteredRows();
            var total = filtered.reduce(function(s,r){ return s + (r.count||0); }, 0);
            var totalEl = document.getElementById('occ-total-today'); if (totalEl) totalEl.textContent = total;
            var countEl = document.getElementById('occ-count'); if (countEl) countEl.textContent = filtered.length;
            if (_occRangeFilter === 'today') {
                var data = new Array(24).fill(null);
                filtered.forEach(function(r){ if (r.hour >= 0 && r.hour < 24) data[r.hour] = r.count; });
                _occChart.data.labels = _OCC_REC_LABEL;
                _occChart.data.datasets[0].data = data;
                _occChart.data.datasets[0].label = 'Person Count (per hour)';
            } else {
                var dayMap = {};
                filtered.forEach(function(r){ var d = r.ts.substring(0,10); dayMap[d] = (dayMap[d]||0) + (r.count||0); });
                var days = Object.keys(dayMap).sort();
                _occChart.data.labels = days;
                _occChart.data.datasets[0].data = days.map(function(d){ return dayMap[d]; });
                _occChart.data.datasets[0].label = 'Total Persons per Day';
            }
            _occChart.update('none');
            _occUpdatePreview();
        }

        function _occUpdatePreview() {
            var tbody = document.getElementById('occ-preview-body');
            if (!tbody) return;
            var filtered = _occGetFilteredRows();
            if (filtered.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:14px;color:var(--text-secondary);">No data for this range&hellip;</td></tr>';
                return;
            }
            tbody.innerHTML = filtered.slice().reverse().map(function(r) {
                return '<tr style="border-bottom:1px solid var(--border);">' +
                    '<td style="padding:6px 12px;color:var(--text-secondary);">' + r.ts + '</td>' +
                    '<td style="padding:6px 12px;text-align:center;font-weight:700;color:#2563eb;">' + String(r.hour).padStart(2,'0') + ':00</td>' +
                    '<td style="padding:6px 12px;text-align:right;font-weight:700;color:#3b82f6;">' + r.count + '</td>' +
                    '<td style="padding:6px 12px;text-align:right;color:var(--text-secondary);">' + r.conf + '</td>' +
                '</tr>';
            }).join('');
        }

        function _occTryRecord() {
            var now = new Date();
            var h = now.getHours();
            var today = now.toISOString().substring(0, 10);
            var dateHour = today + ' ' + String(h).padStart(2, '0');
            // Already recorded for this hour+date?
            if (_occRows.some(function(r){ return r.ts.substring(0,13) === dateHour; })) return;
            // Prune data older than 30 days
            var cutoff = new Date(now - 30*24*60*60*1000).toISOString().substring(0,10);
            _occRows = _occRows.filter(function(r){ return r.ts.substring(0,10) >= cutoff; });
            var countEl = document.getElementById('cam-count-display');
            var confEl  = document.getElementById('cam-confidence-display');
            var count   = parseInt((countEl ? countEl.textContent : '0') || '0', 10);
            var confStr = confEl ? confEl.textContent.replace('%','').trim() : '0';
            var conf    = parseFloat(confStr) || 0;
            var ts = now.toISOString().replace('T',' ').substring(0,19);
            _occRows.push({ ts: ts, hour: h, count: count, conf: conf.toFixed(1) + '%' });
            _occLastHour = h;
            _ssSave('_occRows', _occRows);
            _occRefreshChart();
            showToast('Occupancy at ' + String(h).padStart(2,'0') + ':00 recorded - ' + count + ' persons', 'info');
        }

        function occClear() {
            _occRows = []; _occLastHour = -1;
            sessionStorage.removeItem('_occRows');
            if (_occChart) { _occChart.data.datasets[0].data = new Array(24).fill(null); _occChart.update('none'); }
            _occUpdatePreview();
            var countEl = document.getElementById('occ-count'); if (countEl) countEl.textContent = '0';
            var totalEl = document.getElementById('occ-total-today'); if (totalEl) totalEl.textContent = '0';
            showToast('Occupancy data cleared', 'info');
        }

        function occExportCSV() {
            if (_occRows.length === 0) { showToast('No occupancy data yet', 'error'); return; }
            var header = 'Time,Hour,Person Count,Confidence\\n';
            var body = _occRows.map(function(r) {
                return '"' + r.ts + '",' + String(r.hour).padStart(2,'0') + ':00,' + r.count + ',' + r.conf;
            }).join('\\n');
            var now = new Date();
            var fname = 'occupancy_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
                        String(now.getDate()).padStart(2,'0') + '.csv';
            downloadCSV(fname, header + body);
            showToast(_occRows.length + ' data exported: ' + fname, 'success');
        }

        // Run occupancy check every 60 seconds; reset and record new hour
        setInterval(function() {
            try { _occTryRecord(); } catch(e) {}
        }, 60000);
        // Update "next record in" countdown every second
        setInterval(function() {
            try {
                var now = new Date();
                var minsLeft = 59 - now.getMinutes();
                var secsLeft = 59 - now.getSeconds();
                var el = document.getElementById('occ-next-in');
                if (el) el.textContent = minsLeft + ' min ' + String(secsLeft).padStart(2,'0') + ' sec';
            } catch(e) {}
        }, 1000);
        // Also init chart once DOM is ready (delayed)
        setTimeout(function() {
            try { _occInitChart(); _occRefreshChart(); } catch(e) {}
        }, 500);

        // ==================== MQTT REAL-TIME UPDATE HANDLER ====================
        socket.on('mqtt_update', function(data) {
            updateDashboard();
            
            if (data.type === 'ac') {
                const now = new Date();
                const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
                
                if (charts.temp && charts.temp.data.labels.length < 50) {
                    charts.temp.data.labels.push(timeStr);
                    charts.temp.data.datasets[0].data.push(data.data.temperature);
                    charts.temp.update('none');
                }
                if (charts.hum && charts.hum.data.labels.length < 50) {
                    charts.hum.data.labels.push(timeStr);
                    charts.hum.data.datasets[0].data.push(data.data.humidity);
                    charts.hum.update('none');
                }
            }
            
            if (data.type === 'lamp') {
                const now = new Date();
                const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
                const lamp = data.data;
                const luxAvg    = parseFloat(lamp.lux_avg)    || 0;
                const brightAvg = parseFloat(lamp.brightness_avg) || 0;
                // local setText - does not depend on outer scope
                const _set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };

                // Update Lamp Dashboard lux dan brightness secara real-time
                _set('dash-lux1',    (lamp.lux1    != null ? parseFloat(lamp.lux1)    : 0).toFixed(0));
                _set('dash-lux2',    (lamp.lux2    != null ? parseFloat(lamp.lux2)    : 0).toFixed(0));
                _set('dash-lux3',    (lamp.lux3    != null ? parseFloat(lamp.lux3)    : 0).toFixed(0));
                _set('dash-lux-avg', luxAvg.toFixed(1));
                _set('ctrl-lux1',    (lamp.lux1    != null ? parseFloat(lamp.lux1)    : 0).toFixed(0));
                _set('ctrl-lux2',    (lamp.lux2    != null ? parseFloat(lamp.lux2)    : 0).toFixed(0));
                _set('ctrl-lux3',    (lamp.lux3    != null ? parseFloat(lamp.lux3)    : 0).toFixed(0));
                _set('dash-bright1', Math.round(parseFloat(lamp.brightness1) || 0));
                _set('dash-bright2', Math.round(parseFloat(lamp.brightness2) || 0));

                if (charts.lampLux) {
                    if (charts.lampLux.data.labels.length >= 50) {
                        charts.lampLux.data.labels.shift();
                        charts.lampLux.data.datasets[0].data.shift();
                    }
                    charts.lampLux.data.labels.push(timeStr);
                    charts.lampLux.data.datasets[0].data.push(luxAvg);
                    charts.lampLux.update('none');
                }
                if (charts.lampBright) {
                    if (charts.lampBright.data.labels.length >= 50) {
                        charts.lampBright.data.labels.shift();
                        charts.lampBright.data.datasets[0].data.shift();
                    }
                    charts.lampBright.data.labels.push(timeStr);
                    charts.lampBright.data.datasets[0].data.push(Math.round(brightAvg));
                    charts.lampBright.update('none');
                }
            }
            
            if (data.type === 'energy') {
                const e = data.data;
                // Tampilkan bubble notifikasi jika ada daya
                const pwr = parseFloat(e.power || 0);
                const vlt = parseFloat(e.voltage || 0);
                const cur = parseFloat(e.current || 0);
                if (pwr > 0 || vlt > 0) {
                    showEnergyBubble(pwr, vlt, cur);
                }
            }

            // Real-time optimization (GA->AC / PSO->Lamp) updates
            if (data.type === 'system') {
                const gaFitness = parseFloat(data.data.ga_fitness) || 0;
                const psoFitness = parseFloat(data.data.pso_fitness) || 0;
                const runs = data.data.optimization_runs || 0;
                const gaTemp = data.data.ga_temp || 0;
                const gaFan = data.data.ga_fan || 0;
                // PWM 0-255 values (actual algorithm output)
                const psoPwm1 = data.data.pso_pwm1 != null ? data.data.pso_pwm1 : 0;
                const psoPwm2 = data.data.pso_pwm2 != null ? data.data.pso_pwm2 : 0;
                // Brightness % values (for MQTT/ESP32)
                const psoBrightness1 = data.data.pso_brightness1 != null ? data.data.pso_brightness1 : 0;
                const psoBrightness2 = data.data.pso_brightness2 != null ? data.data.pso_brightness2 : 0;
                
                // Update GA fitness display
                const gaEl = document.getElementById('ga-fitness');
                if (gaEl) {
                    gaEl.textContent = gaFitness.toFixed(2);
                    gaEl.style.color = gaFitness > 0 ? '#3b82f6' : '#94a3b8';
                    if (gaFitness > 0) gaEl.style.textShadow = '0 0 10px rgba(59, 130, 246, 0.5)';
                }
                
                // Update GA solution details (AC settings)
                const gaTempEl = document.getElementById('ga-temp');
                const gaFanEl = document.getElementById('ga-fan');
                if (gaTempEl && gaTemp > 0) gaTempEl.textContent = gaTemp;
                if (gaFanEl && gaFan > 0) gaFanEl.textContent = gaFan;
                
                // Update PSO fitness display
                const psoEl = document.getElementById('pso-fitness');
                if (psoEl) {
                    psoEl.textContent = psoFitness.toFixed(2);
                    psoEl.style.color = psoFitness > 0 ? '#0ea5e9' : '#94a3b8';
                    if (psoFitness > 0) psoEl.style.textShadow = '0 0 10px rgba(14, 165, 233, 0.5)';
                }
                
                console.log('[ML] Optimization Update:', {
                    ga: gaFitness.toFixed(2), ac_temp: gaTemp, ac_fan: gaFan,
                    pso: psoFitness.toFixed(2), pwm1: psoPwm1, pwm2: psoPwm2,
                    b1: psoBrightness1 + '%', b2: psoBrightness2 + '%', runs: runs
                });
                
                // Show toast with actual PWM values
                if (gaFitness > 0 && psoFitness > 0) {
                    showToast('GA->AC: ' + gaTemp + '\xB0C (' + gaFitness.toFixed(1) + ') | PSO->Lamp: PWM1=' + psoPwm1 + '/255 PWM2=' + psoPwm2 + '/255 (err=' + psoFitness.toFixed(1) + ')', 'success');
                } else if (gaFitness > 0) {
                    showToast('GA->AC: ' + gaTemp + '\xB0C Fan:' + gaFan + ' (Fitness: ' + gaFitness.toFixed(2) + ')', 'success');
                } else if (psoFitness > 0) {
                    showToast('PSO->Lamp: PWM1=' + psoPwm1 + '/255 PWM2=' + psoPwm2 + '/255 (err=' + psoFitness.toFixed(2) + ')', 'success');
                }

                // === Update ML Optimization Page ===
                updateMLDisplay(data.data);
                if (gaFitness > 0 || psoFitness > 0) {
                    addMLHistoryRow({
                        ga_fitness: gaFitness, ga_temp: gaTemp, ga_fan: gaFan,
                        pso_fitness: psoFitness, pso_brightness: Math.round((psoPwm1 + psoPwm2) / 2)
                    });
                }
                // Update GA/PSO convergence charts from history arrays
                if (data.data.ga_history && data.data.ga_history.length > 0) {
                    updateMLChart('gaFitness', data.data.ga_history, 'GA');
                }
                if (data.data.pso_history && data.data.pso_history.length > 0) {
                    updateMLChart('psoFitness', data.data.pso_history, 'PSO');
                }
            }
            
            // Real-time camera/person detection update
            if (data.type === 'camera') {
                const personCount = data.data.count || 0;
                const confidence = data.data.confidence || 0;
                const personDetected = data.data.person_detected || false;
                
                // Update count display with color coding
                const countEl = document.getElementById('cam-count');
                const countDisplayEl = document.getElementById('cam-count-display');
                if (countEl) {
                    countEl.textContent = personCount;
                    if (personCount > 0) {
                        countEl.style.color = '#3b82f6';
                        countEl.style.textShadow = '0 0 10px rgba(59, 130, 246, 0.5)';
                    } else {
                        countEl.style.color = '#94a3b8';
                        countEl.style.textShadow = 'none';
                    }
                }
                if (countDisplayEl) {
                    countDisplayEl.textContent = personCount;
                    countDisplayEl.style.color = personCount > 0 ? '#3b82f6' : '#94a3b8';
                    if (personCount > 0) {
                        countDisplayEl.parentElement.style.background = 'rgba(59, 130, 246, 0.1)';
                        countDisplayEl.parentElement.style.borderColor = '#3b82f6';
                    } else {
                        countDisplayEl.parentElement.style.background = '';
                        countDisplayEl.parentElement.style.borderColor = 'var(--border)';
                    }
                }
                
                // Update confidence with color coding
                const confEl = document.getElementById('cam-confidence');
                const confDisplayEl = document.getElementById('cam-confidence-display');
                const confText = confidence + '%';
                
                [confEl, confDisplayEl].forEach(el => {
                    if (el) {
                        el.textContent = confText;
                        if (confidence > 70) {
                            el.style.color = '#3b82f6';
                        } else if (confidence > 50) {
                            el.style.color = '#0ea5e9';
                        } else {
                            el.style.color = '#1e40af';
                        }
                    }
                });
                
                if (confDisplayEl && personCount > 0) {
                    confDisplayEl.parentElement.style.background = 'rgba(59, 130, 246, 0.1)';
                    confDisplayEl.parentElement.style.borderColor = '#3b82f6';
                } else if (confDisplayEl) {
                    confDisplayEl.parentElement.style.background = '';
                    confDisplayEl.parentElement.style.borderColor = 'var(--border)';
                }
                
                // Update person detected card
                const personEl = document.getElementById('cam-person');
                const personCard = document.getElementById('person-detected-card');
                if (personEl) {
                    personEl.textContent = personDetected ? 'Yes' : 'No';
                    personEl.style.color = personDetected ? '#3b82f6' : '#1e40af';
                }
                if (personCard) {
                    if (personDetected) {
                        personCard.style.background = 'rgba(59, 130, 246, 0.1)';
                        personCard.style.borderColor = '#3b82f6';
                    } else {
                        personCard.style.background = '';
                        personCard.style.borderColor = 'var(--border)';
                    }
                }
                
                // Update overlay badge
                const overlayBadge = document.getElementById('overlay-person-badge');
                if (overlayBadge) {
                    if (personDetected && personCount > 0) {
                        overlayBadge.className = 'person-badge detected';
                        overlayBadge.innerHTML = personCount + ' Person(s) - ' + confidence + '%';
                    } else {
                        overlayBadge.className = 'person-badge not-detected';
                        overlayBadge.innerHTML = 'No Person';
                    }
                }
                
                // Show alert when person detected
                if (personDetected && personCount > 0) {
                    showDetectionAlert(personCount, confidence);
                }
                
                // Update "Last Person Detected" display
                const lastSeenEl = document.getElementById('cam-last-seen');
                const lastSeenLabel = document.getElementById('cam-last-seen-label');
                const lastSeenAgo = data.data.last_seen_ago;
                if (lastSeenEl) {
                    if (personDetected && personCount > 0) {
                        lastSeenEl.textContent = 'Sekarang';
                        lastSeenEl.style.color = '#3b82f6';
                        if (lastSeenLabel) lastSeenLabel.textContent = 'Person currently detected';
                    } else if (lastSeenAgo !== undefined && lastSeenAgo >= 0) {
                        const mins = Math.floor(lastSeenAgo / 60);
                        const secs = lastSeenAgo % 60;
                        lastSeenEl.textContent = mins > 0 ? (mins + 'm ' + secs + 's lalu') : (secs + 's lalu');
                        lastSeenEl.style.color = lastSeenAgo > 300 ? '#1e40af' : '#2563eb';
                        if (lastSeenLabel) lastSeenLabel.textContent = 'Since last detection';
                    } else {
                        lastSeenEl.textContent = '--';
                        lastSeenEl.style.color = '#94a3b8';
                        if (lastSeenLabel) lastSeenLabel.textContent = 'Never detected';
                    }
                }
                
                // Update "Auto-OFF AC" countdown
                const autoOffEl = document.getElementById('cam-auto-off-timer');
                const autoOffLabel = document.getElementById('cam-auto-off-label');
                const autoOffIn = data.data.auto_off_in;
                const autoOffTriggered = data.data.auto_off_triggered;
                if (autoOffEl) {
                    if (autoOffTriggered) {
                        autoOffEl.textContent = 'OFF';
                        autoOffEl.style.color = '#1e40af';
                        if (autoOffLabel) autoOffLabel.textContent = 'AC already auto-OFF';
                    } else if (autoOffIn !== undefined && autoOffIn >= 0 && !personDetected) {
                        const offMins = Math.floor(autoOffIn / 60);
                        const offSecs = autoOffIn % 60;
                        autoOffEl.textContent = offMins + 'm ' + offSecs + 's';
                        autoOffEl.style.color = autoOffIn < 120 ? '#1e40af' : '#0ea5e9';
                        if (autoOffLabel) autoOffLabel.textContent = 'Countdown auto-OFF AC';
                    } else {
                        autoOffEl.textContent = '--';
                        autoOffEl.style.color = '#3b82f6';
                        if (autoOffLabel) autoOffLabel.textContent = 'Person detected, AC safe';
                    }
                }
                
                console.log('[CAM] Camera Update:', {
                    count: personCount,
                    confidence: confidence,
                    detected: personDetected
                });
            }
        });

        // ML Optimization status from main.py
        socket.on('ml_status', function(data) {
            console.log('[ML] ML Status:', data);
            const status = data.status || '';
            const algo = (data.algorithm || '').toUpperCase();
            
            if (status === 'running') {
                showToast(algo + ' optimization running...', 'success');
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                });
                // Reset PSO iteration table when new run starts
                if (algo === 'PSO') {
                    const tbody = document.getElementById('pso-iter-tbody');
                    if (tbody) tbody.innerHTML = '';
                    const chart = charts.psoFitness;
                    if (chart) {
                        chart.data.labels = [];
                        chart.data.datasets.forEach(ds => ds.data = []);
                        chart.update('none');
                    }
                }
            } else if (status === 'completed') {
                var modeLabel = '';
                if (data.ga_solution && data.ga_solution.mode) modeLabel = ' Mode:' + data.ga_solution.mode;
                showToast(algo + ' optimization completed! GA: ' + (data.ga_fitness || 0).toFixed(2) + modeLabel + ', PSO: ' + (data.pso_fitness || 0).toFixed(2), 'success');
                // Re-enable run buttons
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                });
                // Update mode display immediately from solution
                if (data.ga_solution && data.ga_solution.mode) {
                    var modeEl = document.getElementById('ml-ga-mode');
                    if (modeEl) modeEl.textContent = data.ga_solution.mode;
                    var rhEl = document.getElementById('ml-ga-rh');
                    if (rhEl && data.ga_solution.set_rh != null) rhEl.textContent = data.ga_solution.set_rh;
                }
                // Directly update GA chart from socket payload (faster than waiting for refreshMLData fetch)
                if (data.ga_history && data.ga_history.length > 0) {
                    try { updateMLChart('gaFitness', data.ga_history, 'GA'); } catch(e) { console.warn('[ML] Direct GA chart update failed:', e); }
                }
                if (data.pso_history && data.pso_history.length > 0) {
                    try { updateMLChart('psoFitness', data.pso_history, 'PSO'); } catch(e) { console.warn('[ML] Direct PSO chart update failed:', e); }
                }
                // Refresh ML data
                refreshMLData();
            } else if (status === 'error') {
                showToast(algo + ' error: ' + (data.message || 'Unknown error'), 'error');
                document.querySelectorAll('.ml-param-grid button').forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                });
            } else if (status === 'busy') {
                showToast('Optimization already in progress', 'warning');
            }
        });

        socket.on('ir_learned', function(data) {
            console.log('[OK] IR Learned event received:', data);
            
            let buttonName = data.button;
            let buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
            let statusElement = document.getElementById('status-' + buttonName);
            
            // Try alternative button names if not found
            if (!buttonElement && currentLearningButton) {
                buttonName = currentLearningButton;
                buttonElement = document.querySelector('[data-button="' + buttonName + '"]');
                statusElement = document.getElementById('status-' + buttonName);
            }
            
            if (data.status === 'error') {
                showToast('[ERROR] IR learning failed: ' + (data.message || 'Unknown error'), 'error');
                if (buttonElement) buttonElement.classList.remove('learning');
                if (statusElement) {
                    statusElement.textContent = 'Failed';
                    statusElement.style.color = '#1e40af';
                }
                return;
            }
            
            // Success!
            if (buttonElement) {
                buttonElement.classList.remove('learning');
                buttonElement.classList.add('learned');
            }
            
            if (statusElement) {
                statusElement.textContent = 'Learned [OK]';
                statusElement.style.color = '#3b82f6';
            }
            
            let message = 'IR Code learned: ' + buttonName;
            if (data.is_toggle) {
                message += ' (Power Toggle)';
            }
            
            showToast(message, 'success');
            learnedCodes[buttonName] = data.code;
            currentLearningButton = null;
        });

        // Debug: Listen to all MQTT messages
        socket.on('mqtt_debug', function(data) {
            console.log('[MQTT Debug]', data.time, '-', data.topic, ':', data.payload);
            
            // If learning mode and IR topic detected, show notification
            if (currentLearningButton && (data.topic.includes('ir') || data.topic.includes('IR'))) {
                console.log('[WARN] IR-related MQTT message while in learning mode!');
            }
        });

        // ==================== INIT ====================
        function loadSavedPreferences() {
            let savedPage = localStorage.getItem('currentPage');
            // Migrate old page IDs to new split pages
            if (savedPage === 'dashboard') savedPage = 'dashboard-ac';
            if (savedPage === 'control') savedPage = 'control-ac';
            if (savedPage === 'power') savedPage = 'energy';
            if (savedPage && document.getElementById(savedPage)) {
                document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
                document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
                
                document.getElementById(savedPage).classList.add('active');
                document.querySelectorAll('.nav-item').forEach(function(item) {
                    const oc = item.getAttribute('onclick') || '';
                    if (oc.indexOf("showPage('" + savedPage + "')") !== -1) {
                        item.classList.add('active');
                    }
                });
            }
            
            const savedRanges = localStorage.getItem('chartRanges');
            if (savedRanges) chartRanges = JSON.parse(savedRanges);
        }

        window.onload = function() {
            console.log('[INIT] Smart Room Dashboard Loading...');
            // Fetch role first so nav is hidden before user sees it
            fetch('/api/auth/role').then(function(r){ return r.json(); }).then(function(d){
                applyRoleRestrictions(d.role, d.username);
            }).catch(function(){ applyRoleRestrictions('user', ''); });
            try {
                var ovInit = document.getElementById('sidebar-overlay');
                if (ovInit) {
                    ovInit.classList.remove('active');
                    ovInit.style.display = 'none';
                    ovInit.style.pointerEvents = 'none';
                }
            } catch(e) {}
            try {
                // Safety net: hidden fixed overlays must not consume clicks.
                var blockers = document.querySelectorAll('body *');
                blockers.forEach(function(el) {
                    if (!el || !el.style) return;
                    if (el.id === 'drp-modal') return; // never touch modal
                    var cs = window.getComputedStyle(el);
                    if (cs.position === 'fixed' && cs.display === 'none') {
                        el.style.pointerEvents = 'none';
                    }
                });
            } catch(e) {}
            try { initCharts(); } catch(e) { console.error('[ERROR] initCharts:', e); }
            // ML charts initialized on first tab open via requestAnimationFrame (canvas must be visible first)
            // Restore recording previews from sessionStorage
            try { _recUpdatePreview(); } catch(e) {}
            try { _tempUpdatePreview(); } catch(e) {}
            try { _luxUpdatePreview(); } catch(e) {}
            // Fetch server-side recorded data (cross-device sync)
            try { _recServerFetch(); } catch(e) {}
            try { _tempServerFetch(); } catch(e) {}
            try { _luxServerFetch(); } catch(e) {}
            try { _occInitChart(); } catch(e) {}
            try { loadSavedPreferences(); } catch(e) { console.error('[ERROR] loadSavedPreferences:', e); }
            try { loadSavedSettings(); } catch(e) { console.error('[ERROR] loadSavedSettings:', e); }
            try { updateSoundToggleUI(); } catch(e) { console.error('[ERROR] updateSoundToggleUI:', e); }
            try { updateCameraToggleUI(); } catch(e) { console.error('[ERROR] updateCameraToggleUI:', e); }
            try { updateDashboard(); } catch(e) { console.error('[ERROR] updateDashboard:', e); }
            try { updateDeviceStatus(); } catch(e) { console.error('[ERROR] updateDeviceStatus:', e); }
            // Auto-show diagnostic panel if MQTT is not connected
            setTimeout(function() {
                fetch('/api/mqtt/status').then(r => r.json()).then(function(d) {
                    if (!d.connected) {
                        var p = document.getElementById('diag-panel');
                        if (p) {
                            p.style.display = 'block';
                            var res = document.getElementById('diag-result');
                            if (res) res.textContent = '[AUTO-DIAGNOSE] MQTT NOT CONNECTED!\\nBroker: ' + d.broker + '\\nError: ' + (d.error || 'Unknown') + '\\n\\nClick "Test Frontend" or "Test MQTT Broker" button below.';
                        }
                    }
                }).catch(function() {});
            }, 2000);
            try { updateLogs(); } catch(e) { console.error('[ERROR] updateLogs:', e); }
            try { checkCameraStatus(); } catch(e) { console.error('[ERROR] checkCameraStatus:', e); }
            try { loadAllEnergyCharts(); } catch(e) { console.error('[ERROR] loadAllEnergyCharts:', e); }
            setTimeout(function() {
                try {
                    if (ensureChartsReady()) {
                        loadAllEnergyCharts();
                    }
                } catch(e) {
                    console.error('[ERROR] delayed energy chart init:', e);
                }
            }, 600);
            const googleFormInput = document.getElementById('google-form-url');
            if (googleFormInput) {
                googleFormInput.value = localStorage.getItem('googleFormUrl') || DEFAULT_GOOGLE_FORM_URL;
            }
            try { loadFeedbackHistory(); } catch(e) { console.error('[ERROR] loadFeedbackHistory:', e); }
            
            // Initialize AC mode UI on page load
            try { updateModeBadges(); } catch(e) { console.error('[ERROR] updateModeBadges:', e); }
            
            Object.keys(chartRanges).forEach(chartName => {
                try { updateChartData(chartName, chartRanges[chartName]); } catch(e) { console.error('[ERROR] updateChartData ' + chartName + ':', e); }
            });
            
            setInterval(function() { try { updateDashboard(); } catch(e) {} }, 1000);
            setInterval(function() { try { updateDeviceStatus(); } catch(e) {} }, 5000);
            setInterval(function() { try { updateLogs(); } catch(e) {} }, 5000);
            
            setInterval(() => {
                Object.keys(chartRanges).forEach(chartName => {
                    try { updateChartData(chartName, chartRanges[chartName]); } catch(e) {}
                });
            }, 30000);

            setInterval(function() {
                try {
                    var energyPage = document.getElementById('energy');
                    if (energyPage && energyPage.classList.contains('active')) {
                        loadAllEnergyCharts();
                    }
                    var outletCtrlPage = document.getElementById('control-outlet');
                    if (outletCtrlPage && outletCtrlPage.classList.contains('active')) {
                        try { refreshOutletStatus(); } catch(e) {}
                    }
                    var outletAnalPage = document.getElementById('outlet-analysis');
                    if (outletAnalPage && outletAnalPage.classList.contains('active')) {
                        try { refreshOutletStatus(); } catch(e) {}
                    }
                } catch(e) {}
            }, 8000);
            
            // ==================== DATE RANGE PICKER ====================
            var _drpCallback = null;
            var _drpAllRows  = null;

            window.showDateRangePicker = function(subtitle, rowsRef, callback) {
                _drpCallback = callback;
                _drpAllRows  = rowsRef;
                var sub = document.getElementById('drp-subtitle');
                if (sub) sub.textContent = subtitle || 'Export data in the selected range';
                var today = new Date().toISOString().substring(0,10);
                var from30 = new Date(Date.now() - 29*24*60*60*1000).toISOString().substring(0,10);
                var fromVal = from30;
                if (rowsRef && rowsRef.length > 0 && rowsRef[0].ts) {
                    var firstDate = rowsRef[0].ts.substring(0,10);
                    fromVal = firstDate < from30 ? firstDate : from30;
                }
                document.getElementById('drp-from').value = fromVal;
                document.getElementById('drp-to').value   = today;
                var modal = document.getElementById('drp-modal');
                modal.style.pointerEvents = 'auto';
                modal.style.display = 'flex';
            };

            window.drpShortcut = function(days) {
                var today = new Date().toISOString().substring(0,10);
                document.getElementById('drp-to').value = today;
                if (days === 0) {
                    var fromVal = today;
                    if (_drpAllRows && _drpAllRows.length > 0 && _drpAllRows[0].ts) {
                        fromVal = _drpAllRows[0].ts.substring(0,10);
                    }
                    document.getElementById('drp-from').value = fromVal;
                } else if (days === 1) {
                    document.getElementById('drp-from').value = today;
                } else {
                    var d = new Date(Date.now() - (days-1)*24*60*60*1000).toISOString().substring(0,10);
                    document.getElementById('drp-from').value = d;
                }
            };

            window.drpConfirm = function() {
                var from = document.getElementById('drp-from').value;
                var to   = document.getElementById('drp-to').value;
                if (!from || !to) { showToast('Please select a date range first', 'error'); return; }
                if (from > to)    { showToast('Start date must be before end date', 'error'); return; }
                var modal = document.getElementById('drp-modal');
                modal.style.display = 'none';
                modal.style.pointerEvents = 'none';
                if (_drpCallback) _drpCallback(from, to);
            };

            window.drpCancel = function() {
                var modal = document.getElementById('drp-modal');
                modal.style.display = 'none';
                modal.style.pointerEvents = 'none';
            };

            function drpFilter(rows, from, to) {
                return rows.filter(function(r){ var d = (r.ts||'').substring(0,10); return d >= from && d <= to; });
            }

            window.energyExportRange = function(device) {
                showDateRangePicker(
                    'Export rekaman energi' + (device && device !== 'all' ? ' (' + device + ')' : ' (All)'),
                    _recRows,
                    function(from, to) {
                        var base = (device && device !== 'all') ? _recRows.filter(function(r){ return r.device === device; }) : _recRows;
                        var rows = drpFilter(base, from, to);
                        if (rows.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                        var header = 'Time,Perangkat,Voltage (V),Current (A),Power Active (W),Energy (kWh),Power Reaktif (VAR),Power Semu (VA),Frequency (Hz),Power Factor\\n';
                        var body = rows.map(function(r) {
                            return '"' + r.ts + '",' + r.device + ',' + r.voltage + ',' + r.arus + ',' +
                                   r.daya + ',' + r.energi + ',' + r.reaktif + ',' + r.semu + ',' + r.freq + ',' + r.pf;
                        }).join('\\n');
                        var suffix = (device && device !== 'all') ? '_' + device.toLowerCase() : '';
                        var fname = 'energy' + suffix + '_' + from + '_to_' + to + '.csv';
                        downloadCSV(fname, header + body);
                        showToast(rows.length + ' rows exported: ' + fname, 'success');
                    }
                );
            };

            window.tempExportRange = function() {
                showDateRangePicker('Export temperature & humidity recording', _tempRows, function(from, to) {
                    var rows = drpFilter(_tempRows, from, to);
                    if (rows.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                    var header = 'Time,Temp Rata2 (C),Humidity (%),Sensor T1 (C),Sensor T2 (C),Sensor T3 (C),Set Temp (C),Fan Speed,Mode AC,Control\\n';
                    var body = rows.map(function(r){ return '"' + r.ts + '",' + r.temp + ',' + r.hum + ',' + r.t1 + ',' + r.t2 + ',' + r.t3 + ',' + r.setT + ',"' + r.fan + '",' + r.mode + ',' + r.ctrl; }).join('\\n');
                    var fname = 'temp_' + from + '_to_' + to + '.csv';
                    downloadCSV(fname, header + body);
                    showToast(rows.length + ' rows exported: ' + fname, 'success');
                });
            };

            window.luxExportRange = function() {
                showDateRangePicker('Export rekaman lux & brightness', _luxRows, function(from, to) {
                    var rows = drpFilter(_luxRows, from, to);
                    if (rows.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                    var header = 'Time,Lux 1 (lx),Lux 2 (lx),Lux 3 (lx),Avg Lux (lx),Brightness 1 (%),Brightness 2 (%)\\n';
                    var body = rows.map(function(r){ return '"' + r.ts + '",' + r.lux1 + ',' + r.lux2 + ',' + r.lux3 + ',' + r.avg + ',' + r.b1 + ',' + r.b2; }).join('\\n');
                    var fname = 'lux_' + from + '_to_' + to + '.csv';
                    downloadCSV(fname, header + body);
                    showToast(rows.length + ' rows exported: ' + fname, 'success');
                });
            };

            window.occExportRange = function() {
                showDateRangePicker('Export rekaman occupancy', _occRows, function(from, to) {
                    var rows = drpFilter(_occRows, from, to);
                    if (rows.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                    var header = 'Time,Hour,Person Count,Confidence\\n';
                    var body = rows.map(function(r){ return '"' + r.ts + '",' + String(r.hour).padStart(2,'0') + ':00,' + r.count + ',' + r.conf; }).join('\\n');
                    var fname = 'occupancy_' + from + '_to_' + to + '.csv';
                    downloadCSV(fname, header + body);
                    showToast(rows.length + ' data exported: ' + fname, 'success');
                });
            };

            window.exportChartRange = function(chartName, valueLabel) {
                var chart = charts[chartName];
                if (!chart || !chart.data.labels || chart.data.labels.length === 0) {
                    showToast('No data for ' + chartName, 'error'); return;
                }
                var pseudoRows = chart.data.labels.map(function(l){ return {ts: l}; });
                showDateRangePicker('Export chart: ' + valueLabel, pseudoRows, function(from, to) {
                    var header = 'Time,' + valueLabel + '\\n';
                    var lines = [];
                    chart.data.labels.forEach(function(label, i) {
                        var d = label.substring(0,10);
                        if (d < from || d > to) return;
                        var val = chart.data.datasets[0].data[i];
                        lines.push('"' + label + '",' + (val !== null && val !== undefined ? val : ''));
                    });
                    if (lines.length === 0) { showToast('No data for range ' + from + ' to ' + to, 'error'); return; }
                    downloadCSV(chartName + '_' + from + '_to_' + to + '.csv', header + lines.join('\\n') + '\\n');
                    showToast(lines.length + ' points exported', 'success');
                });
            };

            // ==================== DB EXPORT HELPERS ====================
            function _dbDateToday() {
                var d = new Date(); 
                return d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0') + '-' + String(d.getDate()).padStart(2,'0');
            }
            function _dbDateNdAgo(n) {
                var d = new Date(); d.setDate(d.getDate() - n + 1);
                return d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0') + '-' + String(d.getDate()).padStart(2,'0');
            }
            window.dbExportSetToday = function(prefix) {
                var t = _dbDateToday();
                var fi = document.getElementById('db-' + prefix + '-from');
                var ti = document.getElementById('db-' + prefix + '-to');
                if (fi) fi.value = t; if (ti) ti.value = t;
            };
            window.dbExportSet7d = function(prefix) {
                var fi = document.getElementById('db-' + prefix + '-from');
                var ti = document.getElementById('db-' + prefix + '-to');
                if (fi) fi.value = _dbDateNdAgo(7); if (ti) ti.value = _dbDateToday();
            };
            window.dbExportSet30d = function(prefix) {
                var fi = document.getElementById('db-' + prefix + '-from');
                var ti = document.getElementById('db-' + prefix + '-to');
                if (fi) fi.value = _dbDateNdAgo(30); if (ti) ti.value = _dbDateToday();
            };
            window.dbExportCSV = function(type) {
                // prefix mapping: all energy types share the same date inputs (db-energy-from / db-energy-to)
                var prefix = (type === 'energy_ac' || type === 'energy_lamp' || type === 'energy_outlet' || type === 'energy_total') ? 'energy' : type;
                var fromEl = document.getElementById('db-' + prefix + '-from');
                var toEl   = document.getElementById('db-' + prefix + '-to');
                var from = fromEl ? fromEl.value : '';
                var to   = toEl   ? toEl.value   : '';
                if (!from || !to) { showToast('Please select a date range first', 'error'); return; }
                if (from > to) { showToast('Start date must be before end date', 'error'); return; }
                showToast('Fetching data from database...', 'success');
                var url = '/api/export/csv?type=' + type + '&from=' + from + '&to=' + to;
                fetch(url).then(function(r) {
                    if (r.ok && r.headers.get('Content-Type') && r.headers.get('Content-Type').indexOf('text/csv') >= 0) {
                        return r.blob().then(function(blob) {
                            var a = document.createElement('a');
                            a.href = URL.createObjectURL(blob);
                            var fname = type + '_' + from + '_to_' + to + '.csv';
                            a.download = fname;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(a.href);
                        });
                    } else {
                        return r.json().then(function(j) {
                            showToast((j && j.error) ? j.error : 'Failed to export data', 'error');
                        });
                    }
                }).catch(function(e) { showToast('Error: ' + e, 'error'); });
            };

            function fetchOutdoorWeather() {
                fetch('/api/outdoor-weather')
                    .then(response => response.json())
                    .then(data => {
                        if (data.outdoor && data.outdoor.fetch_ok) {
                            var tempEl = document.getElementById('dash-outdoor-temp');
                            var humEl = document.getElementById('dash-outdoor-hum');
                            var windEl = document.getElementById('dash-outdoor-wind');
                            var uvEl = document.getElementById('dash-outdoor-uv');
                            var descEl = document.getElementById('dash-weather-desc');
                            var iconEl = document.getElementById('dash-weather-icon');
                            
                            if(tempEl) tempEl.textContent = data.outdoor.temperature;
                            if(humEl) humEl.textContent = data.outdoor.humidity;
                            if(windEl) windEl.textContent = data.outdoor.wind_speed + ' km/h';
                            if(uvEl) uvEl.textContent = data.outdoor.uv_index;
                            if(descEl) descEl.textContent = data.outdoor.weather_desc;
                            if(iconEl) iconEl.textContent = data.outdoor.weather_icon;
                        }
                    })
                    .catch(err => console.error('[WEATHER] Fetch error:', err));
            }
            
            // Initial fetch and interval (every 10 minutes)
            fetchOutdoorWeather();
            setInterval(fetchOutdoorWeather, 600000);

            console.log('[OK] Dashboard Ready!');
        };
    
