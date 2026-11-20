import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL CONSTANTS (DRUDE MODEL)
# ==========================================
EPS_0 = 8.854e-12        # F/m
EPS_INF = 11.7           # Silicon dielectric constant
E_CHARGE = 1.602e-19     # C
M_EFF = 0.26 * 9.109e-31 # Effective carrier mass (kg)
C_SPEED = 2.998e8        # m/s
GAMMA = 3.5e13           # Damping rate (s^-1)

GRID_SIZE = 1000

# ==========================================
# 2. PURE SILICON WAFER DEFECT MAP
# ==========================================
print("Synthesizing raw silicon wafer material defects...")

Y, X = np.ogrid[:GRID_SIZE, :GRID_SIZE]

# Base nominal active implant target: 5.0e19 cm^-3 (5.0e25 m^-3)
N_nominal = 5.0e25
N_map = np.full((GRID_SIZE, GRID_SIZE), N_nominal, dtype=np.float64)

# DEFECT 1: Sweeping Thermal Anneal Arc (RTP furnace heater gradient)
wafer_center_x, wafer_center_y = -3000, -2500
dist_to_center = np.sqrt((X - wafer_center_x)**2 + (Y - wafer_center_y)**2)
thermal_arc = np.sin(dist_to_center / 80.0) * np.exp(-((dist_to_center - 4000) / 600)**2) * 2.2e25
N_map += thermal_arc

# DEFECT 2: Crystalline Slip Line (Thermal Stress Dislocation along (110) plane)
slip_dist = np.abs((X - 100) - (Y - 50))
slip_line = np.exp(-(slip_dist**2) / 3.0) * 4.0e25
N_map += slip_line

# DEFECT 3: Micro-Particle Implant Shadow Void (Blocked ion beam + spray wake)
px, py = 450, 480
particle_dist = np.sqrt((X - px)**2 + (Y - py)**2)
wake = np.exp(-((Y - py)**2) / 30.0) * (X >= px) * (X <= px + 110) * np.exp(-(X - px) / 40.0)
particle_defect = (particle_dist < 16).astype(np.float64) + wake
N_map -= particle_defect * 4.8e25  # Dropout down to substrate level

# DEFECT 4: Implanter Velocity Striations (Scan jitter ripple)
scan_ripple = np.sin((X + 0.2 * Y) / 10.0) * 0.4e25
N_map += scan_ripple

N_map = np.clip(N_map, 2e21, 1e26)

# Ground-truth log10 doping concentration map for HUD
log_N_ground_truth = np.log10(N_map / 1e6)

# ==========================================
# 3. DRUDE REFLECTANCE SOLVER (WITH NOISE)
# ==========================================
def calculate_reflectance(wavelength_nm):
    wavelength_m = wavelength_nm * 1e-9
    omega = 2.0 * np.pi * C_SPEED / wavelength_m
    
    omega_p_sq = (N_map * (E_CHARGE**2)) / (EPS_0 * EPS_INF * M_EFF)
    denom = omega**2 + GAMMA**2
    eps_1 = EPS_INF - (omega_p_sq / denom)
    eps_2 = (omega_p_sq * GAMMA) / (omega * denom)
    
    mag_eps = np.sqrt(eps_1**2 + eps_2**2)
    n = np.sqrt(np.maximum((mag_eps + eps_1) / 2.0, 0))
    k = np.sqrt(np.maximum((mag_eps - eps_1) / 2.0, 0))
    
    R = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)
    
    # Authentic SWIR InGaAs camera shot & sensor noise
    sensor_noise = np.random.normal(0, 0.0018, (GRID_SIZE, GRID_SIZE))
    return np.clip(R + sensor_noise, 0.0, 1.0)

# ==========================================
# 4. GUI & HUD SETUP
# ==========================================
current_wavelength = 600
is_grayscale = False

fig = plt.figure(figsize=(9, 9), facecolor='#06080d')

# Full-bleed camera viewport
ax_cam = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax_cam.axis('off')

initial_frame = calculate_reflectance(current_wavelength)
cam_display = ax_cam.imshow(initial_frame, cmap='magma', vmin=0.30, vmax=0.32)

# --- TOP-RIGHT HUD: DEFECT MAP ---
ax_map = fig.add_axes([0.76, 0.74, 0.21, 0.21])
ax_map.imshow(log_N_ground_truth, cmap='viridis')
ax_map.set_title("DOPING CONC. [LOG10]", fontsize=7.5, color='#00ffcc', pad=4, family="monospace", weight="bold")
ax_map.axis('off')

for spine in ax_map.spines.values():
    spine.set_edgecolor('#00ffcc')
    spine.set_linewidth(1.4)
    spine.set_visible(True)

# --- TOP-RIGHT HUD: METER ---
ax_meter = fig.add_axes([0.76, 0.61, 0.21, 0.11])
ax_meter.set_facecolor('#05070a')
ax_meter.axis('off')

for spine in ax_meter.spines.values():
    spine.set_edgecolor('#00ffcc')
    spine.set_linewidth(1.4)
    spine.set_visible(True)

meter_text = ax_meter.text(
    0.07, 0.50, "", transform=ax_meter.transAxes,
    color="#00ffcc", fontsize=7.8, family="monospace", weight="bold",
    va='center'
)

def update_scene():
    global current_wavelength, is_grayscale
    frame = calculate_reflectance(current_wavelength)
    
    # Auto-Gain dynamic scaling (brings out live sensor grit & punchy defect contrast)
    if current_wavelength < 750:
        band = "VISIBLE [DEFECTS HIDDEN]"
        cam_display.set_clim(0.30, 0.32)
    elif current_wavelength < 1300:
        band = "NIR     [EMERGING CONTRAST]"
        cam_display.set_clim(np.percentile(frame, 1), np.percentile(frame, 99))
    else:
        band = "SWIR    [METROLOGY CONTRAST]"
        cam_display.set_clim(np.percentile(frame, 1), np.percentile(frame, 99))

    cam_display.set_data(frame)
    cam_display.set_cmap('gray' if is_grayscale else 'magma')

    mode_str = "RAW GRAYSCALE" if is_grayscale else "PSEUDOCOLOR"
    meter_text.set_text(
        f"WAVELENGTH : {current_wavelength:4d} nm\n"
        f"SPECTRUM   : {band}\n"
        f"COLOR MODE : {mode_str}\n"
        f"KEYS       : [<-][->] | [G]"
    )
    fig.canvas.draw_idle()

def on_key(event):
    global current_wavelength, is_grayscale
    if event.key in ['right', 'up']:
        current_wavelength = min(current_wavelength + 50, 2450)
        update_scene()
    elif event.key in ['left', 'down']:
        current_wavelength = max(current_wavelength - 50, 450)
        update_scene()
    elif event.key in ['g', 'G']:
        is_grayscale = not is_grayscale
        update_scene()

fig.canvas.mpl_connect('key_press_event', on_key)

update_scene()
plt.show()