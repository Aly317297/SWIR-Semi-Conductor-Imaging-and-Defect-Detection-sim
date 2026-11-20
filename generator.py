import os
import random
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image

# ==========================================
# 1. FIXED PHYSICAL CONSTANTS (DRUDE MODEL)
# ==========================================
EPS_0 = 8.854e-12
EPS_INF = 11.7
E_CHARGE = 1.602e-19
M_EFF = 0.26 * 9.109e-31
C_SPEED = 2.998e8
GAMMA = 3.5e13

GRID_SIZE = 1000
WAVELENGTH_NM = 2450
WAVELENGTH_M = WAVELENGTH_NM * 1e-9
OMEGA = 2.0 * torch.pi * C_SPEED / WAVELENGTH_M
DENOM = OMEGA**2 + GAMMA**2

# Precomputed in float64 with N scaled by 1e25 so float32 stays exact
SCALE_FACTOR = (1e25 * (E_CHARGE**2)) / (EPS_0 * EPS_INF * M_EFF * DENOM)
GAMMA_OVER_OMEGA = GAMMA / OMEGA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coords = torch.arange(GRID_SIZE, device=device, dtype=torch.float32)
Y = coords.view(-1, 1)
X = coords.view(1, -1)
QUANTILES = torch.tensor([0.005, 0.995], device=device, dtype=torch.float32)

# ==========================================
# 2. GPU-ACCELERATED DEFECT ENGINE
# ==========================================
@torch.no_grad()
def generate_random_sample():
    # Work in units of 1e25 cm^-3 (so 4.0e25 -> 4.0)
    N_nominal = random.uniform(4.0, 5.5)
    N_map = torch.full((GRID_SIZE, GRID_SIZE), N_nominal, device=device, dtype=torch.float32)
    defect_mask = torch.zeros((GRID_SIZE, GRID_SIZE), device=device, dtype=torch.bool)

    # 2. Thermal Anneal Gradient (75% chance)
    if random.random() < 0.75:
        theta = random.uniform(0, 2 * torch.pi)
        dist_offset = random.uniform(3000, 5000)
        cx = 500.0 + dist_offset * float(torch.cos(torch.tensor(theta)))
        cy = 500.0 + dist_offset * float(torch.sin(torch.tensor(theta)))

        dist = torch.sqrt((X - cx)**2 + (Y - cy)**2)
        period = random.uniform(60, 100)
        width = random.uniform(400, 700)
        amp = random.uniform(1.8, 2.6)

        thermal_arc = torch.sin(dist / period) * torch.exp(-((dist - dist_offset) / width)**2) * amp
        N_map += thermal_arc
        defect_mask |= (thermal_arc.abs() > (0.28 * N_nominal))

    # 3. Crystal Slip Lines (0 to 3)
    num_slips = random.choices([0, 1, 2, 3], weights=[0.15, 0.45, 0.3, 0.1])[0]
    for _ in range(num_slips):
        direction = 1.0 if random.random() < 0.5 else -1.0
        offset = random.uniform(-400, 400)

        slip_dist = torch.abs((X - 500.0) - direction * (Y - 500.0) + offset)
        slip_intensity = random.uniform(3.5, 4.8)

        x_start = random.uniform(50, 350)
        x_end = random.uniform(x_start + 250, 950)
        span = (X >= x_start) & (X <= x_end)

        slip = torch.exp(-(slip_dist**2) / random.uniform(2.5, 4.0)) * slip_intensity * span
        N_map += slip
        defect_mask |= (slip > (0.25 * N_nominal))

    # 4. Particle Shadow Voids (0 to 4)
    num_particles = random.choices([0, 1, 2, 3, 4], weights=[0.15, 0.35, 0.3, 0.15, 0.05])[0]
    for _ in range(num_particles):
        px = float(random.randint(150, 850))
        py = float(random.randint(150, 850))
        p_rad = random.uniform(10, 22)
        wake_len = random.uniform(60, 140)
        tilt_dir = float(random.choice([1, -1]))

        dx = X - px
        dy = Y - py
        part_dist = torch.sqrt(dx**2 + dy**2)
        wake_span = (tilt_dir * dx >= 0.0) & (tilt_dir * dx <= wake_len)
        wake = (
            torch.exp(-(dy**2) / random.uniform(25.0, 45.0))
            * wake_span
            * torch.exp(-tilt_dir * dx / 40.0)
        )
        particle_defect = (part_dist < p_rad).float() + wake
        N_map -= particle_defect * random.uniform(3.8, 5.0)
        defect_mask |= (particle_defect > 0.18)

    # 5. Implanter Scan Striations
    scan_angle = random.uniform(0.1, 0.3)
    scan_freq = random.uniform(8.0, 14.0)
    scan_amp = random.uniform(0.2, 0.45)
    N_map += torch.sin((X + scan_angle * Y) / scan_freq) * scan_amp

    N_map = torch.clamp(N_map, 2e-4, 10.0)

    # ==========================================
    # 3. GPU DRUDE REFLECTANCE
    # ==========================================
    term = N_map * SCALE_FACTOR
    eps_1 = EPS_INF - term
    eps_2 = term * GAMMA_OVER_OMEGA

    mag_eps = torch.sqrt(eps_1**2 + eps_2**2)
    n = torch.sqrt(torch.clamp((mag_eps + eps_1) * 0.5, min=0.0))
    k = torch.sqrt(torch.clamp((mag_eps - eps_1) * 0.5, min=0.0))

    R = ((n - 1.0)**2 + k**2) / ((n + 1.0)**2 + k**2)

    noise_std = random.uniform(0.0014, 0.0020)
    R_noisy = torch.clamp(R + torch.randn_like(R) * noise_std, 0.0, 1.0)

    # ==========================================
    # 4. CONTRAST NORMALIZATION
    # ==========================================
    qs = torch.quantile(R_noisy[::2, ::2].reshape(-1), QUANTILES)
    p_low, p_high = qs[0], qs[1]

    cam_u8 = torch.clamp((R_noisy - p_low) / (p_high - p_low + 1e-8) * 255.0, 0.0, 255.0).to(torch.uint8)
    mask_u8 = defect_mask.to(torch.uint8) * 255

    return cam_u8.cpu().numpy(), mask_u8.cpu().numpy()

def save_pair(idx, cam_img, mask_img, img_dir, mask_dir):
    sample_name = f"sample_{idx:05d}.png"
    # compress_level=1 is ~8x faster than optimize=True on noisy images
    Image.fromarray(cam_img, mode='L').save(os.path.join(img_dir, sample_name), compress_level=1)
    Image.fromarray(mask_img, mode='L').save(os.path.join(mask_dir, sample_name), compress_level=1)

# ==========================================
# 5. BATCH GENERATOR EXECUTION
# ==========================================
def run_generator():
    user_input = input("How many wafer inspection samples do you want to generate? ")
    try:
        count = int(user_input)
        if count <= 0:
            raise ValueError
    except ValueError:
        print("Invalid number. Please enter a positive integer.")
        return

    output_dir = "dataset"
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    workers = min(8, os.cpu_count() or 4)
    print(f"\n[+] Generating {count} SWIR samples on {device} using {workers} PNG saver threads...")

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(1, count + 1):
            cam_img, mask_img = generate_random_sample()
            futures.append(executor.submit(save_pair, i, cam_img, mask_img, img_dir, mask_dir))

            # Bound in-flight memory to ~32 images
            if len(futures) >= 32:
                futures.pop(0).result()

            if i % 100 == 0 or i == count:
                print(f" -> Generated [{i}/{count}] samples")

        for f in futures:
            f.result()

    print("\n[DONE] Generation complete!")

if __name__ == "__main__":
    run_generator()