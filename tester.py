import matplotlib
matplotlib.use('TkAgg')

import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. HARDWARE SELECTION (RTX 5060 CHECK)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running Inference on: {device}")
if device.type == "cuda":
    print(f"[*] Detected GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[!] Running on CPU. Run the CUDA pip install command to use your RTX 5060.")

# ==========================================
# 2. U-NET ARCHITECTURE
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        )
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))

class DefectUNet(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__()
        c = base_ch
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.dec4 = UpBlock(c * 16, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2)
        self.dec1 = UpBlock(c * 2, c)
        self.final = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.final(d1)

# Load Trained Weights (auto-detects base_ch from checkpoint)
try:
    state_dict = torch.load("defect_unet.pth", map_location=device, weights_only=True)
    detected_base_ch = state_dict["enc1.conv.0.weight"].shape[0]
    model = DefectUNet(base_ch=detected_base_ch).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[+] Loaded defect_unet.pth (base_ch={detected_base_ch})!")
except FileNotFoundError:
    print("[ERROR] defect_unet.pth not found! Run your training script first.")
    exit()

# ==========================================
# 3. ON-THE-FLY PHYSICS GENERATOR
# ==========================================
EPS_0, EPS_INF = 8.854e-12, 11.7
E_CHARGE = 1.602e-19
M_EFF = 0.26 * 9.109e-31
C_SPEED = 2.998e8
GAMMA = 3.5e13
GRID_SIZE = 1000
IMG_SIZE = 768  # Matches training resolution (must be divisible by 16)

WAVELENGTH_M = 2450 * 1e-9
OMEGA = 2.0 * np.pi * C_SPEED / WAVELENGTH_M
DENOM = OMEGA**2 + GAMMA**2
Y, X = np.ogrid[:GRID_SIZE, :GRID_SIZE]

transform_to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def generate_live_sample():
    N_nominal = random.uniform(4.0e25, 5.5e25)
    N_map = np.full((GRID_SIZE, GRID_SIZE), N_nominal, dtype=np.float64)
    defect_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    # Thermal Arc
    if random.random() < 0.75:
        theta = random.uniform(0, 2 * np.pi)
        dist_offset = random.uniform(3000, 5000)
        cx = 500 + dist_offset * np.cos(theta)
        cy = 500 + dist_offset * np.sin(theta)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        arc = np.sin(dist / random.uniform(60, 100)) * np.exp(-((dist - dist_offset)/random.uniform(400, 700))**2) * random.uniform(1.8e25, 2.6e25)
        N_map += arc
        defect_mask |= (np.abs(arc) > 0.28 * N_nominal)

    # Slip Lines
    for _ in range(random.choices([0, 1, 2, 3], weights=[0.15, 0.45, 0.3, 0.1])[0]):
        direction = 1 if random.random() < 0.5 else -1
        slip_dist = np.abs((X - 500) - direction * (Y - 500) + random.uniform(-400, 400))
        x_start = random.uniform(50, 350)
        span = (X >= x_start) & (X <= random.uniform(x_start + 250, 950))
        slip = np.exp(-(slip_dist**2) / random.uniform(2.5, 4.0)) * random.uniform(3.5e25, 4.8e25) * span
        N_map += slip
        defect_mask |= (slip > 0.25 * N_nominal)

    # Particle Voids
    for _ in range(random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]):
        px, py = random.randint(150, 850), random.randint(150, 850)
        tilt_dir = random.choice([1, -1])
        part_dist = np.sqrt((X - px)**2 + (Y - py)**2)
        wake = np.exp(-((Y - py)**2) / 30.0) * (tilt_dir * (X - px) >= 0) * (tilt_dir * (X - px) <= random.uniform(60, 140)) * np.exp(-tilt_dir * (X - px) / 40.0)
        part = ((part_dist < random.uniform(10, 22)).astype(np.float64) + wake)
        N_map -= part * random.uniform(3.8e25, 5.0e25)
        defect_mask |= (part > 0.18)

    # Implanter scan striations
    N_map += np.sin((X + 0.25 * Y) / random.uniform(8.0, 14.0)) * random.uniform(0.2e25, 0.4e25)
    N_map = np.clip(N_map, 2e21, 1e26)

    # Drude Solver
    omega_p_sq = (N_map * (E_CHARGE**2)) / (EPS_0 * EPS_INF * M_EFF)
    eps_1 = EPS_INF - (omega_p_sq / DENOM)
    eps_2 = (omega_p_sq * GAMMA) / (OMEGA * DENOM)
    mag_eps = np.sqrt(eps_1**2 + eps_2**2)
    n = np.sqrt(np.maximum((mag_eps + eps_1) / 2.0, 0))
    k = np.sqrt(np.maximum((mag_eps - eps_1) / 2.0, 0))
    R = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)
    R_noisy = R + np.random.normal(0, random.uniform(0.0014, 0.0020), (GRID_SIZE, GRID_SIZE))

    p_low, p_high = np.percentile(R_noisy, 0.5), np.percentile(R_noisy, 99.5)
    cam_uint8 = np.clip((R_noisy - p_low) / (p_high - p_low + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    mask_uint8 = (defect_mask.astype(np.uint8)) * 255

    return cam_uint8, mask_uint8

# ==========================================
# 4. INTERACTIVE LIVE GUI
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor='#06080d')
plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08)

for ax in axes:
    ax.axis('off')

fig.suptitle("WAFER DEFECT METROLOGY AI TESTER  |  Press [SPACE] or [R] to Reroll",
             color='#00ffcc', fontsize=12, family='monospace', weight='bold')

im_cam = axes[0].imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='gray', vmin=0, vmax=1)
axes[0].set_title("1. LIVE CAMERA SENSOR FEED", color='#00ffcc', fontsize=9.5, family='monospace')

im_gt = axes[1].imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='gray', vmin=0, vmax=1)
axes[1].set_title("2. GROUND TRUTH ANOMALY MASK", color='#00ffcc', fontsize=9.5, family='monospace')

im_pred = axes[2].imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='magma', vmin=0, vmax=1)
axes[2].set_title("3. U-NET AI PREDICTION", color='#00ffcc', fontsize=9.5, family='monospace')

def reroll():
    cam_u8, mask_u8 = generate_live_sample()

    pil_img = Image.fromarray(cam_u8, mode='L')
    pil_mask = Image.fromarray(mask_u8, mode='L')

    tensor_img = transform_to_tensor(pil_img).unsqueeze(0).to(device)
    gt_binary = (transform_to_tensor(pil_mask).squeeze().numpy() > 0.1).astype(np.float32)

    with torch.no_grad():
        logits = model(tensor_img)
        pred_prob = torch.sigmoid(logits).cpu().squeeze().numpy()
        pred_binary = (pred_prob > 0.5).astype(np.float32)

    inter = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    live_dice = (2.0 * inter + 1e-6) / (union + 1e-6)

    im_cam.set_data(tensor_img.cpu().squeeze().numpy())
    im_gt.set_data(gt_binary)
    im_pred.set_data(pred_binary)
    axes[2].set_title(f"3. U-NET AI PREDICTION (Dice: {live_dice:.3f})", color='#00ffcc', fontsize=9.5, family='monospace')
    fig.canvas.draw_idle()

def on_key(event):
    if event.key in [' ', 'r', 'R']:
        reroll()

fig.canvas.mpl_connect('key_press_event', on_key)

print("[READY] Tester started! Press [SPACE] or [R] inside the window to reroll samples.")
reroll()
plt.show()