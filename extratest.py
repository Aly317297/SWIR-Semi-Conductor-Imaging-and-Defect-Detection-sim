import glob, torch
import numpy as np
from PIL import Image
from tester import DefectUNet, generate_live_sample, device  # close GUI if it pops up, or comment out plt.show()

IMG_SIZE = 768   # <-- MUST match what you trained with!
BASE_CH = 24      # <-- MUST match what you trained with!

model = DefectUNet(base_ch=BASE_CH).to(device)
model.load_state_dict(torch.load("defect_unet.pth", map_location=device, weights_only=True))
model.eval()

def test_pair(name, im_pil, mk_pil):
    im_r = im_pil.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    mk_r = mk_pil.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x_np = np.array(im_r, dtype=np.float32) / 255.0
    y_np = (np.array(mk_r) > 30).astype(np.float32)

    x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).cpu().squeeze().numpy()

    pred = (prob > 0.5).astype(np.float32)
    dice = (2 * (pred * y_np).sum() + 1e-6) / (pred.sum() + y_np.sum() + 1e-6)
    print(f"[{name}] Img mean/std: {x_np.mean():.3f}/{x_np.std():.3f} | GT%: {y_np.mean()*100:.1f}% "
          f"| Prob min/max/mean: {prob.min():.3f}/{prob.max():.3f}/{prob.mean():.3f} | Dice: {dice:.4f}")

# 1. Test on a saved dataset sample
im_d = Image.open(sorted(glob.glob("dataset/images/*.png"))[-1])
mk_d = Image.open(sorted(glob.glob("dataset/masks/*.png"))[-1])
test_pair("DATASET", im_d, mk_d)

# 2. Test on a live sample from tester.py
cam_u8, msk_u8 = generate_live_sample()
test_pair("TESTER ", Image.fromarray(cam_u8, "L"), Image.fromarray(msk_u8, "L"))