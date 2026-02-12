import os
import sys
import argparse
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import numpy.fft as fft
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt


def load_image(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    # keep imanmhjge sizes bounded to avoid extremely large images slowing feature extraction
    max_side = 1024
    h, w = arr.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale != 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def rgb_to_gray(img):
    # img: HxWx3 in [0,1]
    return np.clip(0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2], 0.0, 1.0)


def extract_residual(gray, sigma=1.5):
    blurred = gaussian_filter(gray, sigma=sigma)
    residual = gray - blurred
    # normalize
    m = residual.mean()
    s = residual.std() + 1e-12
    residual = (residual - m) / s
    return residual


def fft_stats(gray):
    # compute log-magnitude spectrum and a simple high-frequency ratio
    H, W = gray.shape
    F = fft.fft2(gray)
    Fshift = fft.fftshift(F)
    Fmag = np.log1p(np.abs(Fshift))
    # radial profile -> compute high-frequency energy ratio
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_norm = R / R.max()
    hf_mask = r_norm > 0.5
    hf_ratio = Fmag[hf_mask].sum() / (Fmag.sum() + 1e-12)
    return Fmag, float(hf_ratio)


def lbp_entropy(patch_gray, P=8, R=1):
    # local_binary_pattern works more predictably on integer images (uint8).
    # Convert floating gray [0,1] to uint8 0..255 to avoid numerical issues and speed up histogramming.
    img_uint8 = (np.clip(patch_gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, P=P, R=R, method='uniform')
    # histogram using bincount for speed (lbp values are small integers for 'uniform' method)
    lbp_flat = lbp.ravel().astype(np.int32)
    n_bins = int(lbp_flat.max() + 1)
    if n_bins <= 0:
        return 0.0
    counts = np.bincount(lbp_flat, minlength=n_bins).astype(np.float32)
    probs = counts / (counts.sum() + 1e-12)
    probs = probs + 1e-12
    ent = -np.sum(probs * np.log(probs))
    return float(ent)


def sliding_patch_scores(img_rgb, patch_size=128, stride=64):
    H, W, _ = img_rgb.shape
    gray = rgb_to_gray(img_rgb)
    residual_full = extract_residual(gray)
    Fmag_full, _ = fft_stats(gray)

    scores = []
    coords = []
    for y in range(0, max(1, H - patch_size + 1), stride):
        for x in range(0, max(1, W - patch_size + 1), stride):
            patch = img_rgb[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                # pad
                ph = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                ph[:patch.shape[0], :patch.shape[1]] = patch
                patch = ph
            pg = rgb_to_gray(patch)
            pres = extract_residual(pg)
            # residual score: std (higher natural noise -> more likely real)
            residual_score = np.clip(np.std(pres), 0.0, 10.0)
            # frequency score using patch FFT
            _, hf = fft_stats(pg)
            # texture score: lower entropy -> more repetitive -> more likely AI
            ent = lbp_entropy(pg)
            # convert ent to an anomaly score (lower ent => higher AI-likeliness)
            texture_score = -ent

            scores.append((residual_score, hf, texture_score))
            coords.append((y, x))

    # Convert to arrays
    arr = np.array(scores)  # N x 3
    # Normalize each column to 0..1
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = (maxs - mins) + 1e-12
    norm = (arr - mins) / ranges

    # Map signals to an AI-likelihood per patch:
    # residual: higher std => more natural => lower AI score, so invert it
    residual_inv = 1.0 - norm[:, 0]
    freq = norm[:, 1]  # higher hf_ratio -> more natural -> invert
    freq_inv = 1.0 - freq
    texture = (norm[:, 2] - norm[:, 2].min()) / (norm[:, 2].max() - norm[:, 2].min() + 1e-12)
    # texture here was negative ent; larger -> more AI-like, so keep as-is

    # Weighted fusion
    w_res, w_freq, w_tex = 0.5, 0.25, 0.25
    patch_ai = w_res * residual_inv + w_freq * freq_inv + w_tex * texture

    return patch_ai, coords, (H, W), patch_size, stride


def reconstruct_heatmap(patch_scores, coords, image_shape, patch_size, stride):
    H, W = image_shape
    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for s, (y, x) in zip(patch_scores, coords):
        y2 = min(H, y + patch_size)
        x2 = min(W, x + patch_size)
        h = y2 - y
        w = x2 - x
        heat[y:y2, x:x2] += s
        count[y:y2, x:x2] += 1.0
    count[count == 0] = 1.0
    heat = heat / count
    # smooth
    heat = gaussian_filter(heat, sigma=patch_size / 4.0)
    # normalize
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
    return heat


def overlay_and_save(orig_rgb, heatmap, out_path, alpha=0.5, cmap='jet'):
    plt.figure(figsize=(8, 8))
    plt.imshow(orig_rgb)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_image(path, out_dir=None, patch_size=128, stride=64):
    img = load_image(path)
    patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=patch_size, stride=stride)
    heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
    ai_score = float(np.mean(heat))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fname = Path(path).stem + '_heat.png'
        out_path = os.path.join(out_dir, fname)
        overlay_and_save(np.clip(img, 0, 1), heat, out_path)
    return {'ai_score': ai_score, 'heatmap': heat}


def scan_dataset(dataset_path, out_dir, max_images=None, **kwargs):
    patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.bmp']
    p = Path(dataset_path)
    files = []
    for pat in patterns:
        files.extend(p.glob(pat))
    files = [str(x) for x in sorted(files)]
    if max_images:
        files = files[:max_images]
    results = []
    for i, f in enumerate(files):
        try:
            res = process_image(f, out_dir=out_dir, **kwargs)
            results.append((f, res))
        except Exception as e:
            print(f'Failed {f}: {e}', file=sys.stderr)
    return results


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='path to single image')
    parser.add_argument('--dataset', type=str, help='path to dataset folder')
    parser.add_argument('--out_dir', type=str, default='out', help='where to write overlays')
    parser.add_argument('--max_images', type=int, default=200, help='max images to process when scanning dataset')
    parser.add_argument('--patch', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    args = parser.parse_args()

    if args.image:
        res = process_image(args.image, out_dir=args.out_dir, patch_size=args.patch, stride=args.stride)
        print(f"ai_score: {res['ai_score']:.4f}")
    elif args.dataset:
        print('Scanning dataset, this may take a while...')
        res = scan_dataset(args.dataset, out_dir=args.out_dir, max_images=args.max_images, patch_size=args.patch, stride=args.stride)
        print(f'Processed {len(res)} images, overlays in {args.out_dir}')
    else:
        parser.print_help()


if __name__ == '__main__':
    cli()
