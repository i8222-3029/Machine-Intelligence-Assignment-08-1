import numpy as np
import os

IMG_SIZE = 64
SHELF_THICKNESS = 5
BOX_REGION_TOP = 4
BG_BRIGHTNESS = 0.12
SHELF_BRIGHTNESS = 0.75

def _draw_shelf(img, shelf_top, rng):
    shelf_bottom = shelf_top + SHELF_THICKNESS
    img[shelf_top:shelf_bottom, :] = SHELF_BRIGHTNESS
    n_dividers = rng.integers(2, 5)
    for _ in range(n_dividers):
        x = rng.integers(8, IMG_SIZE - 8)
        img[shelf_top:shelf_bottom, x] = rng.uniform(0.25, 0.40)
    return img

def _draw_boxes(img, shelf_top, n_boxes, max_height, rng):
    for _ in range(n_boxes):
        bw = rng.integers(4, 14)
        bh = rng.integers(4, max_height + 1)
        bx = rng.integers(1, IMG_SIZE - bw - 1)
        by_bottom = shelf_top
        by_top = max(BOX_REGION_TOP, by_bottom - bh)
        brightness = rng.uniform(0.30, 0.55)
        img[by_top:by_bottom, bx:bx + bw] = brightness
        border = brightness * 0.65
        img[by_top, bx:bx + bw] = border
        img[by_top:by_bottom, bx] = border
        img[by_top:by_bottom, min(IMG_SIZE - 1, bx + bw - 1)] = border
    return img

def _draw_crack(img, shelf_top, rng):
    shelf_bottom = shelf_top + SHELF_THICKNESS
    x0 = rng.integers(2, IMG_SIZE // 2)
    y0 = shelf_top + rng.integers(1, SHELF_THICKNESS - 1)
    angle = rng.uniform(-0.3, 0.3)
    length = rng.integers(15, 45)
    for t in range(length):
        x = int(x0 + t * np.cos(angle))
        y = int(y0 + t * np.sin(angle))
        if 0 <= x < IMG_SIZE and shelf_top <= y < shelf_bottom:
            img[y, x] = rng.uniform(0.0, 0.15)
            if y + 1 < shelf_bottom:
                img[y + 1, x] = rng.uniform(0.0, 0.20)
    return img

def _box_area_fraction(img, shelf_top):
    region = img[BOX_REGION_TOP:shelf_top, :]
    return (region > 0.25).mean()

def _random_shelf_top(rng):
    return 48 + rng.integers(-4, 5)

def _add_noise(img, rng):
    img = img + rng.uniform(-0.04, 0.04)
    img = img + rng.normal(0, 0.04, img.shape)
    return np.clip(img, 0, 1)

def generate_normal(rng):
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 5), max_height=15, rng=rng)
    attempts = 0
    while _box_area_fraction(img, shelf_top) > 0.25 and attempts < 10:
        img[BOX_REGION_TOP:shelf_top, :] = BG_BRIGHTNESS
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=12, rng=rng)
        attempts += 1
    return _add_noise(img, rng)

def generate_damaged(rng):
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 5), max_height=15, rng=rng)
    attempts = 0
    while _box_area_fraction(img, shelf_top) > 0.25 and attempts < 10:
        img[BOX_REGION_TOP:shelf_top, :] = BG_BRIGHTNESS
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=12, rng=rng)
        attempts += 1
    n_cracks = rng.integers(1, 3)
    for _ in range(n_cracks):
        _draw_crack(img, shelf_top, rng)
    return _add_noise(img, rng)

def generate_overloaded(rng):
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(6, 10), max_height=35, rng=rng)
    attempts = 0
    while _box_area_fraction(img, shelf_top) < 0.45 and attempts < 20:
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=30, rng=rng)
        attempts += 1
    return _add_noise(img, rng)

def generate_dataset(n_per_class=300, seed=42):
    rng = np.random.default_rng(seed)
    generators = [generate_normal, generate_damaged, generate_overloaded]
    class_names = ["normal", "damaged", "overloaded"]
    images = []
    labels = []
    for class_idx, gen in enumerate(generators):
        for _ in range(n_per_class):
            img = gen(rng)
            images.append(img)
            labels.append(class_idx)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    idx = rng.permutation(len(images))
    images = images[idx]
    labels = labels[idx]
    return images, labels, class_names

if __name__ == '__main__':
    images, labels, class_names = generate_dataset(n_per_class=300, seed=42)
    print(f"Images: {images.shape} (min={images.min():.2f}, max={images.max():.2f})")
    print(f"Labels: {labels.shape}, classes: {class_names}")
    print(f"Class distribution: {[int((labels == i).sum()) for i in range(3)]}")
    np.savez(
        "shelf_images.npz",
        images=images, labels=labels, class_names=class_names,
    )
    print("Saved shelf_images.npz")
