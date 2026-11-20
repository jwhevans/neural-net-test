"""Generate synthetic labelled shape images for training.

Writes 64x64 PNGs to <out>/{train,val}/<class>/NNNN.png so that
torchvision.datasets.ImageFolder can load them directly.
"""
import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle", "star", "rounded_square"]
SIZE = 64
# Shape span (circumscribed-circle diameter) as a fraction of image width.
MIN_SPAN, MAX_SPAN = 0.10, 1.00


def random_color(rng):
    return tuple(rng.randint(0, 255) for _ in range(3))


def contrasting_colors(rng):
    """Pick a background and foreground colour that differ enough to see."""
    while True:
        bg, fg = random_color(rng), random_color(rng)
        if sum(abs(a - b) for a, b in zip(bg, fg)) > 200:
            return bg, fg


def polygon(cx, cy, radius, n_points, rotation, inner_ratio=None):
    """Vertices of a regular polygon (or a star if inner_ratio is given)."""
    pts = []
    steps = n_points * 2 if inner_ratio else n_points
    for i in range(steps):
        r = radius * inner_ratio if (inner_ratio and i % 2) else radius
        a = rotation + 2 * math.pi * i / steps
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def transform(local_pts, cx, cy, rotation):
    """Rotate shape-local (x, y) points and move them to (cx, cy)."""
    c, s = math.cos(rotation), math.sin(rotation)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in local_pts]


def squircle(radius, rng, n_points=64):
    """Superellipse |x|^n + |y|^n = a^n, scaled to fit inside `radius`.

    n between 3.5 and 6 keeps it clearly between a circle (n=2) and a square.
    """
    n = rng.uniform(3.5, 6.0)
    # The farthest point is on the diagonal, at distance a * sqrt(2) * 2^(-1/n).
    a = radius / (math.sqrt(2) * 2 ** (-1 / n))
    pts = []
    for i in range(n_points):
        t = 2 * math.pi * i / n_points
        ct, st = math.cos(t), math.sin(t)
        pts.append((a * math.copysign(abs(ct) ** (2 / n), ct),
                    a * math.copysign(abs(st) ** (2 / n), st)))
    return pts


def rounded_rect(radius, rng, arc_points=12):
    """Square with circular corners, scaled to fit inside `radius`."""
    corner_frac = rng.uniform(0.2, 0.45)  # corner radius / half-side
    # Farthest point: (h - c) * sqrt(2) + c = radius, with c = corner_frac * h.
    h = radius / ((1 - corner_frac) * math.sqrt(2) + corner_frac)
    c = corner_frac * h
    pts = []
    for qx, qy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        ox, oy = qx * (h - c), qy * (h - c)  # corner arc centre
        for i in range(arc_points + 1):
            a = math.radians(start + 90 * i / arc_points)
            pts.append((ox + c * math.cos(a), oy + c * math.sin(a)))
    return pts


def draw_shape(shape, rng):
    bg, fg = contrasting_colors(rng)
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)

    radius = rng.uniform(MIN_SPAN, MAX_SPAN) * SIZE / 2
    cx = rng.uniform(radius, SIZE - radius)
    cy = rng.uniform(radius, SIZE - radius)
    rot = rng.uniform(0, 2 * math.pi)
    # Outlines on tiny shapes are just smudges, so keep those filled.
    filled = radius < 6 or rng.random() < 0.7
    width = rng.randint(1, max(1, min(4, int(radius / 5))))
    kw = {"fill": fg} if filled else {"outline": fg, "width": width}

    if shape == "circle":
        d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], **kw)
    else:
        if shape == "square":
            pts = polygon(cx, cy, radius, 4, rot)
        elif shape == "triangle":
            pts = polygon(cx, cy, radius, 3, rot)
        elif shape == "star":
            pts = polygon(cx, cy, radius, 5, rot, inner_ratio=0.45)
        else:  # rounded_square: half squircles, half rounded rectangles
            local = squircle(radius, rng) if rng.random() < 0.5 else rounded_rect(radius, rng)
            pts = transform(local, cx, cy, rot)
        if filled:
            d.polygon(pts, fill=fg)
        else:
            d.line(pts + [pts[0]], fill=fg, width=width, joint="curve")

    # Light pixel noise so the net can't rely on perfectly flat colours.
    px = img.load()
    for _ in range(rng.randint(0, 150)):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        px[x, y] = random_color(rng)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=2000, help="images per class")
    p.add_argument("--n-val", type=int, default=400, help="images per class")
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        for shape in CLASSES:
            folder = Path(args.out) / split / shape
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                draw_shape(shape, rng).save(folder / f"{i:04d}.png")
        print(f"{split}: {n} images x {len(CLASSES)} classes")


if __name__ == "__main__":
    main()
