"""Generate Stark's PWA icon set into `crates/stark-ui/public/`.

The mark is two strokes crossing — a warm yellow over a cool blue — swept with
Stark's own bundled bristle stamp, and mixing to green where they meet. That is
the app's headline claim (real pigment mixing, §6.7) drawn with its own material.

Run by hand (`python tools/make-icons.py`, needs Pillow + numpy); the PNGs are
checked in. Kept out of build.rs on purpose — a logo is not a build product, and
nothing downstream keys off its bytes, unlike the assets `stark-assetid` hashes.
"""

import math
import os

import numpy as np
from PIL import Image

CRATE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = os.path.join(CRATE, "assets", "shape", "Worn_Bristles.png")
OUT = os.path.join(CRATE, "public")

SS = 4          # supersample factor
N = 1024        # art size after the supersample is resolved
W = N * SS

GROUND = (243, 240, 233)      # off-white gesso
YELLOW = (233, 176, 46)
BLUE = (44, 96, 168)
GREEN = (84, 128, 52)         # what the two make where they cross

# Cool stroke low-left to upper-right; warm stroke crossing it, mirrored in y.
BLUE_PATH = ((0.13, 0.79), (0.38, 0.88), (0.62, 0.12), (0.87, 0.21))
YELLOW_PATH = ((0.13, 0.21), (0.38, 0.12), (0.62, 0.88), (0.87, 0.79))

stamp = np.array(Image.open(STAMP))[..., 1].astype(np.float32) / 255.0
stamp_img = Image.fromarray((stamp * 255).astype(np.uint8), mode="L")


def bezier(p0, p1, p2, p3, t):
    u = 1 - t
    x = u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]
    return x, y


def sweep(pts, r, seed, scale):
    """Accumulate coverage of the bristle stamp swept along a cubic bezier."""
    pts = tuple((0.5 + (p[0] - 0.5) * scale, 0.5 + (p[1] - 0.5) * scale) for p in pts)
    cov = Image.new("L", (W, W), 0)
    rng = np.random.default_rng(seed)
    steps = 1100
    r0, r1 = r[0] * W * scale, r[1] * W * scale
    for i in range(steps):
        t = i / (steps - 1)
        x, y = bezier(*pts, t)
        # tapered nib: a point at the entry, full through the body, a point at the exit
        rad = (r0 + (r1 - r0) * t) * math.sin(math.pi * t) ** 0.32
        if rad < 2:
            continue
        # tangent, so the flat of the nib follows the path
        dx, dy = bezier(*pts, min(t + 1e-3, 1.0))
        ang = math.degrees(math.atan2(dy - y, dx - x))
        size = max(2, int(2 * rad))
        s = stamp_img.resize((size, size), Image.LANCZOS).rotate(
            -ang + rng.uniform(-2, 2), resample=Image.BICUBIC, expand=False
        )
        px = int(x * W - size / 2 + rng.uniform(-1, 1) * SS)
        py = int(y * W - size / 2 + rng.uniform(-1, 1) * SS)
        # max-combine: one pass of paint, not a thousand overlapping ones
        region = cov.crop((px, py, px + size, py + size))
        cov.paste(Image.fromarray(np.maximum(np.array(region), np.array(s))), (px, py))
    return np.array(cov).astype(np.float32) / 255.0


def render(scale):
    blue = sweep(BLUE_PATH, (0.070, 0.056), 7, scale)
    yellow = sweep(YELLOW_PATH, (0.066, 0.054), 21, scale)

    art = np.ones((W, W, 3), np.float32) * (np.array(GROUND, np.float32) / 255.0)
    a = blue[..., None]
    art = art * (1 - a) + (np.array(BLUE, np.float32) / 255.0)[None, None, :] * a
    # The yellow goes on last, but over the blue it becomes green rather than
    # covering it — the one thing this mark is for.
    mix = (
        np.array(YELLOW, np.float32)[None, None, :] * (1 - blue[..., None])
        + np.array(GREEN, np.float32)[None, None, :] * blue[..., None]
    ) / 255.0
    a = yellow[..., None]
    art = art * (1 - a) + mix * a
    return Image.fromarray((np.clip(art, 0, 1) * 255).astype(np.uint8), "RGB").resize(
        (N, N), Image.LANCZOS
    )


os.makedirs(OUT, exist_ok=True)

# Full-bleed, for `purpose: "any"` — the platform draws it as-is.
any_icon = render(1.0)
any_icon.resize((512, 512), Image.LANCZOS).save(os.path.join(OUT, "icon-512.png"))
any_icon.resize((192, 192), Image.LANCZOS).save(os.path.join(OUT, "icon-192.png"))
any_icon.resize((180, 180), Image.LANCZOS).save(os.path.join(OUT, "apple-touch-icon.png"))
any_icon.resize((48, 48), Image.LANCZOS).save(
    os.path.join(OUT, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48)]
)

# `purpose: "maskable"`: the platform may crop to a circle, so the mark has to sit
# inside the 80%-diameter safe zone while the ground still runs to the edge.
render(0.66).resize((512, 512), Image.LANCZOS).save(os.path.join(OUT, "icon-maskable-512.png"))

for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(os.path.join(OUT, f)))
