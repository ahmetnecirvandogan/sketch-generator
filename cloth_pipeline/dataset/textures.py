"""Procedural fabric bitmaps saved under dataset/textures/."""

import os
import random

import numpy as np
from PIL import Image, ImageDraw

from cloth_pipeline.paths import TEXTURES_DIR

TEXTURE_SIZE = 512


def _rand_color():
    return tuple(random.randint(60, 240) for _ in range(3))


def _contrasting_pair():
    c1 = _rand_color()
    c2 = _rand_color()
    while abs(sum(c1) - sum(c2)) < 180:
        c2 = _rand_color()
    return c1, c2


def generate_solid_texture(size=TEXTURE_SIZE):
    color = _rand_color()
    img = Image.new("RGB", (size, size), color)
    return img, "solid", {"color": list(color)}


def generate_stripes_texture(size=TEXTURE_SIZE):
    c1, c2 = _contrasting_pair()
    orientation = random.choice(["horizontal", "vertical", "diagonal"])
    stripe_w = random.randint(8, 64)

    ys, xs = np.mgrid[0:size, 0:size]
    if orientation == "horizontal":
        mask = (ys // stripe_w) % 2 == 1
    elif orientation == "vertical":
        mask = (xs // stripe_w) % 2 == 1
    else:
        mask = ((xs + ys) // stripe_w) % 2 == 1

    arr = np.where(mask[..., None], np.array(c2, dtype=np.uint8), np.array(c1, dtype=np.uint8))
    img = Image.fromarray(arr)
    return img, "stripes", {
        "colors": [list(c1), list(c2)],
        "orientation": orientation,
        "width": stripe_w,
    }


def generate_checkerboard_texture(size=TEXTURE_SIZE):
    c1, c2 = _contrasting_pair()
    cell = random.choice([16, 32, 48, 64])
    ys, xs = np.mgrid[0:size, 0:size]
    mask = ((xs // cell) + (ys // cell)) % 2 == 1
    arr = np.where(mask[..., None], np.array(c2, dtype=np.uint8), np.array(c1, dtype=np.uint8))
    img = Image.fromarray(arr)
    return img, "checkerboard", {"colors": [list(c1), list(c2)], "cell_size": cell}


def generate_plaid_texture(size=TEXTURE_SIZE):
    c_base = np.array(_rand_color(), dtype=np.float32)
    c_h = np.array(_rand_color(), dtype=np.float32)
    c_v = np.array(_rand_color(), dtype=np.float32)
    stripe_w = random.randint(6, 30)
    spacing = random.randint(40, 100)
    alpha = 0.45

    arr = np.full((size, size, 3), c_base, dtype=np.float32)

    ys = np.arange(size)
    h_mask = (ys % spacing) < stripe_w
    arr[h_mask] = arr[h_mask] * (1 - alpha) + c_h * alpha

    xs = np.arange(size)
    v_mask = (xs % spacing) < stripe_w
    arr[:, v_mask] = arr[:, v_mask] * (1 - alpha) + c_v * alpha

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img, "plaid", {
        "colors": [
            c_base.astype(int).tolist(),
            c_h.astype(int).tolist(),
            c_v.astype(int).tolist(),
        ],
        "stripe_width": stripe_w,
        "spacing": spacing,
    }


def generate_polkadot_texture(size=TEXTURE_SIZE):
    c_bg, c_dot = _contrasting_pair()
    dot_r = random.randint(6, 20)
    spacing = random.randint(dot_r * 3, dot_r * 6)
    img = Image.new("RGB", (size, size), c_bg)
    draw = ImageDraw.Draw(img)

    stagger = random.choice([0, spacing // 2])
    for row_i, y in enumerate(range(-dot_r, size + spacing, spacing)):
        row_offset = stagger if row_i % 2 == 1 else 0
        for x in range(-dot_r, size + spacing, spacing):
            draw.ellipse(
                [
                    x + row_offset - dot_r,
                    y - dot_r,
                    x + row_offset + dot_r,
                    y + dot_r,
                ],
                fill=c_dot,
            )

    return img, "polkadot", {
        "colors": [list(c_bg), list(c_dot)],
        "dot_radius": dot_r,
        "spacing": spacing,
    }


def generate_herringbone_texture(size=TEXTURE_SIZE):
    c1, c2 = _contrasting_pair()
    block_w = random.randint(8, 24)
    block_h = block_w * 2

    ys, xs = np.mgrid[0:size, 0:size]
    bx = xs // block_w
    lx = xs % block_w
    ly = ys % block_h
    thresh = ly * block_w // block_h
    even = (bx + ys // block_h) % 2 == 0
    mask = np.where(even, lx < thresh, lx > thresh)

    arr = np.where(mask[..., None], np.array(c2, dtype=np.uint8), np.array(c1, dtype=np.uint8))
    img = Image.fromarray(arr)
    return img, "herringbone", {"colors": [list(c1), list(c2)], "block_width": block_w}


def generate_gradient_texture(size=TEXTURE_SIZE):
    c1 = np.array(_rand_color(), dtype=np.float32)
    c2 = np.array(_rand_color(), dtype=np.float32)
    direction = random.choice(["horizontal", "vertical", "radial"])

    if direction == "horizontal":
        t = np.linspace(0, 1, size, dtype=np.float32)
        arr = c1[None, :] * (1 - t[:, None]) + c2[None, :] * t[:, None]
        arr = np.broadcast_to(arr[None, :, :], (size, size, 3)).copy()
    elif direction == "vertical":
        t = np.linspace(0, 1, size, dtype=np.float32)
        arr = c1[None, :] * (1 - t[:, None]) + c2[None, :] * t[:, None]
        arr = np.broadcast_to(arr[:, None, :], (size, size, 3)).copy()
    else:
        ys, xs = np.mgrid[0:size, 0:size]
        cx_c, cy_c = size / 2, size / 2
        dist = np.sqrt((xs - cx_c) ** 2 + (ys - cy_c) ** 2)
        t = np.clip(dist / (np.sqrt(cx_c**2 + cy_c**2)), 0, 1).astype(np.float32)
        arr = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img, "gradient", {
        "colors": [c1.astype(int).tolist(), c2.astype(int).tolist()],
        "direction": direction,
    }


def generate_houndstooth_texture(size=TEXTURE_SIZE):
    c1, c2 = _contrasting_pair()
    cell = random.choice([16, 24, 32])

    ys, xs = np.mgrid[0:size, 0:size]
    cx_mod = xs % (cell * 2)
    cy_mod = ys % (cell * 2)
    in_check = (cx_mod < cell) ^ (cy_mod < cell)
    lx = cx_mod % cell
    ly = cy_mod % cell
    in_tooth = (lx + ly) < cell
    mask = in_check ^ in_tooth

    arr = np.where(mask[..., None], np.array(c2, dtype=np.uint8), np.array(c1, dtype=np.uint8))
    img = Image.fromarray(arr)
    return img, "houndstooth", {"colors": [list(c1), list(c2)], "cell_size": cell}


TEXTURE_GENERATORS = [
    generate_solid_texture,
    generate_stripes_texture,
    generate_checkerboard_texture,
    generate_plaid_texture,
    generate_polkadot_texture,
    generate_herringbone_texture,
    generate_gradient_texture,
    generate_houndstooth_texture,
]


def generate_random_texture(frame_str: str) -> tuple:
    gen_fn = random.choice(TEXTURE_GENERATORS)
    img, pattern_name, params = gen_fn()
    tex_path = os.path.join(TEXTURES_DIR, f"texture_{frame_str}.png")
    img.save(tex_path)
    return tex_path, pattern_name, params
