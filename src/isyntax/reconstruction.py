import os
import re
import struct
from typing import Dict, Tuple, List

import numpy as np

TARGET_SCALE = 3
TARGET_TX = 10
TARGET_TY = 10
TARGET_COLOR = 0


BIN_RE = re.compile(
    r"isy_s(?P<scale>\d+)_tx(?P<tx>\d+)_ty(?P<ty>\d+)"
    r"_r(?P<r>\d+)_(?P<band>LL|HL|LH|HH)"
    r"_c(?P<c>\d+)_x0_(?P<x>\d+)_y0_(?P<y>\d+)\.bin"
)



def read_isy_cblk_dump(path: str) -> tuple[int,int,int,int,np.ndarray]:
    with open(path, "rb") as f:
        x0 = y0 = w = h = None
        fmt = None

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: EOF before DATA_BEGIN")

            s = line.decode("ascii", errors="strict").strip()

            if s.startswith("x0="):
                x0 = int(s[3:])
            elif s.startswith("y0="):
                y0 = int(s[3:])
            elif s.startswith("w="):
                w = int(s[2:])
            elif s.startswith("h="):
                h = int(s[2:])
            elif s.startswith("format="):
                fmt = s[len("format="):]
            elif s == "DATA_BEGIN":
                break

        if w is None or h is None or w <= 0 or h <= 0:
            raise ValueError(f"{path}: missing/invalid w,h in header")

        n = w * h
        payload = f.read()  # read the rest

        if fmt == "int16_rowmajor":
            expected = n * 2
            if len(payload) != expected:
                raise ValueError(f"{path}: short payload {len(payload)} bytes, expected {expected}")
            data = np.frombuffer(payload, dtype="<i2").astype(np.int32).reshape((h, w))

        elif fmt in (None, "int32_rowmajor"):
            expected = n * 4
            if len(payload) != expected:
                raise ValueError(f"{path}: short payload {len(payload)} bytes, expected {expected}")
            data = np.frombuffer(payload, dtype="<i4").reshape((h, w))

        else:
            raise ValueError(f"{path}: unknown format={fmt}")

        return x0 or 0, y0 or 0, w, h, data

def load_full_plane(path: str) -> np.ndarray:
    x0, y0, w, h, data = read_isy_cblk_dump(path)
    return data  # int32 already

def load_band_image(
    base_dir: str,
    r: int,
    band: str,
    c: int,
    band_w: int,
    band_h: int,
    block_w: int = 64,
    block_h: int = 64,
) -> np.ndarray:
    img = np.zeros((band_h, band_w), dtype=np.int32)

    # For speed, list once and filter by regex
    for name in os.listdir(base_dir):
        m = BIN_RE.match(name)
        if not m:
            continue

        # filter tile identity first
        if int(m.group("scale")) != TARGET_SCALE:
            continue
        if int(m.group("tx")) != TARGET_TX:
            continue
        if int(m.group("ty")) != TARGET_TY:
            continue

        # then filter which band you're loading
        if int(m.group("r")) != r:
            continue
        if m.group("band") != band:
            continue
        if int(m.group("c")) != c:
            continue

        bx = int(m.group("x"))
        by = int(m.group("y"))

        path = os.path.join(base_dir, name)
        x0, y0, w, h, blk = read_isy_cblk_dump(path)

        if bx + block_w > band_w or by + block_h > band_h:
            raise ValueError(
                f"{name}: block at ({bx},{by}) overflows band ({band_w}x{band_h})"
            )

        img[by : by + block_h, bx : bx + block_w] = blk

    return img


def idwt53_1d_cas1_inplace(a: np.ndarray) -> None:
    """
    Inverse 5/3 lifting for cas=1 (first sample at odd coordinate).
    This is a direct translation of opj_idwt53_h_cas1 from libisyntax.
    
    Input layout: [H (dn samples) | L (sn samples)]
    where dn = n//2, sn = (n+1)//2 (for equal subbands, dn == sn)
    """
    n = a.shape[0]
    if n < 2:
        return
    
    sn = (n + 1) // 2  # number of low-pass (even) samples
    dn = n // 2       # number of high-pass (odd) samples
    
    # cas=1 layout: [H | L] i.e. odd samples first, then even
    in_odd = a[:dn].copy()    # H samples at start
    in_even = a[dn:].copy()   # L samples after
    
    tmp = np.zeros(n, dtype=np.int32)
    
    if n == 2:
        # Special case for length 2
        tmp[1] = in_odd[0] - ((in_even[0] + 1) >> 1)
        tmp[0] = in_even[0] + tmp[1]
    else:
        # Direct translation of opj_idwt53_h_cas1
        s1 = int(in_even[1])
        dc = int(in_odd[0]) - ((int(in_even[0]) + s1 + 2) >> 2)
        tmp[0] = int(in_even[0]) + dc
        
        i = 1
        j = 1
        # Loop bound: i < (len - 2 - !(len & 1))
        # For even n: i < n - 3
        # For odd n:  i < n - 2
        limit = n - 2 - (0 if (n & 1) else 1)
        
        while i < limit:
            s2 = int(in_even[j + 1])
            dn_val = int(in_odd[j]) - ((s1 + s2 + 2) >> 2)
            tmp[i] = dc
            tmp[i + 1] = s1 + ((dn_val + dc) >> 1)
            dc = dn_val
            s1 = s2
            i += 2
            j += 1
        
        tmp[i] = dc
        
        if not (n & 1):  # even length
            dn_val = int(in_odd[n // 2 - 1]) - ((s1 + 1) >> 1)
            tmp[n - 2] = s1 + ((dn_val + dc) >> 1)
            tmp[n - 1] = dn_val
        else:  # odd length
            tmp[n - 1] = s1 + dc
    
    a[:] = tmp


def idwt53_1d_inplace(a: np.ndarray) -> None:
    """Wrapper that calls cas=1 version (matching libisyntax)."""
    idwt53_1d_cas1_inplace(a)

def idwt53_2d_inplace(img: np.ndarray) -> None:
    h, w = img.shape

    # rows: each row is [low | high]
    for y in range(h):
        idwt53_1d_inplace(img[y, :])

    # cols: each col is [low; high]
    for x in range(w):
        idwt53_1d_inplace(img[:, x])

def merge_quadrants_to_full(LL: np.ndarray, HL: np.ndarray, LH: np.ndarray, HH: np.ndarray) -> np.ndarray:
    """
    Merge 2D subbands into a single 2W x 2H array for cas=1 IDWT.
    
    For cas=1, each 1D IDWT expects [H | L] layout.
    So horizontally: [HL | LL] and [HH | LH]
    And vertically: top rows are H, bottom rows are L
    
    Final layout:
        [ HL | LL ]   <- high rows (LH/HH vertical)
        [ HH | LH ]
    Wait, that's wrong. Let me think again...
    
    Actually the C code arranges in memory as:
        quadrant[0]=LL at (0,0)
        quadrant[1]=HL at (0, quadrant_width)  
        quadrant[2]=LH at (quadrant_height*stride, 0)
        quadrant[3]=HH at (quadrant_height*stride, quadrant_width)
    
    So the memory layout for one row is: [LL_row | HL_row]
    For cas=1 horizontal IDWT, it reads the first half as H and second half as L.
    
    This means the C code stores [LL | HL] but the IDWT interprets the first half as H.
    So LL is being treated as H, and HL as L in the horizontal pass!
    
    To match this exactly, we need:
        [ LL | HL ]  (LL treated as H, HL treated as L horizontally)
        [ LH | HH ]  (LH treated as H, HH treated as L horizontally)
    """
    h, w = LL.shape
    full = np.zeros((2 * h, 2 * w), dtype=np.int32)
    # Match C layout exactly:
    full[0:h, 0:w] = LL      # quadrant[0]
    full[0:h, w:2*w] = HL    # quadrant[1]
    full[h:2*h, 0:w] = LH    # quadrant[2]
    full[h:2*h, w:2*w] = HH  # quadrant[3]
    return full


def write_pgm16(path: str, img_i32: np.ndarray) -> None:
    """
    Write as 16-bit big-endian PGM (P5). Viewers usually expect unsigned.
    We'll clamp to [0, 65535] for inspection. For exact diffing, keep raw arrays.
    """
    img = img_i32.astype(np.float64)
    mn = img.min()
    mx = img.max()
    if mx == mn:
        scaled = np.zeros_like(img, dtype=np.uint16)
    else:
        scaled = (img - mn) * (65535.0 / (mx - mn))
        scaled = np.clip(scaled, 0, 65535).astype(np.uint16)

    img_be = scaled.astype('>u2')  # big-endian uint16
    h, w = img_be.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n65535\n".encode("ascii"))
        f.write(img_be.tobytes())

def pack_subbands(LL, HL, LH, HH):
    h, w = LL.shape
    out = np.zeros((2*h, 2*w), dtype=np.int32)

    out[0:h,     0:w]     = LL
    out[0:h,     w:2*w]   = HL
    out[h:2*h,   0:w]     = LH
    out[h:2*h,   w:2*w]   = HH

    return out

def write_pgm8_fixed(path: str, img_i32: np.ndarray, level_shift: int = 0) -> None:
    """
    Write P5 8-bit PGM with a fixed mapping:
      out = clip(img + level_shift, 0..255)
    No normalization/rescaling.
    """
    img = img_i32.astype(np.int32)

    cand = []
    for s in (0, 128):
        x = img + s
        clip_low = np.mean(x < 0)
        clip_high = np.mean(x > 255)
        cand.append((clip_low + clip_high, s, x))

    _, best_shift, best = min(cand, key=lambda t: t[0])

    img8 = np.clip(best, 0, 255).astype(np.uint8)
    h, w = img8.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(img8.tobytes())

    print(f"[write_pgm8_best] {path}: chose shift={best_shift}, "
          f"clip_low={np.mean(best < 0):.3f}, clip_high={np.mean(best > 255):.3f}, "
          f"min={best.min()}, max={best.max()}, mean={best.mean():.2f}")

def reconstruct_one_level_from_bins(
    base_dir: str,
    r: int,
    color: int,
    band_w: int,
    band_h: int,
    out_path: str,
) -> None:
    LL = load_band_image(base_dir, r=r, band="LL", c=color, band_w=band_w, band_h=band_h)
    HL = load_band_image(base_dir, r=r, band="HL", c=color, band_w=band_w, band_h=band_h)
    LH = load_band_image(base_dir, r=r, band="LH", c=color, band_w=band_w, band_h=band_h)
    HH = load_band_image(base_dir, r=r, band="HH", c=color, band_w=band_w, band_h=band_h)

    full = merge_quadrants_to_full(LL, HL, LH, HH)

    # One inverse DWT level
    idwt53_2d_inplace(full)

    print("recon stats:",
      full.min(),
      full.max(),
      full.mean())

    # --- baseline ---
    packed = pack_subbands(LL, HL, LH, HH)
    idwt53_2d_inplace(packed)

    # Load libisyntax post-IDWT and compare
    post_path = os.path.join(
        base_dir,
        f"isy_postidwt_s{TARGET_SCALE}_tx{TARGET_TX}_ty{TARGET_TY}_c{color}.bin"
    )
    x0, y0, w, h, post = read_isy_cblk_dump(post_path)

    if post.shape != packed.shape:
        raise ValueError(f"shape mismatch: post {post.shape} vs packed {packed.shape}")

    diff = packed.astype(np.int32) - post.astype(np.int32)
    print("POST compare:")
    print("  post min/max/mean:", post.min(), post.max(), post.mean())
    print("  py   min/max/mean:", packed.min(), packed.max(), packed.mean())
    print("  diff min/max/mean:", diff.min(), diff.max(), diff.mean())
    print("  diff abs max:", np.max(np.abs(diff)))
    print("  diff abs mean:", np.mean(np.abs(diff)))
    print("  diff nonzero %:", 100.0 * np.mean(diff != 0))

    write_pgm8_fixed(out_path, packed, level_shift=0)

def reconstruct_full_padded(base_dir: str, out_path: str, color: int, block_w: int, block_h: int):
    prefix = f"isy_full_s{TARGET_SCALE}_tx{TARGET_TX}_ty{TARGET_TY}_r0"
    LL = load_full_plane(os.path.join(base_dir, f"{prefix}_LL_c{color}.bin"))
    HL = load_full_plane(os.path.join(base_dir, f"{prefix}_HL_c{color}.bin"))
    LH = load_full_plane(os.path.join(base_dir, f"{prefix}_LH_c{color}.bin"))
    HH = load_full_plane(os.path.join(base_dir, f"{prefix}_HH_c{color}.bin"))

    # Infer padding from quadrant size
    qh, qw = LL.shape
    pad_x_total = qw - block_w
    pad_y_total = qh - block_h
    if pad_x_total < 0 or pad_y_total < 0:
        raise ValueError(f"full dump smaller than block: quadrant={qw}x{qh}, block={block_w}x{block_h}")

    # assume symmetric padding (pad_l == pad_r)
    if pad_x_total % 2 != 0 or pad_y_total % 2 != 0:
        raise ValueError(f"non-even padding total: pad_x_total={pad_x_total}, pad_y_total={pad_y_total}")

    pad_x = pad_x_total // 2
    pad_y = pad_y_total // 2
    print(f"[pad] inferred pad_x={pad_x} pad_y={pad_y} from quadrant={qw}x{qh}")

    full = merge_quadrants_to_full(LL, HL, LH, HH)
    idwt53_2d_inplace(full)

    # Crop center region to match your POST dump region
    out_w = 2 * block_w
    out_h = 2 * block_h
    crop = full[pad_y:pad_y + out_h, pad_x:pad_x + out_w].copy()

    # Compare to libisyntax POST
    post_path = os.path.join(
        base_dir,
        f"isy_postidwt_s{TARGET_SCALE}_tx{TARGET_TX}_ty{TARGET_TY}_c{color}.bin"
    )
    _, _, _, _, post = read_isy_cblk_dump(post_path)
    if post.shape != crop.shape:
        raise ValueError(f"shape mismatch: post {post.shape} vs crop {crop.shape}")

    diff = crop.astype(np.int32) - post.astype(np.int32)
    print("POST compare (FULL PADDED -> CROPPED):")
    print("  post min/max/mean:", post.min(), post.max(), post.mean())
    print("  py   min/max/mean:", crop.min(), crop.max(), crop.mean())
    print("  diff min/max/mean:", diff.min(), diff.max(), diff.mean())
    print("  diff abs max:", np.max(np.abs(diff)))
    print("  diff abs mean:", np.mean(np.abs(diff)))
    print("  diff nonzero %:", 100.0 * np.mean(diff != 0))

    write_pgm8_fixed(out_path, crop, level_shift=0)
    return crop



if __name__ == "__main__":
    # Adjust these to match your run:
    # - base_dir is where your isy_r*_*.bin files are dumped
    # - band_w/band_h should match block_width/block_height in your code (often 128)
    BASE_DIR = "/Users/yaellyshkow/Desktop/isyntaxtoj2k/libisyntax"
    R = 0
    COLOR = 0
    BAND_W = 128
    BAND_H = 128
    OUT = f"recon_s{TARGET_SCALE}_tx{TARGET_TX}_ty{TARGET_TY}_r{R}_c{COLOR}.pgm"

    # reconstruct_one_level_from_bins(BASE_DIR, r=R, color=COLOR, band_w=BAND_W, band_h=BAND_H, out_path=OUT)
    OUT = f"recon_FULLPAD_s{TARGET_SCALE}_tx{TARGET_TX}_ty{TARGET_TY}_c{COLOR}.pgm"
    reconstruct_full_padded(BASE_DIR, out_path=OUT, color=COLOR, block_w=BAND_W, block_h=BAND_H)

