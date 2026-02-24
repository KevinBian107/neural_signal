#!/usr/bin/env python
"""
Generate 12 hero background video clips (3 rows × 4 columns).

Grid layout:
  Row 1 (Social):     behavior | cluster0 | cluster1 | combined
  Row 2 (Solo):       behavior | cluster0 | cluster1 | combined
  Row 3 (Transition): behavior | cluster0 | cluster1 | combined

Each clip preserves the original aspect ratio of its source:
  - Behavior clips: native video aspect ratio (auto-detected)
  - Spatial clips: square (matching FOV footprints)
Resolution: ~480px height, 15 fps, ~6 seconds, H.264 CRF 28.
Output: docs/hero/{social,solo,transition}_{behavior,cluster0,cluster1,combined}.mp4

Usage:
    conda run -n sleap python scripts/create_hero_videos.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
import h5py
import sleap_io as sio

from src.constants import IMAGING_FPS, BEHAVIOR_FPS
from src.data import (download_data, load_entrances, load_behavior,
                      load_imaging, align_all_sessions, get_epoch_durations)
from src.analysis import compute_neuron_spectral_profiles, cluster_neurons

# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT_FPS    = 15
CLIP_DURATION = 6        # seconds per clip
N_CLIP_FRAMES = OUTPUT_FPS * CLIP_DURATION   # 90 output frames
CLIP_H        = 480      # target height; width derived from source aspect
SMOOTH_FRAMES = 15
SESSION_VID   = 5
MIN_BOUT_S    = 3.0      # minimum bout duration for window selection

SKELETON_COLORS_BGR = [
    (113, 204, 46),   # green (#2ecc71)
    (34, 126, 230),   # orange (#e67e22)
]

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'hero')

# ═══════════════════════════════════════════════════════════════════════════
# Load data (same pipeline as create_two_videos.py)
# ═══════════════════════════════════════════════════════════════════════════
print("[1/6] Loading data …")
DATA_DIR  = download_data()
entrances = load_entrances(DATA_DIR)
n_sess    = len(entrances)
behavior  = load_behavior(n_sess, DATA_DIR)
imaging   = load_imaging(n_sess, DATA_DIR)

aligned_cal, aligned_beh, sess_info, sess_df = \
    align_all_sessions(imaging, behavior, entrances)

# Session 5
s5_ai    = next(i for i, d in enumerate(sess_info) if d['session_idx'] == SESSION_VID)
cal_s5   = aligned_cal[s5_ai]
beh_s5   = aligned_beh[s5_ai]
entry_s5 = int(entrances.iloc[SESSION_VID]['Int_Entry'])
nn_s5    = cal_s5.shape[1]
n_total  = cal_s5.shape[0]

# Spatial footprints
sp_path = DATA_DIR / 'spatial_footprints_5-2.h5'
with h5py.File(sp_path, 'r') as f:
    A_flat = f['A'][:]
    Cn     = f['Cn'][:]
n_sp = A_flat.shape[0]
side = int(np.sqrt(A_flat.shape[1]))
A = A_flat.reshape(n_sp, side, side)
print(f"  Session {SESSION_VID}: {cal_s5.shape}, Spatial: {n_sp} neurons, {side}x{side}")

# ═══════════════════════════════════════════════════════════════════════════
# Spectral clustering
# ═══════════════════════════════════════════════════════════════════════════
print("[2/6] Computing spectral clusters …")
prof_df = compute_neuron_spectral_profiles(aligned_cal, sess_info)
labels_all, _, best_k, sil = cluster_neurons(prof_df)
prof_df['cluster'] = labels_all

cmap_s5 = np.full(nn_s5, -1, dtype=int)
for _, r in prof_df[prof_df['session_idx'] == SESSION_VID].iterrows():
    cmap_s5[int(r['neuron_id'])] = int(r['cluster'])
c0_s5 = np.where(cmap_s5 == 0)[0]
c1_s5 = np.where(cmap_s5 == 1)[0]
print(f"  Clusters: C0={len(c0_s5)}, C1={len(c1_s5)}")

# ═══════════════════════════════════════════════════════════════════════════
# Z-score full session & spatial activation engine
# ═══════════════════════════════════════════════════════════════════════════
print("[3/6] Preparing spatial activation …")
cal_z_s5 = np.zeros_like(cal_s5, dtype=np.float64)
for ni in range(nn_s5):
    tr = cal_s5[:, ni].astype(np.float64)
    m, s = tr.mean(), tr.std()
    cal_z_s5[:, ni] = (tr - m) / (s + 1e-12)

A_c0f = A[c0_s5].reshape(len(c0_s5), -1).astype(np.float64)
A_c1f = A[c1_s5].reshape(len(c1_s5), -1).astype(np.float64)
ms    = Cn.shape
Cn_n  = (Cn - Cn.min()) / (Cn.max() - Cn.min() + 1e-12)


def spatial_maps(ci):
    """Compute spatial activation maps at absolute calcium frame index ci."""
    lo = max(0, ci - SMOOTH_FRAMES // 2)
    hi = min(n_total, ci + SMOOTH_FRAMES // 2 + 1)
    a0 = np.clip(cal_z_s5[lo:hi, c0_s5].mean(axis=0), 0, None)
    a1 = np.clip(cal_z_s5[lo:hi, c1_s5].mean(axis=0), 0, None)
    return (a0 @ A_c0f).reshape(ms), (a1 @ A_c1f).reshape(ms)


# Calibrate vmax from samples across full session
_samples = [spatial_maps(ci) for ci in range(0, n_total, 60)]
_v0 = [np.percentile(m[m > 0], 95) for m, _ in _samples if (m > 0).any()]
_v1 = [np.percentile(m[m > 0], 95) for _, m in _samples if (m > 0).any()]
vmax0 = np.percentile(_v0, 85) if _v0 else 0.01
vmax1 = np.percentile(_v1, 85) if _v1 else 0.01
print(f"  vmax0={vmax0:.4f}, vmax1={vmax1:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# SLEAP data
# ═══════════════════════════════════════════════════════════════════════════
print("[4/6] Loading SLEAP predictions …")
preds = sio.load_file(str(DATA_DIR / 'behavior_tracking.slp'))
preds.video.replace_filename(str(DATA_DIR / 'behavior_video.mp4'))
frame_to_lf = {lf.frame_idx: lf for lf in preds}

# ═══════════════════════════════════════════════════════════════════════════
# Detect time windows (social, solo, transition)
# ═══════════════════════════════════════════════════════════════════════════
print("[5/6] Detecting time windows …")
clip_cal_frames = int(CLIP_DURATION * IMAGING_FPS)   # frames per clip at 30fps

starts, ends, epoch_labels, durations = get_epoch_durations(beh_s5, IMAGING_FPS)

# Social: longest social bout >= CLIP_DURATION
social_bouts = [(s, e, d) for s, e, l, d in zip(starts, ends, epoch_labels, durations)
                if l == 1 and d >= CLIP_DURATION]
if not social_bouts:
    # Fallback: longest social bout of any length
    social_bouts = [(s, e, d) for s, e, l, d in zip(starts, ends, epoch_labels, durations)
                    if l == 1]
social_bouts.sort(key=lambda x: x[2], reverse=True)
sb = social_bouts[0]
social_start = sb[0] + max(0, (sb[1] - sb[0] - clip_cal_frames) // 2)

# Solo: longest solo bout >= CLIP_DURATION (post first few seconds)
solo_bouts = [(s, e, d) for s, e, l, d in zip(starts, ends, epoch_labels, durations)
              if l == 0 and d >= CLIP_DURATION and s > IMAGING_FPS]  # skip first second
if not solo_bouts:
    solo_bouts = [(s, e, d) for s, e, l, d in zip(starts, ends, epoch_labels, durations)
                  if l == 0 and s > IMAGING_FPS]
solo_bouts.sort(key=lambda x: x[2], reverse=True)
slb = solo_bouts[0]
solo_start = slb[0] + max(0, (slb[1] - slb[0] - clip_cal_frames) // 2)

# Transition: solo→social boundary with padding on each side
half = clip_cal_frames // 2
trans_start = None
for i in range(1, len(starts)):
    if epoch_labels[i] == 1 and epoch_labels[i - 1] == 0:
        boundary = starts[i]
        pre_pad = boundary - starts[i - 1]
        post_pad = ends[i] - boundary
        if pre_pad >= half and post_pad >= half:
            trans_start = boundary - half
            break
# Fallback: use first solo→social boundary, clamp
if trans_start is None:
    for i in range(1, len(starts)):
        if epoch_labels[i] == 1 and epoch_labels[i - 1] == 0:
            trans_start = max(0, starts[i] - half)
            break
# Ultimate fallback: midpoint of session
if trans_start is None:
    trans_start = n_total // 2 - half

windows = {
    'social':     social_start,
    'solo':       solo_start,
    'transition': trans_start,
}

for name, start in windows.items():
    t = start / IMAGING_FPS
    print(f"  {name:12s}: calcium frame {start}, t = {t:.1f}s post-entry")


# ═══════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ═══════════════════════════════════════════════════════════════════════════

def draw_skeleton_cv(frame_bgr, vid_idx):
    """Draw SLEAP skeleton directly on an OpenCV frame (in-place)."""
    lf = frame_to_lf.get(vid_idx)
    if lf is None:
        return
    for inst in lf:
        pts = inst.numpy()
        ti = preds.tracks.index(inst.track)
        col = SKELETON_COLORS_BGR[ti]
        for s, d in inst.skeleton.edge_inds:
            if not (np.isnan(pts[s]).any() or np.isnan(pts[d]).any()):
                p1 = tuple(pts[s].astype(int))
                p2 = tuple(pts[d].astype(int))
                cv2.line(frame_bgr, p1, p2, col, 2, cv2.LINE_AA)
                cv2.circle(frame_bgr, p1, 3, col, -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, p2, 3, col, -1, cv2.LINE_AA)


def render_spatial(m0, m1, mode):
    """Render a spatial map frame as a BGR image at full FOV resolution.

    mode: 'cluster0', 'cluster1', or 'combined'
    Returns BGR uint8 array at (side, side, 3).
    """
    a0 = np.clip(m0 / (vmax0 + 1e-12), 0, 1)
    a1 = np.clip(m1 / (vmax1 + 1e-12), 0, 1)

    # Dim correlation image as background
    bg = (Cn_n * 80).astype(np.float64)  # dim gray
    rgb = np.stack([bg, bg, bg], axis=-1)

    if mode == 'cluster0':
        # Blue overlay (BGR: channel 0)
        rgb[..., 0] += a0 * 220
    elif mode == 'cluster1':
        # Red overlay (BGR: channel 2)
        rgb[..., 2] += a1 * 220
    else:  # combined
        rgb[..., 0] += a0 * 200  # blue
        rgb[..., 2] += a1 * 200  # red

    return np.clip(rgb, 0, 255).astype(np.uint8)


def cal_to_vid(ci):
    """Convert absolute calcium frame index to behavior video frame index."""
    return entry_s5 + int(ci * BEHAVIOR_FPS / IMAGING_FPS)


# ═══════════════════════════════════════════════════════════════════════════
# Render 12 clips
# ═══════════════════════════════════════════════════════════════════════════
print("[6/6] Rendering clips …")
os.makedirs(OUT_DIR, exist_ok=True)

columns = ['behavior', 'cluster0', 'cluster1', 'combined']
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

cap = cv2.VideoCapture(str(DATA_DIR / 'behavior_video.mp4'))

# Detect native behavior video aspect ratio
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
beh_w = int(vid_w * CLIP_H / vid_h)
beh_h = CLIP_H
# Ensure even dimensions for H.264
beh_w = beh_w + (beh_w % 2)
beh_h = beh_h + (beh_h % 2)

# Spatial clips: square (matching the FOV footprints)
sp_size = CLIP_H
sp_size = sp_size + (sp_size % 2)

print(f"  Behavior video native: {vid_w}×{vid_h}")
print(f"  Output behavior clips: {beh_w}×{beh_h}")
print(f"  Output spatial clips:  {sp_size}×{sp_size}")

for row_name, win_start in windows.items():
    print(f"\n  Row: {row_name}")

    # Open 4 writers — behavior at native aspect, spatial square
    tmp_paths = {}
    final_paths = {}
    writers = {}
    for col in columns:
        tmp = os.path.join(OUT_DIR, f'{row_name}_{col}_tmp.mp4')
        final = os.path.join(OUT_DIR, f'{row_name}_{col}.mp4')
        tmp_paths[col] = tmp
        final_paths[col] = final
        if col == 'behavior':
            writers[col] = cv2.VideoWriter(tmp, fourcc, OUTPUT_FPS, (beh_w, beh_h))
        else:
            writers[col] = cv2.VideoWriter(tmp, fourcc, OUTPUT_FPS, (sp_size, sp_size))

    for fi in range(N_CLIP_FRAMES):
        t = fi / OUTPUT_FPS  # time in seconds into clip
        ci = win_start + min(int(t * IMAGING_FPS), clip_cal_frames - 1)
        vi = cal_to_vid(ci)

        # --- Behavior clip (native aspect) ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, vi)
        ok, frame = cap.read()
        if ok:
            draw_skeleton_cv(frame, vi)
            frame = cv2.resize(frame, (beh_w, beh_h))
            writers['behavior'].write(frame)

        # --- Spatial clips (square, native FOV shape) ---
        m0, m1 = spatial_maps(ci)
        for mode in ['cluster0', 'cluster1', 'combined']:
            sp_frame = render_spatial(m0, m1, mode)
            sp_frame = cv2.resize(sp_frame, (sp_size, sp_size))
            writers[mode].write(sp_frame)

        if (fi + 1) % 30 == 0 or fi == 0:
            print(f"    frame {fi + 1}/{N_CLIP_FRAMES}")

    # Close writers and re-encode
    for col in columns:
        writers[col].release()
        tmp = tmp_paths[col]
        final = final_paths[col]
        os.system(
            f'ffmpeg -y -i "{tmp}" '
            f'-c:v libx264 -preset slow -crf 28 '
            f'-pix_fmt yuv420p -an '
            f'"{final}" -loglevel warning'
        )
        os.remove(tmp)
        sz = os.path.getsize(final) / 1024
        print(f"    {os.path.basename(final):40s} {sz:.0f} KB")

cap.release()
print("\nDone! Output:", OUT_DIR)
