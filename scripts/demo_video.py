#!/usr/bin/env python
"""
Generate a 15s demo video with 3 panels for research presentation:
  +-------------------------+-------------------------+
  |  SLEAP Tracked          |  Nose→Body Distance     |
  |  Behavior Video         |  + Social bar           |
  +-------------------------+-------------------------+
  |     Live Neural Activation (all neurons)          |
  +---------------------------------------------------+

Usage:
    conda run -n sleap python scripts/demo_video.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
import cv2
import sleap_io as sio
import h5py

from src.constants import IMAGING_FPS, BEHAVIOR_FPS
from src.data import (download_data, load_entrances, load_behavior,
                      load_imaging, align_all_sessions)

# ── Parameters ──
VIDEO_FPS    = 25
DURATION_S   = 15
N_FRAMES     = VIDEO_FPS * DURATION_S
START_S      = 38
SESSION_VID  = 5
OUTPUT_W, OUTPUT_H = 1920, 1080
DPI          = 100
SMOOTH       = 15
DIST_THRESH  = 10

SKELETON_COLORS = ['#2ecc71', '#e67e22']

# Dark theme
BG       = '#0d1117'
PANEL_BG = '#161b22'
GRID_C   = '#30363d'
SPINE_C  = '#30363d'
TEXT_C   = '#e6edf3'
MUTED_C  = '#8b949e'

OUT_PATH = '/Users/ketan/Downloads/dataset_demo.mp4'

def style(ax, grid=True):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED_C, labelsize=9)
    if grid:
        ax.grid(True, color=GRID_C, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(SPINE_C)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)

# ── Load data ──
print("[1/5] Loading data...")
DATA_DIR  = download_data()
entrances = load_entrances(DATA_DIR)
n_sess    = len(entrances)
behavior  = load_behavior(n_sess, DATA_DIR)
imaging   = load_imaging(n_sess, DATA_DIR)
aligned_cal, aligned_beh, sess_info, sess_df = \
    align_all_sessions(imaging, behavior, entrances)

s5_ai    = next(i for i, d in enumerate(sess_info) if d['session_idx'] == SESSION_VID)
cal_s5   = aligned_cal[s5_ai]
beh_s5   = aligned_beh[s5_ai]
entry_s5 = int(entrances.iloc[SESSION_VID]['Int_Entry'])
nn_s5    = cal_s5.shape[1]

# Spatial footprints
print("[2/5] Loading spatial footprints...")
sp_path = DATA_DIR / 'spatial_footprints_5-2.h5'
with h5py.File(sp_path, 'r') as f:
    A_flat = f['A'][:]
    Cn     = f['Cn'][:]
n_sp = A_flat.shape[0]
side = int(np.sqrt(A_flat.shape[1]))
A = A_flat.reshape(n_sp, side, side)
ms = Cn.shape
Cn_n = (Cn - Cn.min()) / (Cn.max() - Cn.min() + 1e-12)

# ── Time windows ──
n_cal = int(DURATION_S * IMAGING_FPS)
start_cal = int(START_S * IMAGING_FPS)
start_beh_vid = entry_s5 + int(START_S * BEHAVIOR_FPS)

# Z-score calcium traces
cal_z_s5 = np.zeros_like(cal_s5, dtype=np.float64)
for ni in range(nn_s5):
    tr = cal_s5[:, ni].astype(np.float64)
    m, s = tr.mean(), tr.std()
    cal_z_s5[:, ni] = (tr - m) / (s + 1e-12)
cal_z_win = cal_z_s5[start_cal:start_cal + n_cal]

# ── Spatial activation engine (all neurons) ──
print("[3/5] Preparing spatial activation...")
A_all = A[:nn_s5].reshape(nn_s5, -1).astype(np.float64)

def spatial_map(ci):
    lo = max(0, ci - SMOOTH // 2)
    hi = min(n_cal, ci + SMOOTH // 2 + 1)
    act = np.clip(cal_z_win[lo:hi].mean(axis=0), 0, None)
    return (act @ A_all).reshape(ms)

# Calibrate color range
_samples = [spatial_map(ci) for ci in range(0, n_cal, 30)]
_vals = [np.percentile(m[m > 0], 95) for m in _samples if (m > 0).any()]
vmax = np.percentile(_vals, 85) if _vals else 0.01

# ── SLEAP data ──
print("[4/5] Loading SLEAP predictions...")
preds = sio.load_file(str(DATA_DIR / 'behavior_tracking.slp'))
preds.video.replace_filename(str(DATA_DIR / 'behavior_video.mp4'))
frame_to_lf = {lf.frame_idx: lf for lf in preds}
nose_node = preds.skeleton.index('nose')

distances = np.full(N_FRAMES, np.nan)
for i in range(N_FRAMES):
    lf = frame_to_lf.get(start_beh_vid + i)
    if lf is None:
        continue
    trk = {}
    for inst in lf:
        trk[preds.tracks.index(inst.track)] = inst.numpy()
    if 0 in trk and 1 in trk:
        rn = trk[0][nose_node]
        ib = trk[1]
        if not np.isnan(rn).any():
            d = np.sqrt(np.nansum((ib - rn[None, :]) ** 2, axis=-1))
            distances[i] = np.nanmin(d)
t_vid = np.arange(N_FRAMES) / VIDEO_FPS

def draw_skeleton(ax, vid_idx, art_list):
    lf = frame_to_lf.get(vid_idx)
    if lf is None:
        return
    for inst in lf:
        pts = inst.numpy()
        ti  = preds.tracks.index(inst.track)
        col = SKELETON_COLORS[ti]
        for s, d in inst.skeleton.edge_inds:
            if not (np.isnan(pts[s]).any() or np.isnan(pts[d]).any()):
                ln, = ax.plot(pts[[s, d], 0], pts[[s, d], 1],
                              '.-', color=col, lw=2.5, ms=6, zorder=3)
                art_list.append(ln)
        if not np.isnan(pts[nose_node]).any():
            tx = ax.text(pts[nose_node, 0] + 5, pts[nose_node, 1] - 5,
                         inst.track.name, color=col, fontsize=9,
                         fontweight='bold', zorder=4)
            art_list.append(tx)

# ── Build composite video ──
print("[5/5] Rendering video...")
fig = plt.figure(figsize=(OUTPUT_W / DPI, OUTPUT_H / DPI), dpi=DPI, facecolor=BG)

gs = GridSpec(2, 2, height_ratios=[1, 0.85], width_ratios=[1, 1],
              left=0.03, right=0.99, top=0.93, bottom=0.03,
              hspace=0.08, wspace=0.05, figure=fig)

# ── Top-left: SLEAP video ──
cap = cv2.VideoCapture(str(DATA_DIR / 'behavior_video.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_beh_vid)
_, first = cap.read()
first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)

ax_vid = fig.add_subplot(gs[0, 0])
ax_vid.set_facecolor('black')
im_vid = ax_vid.imshow(first_rgb)
ax_vid.set_xlim(0, first_rgb.shape[1])
ax_vid.set_ylim(first_rgb.shape[0], 0)
ax_vid.set_xticks([]); ax_vid.set_yticks([])
for sp in ax_vid.spines.values():
    sp.set_color(SPINE_C)
ax_vid.set_title('Behavior Video + SLEAP Tracking', fontsize=13,
                 fontweight='bold', color=TEXT_C, pad=4)
txt_vid = ax_vid.text(0.02, 0.92, '', transform=ax_vid.transAxes,
                      fontsize=11, color='white', fontweight='bold', va='top',
                      bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.65))

# ── Top-right: Distance + social bar ──
dist_gs = GridSpecFromSubplotSpec(2, 1, gs[0, 1], height_ratios=[3, 1], hspace=0.10)
ax_dist = fig.add_subplot(dist_gs[0])
ax_sbar = fig.add_subplot(dist_gs[1], sharex=ax_dist)

style(ax_dist)
ax_dist.plot(t_vid, distances, color='#58a6ff', lw=0.8, alpha=0.85)
ax_dist.scatter(t_vid[::2], distances[::2], s=3, color='#58a6ff',
                alpha=0.35, zorder=6, linewidths=0)
ax_dist.axhline(DIST_THRESH, color='#f85149', lw=2, ls='--',
                label=f'Threshold = {DIST_THRESH} px', zorder=7)
ymax_d = min(180, np.nanmax(distances) * 1.1) if np.any(~np.isnan(distances)) else 180
ax_dist.set_xlim(0, DURATION_S)
ax_dist.set_ylim(0, ymax_d)
ax_dist.set_ylabel('Distance (px)', fontsize=10, color=TEXT_C)
ax_dist.set_title('Nose \u2192 Body Distance', fontsize=13,
                  fontweight='bold', color=TEXT_C, pad=4)
ax_dist.tick_params(axis='x', labelbottom=False)
ax_dist.legend(loc='upper right', fontsize=9, framealpha=0.7,
               facecolor=PANEL_BG, edgecolor=SPINE_C, labelcolor=TEXT_C)
mask_dist = Rectangle((0, 0), DURATION_S, ymax_d, fc=PANEL_BG, ec='none', zorder=4)
ax_dist.add_patch(mask_dist)
cur_dist = ax_dist.axvline(0, color='#f0f6fc', lw=2, alpha=0.9, zorder=5)

style(ax_sbar, grid=False)
is_close = distances < DIST_THRESH
ax_sbar.fill_between(t_vid, 0, is_close.astype(float),
                     color='#f85149', alpha=0.7, step='mid')
ax_sbar.set_xlim(0, DURATION_S)
ax_sbar.set_ylim(-0.05, 1.15)
ax_sbar.set_ylabel('Social', fontsize=10, color=TEXT_C)
ax_sbar.set_xlabel('Time (s)', fontsize=10, color=TEXT_C)
ax_sbar.set_yticks([0, 1])
ax_sbar.set_yticklabels(['Solo', 'Social'], fontsize=8, color=MUTED_C)
mask_sbar = Rectangle((0, -0.05), DURATION_S, 1.2, fc=PANEL_BG, ec='none', zorder=4)
ax_sbar.add_patch(mask_sbar)
cur_sbar = ax_sbar.axvline(0, color='#f0f6fc', lw=2, alpha=0.9, zorder=5)

# ── Bottom: Live neural activation (full width) ──
ax_act = fig.add_subplot(gs[1, :])
ax_act.set_facecolor('black')
ax_act.imshow(Cn_n, cmap='gray', alpha=0.6)
im_act = ax_act.imshow(np.zeros((*ms, 4)))
ax_act.set_xticks([]); ax_act.set_yticks([])
for sp in ax_act.spines.values():
    sp.set_color(SPINE_C)
ax_act.set_title(f'Live Neural Activation — {nn_s5} Neurons (Prefrontal Cortex)',
                 fontsize=13, fontweight='bold', color=TEXT_C, pad=4)

fig.suptitle('Dataset Demo — Calcium Imaging + Behavior (Session 5)',
             fontsize=16, color=TEXT_C, fontweight='bold', y=0.98)

# ── Render frames ──
skel_art = []
out_tmp = OUT_PATH.replace('.mp4', '_tmp.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_tmp, fourcc, VIDEO_FPS, (OUTPUT_W, OUTPUT_H))

for fi in range(N_FRAMES):
    t = fi / VIDEO_FPS
    cal_idx = min(int(t * IMAGING_FPS), n_cal - 1)
    vid_idx = start_beh_vid + fi

    # Video frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
    ok, frm = cap.read()
    if ok:
        im_vid.set_data(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    for a in skel_art:
        a.remove()
    skel_art.clear()
    draw_skeleton(ax_vid, vid_idx, skel_art)
    txt_vid.set_text(f't = {START_S + t:.1f}s')

    # Distance panel progressive reveal
    remaining = DURATION_S - t
    mask_dist.set_x(t); mask_dist.set_width(remaining)
    mask_sbar.set_x(t); mask_sbar.set_width(remaining)
    cur_dist.set_xdata([t, t])
    cur_sbar.set_xdata([t, t])

    # Live spatial activation (single green heatmap)
    m = spatial_map(cal_idx)
    a = np.clip(m / (vmax + 1e-12), 0, 1)
    rgba = np.zeros((*ms, 4))
    rgba[..., 1] = 1.0       # green channel
    rgba[..., 3] = a * 0.8   # alpha from activation
    im_act.set_data(rgba)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

    if (fi + 1) % 25 == 0 or fi == 0:
        print(f"  Frame {fi + 1:3d}/{N_FRAMES}  ({(fi + 1) / N_FRAMES * 100:.0f}%)")

writer.release()
cap.release()
plt.close(fig)

print("  Re-encoding H.264...")
os.system(f'ffmpeg -y -i "{out_tmp}" -c:v libx264 -preset fast -crf 18 '
          f'-pix_fmt yuv420p "{OUT_PATH}" -loglevel warning')
os.remove(out_tmp)
print(f"Done! → {OUT_PATH}")
