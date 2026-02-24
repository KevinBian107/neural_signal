#!/usr/bin/env python
"""
Generate two presentation-quality composite videos (30s each, dark theme).

Video 1 — SLEAP + Spatial Activation:
  +---------------------+---------------------+
  |  SLEAP Tracked      |  Spatial Cluster     |
  |  Behavior Video     |  Activation (FOV)    |
  +---------------------+---------------------+

Video 2 — SLEAP + Distance + Band Decomposition (2 neurons):
  +---------------------+---------------------+
  |  SLEAP Tracked      |  Nose→Body Distance  |
  |  Behavior Video     |  + Social bar        |
  +---------------------+---------------------+
  |  Cluster 0 Neuron   |  Cluster 1 Neuron    |
  |  Band Decomposition |  Band Decomposition  |
  +---------------------+---------------------+

Usage:
    conda run -n sleap python scripts/create_two_videos.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cv2
import sleap_io as sio
from scipy.signal import detrend

from src.constants import IMAGING_FPS, BEHAVIOR_FPS, FREQ_BANDS, BAND_COLORS
from src.data import (download_data, load_entrances, load_behavior,
                      load_imaging, align_all_sessions)
from src.signal_processing import bandpass_filter, band_envelope
from src.analysis import compute_neuron_spectral_profiles, cluster_neurons

# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════
VIDEO_FPS          = 25
DURATION_S         = 30
N_FRAMES           = VIDEO_FPS * DURATION_S
START_POST_ENTRY_S = 38       # best 30s window
SESSION_VID        = 5
OUTPUT_W, OUTPUT_H = 1920, 1080
DPI                = 100
SMOOTH_FRAMES      = 15
DIST_THRESH        = 10

SKELETON_COLORS = ['#2ecc71', '#e67e22']

# Unified dark theme
BG       = '#0d1117'
PANEL_BG = '#161b22'
GRID_C   = '#30363d'
SPINE_C  = '#30363d'
TEXT_C   = '#e6edf3'
MUTED_C  = '#8b949e'

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def style(ax, grid=True):
    """Apply unified dark theme."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED_C, labelsize=10)
    if grid:
        ax.grid(True, color=GRID_C, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(SPINE_C)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
print("[1/9] Loading data …")
DATA_DIR  = download_data()
entrances = load_entrances(DATA_DIR)
n_sess    = len(entrances)
behavior  = load_behavior(n_sess, DATA_DIR)
imaging   = load_imaging(n_sess, DATA_DIR)

aligned_cal, aligned_beh, sess_info, sess_df = \
    align_all_sessions(imaging, behavior, entrances)

# Session 5 — video / behavior / Fourier
s5_ai    = next(i for i, d in enumerate(sess_info) if d['session_idx'] == SESSION_VID)
cal_s5   = aligned_cal[s5_ai]
beh_s5   = aligned_beh[s5_ai]
entry_s5 = int(entrances.iloc[SESSION_VID]['Int_Entry'])
nn_s5    = cal_s5.shape[1]
animal_s5 = entrances.iloc[SESSION_VID]['Animal']

# Session 5 — spatial footprints (same animal as behavior video)
import h5py
sp_path = DATA_DIR / 'spatial_footprints_5-2.h5'
with h5py.File(sp_path, 'r') as f:
    A_flat = f['A'][:]   # (n_neurons, 360000)
    Cn     = f['Cn'][:]  # (600, 600)
n_sp = A_flat.shape[0]
side = int(np.sqrt(A_flat.shape[1]))
A = A_flat.reshape(n_sp, side, side)
print(f"  Session {SESSION_VID}: {cal_s5.shape}, Animal: {animal_s5}, "
      f"Spatial: {n_sp} neurons, {side}x{side} FOV")

# ═══════════════════════════════════════════════════════════════════════════
# Spectral clustering
# ═══════════════════════════════════════════════════════════════════════════
print("[2/9] Computing spectral clusters …")
prof_df = compute_neuron_spectral_profiles(aligned_cal, sess_info)
labels_all, _, best_k, sil = cluster_neurons(prof_df)
prof_df['cluster'] = labels_all

# Session 5 clusters (for both spatial map and band decomposition)
cmap_s5 = np.full(nn_s5, -1, dtype=int)
for _, r in prof_df[prof_df['session_idx'] == SESSION_VID].iterrows():
    cmap_s5[int(r['neuron_id'])] = int(r['cluster'])
c0_s5 = np.where(cmap_s5 == 0)[0]
c1_s5 = np.where(cmap_s5 == 1)[0]
print(f"  Session 5 clusters: C0={len(c0_s5)}, C1={len(c1_s5)}")

# ═══════════════════════════════════════════════════════════════════════════
# Pick representative neurons from Session 5
# ═══════════════════════════════════════════════════════════════════════════
print("[3/9] Selecting representative neurons …")
s5_prof = prof_df[prof_df['session_idx'] == SESSION_VID].copy()
frac_cols = [f'{b}_frac' for b in FREQ_BANDS.keys()]

# For each cluster, pick neuron closest to the cluster centroid
rep_neurons = {}
for cl in [0, 1]:
    sub = s5_prof[s5_prof['cluster'] == cl]
    centroid = sub[frac_cols].mean().values
    dists = np.linalg.norm(sub[frac_cols].values - centroid, axis=1)
    best_idx = sub.index[np.argmin(dists)]
    nid = int(sub.loc[best_idx, 'neuron_id'])
    rep_neurons[cl] = nid
    print(f"  Cluster {cl} representative: Neuron {nid}")

# ═══════════════════════════════════════════════════════════════════════════
# Time windows
# ═══════════════════════════════════════════════════════════════════════════
n_cal = int(DURATION_S * IMAGING_FPS)
t_cal = np.arange(n_cal) / IMAGING_FPS

start_cal5    = int(START_POST_ENTRY_S * IMAGING_FPS)
beh_win5      = beh_s5[start_cal5:start_cal5 + n_cal]
start_beh_vid = entry_s5 + int(START_POST_ENTRY_S * BEHAVIOR_FPS)

# Session 5 z-scored for spatial maps
cal_z_s5 = np.zeros_like(cal_s5, dtype=np.float64)
for ni in range(nn_s5):
    tr = cal_s5[:, ni].astype(np.float64)
    m, s = tr.mean(), tr.std()
    cal_z_s5[:, ni] = (tr - m) / (s + 1e-12)
cal_z5_win = cal_z_s5[start_cal5:start_cal5 + n_cal]

# ═══════════════════════════════════════════════════════════════════════════
# Band decomposition for the two representative neurons
# ═══════════════════════════════════════════════════════════════════════════
print("[4/9] Computing band decompositions …")
neuron_bands = {}
neuron_envs  = {}
for cl in [0, 1]:
    nid = rep_neurons[cl]
    sig_full = detrend(cal_s5[:, nid].astype(np.float64))
    neuron_bands[cl] = {}
    neuron_envs[cl]  = {}
    for bn, (lo, hi) in FREQ_BANDS.items():
        neuron_bands[cl][bn] = bandpass_filter(sig_full, lo, hi, IMAGING_FPS)[start_cal5:start_cal5 + n_cal]
        neuron_envs[cl][bn]  = band_envelope(sig_full, lo, hi, IMAGING_FPS)[start_cal5:start_cal5 + n_cal]
    # Also store raw windowed trace
    neuron_bands[cl]['raw'] = sig_full[start_cal5:start_cal5 + n_cal]

# Cluster population mean band decomposition (for Video 3)
print("  Computing population mean band decomposition (Video 3) …")
pop_bands = {}
pop_envs  = {}
for cl, neurons in [(0, c0_s5), (1, c1_s5)]:
    sig = detrend(cal_s5[:, neurons].mean(axis=1).astype(np.float64))
    pop_bands[cl] = {}
    pop_envs[cl]  = {}
    for bn, (lo, hi) in FREQ_BANDS.items():
        pop_bands[cl][bn] = bandpass_filter(sig, lo, hi, IMAGING_FPS)[start_cal5:start_cal5 + n_cal]
        pop_envs[cl][bn]  = band_envelope(sig, lo, hi, IMAGING_FPS)[start_cal5:start_cal5 + n_cal]
    pop_bands[cl]['raw'] = sig[start_cal5:start_cal5 + n_cal]

# ═══════════════════════════════════════════════════════════════════════════
# SLEAP data
# ═══════════════════════════════════════════════════════════════════════════
print("[5/9] Loading SLEAP predictions …")
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

# ═══════════════════════════════════════════════════════════════════════════
# Spatial activation engine (Video 1)
# ═══════════════════════════════════════════════════════════════════════════
print("[6/9] Preparing spatial activation …")
A_c0f = A[c0_s5].reshape(len(c0_s5), -1).astype(np.float64)
A_c1f = A[c1_s5].reshape(len(c1_s5), -1).astype(np.float64)
ms    = Cn.shape
Cn_n  = (Cn - Cn.min()) / (Cn.max() - Cn.min() + 1e-12)


def spatial_maps(ci):
    lo = max(0, ci - SMOOTH_FRAMES // 2)
    hi = min(n_cal, ci + SMOOTH_FRAMES // 2 + 1)
    a0 = np.clip(cal_z5_win[lo:hi, c0_s5].mean(axis=0), 0, None)
    a1 = np.clip(cal_z5_win[lo:hi, c1_s5].mean(axis=0), 0, None)
    return (a0 @ A_c0f).reshape(ms), (a1 @ A_c1f).reshape(ms)


_samples = [spatial_maps(ci) for ci in range(0, n_cal, 30)]
_v0 = [np.percentile(m[m > 0], 95) for m, _ in _samples if (m > 0).any()]
_v1 = [np.percentile(m[m > 0], 95) for _, m in _samples if (m > 0).any()]
vmax0 = np.percentile(_v0, 85) if _v0 else 0.01
vmax1 = np.percentile(_v1, 85) if _v1 else 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Helper: draw skeleton on an axes
# ═══════════════════════════════════════════════════════════════════════════
def draw_skeleton(ax, vid_idx, art_list):
    """Draw SLEAP skeleton on ax, appending artists to art_list."""
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


def setup_video_panel(ax, first_rgb, title):
    """Configure a SLEAP video panel."""
    ax.set_facecolor('black')
    im = ax.imshow(first_rgb)
    ax.set_xlim(0, first_rgb.shape[1])
    ax.set_ylim(first_rgb.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(SPINE_C)
    ax.set_title(title, fontsize=14, fontweight='bold', color=TEXT_C, pad=4)
    return im


def make_band_panel(ax, title_str, b_data, e_data, show_social=True,
                    dot_step=5, title_size=13):
    """Set up a band decomposition panel with scatter dots + progressive mask.

    b_data: dict with 'raw', 'infraslow', 'slow', 'delta', 'theta'
    e_data: dict with 'infraslow', 'slow', 'delta', 'theta'
    """
    from matplotlib.patches import Rectangle
    style(ax, grid=False)
    ax.set_title(title_str, fontsize=title_size, fontweight='bold',
                 color=TEXT_C, pad=4)

    spacing = 2.8
    bands_list = [
        ('Raw trace',               b_data['raw'],       '#8b949e', None),
        ('Infraslow  0.01\u20130.1 Hz',  b_data['infraslow'],
         BAND_COLORS['infraslow'],  e_data['infraslow']),
        ('Slow  0.1\u20131 Hz',          b_data['slow'],
         BAND_COLORS['slow'],       e_data['slow']),
        ('Delta  1\u20134 Hz',           b_data['delta'],
         BAND_COLORS['delta'],      e_data['delta']),
        ('Theta  4\u20137 Hz',           b_data['theta'],
         BAND_COLORS['theta'],      e_data['theta']),
    ]

    for i, (lab, sig, col, env) in enumerate(bands_list):
        off = (len(bands_list) - 1 - i) * spacing
        pk  = np.max(np.abs(sig)) + 1e-12
        sn  = sig / pk * 1.1
        ax.plot(t_cal, sn + off, color=col, lw=0.8, alpha=0.9)
        if env is not None:
            en = env / pk * 1.1
            ax.fill_between(t_cal, -en + off, en + off,
                            color=col, alpha=0.18)
        # Scatter dots visible above mask
        ax.scatter(t_cal[::dot_step], (sn + off)[::dot_step],
                   s=2, color=col, alpha=0.4, zorder=6, linewidths=0)
        ax.text(0.3, off + 0.15, lab, color=col, fontsize=10,
                va='bottom', fontweight='bold', zorder=7)

    # Social bar at bottom
    bar_y = -spacing * 0.55
    if show_social:
        ax.fill_between(t_cal, bar_y - 0.4, bar_y + 0.4,
                        where=beh_win5.astype(bool),
                        color='#f85149', alpha=0.6, step='mid')
        ax.fill_between(t_cal, bar_y - 0.4, bar_y + 0.4,
                        where=~beh_win5.astype(bool),
                        color='#388bfd', alpha=0.2, step='mid')
        ax.text(0.3, bar_y + 0.15, 'Social / Solo', color=TEXT_C,
                fontsize=10, va='bottom', fontweight='bold', zorder=7)

    ylo = bar_y - 1.2 if show_social else -1.0
    yhi = (len(bands_list) - 1) * spacing + 1.8
    ax.set_xlim(0, DURATION_S)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel('Time (s)', fontsize=11, color=TEXT_C)
    ax.set_yticks([])

    # Mask rectangle to progressively reveal
    mask = Rectangle((0, ylo), DURATION_S, yhi - ylo,
                      fc=PANEL_BG, ec='none', zorder=4)
    ax.add_patch(mask)
    cursor = ax.axvline(0, color='#f0f6fc', lw=2, alpha=0.9, zorder=5)
    return cursor, mask


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO 1 — SLEAP + Band Decomposition + Spatial per cluster
# ═══════════════════════════════════════════════════════════════════════════
print("[7/8] Rendering Video 1 — SLEAP + Bands + Spatial …")
fig1 = plt.figure(figsize=(OUTPUT_W / DPI, OUTPUT_H / DPI), dpi=DPI,
                  facecolor=BG)

gs1 = GridSpec(3, 2, height_ratios=[0.8, 1, 0.8], width_ratios=[1, 1],
               left=0.04, right=0.98, top=0.94, bottom=0.03,
               hspace=0.14, wspace=0.08, figure=fig1)

# -- Row 0: SLEAP video (spans both columns) --
cap = cv2.VideoCapture(str(DATA_DIR / 'behavior_video.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_beh_vid)
_, first = cap.read()
first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)

ax1_vid  = fig1.add_subplot(gs1[0, :])
im1_vid  = setup_video_panel(ax1_vid, first_rgb,
                             'SLEAP Tracked Behavior (Session 5)')
txt1_vid = ax1_vid.text(0.02, 0.92, '', transform=ax1_vid.transAxes,
                        fontsize=11, color='white', fontweight='bold',
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.65))

# -- Row 1: Band decomposition per cluster (population mean) --
ax1_b0 = fig1.add_subplot(gs1[1, 0])
ax1_b1 = fig1.add_subplot(gs1[1, 1])
cur1_b0, mask1_b0 = make_band_panel(
    ax1_b0,
    f'Cluster 0 — Low-Frequency  (n={len(c0_s5)} neurons)',
    pop_bands[0], pop_envs[0], show_social=True, title_size=12)
cur1_b1, mask1_b1 = make_band_panel(
    ax1_b1,
    f'Cluster 1 — High-Frequency  (n={len(c1_s5)} neurons)',
    pop_bands[1], pop_envs[1], show_social=True, title_size=12)

# -- Row 2: Spatial activation per cluster --
ax1_s0 = fig1.add_subplot(gs1[2, 0])
ax1_s0.set_facecolor('black')
ax1_s0.imshow(Cn_n, cmap='gray', alpha=0.6)
im1_c0 = ax1_s0.imshow(np.zeros((*ms, 4)))
ax1_s0.set_xticks([]); ax1_s0.set_yticks([])
for sp in ax1_s0.spines.values():
    sp.set_color(SPINE_C)
ax1_s0.set_title(f'Cluster 0 — Low-Freq Spatial  (n={len(c0_s5)})',
                 fontsize=12, fontweight='bold', color=TEXT_C, pad=3)

ax1_s1 = fig1.add_subplot(gs1[2, 1])
ax1_s1.set_facecolor('black')
ax1_s1.imshow(Cn_n, cmap='gray', alpha=0.6)
im1_c1 = ax1_s1.imshow(np.zeros((*ms, 4)))
ax1_s1.set_xticks([]); ax1_s1.set_yticks([])
for sp in ax1_s1.spines.values():
    sp.set_color(SPINE_C)
ax1_s1.set_title(f'Cluster 1 — High-Freq Spatial  (n={len(c1_s5)})',
                 fontsize=12, fontweight='bold', color=TEXT_C, pad=3)

fig1.suptitle('Neural Activity During Social Interaction',
              fontsize=16, color=TEXT_C, fontweight='bold', y=0.99)

skel_art1 = []

out1_tmp = os.path.join(OUT_DIR, 'video1_tmp.mp4')
out1     = os.path.join(OUT_DIR, 'sleap_spatial.mp4')
os.makedirs(OUT_DIR, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
w1 = cv2.VideoWriter(out1_tmp, fourcc, VIDEO_FPS, (OUTPUT_W, OUTPUT_H))

for fi in range(N_FRAMES):
    t       = fi / VIDEO_FPS
    cal_idx = min(int(t * IMAGING_FPS), n_cal - 1)
    vid_idx = start_beh_vid + fi

    # Video frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
    ok, frm = cap.read()
    if ok:
        im1_vid.set_data(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    for a in skel_art1:
        a.remove()
    skel_art1.clear()
    draw_skeleton(ax1_vid, vid_idx, skel_art1)
    txt1_vid.set_text(f't = {START_POST_ENTRY_S + t:.1f}s')

    # Progressive reveal on band panels
    remaining = DURATION_S - t
    mask1_b0.set_x(t);  mask1_b0.set_width(remaining)
    mask1_b1.set_x(t);  mask1_b1.set_width(remaining)
    cur1_b0.set_xdata([t, t])
    cur1_b1.set_xdata([t, t])

    # Spatial maps — separate per cluster
    m0, m1 = spatial_maps(cal_idx)

    rgba0 = np.zeros((*ms, 4))
    rgba0[..., 2] = 1.0
    rgba0[..., 3] = np.clip(m0 / (vmax0 + 1e-12), 0, 1) * 0.6
    im1_c0.set_data(rgba0)

    rgba1 = np.zeros((*ms, 4))
    rgba1[..., 0] = 1.0
    rgba1[..., 3] = np.clip(m1 / (vmax1 + 1e-12), 0, 1) * 0.75
    im1_c1.set_data(rgba1)

    fig1.canvas.draw()
    buf = np.asarray(fig1.canvas.buffer_rgba())[:, :, :3]
    w1.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

    if (fi + 1) % 50 == 0 or fi == 0:
        print(f"  V1 frame {fi + 1:3d}/{N_FRAMES}  ({(fi + 1) / N_FRAMES * 100:.0f}%)")

w1.release()
cap.release()
plt.close(fig1)
print("  Re-encoding Video 1 with H.264 …")
os.system(f'ffmpeg -y -i "{out1_tmp}" -c:v libx264 -preset fast -crf 18 '
          f'-pix_fmt yuv420p "{out1}" -loglevel warning')
os.remove(out1_tmp)
print(f"  Video 1 → {os.path.abspath(out1)}")


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO 2 — SLEAP + Distance + Band Decomposition (2 neurons)
# ═══════════════════════════════════════════════════════════════════════════
print("[8/8] Rendering Video 2 — SLEAP + Distance + Bands …")
fig2 = plt.figure(figsize=(OUTPUT_W / DPI, OUTPUT_H / DPI), dpi=DPI,
                  facecolor=BG)

gs2 = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.1],
               left=0.05, right=0.98, top=0.93, bottom=0.05,
               wspace=0.10, hspace=0.18, figure=fig2)

# -- top-left: SLEAP video --
cap2 = cv2.VideoCapture(str(DATA_DIR / 'behavior_video.mp4'))
cap2.set(cv2.CAP_PROP_POS_FRAMES, start_beh_vid)
_, first2 = cap2.read()
first2_rgb = cv2.cvtColor(first2, cv2.COLOR_BGR2RGB)

ax2_vid  = fig2.add_subplot(gs2[0, 0])
im2_vid  = setup_video_panel(ax2_vid, first2_rgb,
                             'SLEAP Tracked Behavior')
txt2_vid = ax2_vid.text(0.02, 0.96, '', transform=ax2_vid.transAxes,
                        fontsize=11, color='white', fontweight='bold',
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.65))

# -- top-right: distance + social bar --
from matplotlib.patches import Rectangle

dist_gs = GridSpecFromSubplotSpec(2, 1, gs2[0, 1],
                                  height_ratios=[3, 1], hspace=0.10)
ax2_dist = fig2.add_subplot(dist_gs[0])
ax2_sbar = fig2.add_subplot(dist_gs[1], sharex=ax2_dist)

style(ax2_dist)
ax2_dist.plot(t_vid, distances, color='#58a6ff', lw=0.8, alpha=0.85)
# Scatter dots visible from the start (above mask)
ax2_dist.scatter(t_vid[::2], distances[::2], s=3, color='#58a6ff',
                 alpha=0.35, zorder=6, linewidths=0)
ax2_dist.axhline(DIST_THRESH, color='#f85149', lw=2, ls='--',
                 label=f'Threshold = {DIST_THRESH} px', zorder=7)
ymax_d = min(180, np.nanmax(distances) * 1.1) if np.any(~np.isnan(distances)) else 180
ax2_dist.set_xlim(0, DURATION_S)
ax2_dist.set_ylim(0, ymax_d)
ax2_dist.set_ylabel('Distance (px)', fontsize=11, color=TEXT_C)
ax2_dist.set_title('Nose \u2192 Body Distance', fontsize=14,
                   fontweight='bold', color=TEXT_C, pad=4)
ax2_dist.tick_params(axis='x', labelbottom=False)
ax2_dist.legend(loc='upper right', fontsize=10, framealpha=0.7,
                facecolor=PANEL_BG, edgecolor=SPINE_C, labelcolor=TEXT_C)
# Mask rectangle to progressively reveal the distance trace
mask_dist = Rectangle((0, 0), DURATION_S, ymax_d,
                       fc=PANEL_BG, ec='none', zorder=4)
ax2_dist.add_patch(mask_dist)
cur_dist = ax2_dist.axvline(0, color='#f0f6fc', lw=2, alpha=0.9, zorder=5)

style(ax2_sbar, grid=False)
is_close = distances < DIST_THRESH
ax2_sbar.fill_between(t_vid, 0, is_close.astype(float),
                      color='#f85149', alpha=0.7, step='mid')
ax2_sbar.set_xlim(0, DURATION_S)
ax2_sbar.set_ylim(-0.05, 1.15)
ax2_sbar.set_ylabel('Social', fontsize=11, color=TEXT_C)
ax2_sbar.set_xlabel('Time (s)', fontsize=11, color=TEXT_C)
ax2_sbar.set_yticks([0, 1])
ax2_sbar.set_yticklabels(['Solo', 'Social'], fontsize=9, color=MUTED_C)
# Mask for social bar
mask_sbar = Rectangle((0, -0.05), DURATION_S, 1.2,
                       fc=PANEL_BG, ec='none', zorder=4)
ax2_sbar.add_patch(mask_sbar)
cur_sbar = ax2_sbar.axvline(0, color='#f0f6fc', lw=2, alpha=0.9, zorder=5)


# -- bottom panels: band decomposition for 2 neurons --
iso_label = entrances.iloc[SESSION_VID]['Isolation Length']
ax2_b0 = fig2.add_subplot(gs2[1, 0])
ax2_b1 = fig2.add_subplot(gs2[1, 1])
cur_b0, mask_b0 = make_band_panel(
    ax2_b0,
    f'Cluster 0 (Low-Freq) — Animal {animal_s5}, Neuron #{rep_neurons[0]}  ({iso_label})',
    neuron_bands[0], neuron_envs[0])
cur_b1, mask_b1 = make_band_panel(
    ax2_b1,
    f'Cluster 1 (High-Freq) — Animal {animal_s5}, Neuron #{rep_neurons[1]}  ({iso_label})',
    neuron_bands[1], neuron_envs[1])

fig2.suptitle('Spectral Decomposition of Individual Neurons',
              fontsize=17, color=TEXT_C, fontweight='bold', y=0.99)

skel_art2 = []

out2_tmp = os.path.join(OUT_DIR, 'video2_tmp.mp4')
out2     = os.path.join(OUT_DIR, 'distance_bands.mp4')
w2 = cv2.VideoWriter(out2_tmp, fourcc, VIDEO_FPS, (OUTPUT_W, OUTPUT_H))

for fi in range(N_FRAMES):
    t       = fi / VIDEO_FPS
    vid_idx = start_beh_vid + fi

    # Video frame
    cap2.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
    ok, frm = cap2.read()
    if ok:
        im2_vid.set_data(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    for a in skel_art2:
        a.remove()
    skel_art2.clear()
    draw_skeleton(ax2_vid, vid_idx, skel_art2)
    txt2_vid.set_text(f't = {START_POST_ENTRY_S + t:.1f}s')

    # Update mask rectangles — reveal data up to current time
    remaining = DURATION_S - t
    mask_dist.set_x(t)
    mask_dist.set_width(remaining)
    mask_sbar.set_x(t)
    mask_sbar.set_width(remaining)
    mask_b0.set_x(t)
    mask_b0.set_width(remaining)
    mask_b1.set_x(t)
    mask_b1.set_width(remaining)

    # Cursors at leading edge
    cur_dist.set_xdata([t, t])
    cur_sbar.set_xdata([t, t])
    cur_b0.set_xdata([t, t])
    cur_b1.set_xdata([t, t])

    fig2.canvas.draw()
    buf = np.asarray(fig2.canvas.buffer_rgba())[:, :, :3]
    w2.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

    if (fi + 1) % 50 == 0 or fi == 0:
        print(f"  V2 frame {fi + 1:3d}/{N_FRAMES}  ({(fi + 1) / N_FRAMES * 100:.0f}%)")

w2.release()
cap2.release()
plt.close(fig2)
# Re-encode H.264
print("  Re-encoding Video 2 with H.264 …")
os.system(f'ffmpeg -y -i "{out2_tmp}" -c:v libx264 -preset fast -crf 18 '
          f'-pix_fmt yuv420p "{out2}" -loglevel warning')
os.remove(out2_tmp)
print(f"  Video 2 → {os.path.abspath(out2)}")
print("\nDone!")
