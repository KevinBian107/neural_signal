"""Visualization helpers for neural signal analysis."""

import numpy as np
import matplotlib.pyplot as plt

from .constants import FREQ_BANDS, BAND_COLORS


def shade_social(ax, beh, t):
    """Add red shading for social epochs on an axes."""
    mask = beh.astype(bool)
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], alpha=0.15, color='#e74c3c')


def add_band_shading(ax, freq_bands=None, band_colors=None):
    """Add transparent frequency-band shading to an axes."""
    if freq_bands is None:
        freq_bands = FREQ_BANDS
    if band_colors is None:
        band_colors = BAND_COLORS
    for name, (lo, hi) in freq_bands.items():
        ax.axvspan(lo, hi, alpha=0.08, color=band_colors[name])


def add_band_hlines(ax, freq_bands=None, max_freq=None, color='white',
                    lw=0.5, alpha=0.4, ls='--'):
    """Add horizontal dashed lines at band boundaries."""
    if freq_bands is None:
        freq_bands = FREQ_BANDS
    for _, (lo, hi) in freq_bands.items():
        if max_freq and hi > max_freq:
            continue
        ax.axhline(lo, color=color, lw=lw, alpha=alpha, ls=ls)
