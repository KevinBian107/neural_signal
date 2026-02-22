"""Shared constants for neural signal analysis."""

BEHAVIOR_FPS = 25
IMAGING_FPS = 30

FREQ_BANDS = {
    'infraslow': (0.01, 0.1),
    'slow': (0.1, 1.0),
    'delta': (1.0, 4.0),
    'theta': (4.0, 7.0),
}

BAND_COLORS = {
    'infraslow': '#f39c12',
    'slow': '#2ecc71',
    'delta': '#3498db',
    'theta': '#9b59b6',
}

ISO_COLORS = {
    'GH (7d)': '#3498db',
    'GH (24hr)': '#2ecc71',
    '24hr': '#f39c12',
    '7d': '#e74c3c',
}

MIN_BOUT_S = 0.5
PRE_S = 3.0
POST_S = 3.0
