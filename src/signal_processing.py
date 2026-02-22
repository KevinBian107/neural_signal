"""Signal processing utilities: filtering, PSD, CWT."""

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert, detrend, stft
import pywt

from .constants import IMAGING_FPS, FREQ_BANDS


def bandpass_filter(sig, lo, hi, fs=IMAGING_FPS, order=4):
    """Bandpass filter using SOS form (numerically stable at low frequencies)."""
    nyq = fs / 2
    lo_n, hi_n = lo / nyq, hi / nyq
    lo_n = max(lo_n, 1e-5)
    hi_n = min(hi_n, 0.9999)
    if lo_n >= hi_n:
        return np.zeros_like(sig)
    sos = butter(order, [lo_n, hi_n], btype='band', output='sos')
    return sosfiltfilt(sos, sig)


def compute_welch_psd(data, fs=IMAGING_FPS, nperseg=None, noverlap=None,
                      nfft=None):
    """Compute Welch PSD for a 2-D array (frames x neurons).

    Returns (freqs, psd_array) where psd_array is (neurons x freq_bins).
    """
    if nperseg is None:
        nperseg = int(10 * fs)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = max(1024, nperseg * 2)

    f_w, psd_all = welch(data.T, fs=fs, nperseg=nperseg,
                         noverlap=noverlap, nfft=nfft, axis=1)
    return f_w, psd_all


def compute_stft(sig, fs=IMAGING_FPS, nperseg=None):
    """Compute STFT for a single signal.

    Returns (frequencies, times, Zxx).
    """
    if nperseg is None:
        nperseg = int(5 * fs)
    return stft(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)


def compute_cwt(sig, freqs=None, fs=IMAGING_FPS, wavelet='cmor1.5-1.0'):
    """Compute continuous wavelet transform (Morlet).

    Returns (coefficients, frequencies, power).
    """
    if freqs is None:
        freqs = np.linspace(0.5, 9, 50)
    scales = pywt.central_frequency(wavelet) * fs / freqs
    coeffs, freqs_out = pywt.cwt(sig, scales, wavelet,
                                 sampling_period=1 / fs)
    power = np.abs(coeffs) ** 2
    return coeffs, freqs_out, power


def band_envelope(sig, lo, hi, fs=IMAGING_FPS):
    """Bandpass filter a signal and return its analytic envelope."""
    filt = bandpass_filter(sig, lo, hi, fs)
    return np.abs(hilbert(filt))


def compute_band_powers(psd, freqs, bands=None):
    """Integrate PSD within each frequency band.

    Parameters
    ----------
    psd : 1-D or 2-D array (neurons x freq_bins)
    freqs : 1-D frequency axis
    bands : dict mapping band_name -> (lo, hi)

    Returns dict mapping band_name -> power (scalar or 1-D array).
    """
    if bands is None:
        bands = FREQ_BANDS
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            result[name] = 0.0
            continue
        if psd.ndim == 1:
            result[name] = _trapz(psd[mask], freqs[mask])
        else:
            result[name] = np.array([_trapz(p[mask], freqs[mask])
                                     for p in psd])
    return result
