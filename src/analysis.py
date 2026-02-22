"""Analysis functions: windowed feature extraction, clustering, classification."""

import numpy as np
from scipy.signal import welch, detrend
from scipy.stats import ranksums, entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from .constants import IMAGING_FPS, FREQ_BANDS
from .signal_processing import compute_band_powers


# ---------------------------------------------------------------------------
# Windowed feature extraction
# ---------------------------------------------------------------------------

def extract_labeled_windows(cal, beh, win_sec=1.0, purity=0.70,
                            fs=IMAGING_FPS, neuron_mask=None):
    """Slide non-overlapping windows and label social/solo.

    Parameters
    ----------
    cal : (T, N) calcium array
    beh : (T,) binary social labels
    win_sec : window length in seconds
    purity : fraction threshold to label a window
    fs : sampling rate
    neuron_mask : optional boolean array selecting neurons

    Returns (social_bps, solo_bps) where each is a list of band-power dicts.
    """
    win_frames = int(win_sec * fs)
    nperseg = win_frames
    noverlap = nperseg // 2
    nfft = max(1024, nperseg * 2)
    n_frames = cal.shape[0]

    social_bps, solo_bps = [], []

    for start in range(0, n_frames - win_frames + 1, win_frames):
        end = start + win_frames
        frac = beh[start:end].mean()

        if frac >= purity:
            label = 'social'
        elif frac <= (1 - purity):
            label = 'solo'
        else:
            continue

        cal_win = cal[start:end]
        if neuron_mask is not None:
            cal_win = cal_win[:, neuron_mask]

        cal_dt = detrend(cal_win, axis=0)
        f_w, psd_all = welch(cal_dt.T, fs=fs, nperseg=nperseg,
                             noverlap=noverlap, nfft=nfft, axis=1)
        psd_mean = psd_all.mean(axis=0)
        bp = compute_band_powers(psd_mean, f_w, FREQ_BANDS)

        if label == 'social':
            social_bps.append(bp)
        else:
            solo_bps.append(bp)

    return social_bps, solo_bps


def bp_list_to_array(bp_list, band_names=None):
    """Convert list of band-power dicts to (n_windows, n_bands) array."""
    if band_names is None:
        band_names = list(FREQ_BANDS.keys())
    return np.array([[bp[b] for b in band_names] for bp in bp_list])


def cohens_d(x, y):
    """Cohen's d with pooled standard deviation."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / sp


# ---------------------------------------------------------------------------
# Per-neuron spectral profiling
# ---------------------------------------------------------------------------

def compute_neuron_spectral_profiles(aligned_calcium, session_info, fs=IMAGING_FPS,
                                     nperseg=None):
    """Compute fractional band power profile for every neuron.

    Returns DataFrame with columns: session_idx, neuron_id, animal, isolation,
    infraslow_frac, slow_frac, delta_frac, theta_frac, total_power.
    """
    if nperseg is None:
        nperseg = int(10 * fs)
    noverlap = nperseg // 2
    nfft = max(1024, nperseg * 2)
    band_names = list(FREQ_BANDS.keys())

    rows = []
    for si, cal in enumerate(aligned_calcium):
        n_frames, n_neurons = cal.shape
        info = session_info[si]

        if n_frames < nperseg:
            continue

        for ni in range(n_neurons):
            trace = detrend(cal[:, ni].astype(np.float64))
            freqs, psd = welch(trace, fs=fs, nperseg=nperseg,
                               noverlap=noverlap, nfft=nfft, window='hann')
            bp = compute_band_powers(psd, freqs, FREQ_BANDS)
            total = sum(bp.values())
            if total == 0:
                continue

            row = {
                'session_idx': info['session_idx'],
                'neuron_id': ni,
                'animal': info['animal'],
                'isolation': info['isolation'],
                'total_power': total,
            }
            for band in band_names:
                row[f'{band}_frac'] = bp[band] / total
            rows.append(row)

    import pandas as pd
    return pd.DataFrame(rows)


def cluster_neurons(profile_df, k_range=range(2, 7), random_state=42):
    """Cluster neurons by fractional band power. Returns (labels, centroids, best_k, scores)."""
    band_names = list(FREQ_BANDS.keys())
    frac_cols = [f'{b}_frac' for b in band_names]
    X = profile_df[frac_cols].values

    scores = {}
    models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state, max_iter=500)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
        models[k] = km

    best_k = max(scores, key=scores.get)
    best_km = models[best_k]
    labels = best_km.predict(X)
    return labels, best_km.cluster_centers_, best_k, scores


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def extract_feature_matrix(aligned_calcium, aligned_behavior, session_info,
                           neuron_selector=None, win_sec=1.0, purity=0.70,
                           fs=IMAGING_FPS):
    """Build feature matrix X, labels y, and group indices for classification.

    neuron_selector: callable(si, n_neurons) -> array of neuron indices, or None for all.
    Features: band powers (4) + spectral entropy + theta/delta ratio = 6.
    """
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    band_names = list(FREQ_BANDS.keys())
    win_frames = int(win_sec * fs)
    nperseg = win_frames
    noverlap = nperseg // 2
    nfft = max(1024, nperseg * 2)

    X_list, y_list, g_list = [], [], []

    for si in range(len(aligned_calcium)):
        cal = aligned_calcium[si]
        beh = aligned_behavior[si]

        if neuron_selector is not None:
            neurons = neuron_selector(si, cal.shape[1])
            if len(neurons) == 0:
                continue
            mask = np.zeros(cal.shape[1], dtype=bool)
            mask[neurons] = True
        else:
            mask = np.ones(cal.shape[1], dtype=bool)

        for start in range(0, len(cal) - win_frames, win_frames):
            frac = beh[start:start + win_frames].mean()
            if frac >= purity:
                label = 1
            elif frac <= 1 - purity:
                label = 0
            else:
                continue

            cal_win = detrend(cal[start:start + win_frames][:, mask], axis=0)
            f_w, psd_all = welch(cal_win.T, fs=fs, nperseg=nperseg,
                                 noverlap=noverlap, nfft=nfft, axis=1)
            psd_mean = psd_all.mean(axis=0)

            bp = compute_band_powers(psd_mean, f_w, FREQ_BANDS)
            feats = [bp[b] for b in band_names]

            # Spectral entropy
            psd_norm = psd_mean / (psd_mean.sum() + 1e-12)
            feats.append(entropy(psd_norm + 1e-12))

            # Theta/delta ratio
            delta_p = bp['delta'] if bp['delta'] > 0 else 1e-12
            feats.append(bp['theta'] / delta_p)

            X_list.append(feats)
            y_list.append(label)
            g_list.append(si)

    feat_names = band_names + ['spectral_entropy', 'theta_delta_ratio']
    return np.array(X_list), np.array(y_list), np.array(g_list), feat_names


def run_classification(X, y, groups, n_splits=5):
    """Run LDA, SVM, LogReg with GroupKFold. Returns dict of results."""
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': SVC(kernel='linear', class_weight='balanced', probability=True,
                   random_state=42),
        'LogReg': LogisticRegression(class_weight='balanced', max_iter=1000,
                                     random_state=42),
    }

    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return {}

    gkf = GroupKFold(n_splits=n_splits)
    results = {}

    for name, clf in classifiers.items():
        aucs, accs, coefs = [], [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf_copy = clf.__class__(**clf.get_params())
            clf_copy.fit(X_tr_s, y_tr)

            y_pred = clf_copy.predict(X_te_s)
            accs.append(accuracy_score(y_te, y_pred))

            if len(np.unique(y_te)) >= 2:
                proba = clf_copy.predict_proba(X_te_s)[:, 1]
                aucs.append(roc_auc_score(y_te, proba))

            if name == 'LogReg' and hasattr(clf_copy, 'coef_'):
                coefs.append(clf_copy.coef_[0])

        results[name] = {
            'auc_mean': np.mean(aucs) if aucs else 0,
            'auc_std': np.std(aucs) if aucs else 0,
            'acc_mean': np.mean(accs) if accs else 0,
            'acc_std': np.std(accs) if accs else 0,
        }
        if coefs:
            results[name]['coefs'] = np.mean(coefs, axis=0)

    return results


def run_permutation_test(X, y, groups, n_perm=100, n_splits=5):
    """Permutation test using LogReg with GroupKFold. Returns (actual_auc, perm_aucs, p)."""
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    def _cv_auc(X, y, groups):
        aucs = []
        for tr, te in gkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr])
            X_te = sc.transform(X[te])
            lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
            lr.fit(X_tr, y[tr])
            aucs.append(roc_auc_score(y[te], lr.predict_proba(X_te)[:, 1]))
        return np.mean(aucs) if aucs else 0.5

    actual = _cv_auc(X, y, groups)
    perm_aucs = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        perm_aucs.append(_cv_auc(X, y_perm, groups))

    perm_aucs = np.array(perm_aucs)
    p = (perm_aucs >= actual).sum() / len(perm_aucs)
    return actual, perm_aucs, p
