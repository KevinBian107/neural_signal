"""Data loading, downloading, and session alignment."""

import numpy as np
import pandas as pd
import h5py
import gdown
from pathlib import Path

from .constants import BEHAVIOR_FPS, IMAGING_FPS


DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw'

FILES = {
    'calcium.00.h5':                 '1UthpsskvkHbKKDsbQjUxyVN4Xkd-ZJuN',
    'social_bouts.00.h5':            '1Mh8oGKNyKpT5WS0Wu92SULFFanvqmSMf',
    'SI3_2022_Entrance_Frames.xlsx': '1POpRqpA_QaWfZhxswQvLSs9uBnnHrmhZ',
    'spatial_footprints.00.h5':      '1PYLeqT88IH_9JWNPwYaUC9WJVT2IqevL',
    'behavior_video.mp4':            '1SAfse1kJU4AGk8AFxbt34GsElpUxCFMj',
    'behavior_tracking.slp':         '1ROllUZbwevCP3oAjSNwbxq-zVotzzxJ4',
}

BEHAVIOR_KEYS = [
    'is_ag_sniffed', 'is_ag_sniffing', 'is_of_sniffed', 'is_of_sniffing',
    'is_social', 'is_social_receiver', 'is_social_sender',
    'is_touched', 'is_touching',
]


def download_data(data_dir=None):
    """Download raw data files from Google Drive if not present."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_id in FILES.items():
        out_path = data_dir / filename
        if out_path.exists():
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f'  [skip] {filename} ({size_mb:.1f} MB)')
        else:
            print(f'  Downloading {filename}...')
            gdown.download(
                f'https://drive.google.com/uc?id={file_id}',
                str(out_path), quiet=False,
            )
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f'  [ok] {filename} ({size_mb:.1f} MB)')

    return data_dir


def load_entrances(data_dir=None):
    """Load session entrance metadata."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    entrances = pd.read_excel(data_dir / 'SI3_2022_Entrance_Frames.xlsx')
    return entrances


def load_behavior(n_sessions, data_dir=None):
    """Load behavioral label arrays for all sessions."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    behavior = []
    with h5py.File(data_dir / 'social_bouts.00.h5', 'r') as f:
        for sess in range(n_sessions):
            sd = {}
            for key in BEHAVIOR_KEYS:
                sd[key] = f[f'session_{sess}'][key][:]
            behavior.append(sd)
    return behavior


def load_imaging(n_sessions, data_dir=None):
    """Load calcium imaging matrices for all sessions."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    imaging = []
    with h5py.File(data_dir / 'calcium.00.h5', 'r') as f:
        for sess in range(n_sessions):
            C = f[f'session_{sess}']['C'][:]
            imaging.append(C)
    return imaging


def align_session(calcium_C, beh_dict, entry_beh, beh_fps=BEHAVIOR_FPS,
                  img_fps=IMAGING_FPS):
    """Align behavior labels to calcium imaging for one session.

    Returns (calcium_cropped, behavior_resampled) or (None, None) if
    the entry frame exceeds the calcium recording length.
    """
    entry_img = int(entry_beh * (img_fps / beh_fps))
    if entry_img >= calcium_C.shape[0]:
        return None, None

    cal = calcium_C[entry_img:]
    beh = beh_dict['is_social'][int(entry_beh):]

    # Nearest-neighbor resample: beh_fps -> img_fps
    n_beh = len(beh)
    target = int(n_beh * (img_fps / beh_fps))
    idx = np.round(np.linspace(0, n_beh - 1, target)).astype(int)
    beh_rs = beh[idx]

    common = min(len(cal), len(beh_rs))
    return cal[:common], beh_rs[:common]


def align_all_sessions(imaging, behavior, entrances,
                       beh_fps=BEHAVIOR_FPS, img_fps=IMAGING_FPS):
    """Align all sessions. Returns aligned_calcium, aligned_behavior, session_info."""
    aligned_calcium = []
    aligned_behavior = []
    session_info = []
    n_sessions = len(entrances)

    for i in range(n_sessions):
        entry = int(entrances.iloc[i]['Int_Entry'])
        cal, beh = align_session(imaging[i], behavior[i], entry, beh_fps, img_fps)
        if cal is None:
            print(f'  Session {i:2d}: SKIPPED')
            continue

        aligned_calcium.append(cal)
        aligned_behavior.append(beh)
        info = {
            'session_idx': i,
            'animal': entrances.iloc[i]['Animal'],
            'isolation': entrances.iloc[i]['Isolation Length'],
            'n_frames': len(cal),
            'n_neurons': cal.shape[1],
            'duration_s': len(cal) / img_fps,
            'social_frac': beh.mean(),
        }
        session_info.append(info)
        print(f'  Session {i:2d}: {info["n_frames"]:6d} frames, '
              f'{info["n_neurons"]:3d} neurons, '
              f'{info["duration_s"]:.0f}s, '
              f'{info["social_frac"]*100:.1f}% social, '
              f'{info["isolation"]}')

    session_df = pd.DataFrame(session_info)
    print(f'\nAligned {len(aligned_calcium)} / {n_sessions} sessions.')
    return aligned_calcium, aligned_behavior, session_info, session_df


def load_spatial_footprints(data_dir=None):
    """Load spatial footprints (A) and correlation image (Cn) for Session 0.

    Returns dict with keys: A (n_neurons, 600, 600), Cn (600, 600), Fs.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    path = data_dir / 'spatial_footprints.00.h5'
    with h5py.File(path, 'r') as f:
        A_flat = f['A'][:]        # (n_neurons, 360000)
        Cn = f['Cn'][:]           # (600, 600)
        Fs = float(f['Fs'][:].flat[0])
    n_neurons = A_flat.shape[0]
    side = int(np.sqrt(A_flat.shape[1]))
    A = A_flat.reshape(n_neurons, side, side)
    return {'A': A, 'Cn': Cn, 'Fs': Fs, 'n_neurons': n_neurons}


def get_epoch_durations(labels, fps):
    """Extract epoch boundaries and durations from a binary label array.

    Returns (starts, ends, labels_per_epoch, durations_s).
    """
    d = np.diff(labels.astype(int))
    bounds = np.where(d != 0)[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(labels)]])
    labs = np.array([labels[s] for s in starts])
    durs = (ends - starts) / fps
    return starts, ends, labs, durs
