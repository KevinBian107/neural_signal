# Decoding Social Intent from Neural Oscillations

**COGS 118C — Signal Processing Course Project**

## Research Question

Can we classify whether an animal is interacting socially vs. exploring alone based on the spectral features of its neural calcium signals?

## Key Questions

1. **Band-Behavior Link:** Do specific frequency bands of the population-average calcium signal differentiate social from solo epochs?
2. **Neuron-Band Heterogeneity:** Do different subpopulations of neurons contribute differently to each frequency band?
3. **Spectral Subpopulation → Classification:** Do neuron subpopulations with stronger high-frequency spectral profiles drive behavior classification?

## Key Findings

- **Theta (4-7 Hz) is the dominant band** for social-vs-solo discrimination (Cohen's d = +0.235, p = 6.5e-17 after Bonferroni correction), consistent across 14/18 sessions
- **Neurons split into two spectral clusters:** a majority (70%) with power below 1 Hz, and a minority (30%) with 3x more delta+theta power — present in all sessions
- **Only the 30% high-frequency neurons significantly classify behavior** (permutation p = 0.020). Using all neurons fails the permutation test — the majority population dilutes the signal
- The high-frequency subpopulation uses **theta/delta ratio** (spectral shape) as its top feature, while the full population relies on raw theta power — qualitatively different coding strategies

## Background

Electrophysiology studies show that theta-band (4-7 Hz) power increases in the **prefrontal cortex** during social contexts (Tzilivaki et al. 2022, eLife). We test whether these spectral signatures are recoverable from calcium imaging data — a slower, indirect proxy for neural spiking — using signal processing techniques appropriate for the data's temporal resolution.

## Approach

1. **Preprocessing** — Detrend (remove photobleaching), correct motion artifacts, z-score normalize calcium traces (ΔF/F)
2. **Epoching** — Segment recordings into social and solo windows (1s, 70% purity) using behavioral annotations
3. **Spectral feature extraction** — Compute PSD (Welch's method), wavelet scalograms (Morlet CWT), bandpass-filtered amplitude in frequency bands (infraslow, slow, delta, theta)
4. **Neuron clustering** — Cluster neurons by spectral profile (k-means on fractional band power)
5. **Classification** — Train linear classifiers (LDA, SVM, Logistic Regression) on spectral features with GroupKFold cross-validation by session
6. **Validation** — Permutation tests for statistical significance

## Hypothesis

Spectral features of neuronal calcium signals — specifically power in the infraslow (0.01-0.1 Hz), delta (1-4 Hz), and theta (4-7 Hz) bands, as well as spectral entropy — differ systematically between social interaction and solo exploration epochs, and are sufficient for above-chance binary classification using a linear classifier.

## Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   └── 02_band_neuron_behavior.ipynb   # Band-neuron-behavior investigation
├── src/
│   ├── constants.py                    # Shared constants (FPS, freq bands, colors)
│   ├── data.py                         # Data loading, alignment, epoch extraction
│   ├── signal_processing.py            # Filtering, PSD, CWT, band power
│   ├── analysis.py                     # Windowed extraction, clustering, classification
│   └── visualization.py                # Plotting helpers
├── data/                               # Downloaded data (gitignored)
│   └── raw/                            # Raw HDF5 + Excel files
├── scratch/                            # Investigation working directory (gitignored)
├── data.md                             # Links to EDGE data sources
├── .claude/                            # Propel workflow config
├── environment.yml                     # Conda environment spec
└── README.md
```

## Data Sources

Data comes from EDGE (Education in Data and Guided Exploration) course notebooks:

| Notebook | Content |
|----------|---------|
| Finding social behaviors | Behavioral annotations (social vs. non-social labels) |
| Calcium demo | Raw calcium fluorescence traces |
| Demixed calcium | Source-separated calcium signals |
| Neural signals of social isolation | Calcium traces + social condition labels |
| Analyzing social isolation | Behavioral analysis of isolated vs. group-housed animals |
| social_bouts.00 | Compiled social bout timing data |

See `data.md` for full links.

## Key Signal Processing Methods

- **Welch PSD** — Primary spectral estimation (scipy.signal.welch)
- **Morlet wavelet CWT** — Time-frequency analysis for non-stationary signals (pywt)
- **Butterworth bandpass filtering** — Frequency band isolation (scipy.signal)
- **Spectral feature engineering** — Band power, spectral entropy, theta/delta ratio
- **K-means clustering** — Neuron subpopulation identification by spectral profile

## Setup

```bash
conda env create -f environment.yml
conda activate neural_signal
```

Then launch the notebooks:

```bash
jupyter notebook notebooks/
```

## References

- Tzilivaki et al. (2022). Prefrontal-amygdalar oscillations related to social behavior in mice. *eLife*.
- Bhatt et al. (2013). Bhatt DH, et al. Spectral analysis of calcium oscillations. *Frontiers in Neural Circuits*.
- Cohen MX (2019). A better way to define and describe Morlet wavelets. *NeuroImage*.
