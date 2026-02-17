# Decoding Social Intent from Neural Oscillations

**COGS 118C — Signal Processing Course Project**

## Research Question

Can we classify whether an animal is interacting socially vs. exploring alone based on the spectral features of its neural calcium signals?

## Background

Electrophysiology studies show that theta-band (4-7 Hz) power increases in the prefrontal cortex during social contexts (Tzilivaki et al. 2022, eLife). We test whether these spectral signatures are recoverable from calcium imaging data — a slower, indirect proxy for neural spiking — using signal processing techniques appropriate for the data's temporal resolution.

## Approach

1. **Preprocessing** — Detrend (remove photobleaching), correct motion artifacts, z-score normalize calcium traces (ΔF/F)
2. **Epoching** — Segment recordings into social and solo windows using behavioral annotations
3. **Spectral feature extraction** — Compute PSD (Welch's method), wavelet scalograms (Morlet CWT), bandpass-filtered amplitude in frequency bands (infraslow, delta, theta)
4. **Classification** — Train linear classifiers (LDA, SVM, Logistic Regression) on spectral features with block cross-validation
5. **Validation** — Permutation tests for statistical significance, movement-speed confound control

## Hypothesis

Spectral features of neuronal calcium signals — specifically power in the infraslow (0.01-0.1 Hz), delta (1-4 Hz), and theta (4-7 Hz) bands, as well as spectral entropy — differ systematically between social interaction and solo exploration epochs, and are sufficient for above-chance binary classification.

## Project Structure

```
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── data/                      # Downloaded data (gitignored)
│   ├── raw/                   # Raw data files
│   └── source_notebooks/      # EDGE course notebooks
├── data.md                    # Links to EDGE data sources
├── .claude/                   # Propel workflow config
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
- **Spectral feature engineering** — Band power, spectral entropy, peak frequency, spectral centroid, band power ratios

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
