# [Decoding Social Intent from Neural Oscillations](https://kbian.org/neural_signal/)

## Research Question

Can we classify whether an animal is interacting socially vs. exploring alone based on the spectral features of its neural calcium signals?
- **Band–Behavior Link:** Do specific frequency bands of the calcium signal differentiate social from solo epochs?
- **Neuron–Band Heterogeneity:** Do neurons split into subpopulations with distinct spectral profiles?
- **Subpopulation Classification:** Does a spectrally defined subpopulation drive behavior classification?

## Key Findings

- **Theta (4-7 Hz) is the dominant band** for social-vs-solo discrimination (Cohen's d = 0.235, p < 0.001)
- **Neurons split into two spectral clusters:** a slow-dominated majority (70%) and a theta-enriched minority (30%)
- **Only the 30% minority significantly classifies behavior** (AUC = 0.570, permutation p = 0.020) — the majority dilutes the signal
- The minority uses **theta/delta ratio** (spectral shape) as its top feature, not raw power

## Pipeline

1. **Preprocess** — Detrend, z-score calcium traces (30 Hz, 3,938 neurons, 18 sessions)
2. **Extract features** — Welch PSD → band powers + spectral entropy + theta/delta ratio
3. **Cluster neurons** — K-means on fractional band power → 2 spectral subpopulations
4. **Classify** — LDA / SVM / LogReg with GroupKFold CV by session
5. **Validate** — Permutation tests (100 shuffles)

## Setup

```bash
conda env create -f environment.yml
conda activate neural_signal
jupyter notebook notebooks/
```

## Project Structure

```
src/           Analysis modules (data, signal processing, classification)
notebooks/     Jupyter notebooks (EDA → final report)
docs/          Project website
data/raw/      HDF5 data files (gitignored)
scripts/       Figure and video export
```

Data from EDGE (Talmo Pereira and colleagues), UC San Diego.
