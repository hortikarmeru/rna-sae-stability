# Seed stability of SAE features in RiNALMo

Sparse autoencoders (SAEs) are used to split a language model's internal activations into
interpretable features. If an SAE trained with a different random seed finds different features,
which of them can be trusted? This project trains 10 SAEs per layer on activations from the RNA
language model RiNALMo (giga, 650M parameters), differing only by seed, and asks (1) what
fraction of features reappear across seeds? and (2) are the features that reappear (stable)
more tied to RNA secondary structure than the ones that don't?

## Status

Complete for layer 9 and layer 18 of RiNALMo-giga.

- [x] Activations: 5,000 RNAcentral sequences (training), 500 bpRNA-90 sequences with
      per-nucleotide structure labels (interpretation)
- [x] L1 SAEs (10 seeds x 2 layers): failed (see Results)
- [x] TopK SAEs (K = 32, 10 seeds x 2 layers)
- [x] Seed stability, 45 seed pairs per layer
- [x] Structural enrichment vs. stability (layer 9, seed 0)

## Results

**L1-penalized SAEs collapsed.** About 99% of the 8,192 features never fired, and the ~100 that
did fired on 70–85% of all nucleotides. Their structural enrichment was 1.01–1.10x, i.e. none,
so stability could not be related to biology with these SAEs.

**TopK SAEs (K = 32) trained well.** About 98% of features are alive, the median feature fires on
0.19% of inputs, and reconstruction explains about 85% of activation variance on layer 9.

| TopK, 45 seed pairs per layer | Layer 9 | Layer 18 |
|---|---|---|
| Alive features per SAE | ~8,050 | ~8,160 |
| Features stable across a seed pair (cosine >= 0.7) | 38.9% | 39.4% |
| Mean matched cosine similarity | 0.59 | 0.60 |

**Stable features are somewhat more structure-specific** (layer 9, seed 0; 6,809 features with
at least 30 firings on bpRNA-90):

| | Stable | Unstable | Test |
|---|---|---|---|
| Mean max enrichment | 3.40x | 2.78x | Welch t = 9.1, p < 1e-18 |
| Same, features with >= 100 firings | 2.69x | 2.46x | p = 3e-4 |
| Pearson r, stability vs. log enrichment | | | r = 0.16 |
| Partial r, controlling for log firing count | | | r = 0.09, p < 1e-12 |

The effect is real but small. Part of the raw correlation comes from firing frequency: rarely
firing features are both less stable and have noisier, more extreme enrichment, so the partial
correlation is the fairer estimate. Exact numbers for every run are in `results/`.

## Installation

```bash
conda env create -f environment.yml
conda activate rna-sae
pytest
```

Runs on one 12 GB GPU (RTX 4070). Activations and checkpoints take about 10 GB in `data/`.

## Pipeline

Scripts are numbered in run order and read every setting from `configs/config.yaml`. Each writes
its output plus a `*.meta.json` with the git commit, the full config and package versions.
Scripts 02–06 take the SAE type, `l1` or `topk`.

| Script | Output |
|---|---|
| `00_cache_activations.py` | `data/activations/`: RiNALMo hidden states, 5,000 RNAcentral sequences |
| `01_cache_bprna_activations.py` | `data/bprna_activations/`: same for bpRNA-90, with structure labels |
| `02_train_saes.py` | `data/checkpoints/<arch>/`: one SAE per layer x seed; `results/<arch>/training.json` |
| `03_feature_diagnostics.py` | `results/<arch>/feature_diagnostics.csv`: alive and selective features |
| `04_stability.py` | `results/<arch>/stability_pairs.csv`, `stability_summary.csv` |
| `05_enrichment.py` | `results/<arch>/enrichment_layer9_seed0.csv`: stability and enrichment per feature |
| `06_enrichment_stats.py` | `results/<arch>/enrichment_stats.json` |

## Repository layout

```
configs/           config.yaml: model, data, SAE and analysis settings
scripts/           numbered pipeline steps
src/rna_sae/       SAE models and training, activation handling, feature matching
tests/             pytest
results/           result tables for the L1 and TopK SAEs
```

## Data

- Model: [multimolecule/rinalmo-giga](https://huggingface.co/multimolecule/rinalmo-giga) (RiNALMo, Penić et al. 2025)
- Training sequences: [multimolecule/rnacentral.1024](https://huggingface.co/datasets/multimolecule/rnacentral.1024)
- Structure labels: [multimolecule/bprna-90](https://huggingface.co/datasets/multimolecule/bprna-90)

## License

MIT
