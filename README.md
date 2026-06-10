# Seed stability of SAE features in RiNALMo

Sparse autoencoders (SAEs) are used to split a language model's internal activations into
interpretable features. If an SAE trained with a different random seed finds different features,
which of them can be trusted? This project trains 10 SAEs per layer on activations from the RNA
language model RiNALMo (giga, 650M parameters), differing only by seed, and asks (1) what
fraction of features reappear across seeds? and (2) are the features that reappear (stable)
more tied to RNA secondary structure than the ones that don't?

## Installation

```bash
conda env create -f environment.yml
conda activate rna-sae
pytest
```

Runs on one 12 GB GPU (RTX 4070). Activations and checkpoints take about 10 GB in `data/`.

## Data

- Model: [multimolecule/rinalmo-giga](https://huggingface.co/multimolecule/rinalmo-giga) (RiNALMo, Penić et al. 2025)
- Training sequences: [multimolecule/rnacentral.1024](https://huggingface.co/datasets/multimolecule/rnacentral.1024)
- Structure labels: [multimolecule/bprna-90](https://huggingface.co/datasets/multimolecule/bprna-90)

## License

MIT
