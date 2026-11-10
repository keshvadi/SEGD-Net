# SEGD-Net

Reference implementation of the Semi-Supervised Ensemble Graph Distillation
Network for breast histopathology experiments. The graph filtration is a
sequence of expanding spatial contexts ("pseudo-time"), not a temporal
acquisition of the tissue.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The package does not download datasets or pretrained weights automatically.
Put BreaKHis or BACH images in the directory structure accepted by
`segdnet.data.datasets`; see the CLI help for dataset arguments. Patient/slide
identifiers are used as split groups where they can be recovered. BACH does
not provide patient IDs in its standard image folders, so each image is a
separate group there.

## Quick checks

```bash
pytest tests/test_smoke_pipeline.py
python scripts/generate_xai_fig8.py --output ../images/Fig8.png
python scripts/generate_training_time_fig9.py --output ../images/Fig9.png
```

The smoke test uses synthetic images and does not establish clinical
performance. Dataset-level accuracy claims require running the experiments on
the public datasets with the documented split protocol.

## Training

```bash
python scripts/train_eval_ssl.py \
  --dataset breakhis --data-root /path/to/BreaKHis \
  --config configs/breakhis_segdnet.yaml --output runs/breakhis
```

The supplied configurations specify the requested model defaults. Use
`run_all_experiments.sh` to launch the training, noise-sweep, ablation, and XAI
commands in sequence. Results are written to the selected output directory.

## Package map

- `segdnet/data`: image datasets, patient/slide folds, SSL splits, augmentations
- `segdnet/preprocessing`: Macenko normalization and hybrid nuclear segmentation
- `segdnet/features`: 16 morphology + 4 texture descriptors and node embeddings
- `segdnet/models`: graph filtration, recurrent GNN, CNNs, heterogeneous ensemble
- `segdnet/training`: dual-criterion filtering, losses, Algorithm 1 trainer
- `segdnet/evaluation`: metrics, paired tests, ablations, runtime profiling
- `segdnet/explainability`: Grad-CAM and integrated-gradient graph attributions

## Data, ethics, and interpretation

Only publicly available, de-identified benchmark data should be used. The
included XAI illustration is synthetic and is not a model prediction, a
pathology slide, or evidence of clinical validity. SEGD-Net is research
software, not a medical device or diagnostic aid.

## License

MIT; see `LICENSE`.
