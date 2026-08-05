# UAV Identity-Preserving Tracking Reproduction

This repository provides a self-contained reproduction package for UAV
multi-object tracking and identity-preservation experiments. It includes the
bundled input data, runnable scripts, model files, tracker implementations,
and TrackEval.

The package is portable: runtime paths are resolved from the repository
location, so it can be cloned or moved to another directory without editing
machine-specific paths.

## Run the full comparison

From the repository root:

```powershell
python -m pip install -r requirements.txt
python runall.py
```

The command runs the complete comparison on all included datasets and writes
the tables and evaluation summaries to the generated output directory.

To enable the separate ablation run:

```powershell
python runall.py --ablation --force
```

For a CPU-only environment, set the execution device in the runner as needed.
The package was tested with Python 3.10 and CUDA-enabled PyTorch; CPU
execution is also supported but may take longer.

## Included datasets

The package contains six UAV tracking datasets:

- UAVSwarm-39
- UAVSwarm-25
- UAVSwarm-61
- Watercraft
- Court-3
- Court-2

Each sequence is stored under `data/k3_sequences/` with its images,
detections, and annotations.

## Repository layout

```text
runall.py                      one-command entrypoint
code/                          final reproduction code
data/k3_sequences/             six bundled datasets
deps/TrackEval/                bundled evaluation code
weights/                       required model files
```

Generated tables, tracker outputs, logs, caches, and temporary evaluation
files are created locally and are not part of the repository.
