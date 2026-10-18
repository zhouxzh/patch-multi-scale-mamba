# Patch/Multi-Scale Mamba

Standalone local research runtime for the PMS-Mamba architecture. This checkout is intentionally separate from the paper/evidence repository and has no configured remote.

## Source and identity

The runtime combines the historical PMS-Mamba snapshot under `programs/pms_mamba/source` with the complete dependency closure from the parent repository. The frozen protocol identity is preserved exactly:

- protocol: `pms_mamba_v3_20260804a`
- config: `configs/pms_mamba_v3.json`
- source provenance: [`provenance.json`](provenance.json)

Retrospective financial configs remain under `configs/`; they are not current SAPR evidence and must not be pooled with SAPR results.

## Commands

Run from this checkout root:

```text
python -m pms_mamba.cli --help
python -m pms_mamba.cli validate-config
python -m pms_mamba.cli preflight --dry-run
python -m pms_mamba.cli model-smoke
```

`validate-config` and `preflight --dry-run` do not write. `model-smoke` constructs a tiny model and runs one forward pass when compatible `torch` and `mamba-ssm` are installed; otherwise it exits with an explicit dependency message. Training requires the optional runtime/research dependencies and controlled data supplied through `--data-root`/`--results-root` or the historical experiment entrypoints.

## Layout

- `pms_mamba/`: standalone validation and smoke CLI.
- `model/`, `layers/`, `experiments/`, `data_provider/`, `utils/`: executable dependency closure.
- `configs/`: frozen protocol and historical config identities.
- `data/`, `results/`, `checkpoints/`, `tensorboard/`: ignored local runtime locations.
- `tests/`: read-only config and launcher tests.

No raw datasets, checkpoints, prediction arrays, TensorBoard files, or experiment results are committed here.
