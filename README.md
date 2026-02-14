# xray_fracture_benchmark

LLM-driven experimentation for X-ray classification on a small open dataset.

## Dataset

This project uses `PneumoniaMNIST` from MedMNIST:
- Modality: chest X-ray
- Task: binary classification (normal vs pneumonia)
- Size: small enough for quick CPU/GPU iteration

## Setup

```powershell
cd C:\Users\Max\code
C:\Users\Max\code\xray_fracture_benchmark_venv\Scripts\Activate.ps1
cd .\xray_fracture_benchmark
python -m pip install -r requirements.txt
```

## Prepare Data

```powershell
python .\scripts\prepare_pneumoniamnist.py --output-dir .\data\pneumoniamnist
```

## Run A Baseline Experiment

```powershell
python .\scripts\run_llm_experiment.py `
  --config .\configs\baseline.yaml `
  --output-dir .\results\baseline_run `
  --use-llm false
```

## Run LLM-Driven Experiment

Set `OPENAI_API_KEY` and run:

```powershell
python .\scripts\run_llm_experiment.py `
  --config .\configs\baseline.yaml `
  --output-dir .\results\llm_run `
  --use-llm true `
  --llm-model gpt-5-mini
```

Artifacts saved per run:
- `resolved_config.yaml`
- `llm_response.txt` (when LLM is used)
- `best_model.pt`
- `metrics.json`
