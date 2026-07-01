# Machine-Learning Analysis Pipeline

This repository contains the Python script used for the machine-learning analysis.

## Files

- `ml_pipeline.py`: full analysis script
- `requirements.txt`: required Python packages

## Usage

```bash
pip install -r requirements.txt
python ml_pipeline.py --input Dataset.xlsx
```

The input file should use the same de-identified variable names as the script, including `Group` for the outcome and `feature_01`, `feature_02`, etc. for predictors. The final de-identified feature set reflects the features retained after feature assessment. If the input file is unavailable, the script automatically generates a simulated de-identified dataset for testing.

The script exports a summary workbook named `analysis_summary.xlsx` by default. A different output path can be specified with:

```bash
python ml_pipeline.py --input Dataset.xlsx --summary-output analysis_summary.xlsx
```
