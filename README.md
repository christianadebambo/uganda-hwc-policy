# Uncertainty-Aware Decision Support for Human-Wildlife Conflict in Uganda

This repository contains the code and dataset for:

**Christian Adeoye Adebambo**  
*Uncertainty-Aware Decision Support for Human-Wildlife Conflict in Uganda*

doi: https://doi.org/10.5281/zenodo.17247146 

## Overview

We develop a reproducible pipeline for human–wildlife conflict (HWC) decision support in Uganda’s Kasese District, combining:

- Severity modelling (logistic regression, TabTransformer)  
- Calibration and uncertainty (Platt scaling, temperature scaling, conformal prediction)  
- Uplift modelling (multi-arm T-learners with XGBoost)  
- Off-policy evaluation (IPS, overlap weighting, doubly robust)  

## Data

Dataset: _kasese-hwc-data-2021-combined-2021-2022-partly-cleaned.csv_

## Installation

Clone this repository:
```bash
git clone https://github.com/christianadebambo/uganda-hwc-policy.git
cd uganda-hwc-policy
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open the notebook:
```bash
jupyter notebook uganda-hwc-policy.ipynb
```

The notebook runs end-to-end:
- Loads raw CSV (update path if running locally)
- Cleans and engineers features
- Trains models
- Saves outputs under _outputs/_
  
_**N.B: Adjust file paths accordingly**_

## Outputs
Key artefacts saved in _outputs/_:
- _reliability_valid.png_ - calibration diagram
- _logistic_top_weights.csv_ - logistic regression weights
- _tabtransformer_perm_importance.csv_ - feature importance
- _uplift_recommendations_by_parish.csv_ - parish-level recommendations

## Citation

If you use this repository, please cite:

```bibtex
@article{adebambo2025ugandahwc,
  title   = {Uncertainty-Aware Decision Support for Human-Wildlife Conflict in Uganda},
  author  = {Adebambo, Christian Adeoye},
  year    = {2025}
}
```
