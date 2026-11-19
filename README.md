# Uncertainty-Aware Ensemble Learning for Reliable Prediction of Experimental Band Gaps – code, results and dataset

ICONN-2027 study package. Everything needed to reproduce the results is in this archive.

## Contents
| Path | What it is |
|---|---|
| `T2_Uncertainty_Ensembles/code/t2_uncertainty_ensembles.py` | Full study pipeline (data loading, features, models, evaluation, figures) |
| `common/features.py` | Shared descriptor and metric helpers used by the pipeline |
| `common/prepare_jarvis.py` | Rebuilds the JARVIS tables from the raw JARVIS zip (only needed if the study uses JARVIS) |
| `T2_Uncertainty_Ensembles/results/` | Results produced by the code: `metrics.json`, CSV tables, `figures/` |
| `T2_Uncertainty_Ensembles/results_run.log` | Console log of the run that produced the results |
| `requirements.txt` | Exact package versions used (Python 3.14) |

## Datasets (real, public, unmodified)
| File | Source |
|---|---|
| `data_raw/matbench_expt_gap.json.gz` | Matbench v0.1 `matbench_expt_gap` (4,604 experimental band gaps) |

No synthetic, augmented or imputed target values are used.

## How to reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python T2_Uncertainty_Ensembles/code/t2_uncertainty_ensembles.py
```
Run from the folder that contains this README. Runtime on an Apple M1 laptop: about 15 minutes.
Results are written to `T2_Uncertainty_Ensembles/results/` (existing files are overwritten). Cached descriptors go to `data_cache/`.

---

# Uncertainty-Aware Ensemble Learning for Reliable Band-Gap Prediction

ICONN-2027 (SRMIST, 27–29 Jan 2027) · theme: Computational Materials Science and AI/ML

## Data (real, public; no synthetic or augmented records)
Matbench v0.1 `matbench_expt_gap` (4,604 experimental band gaps), official Matbench 5-fold split (KFold, shuffle, random_state = 18012019).

## How to run
```bash
cd "/Users/alexdavid/Documents/Claude/SRM Jan Conference"
.venv/bin/python T2_Uncertainty_Ensembles/code/t2_uncertainty_ensembles.py
```

## Enhancements implemented from the downloaded literature
| Enhancement | Source papers (arXiv IDs in papers/) |
|---|---|
| M1–M5 RF variance, bootstrap XGBoost, LightGBM quantile, Gaussian process, MLP deep ensemble | 2107.07997, 2211.02235, 2310.13136, 2504.12627, 2311.02495 |
| E1 Post-hoc σ recalibration (σ′ = aσ + b, NLL-optimal) | 2107.07997 |
| E2 Normalised split-conformal intervals and conformalised quantile regression | 2406.15650 |
| E3 Domain guidance via k-NN distance to training data | 2406.15650 |
| E4 Out-of-distribution leave-cluster-out evaluation | 2401.08032 |
| E5 Uncertainty-aware screening for the 1.1–1.8 eV photovoltaic window | 2610.06985 |

## Key results (auto-generated from results/metrics.json)
- **data**
  - n: 4604
  - n_features: 154
  - metals: 2450
  - gap_mean: 0.976
  - gap_max: 11.7
- **summary_mean_over_folds**
  - **M1_RandomForest**
    - fit_s: 3.74
    - raw_MAE: 0.4541
    - raw_NLL: 1.9078
    - raw_coverage90: 0.9689
    - raw_mean_width90: 2.3216
    - raw_spearman_err_sigma: 0.8813
    - raw_miscalibration_area: 0.1933
    - recal_NLL: 3.7792
    - recal_coverage90: 0.9299
    - recal_mean_width90: 1.708
    - recal_spearman_err_sigma: 0.8813
    - recal_miscalibration_area: 0.1168
    - conformal_coverage90: 0.8999
    - conformal_width90: 1.2893
    - MAE_in_domain: 0.451
    - MAE_out_domain: 0.5175
    - frac_out_domain: 0.053
    - recal_a: 0.7344
    - recal_b: 0.0
  - **M2_XGB_BootstrapEnsemble**
    - fit_s: 17.16
    - raw_MAE: 0.4328
    - raw_NLL: 6.3346
    - raw_coverage90: 0.5971
    - raw_mean_width90: 0.6022
    - raw_spearman_err_sigma: 0.7958
    - raw_miscalibration_area: 0.1606
    - recal_NLL: 0.6481
    - recal_coverage90: 0.9233
    - recal_mean_width90: 1.8749
    - recal_spearman_err_sigma: 0.7958
    - recal_miscalibration_area: 0.1321
    - conformal_coverage90: 0.8975
    - conformal_width90: 1.3934
    - MAE_in_domain: 0.4305
    - MAE_out_domain: 0.4785
    - frac_out_domain: 0.053
    - recal_a: 3.1102
    - recal_b: 0.0
  - **M3_LGBM_Quantile**
    - fit_s: 14.78
    - raw_MAE: 0.4206
    - raw_NLL: 1.5116
    - raw_coverage90: 0.8923
    - raw_mean_width90: 1.7565
    - raw_spearman_err_sigma: 0.7099
    - raw_miscalibration_area: 0.1565
    - recal_NLL: 0.7434
    - recal_coverage90: 0.9233
    - recal_mean_width90: 2.0211
    - recal_spearman_err_sigma: 0.7099
    - recal_miscalibration_area: 0.1909
    - conformal_coverage90: 0.9031
    - conformal_width90: 1.8596
    - MAE_in_domain: 0.4214
    - MAE_out_domain: 0.4173
    - frac_out_domain: 0.053
    - recal_a: 1.0967
    - recal_b: 0.0287
  - **M4_GaussianProcess**
    - fit_s: 14.32
    - raw_MAE: 0.5146
    - raw_NLL: 1.3055
    - raw_coverage90: 0.907
    - raw_mean_width90: 2.9977
    - raw_spearman_err_sigma: -0.094
    - raw_miscalibration_area: 0.1727
    - recal_NLL: 1.2948
    - recal_coverage90: 0.9057
    - recal_mean_width90: 2.8938
    - recal_spearman_err_sigma: -0.094
    - recal_miscalibration_area: 0.1697
    - conformal_coverage90: 0.8999
    - conformal_width90: 2.1186
    - MAE_in_domain: 0.5108
    - MAE_out_domain: 0.5834
    - frac_out_domain: 0.053
    - recal_a: 0.2941
    - recal_b: 0.6127
  - **M5_MLP_DeepEnsemble**
    - fit_s: 3.42
    - raw_MAE: 0.4691
    - raw_NLL: 4046095558.5557
    - raw_coverage90: 0.5864
    - raw_mean_width90: 0.5831
    - raw_spearman_err_sigma: 0.7585
    - raw_miscalibration_area: 0.2198
    - recal_NLL: 111474357.7117
    - recal_coverage90: 0.944
    - recal_mean_width90: 2.6124
    - recal_spearman_err_sigma: 0.7585
    - recal_miscalibration_area: 0.1787
    - conformal_coverage90: 0.9092
    - conformal_width90: 1.7088
    - MAE_in_domain: 0.4715
    - MAE_out_domain: 0.4238
    - frac_out_domain: 0.053
    - recal_a: 4.4005
    - recal_b: 0.019
- **ood_leave_cluster_out_mean**
  - **M1_RandomForest**
    - MAE: 0.7448
    - conformal_coverage90: 0.8096
    - recal_coverage90: 0.8809
  - **M2_XGB_BootstrapEnsemble**
    - MAE: 0.7081
    - conformal_coverage90: 0.7758
    - recal_coverage90: 0.813
  - **M3_LGBM_Quantile**
    - MAE: 0.6266
    - conformal_coverage90: 0.8591
    - recal_coverage90: 0.8739
- **screening_PV_window_1.1-1.8eV**
  - n_true_in_window: 432
  - base_rate: 0.0938
  - **topN**
    - **50**
      - precision_by_mean_closeness: 0.32
      - precision_by_probability_in_window: 0.8
    - **100**
      - precision_by_mean_closeness: 0.3
      - precision_by_probability_in_window: 0.74
    - **200**
      - precision_by_mean_closeness: 0.39
      - precision_by_probability_in_window: 0.68

## Output files
Tables: `results/predictions_oof_with_uncertainty.csv`, `results/table_confidence_curve_mae.csv`, `results/table_ood_leave_cluster_out.csv`, `results/table_per_fold.csv`, `results/table_summary_mean_std.csv`

Figures: `results/figures/calibration_curves.png`, `results/figures/confidence_curves.png`, `results/figures/domain_distance_vs_error.png`, `results/figures/reliability_rmse_vs_rmv.png`

## Papers (17 open-access PDFs in `papers/`, index in `papers/papers_index.csv`)
- 2024 · Machine Learning Materials Properties with Accurate Predictions, Uncertainty Estimates, Domain Guidance, and Persistent Online Accessibility · https://arxiv.org/abs/2406.15650v1
- 2022 · Uncertainty-Aware Mixed-Variable Machine Learning for Materials Design · https://arxiv.org/abs/2207.04994v3
- 2021 · Uncertainty Prediction for Machine Learning Models of Material Properties · https://arxiv.org/abs/2107.07997v1
- 2023 · Uncertainty Quantification in Multivariable Regression for Material Property Prediction with Bayesian Neural Networks · https://arxiv.org/abs/2311.02495v4
- 2025 · A Simple and Scalable Kernel Density Approach for Reliable Uncertainty Quantification in Atomistic Machine Learning · https://arxiv.org/abs/2508.14613v1
- 2025 · Uncertainty Quantification in Graph Neural Networks with Shallow Ensembles · https://arxiv.org/abs/2504.12627v1
- 2024 · Electronic structure prediction of medium and high entropy alloys across composition space · https://arxiv.org/abs/2410.08294v3
- 2026 · AutoMatBench: An Automatic Optimization Toolkit for the Acceleration of Material Properties Prediction Benchmarking · https://arxiv.org/abs/2607.11526v1
- 2020 · Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm · https://arxiv.org/abs/2005.00707v2
- 2024 · Structure-based out-of-distribution (OOD) materials property prediction: a benchmark study · https://arxiv.org/abs/2401.08032v1
- 2026 · Composition-Weighted Symbolic Regression for General-Purpose Property Prediction · https://arxiv.org/abs/2605.02267v1
- 2025 · Enhancing composition-based materials property prediction by cross-modal knowledge transfer · https://arxiv.org/abs/2511.03371v1
- 2026 · CrystalJev: calibrated foundation-model probabilities decide what to compute next in materials discovery · https://arxiv.org/abs/2610.06985v2
- 2025 · Combining feature-based approaches with graph neural networks and symbolic regression for synergistic performance and interpretability · https://arxiv.org/abs/2509.03547v2
- 2022 · Materials Property Prediction with Uncertainty Quantification: A Benchmark Study · https://arxiv.org/abs/2211.02235v2
- 2023 · Approaches for Uncertainty Quantification of AI-predicted Material Properties: A Comparison · https://arxiv.org/abs/2310.13136v1
- 2016 · A General-Purpose Machine Learning Framework for Predicting Properties of Inorganic Materials · https://arxiv.org/abs/1606.09551v2
