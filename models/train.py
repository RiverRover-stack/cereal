"""Train and honestly evaluate the Trends Arc demand forecaster.

Single command:

    backend/.venv/Scripts/python.exe models/train.py

Everything printed is measured in this run. Nothing is asserted from memory.

What this script does, in order (implementations in train_pipeline.py):
  0. Proves the leakage guards (features.assert_no_target_leakage).
  1. Coarse hyperparameter sweep + feature-set ablation, on a SELECTION block of
     folds that is disjoint from every fold reported in step 3.
  2. A/B: identical model on the raw target vs the cube-root target, scored in
     original units on identical folds, plus per-decile MAE.
  3. Full 5-fold rolling-origin backtest of the winner in all three horizon regimes,
     against three no-fit baselines.
  4. Directional accuracy / F1 with class balance.
  5. Reproducibility check (two fits, identical seed, bit-identical predictions).
  6. SHAP attribution on the final model + templated explanation sentences.
  7. Saves the artefact and feature metadata to models/artifacts/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as bt
from features import FEATURES
from train_pipeline import (
    step0_data_and_leakage,
    step1_hyperparameter_sweep,
    step1b_feature_ablation,
    step2_target_transform_ab,
    step3_backtest,
    step4_directional,
    step5_reproducibility,
    step6_final_model_and_shap,
    step7_save_artifacts,
)

SEED = 20260721
ROOT = Path(__file__).resolve().parents[1]
ART = Path(__file__).resolve().parent / "artifacts"
N_FOLDS = 5
HORIZON = 30
# Model selection (hyperparameters + feature set) runs on its own block of 4 folds
# that sit immediately BEFORE the 5 reported folds and never overlap them. Earlier
# revisions tuned on reported folds 1-3; that biased the reported numbers and, worse,
# the ablation picked a feature set on 3 folds that lost across all 5.
N_SELECT_FOLDS = 4

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

# Features known in advance for any future date -- the "known-covariates only" set.
KNOWN_ONLY_FEATURES = [f for f in FEATURES if f not in ("units_lag_7", "units_rolling_7d_avg")]


def main() -> None:
    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)

    df, folds, sel_folds = step0_data_and_leakage(
        N_FOLDS, HORIZON, N_SELECT_FOLDS,
    )

    sweep, best_params = step1_hyperparameter_sweep(df, sel_folds, SEED)

    abl, selected_fs_name, selected_features, feature_sets = step1b_feature_ablation(
        df, sel_folds, best_params, KNOWN_ONLY_FEATURES, SEED,
    )

    ab, use_cbrt, dec = step2_target_transform_ab(
        df, folds, best_params, feature_sets, selected_fs_name, SEED,
    )

    res, hzdf, dir_store, best_iters = step3_backtest(
        df, folds, best_params, FEATURES, KNOWN_ONLY_FEATURES,
        selected_fs_name, use_cbrt, SEED,
    )

    ddf = step4_directional(folds, dir_store)

    repro_train, repro_test = bt.split(df, folds[-1])
    step5_reproducibility(
        repro_train, repro_test, best_params, selected_features, use_cbrt, SEED,
    )

    final, final_params, shap_table = step6_final_model_and_shap(
        df, repro_test, selected_features, use_cbrt, best_iters, best_params, SEED,
    )

    step7_save_artifacts(
        ART, ROOT, df, final, SEED, use_cbrt, selected_features,
        selected_fs_name, FEATURES, final_params, best_iters,
        sweep, abl, ab, res, hzdf, ddf, dec, shap_table,
    )

    print(f"\ntotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
