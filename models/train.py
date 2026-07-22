"""Train and honestly evaluate the Trends Arc demand forecaster.

Single command:

    backend/.venv/Scripts/python.exe models/train.py

Everything printed is measured in this run. Nothing is asserted from memory.

What this script does, in order:
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

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import xgboost as xgb

import backtest as bt
from features import (
    BANNED_FEATURES,
    CATEGORICAL,
    FEATURES,
    TARGET_CBRT,
    TARGET_RAW,
    UNUSED_REDUNDANT,
    assert_no_target_leakage,
    inverse_cbrt,
    load_frame,
    signed_cbrt,
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


def make_model(params: dict, early_stop: int | None = 50) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=4,
        early_stopping_rounds=early_stop,
        eval_metric="rmse",
        **params,
    )


def fit(params: dict, train: pd.DataFrame, features: list[str], cbrt: bool):
    """Fit with a TIME-ORDERED early-stopping holdout carved off the end of train."""
    inner, valid = bt.time_ordered_valid(train, valid_days=30)
    tcol = TARGET_CBRT if cbrt else TARGET_RAW
    model = make_model(params)
    model.fit(
        inner[features],
        inner[tcol],
        eval_set=[(valid[features], valid[tcol])],
        verbose=False,
    )
    return model


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> None:
    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)

    df = load_frame()
    folds = bt.make_folds(df, N_FOLDS, HORIZON)
    sel_folds = bt.make_folds(
        df, N_SELECT_FOLDS, HORIZON,
        last_test_end=folds[0].test_start - pd.Timedelta(days=1),
    )

    banner("0. DATA + LEAKAGE GUARDS")
    print(f"rows={len(df)}  skus={df['sku'].nunique()}  "
          f"dates {df['date'].min():%Y-%m-%d}..{df['date'].max():%Y-%m-%d}  "
          f"({df['date'].nunique()} distinct)")
    # Was 40 before the source fix (one row per SKU lost to the compensating shift).
    print(f"rows dropped for NaN features: {df.attrs['rows_dropped_for_nan_features']} "
          f"(was 40 before the rolling-window fix moved to the build script)")
    ev = assert_no_target_leakage(df)
    print("\nleakage evidence:")
    for k, v in ev.items():
        print(f"  {k}: {v}")
    print(f"\nfeatures used ({len(FEATURES)}): {FEATURES}")
    print(f"categorical (pandas category dtype, enable_categorical=True): {CATEGORICAL}")
    print(f"EXCLUDED as target leakage / target itself: {BANNED_FEATURES}")
    print(f"EXCLUDED as redundant (monotone transforms): {UNUSED_REDUNDANT}")

    for title, fl in (("SELECTION folds (tuning only, never reported)", sel_folds),
                      ("REPORTED folds (never used for selection)", folds)):
        print(f"\n{title} -- expanding train window, disjoint 30-day test windows:")
        for f in fl:
            tr, te = f.label(df)
            n_tr = int((df["date"] < f.train_end).sum())
            n_te = int(((df["date"] >= f.test_start) & (df["date"] <= f.test_end)).sum())
            print(f"  fold {f.idx}: train {tr} (n={n_tr})   test {te} (n={n_te})")
    assert sel_folds[-1].test_end < folds[0].test_start, "selection/report overlap"

    # ------------------------------------------------------------------ #
    # 1. Hyperparameter sweep -- validated, not asserted.
    # ------------------------------------------------------------------ #
    banner("1. HYPERPARAMETER SWEEP (recursive 30-day backtest, SELECTION folds, cbrt target)")
    grid = []
    for depth in (3, 4, 5, 6, 8):
        for lr in (0.05, 0.1):
            grid.append(
                dict(
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=600,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_lambda=1.0,
                )
            )

    sweep_rows = []
    for params in grid:
        maes, rmses, mapes, iters, train_maes = [], [], [], [], []
        for f in sel_folds:
            train, test = bt.split(df, f)
            m = fit(params, train, FEATURES, cbrt=True)
            pred = bt.predict_recursive(m, train, test, FEATURES, cbrt=True)
            r = bt.metrics(test[TARGET_RAW], pred)
            maes.append(r["MAE"])
            rmses.append(r["RMSE"])
            mapes.append(r["MAPE"])
            iters.append(int(m.best_iteration))
            tr_pred = inverse_cbrt(m.predict(train[FEATURES]))
            train_maes.append(float(np.mean(np.abs(tr_pred - train[TARGET_RAW]))))
        sweep_rows.append(
            {
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "train_MAE": np.mean(train_maes),
                "backtest_MAE": np.mean(maes),
                "backtest_RMSE": np.mean(rmses),
                "backtest_MAPE": np.mean(mapes),
                "gap(bt-train)": np.mean(maes) - np.mean(train_maes),
                "best_iter": np.mean(iters),
            }
        )
    sweep = pd.DataFrame(sweep_rows).sort_values("backtest_MAE").reset_index(drop=True)
    print(sweep.round(4).to_string(index=False))
    print("\n(train_MAE falling while backtest_MAE rises with depth = memorisation.)")

    best = sweep.iloc[0]
    BEST_PARAMS = dict(
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        n_estimators=600,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
    )
    print(f"\nselected: {BEST_PARAMS}")

    # ------------------------------------------------------------------ #
    # 1b. Feature-set ablation.
    #
    # `time_idx` is a monotone counter. A tree cannot extrapolate: every future date
    # has a time_idx larger than anything in training, so every test row falls into
    # the single right-most leaf of every split on it. That can act as a constant
    # offset learned from the tail of the training window rather than a trend. This
    # is measured, not assumed.
    # ------------------------------------------------------------------ #
    banner("1b. FEATURE-SET ABLATION (recursive, SELECTION folds, cbrt target)")
    FEATURE_SETS = {
        "full": FEATURES,
        "no_time_idx": [f for f in FEATURES if f != "time_idx"],
        "known_covariates_only": KNOWN_ONLY_FEATURES,
    }
    abl_rows = []
    for nm, fs in FEATURE_SETS.items():
        maes, rmses, mapes = [], [], []
        for f in sel_folds:
            train, test = bt.split(df, f)
            m = fit(BEST_PARAMS, train, fs, cbrt=True)
            pred = bt.predict_recursive(m, train, test, fs, cbrt=True)
            r = bt.metrics(test[TARGET_RAW], pred)
            maes.append(r["MAE"])
            rmses.append(r["RMSE"])
            mapes.append(r["MAPE"])
        abl_rows.append({"feature_set": nm, "n_features": len(fs),
                         "MAE": np.mean(maes), "RMSE": np.mean(rmses),
                         "MAPE": np.mean(mapes)})
    abl = pd.DataFrame(abl_rows).sort_values("MAE").reset_index(drop=True)
    print(abl.round(4).to_string(index=False))
    SELECTED_FS_NAME = abl.iloc[0]["feature_set"]
    SELECTED_FEATURES = FEATURE_SETS[SELECTED_FS_NAME]
    print(f"\nselected feature set: {SELECTED_FS_NAME} -> {SELECTED_FEATURES}")

    # ------------------------------------------------------------------ #
    # 2. A/B: raw target vs cube-root target
    # ------------------------------------------------------------------ #
    banner("2. A/B TARGET TRANSFORM -- identical model, identical folds, "
           "metrics in ORIGINAL UNITS")
    # Run the A/B on BOTH feature sets, all 5 folds. The transform's effect is not
    # guaranteed to be independent of the feature set, so reporting it on only one
    # would be a cherry-pick.
    ab_rows, dec_store = [], {}
    for fs_name in ("full", SELECTED_FS_NAME) if SELECTED_FS_NAME != "full" else ("full",):
        fs = FEATURE_SETS[fs_name]
        for cbrt in (False, True):
            name = "cbrt" if cbrt else "raw"
            all_y, all_p = [], []
            for f in folds:
                train, test = bt.split(df, f)
                m = fit(BEST_PARAMS, train, fs, cbrt=cbrt)
                pred = bt.predict_recursive(m, train, test, fs, cbrt=cbrt)
                r = bt.metrics(test[TARGET_RAW], pred)
                ab_rows.append({"feature_set": fs_name, "target": name,
                                "fold": f.idx, **r})
                all_y.append(test[TARGET_RAW].to_numpy(float))
                all_p.append(pred)
            y, p = np.concatenate(all_y), np.concatenate(all_p)
            ab_rows.append({"feature_set": fs_name, "target": name, "fold": "ALL",
                            **bt.metrics(y, p)})
            if fs_name == SELECTED_FS_NAME:
                dec_store[name] = bt.decile_mae(y, p)

    ab = pd.DataFrame(ab_rows)
    print(ab.round(4).to_string(index=False))

    print(f"\nPer-decile MAE and bias, feature set '{SELECTED_FS_NAME}' "
          "(deciles of ACTUAL units; bias>0 = over-predicts)")
    dec = dec_store["raw"].join(dec_store["cbrt"], lsuffix="_raw", rsuffix="_cbrt")
    print(dec[["actual_mean_raw", "MAE_raw", "bias_raw", "MAE_cbrt", "bias_cbrt",
               "n_raw"]].round(3).to_string())

    sel = ab[(ab.feature_set == SELECTED_FS_NAME) & (ab.fold == "ALL")]
    raw_all = sel[sel.target == "raw"].iloc[0]
    cb_all = sel[sel.target == "cbrt"].iloc[0]
    # Decision rule, fixed in advance: pooled MAE in ORIGINAL UNITS on the selected
    # feature set. MAPE is reported alongside because the two can and do disagree.
    USE_CBRT = bool(cb_all["MAE"] < raw_all["MAE"])
    print(f"\ndecision rule = pooled MAE in original units on '{SELECTED_FS_NAME}'")
    print(f"  raw : MAE={raw_all['MAE']:.4f} RMSE={raw_all['RMSE']:.4f} "
          f"MAPE={raw_all['MAPE']:.2f}")
    print(f"  cbrt: MAE={cb_all['MAE']:.4f} RMSE={cb_all['RMSE']:.4f} "
          f"MAPE={cb_all['MAPE']:.2f}")
    print(f"  -> keeping {'CUBE-ROOT' if USE_CBRT else 'RAW'} target "
          f"(MAE delta {cb_all['MAE'] - raw_all['MAE']:+.4f}, "
          f"{(cb_all['MAE'] - raw_all['MAE']) / raw_all['MAE'] * 100:+.1f}%)")

    # ------------------------------------------------------------------ #
    # 3. Full backtest: three horizon regimes vs three baselines
    # ------------------------------------------------------------------ #
    banner("3. ROLLING-ORIGIN BACKTEST -- winner vs baselines, per fold")
    rows, hz_rows, dir_store, best_iters = [], [], [], []
    for f in folds:
        train, test = bt.split(df, f)
        y = test[TARGET_RAW].to_numpy(float)

        # Model A: full feature set (has lags) -- run in BOTH horizon regimes so the
        # cost of recursion is visible rather than hidden by the ablation choice.
        m_full = fit(BEST_PARAMS, train, FEATURES, cbrt=USE_CBRT)
        p_rec = bt.predict_recursive(m_full, train, test, FEATURES, cbrt=USE_CBRT)
        p_ora = bt.predict_oracle(m_full, test, FEATURES, cbrt=USE_CBRT)

        # Model B: known-covariates only -- no lags, so recursive and oracle are the
        # same computation by construction. Zero compounding.
        m_known = fit(BEST_PARAMS, train, KNOWN_ONLY_FEATURES, cbrt=USE_CBRT)
        p_known = bt.predict_oracle(m_known, test, KNOWN_ONLY_FEATURES, cbrt=USE_CBRT)

        m_sel = m_known if SELECTED_FS_NAME == "known_covariates_only" else m_full
        p_sel = p_known if SELECTED_FS_NAME == "known_covariates_only" else p_rec
        best_iters.append(int(m_sel.best_iteration))

        cand = {
            "xgb_full_recursive": p_rec,
            "xgb_full_oracle_lags(NOT deployable)": p_ora,
            "xgb_known_covariates_only": p_known,
            "baseline_flat_last7(deployable 30d)": bt.baseline_flat_last7(train, test),
            "baseline_lag7(1-step-ahead)": bt.baseline_lag7(test),
            "baseline_rolling7_shifted(1-step-ahead)": bt.baseline_rolling(test),
            "baseline_rolling7_LEAKY(1-step + leaks)": bt.baseline_rolling_leaky(test),
        }
        for nm, p in cand.items():
            ok = ~np.isnan(p)
            rows.append({"model": nm, "fold": f.idx, **bt.metrics(y[ok], p[ok])})

        for nm, p in (("full_recursive", p_rec), ("known_only", p_known),
                      ("baseline_flat_last7", bt.baseline_flat_last7(train, test)),
                      ("baseline_rolling7_shifted_1step", bt.baseline_rolling(test))):
            hz = bt.horizon_buckets(test, p, f).reset_index()
            hz["fold"] = f.idx
            hz["model"] = nm
            hz_rows.append(hz)
        dir_store.append((test, p_sel))

    res = pd.DataFrame(rows)
    piv = res.pivot_table(index="model", columns="fold", values="MAE").round(3)
    print("MAE by fold\n" + piv.to_string())
    print("\nfull per-fold detail")
    print(res.round(4).to_string(index=False))
    print("\npooled across folds (mean of fold metrics, and worst fold)")
    summary = res.groupby("model").agg(
        MAE_mean=("MAE", "mean"), MAE_worst=("MAE", "max"),
        RMSE_mean=("RMSE", "mean"), RMSE_worst=("RMSE", "max"),
        MAPE_mean=("MAPE", "mean"), MAPE_worst=("MAPE", "max"),
    ).sort_values("MAE_mean")
    print(summary.round(4).to_string())

    print(f"\nSELECTED PRODUCTION MODEL: {SELECTED_FS_NAME}, "
          f"target={'cube-root' if USE_CBRT else 'raw'}")
    sel_name = ("xgb_known_covariates_only"
                if SELECTED_FS_NAME == "known_covariates_only" else "xgb_full_recursive")
    for nm, bl_name in [
        ("like-for-like (both forecast 30d with no in-window actuals)",
         "baseline_flat_last7(deployable 30d)"),
        ("unfair to the model (baseline sees in-window actuals)",
         "baseline_rolling7_shifted(1-step-ahead)"),
    ]:
        bl = summary.loc[bl_name, "MAE_mean"]
        for m_name in (sel_name, "xgb_full_recursive"):
            sm = summary.loc[m_name, "MAE_mean"]
            print(f"  {m_name} MAE {sm:.4f} vs {bl_name} {bl:.4f} -> "
                  f"{'BEATS' if sm < bl else 'LOSES TO'} it "
                  f"({(sm - bl) / bl * 100:+.1f}%)  [{nm}]")

    banner("3b. HORIZON DEGRADATION -- error by day-into-the-30-day-window")
    hzdf = pd.concat(hz_rows)
    print("MAE by bucket x fold, recursive model (lags rebuilt from own predictions)")
    print(hzdf[hzdf.model == "full_recursive"]
          .pivot_table(index="bucket", columns="fold", values="MAE", observed=True)
          .round(3).to_string())
    print("\npooled by bucket, both models "
          "(known-only has no lags, so it cannot compound)")
    print(hzdf.groupby(["model", "bucket"], observed=True)[["MAE", "RMSE", "MAPE"]]
          .mean().round(3).to_string())

    # ------------------------------------------------------------------ #
    # 4. Directional
    # ------------------------------------------------------------------ #
    banner("4. DIRECTIONAL ACCURACY / F1 (post-hoc from the regressor's output)")
    drows = []
    for f, (test, p) in zip(folds, dir_store):
        drows.append({"fold": f.idx, **bt.directional(test, p)})
    ddf = pd.DataFrame(drows)
    print(ddf.round(4).to_string(index=False))
    print("\nmean: " + ddf.drop(columns=["fold"]).mean().round(4).to_dict().__str__())
    print("class balance = actual_up_share; F1 is reported because accuracy alone on "
          "an imbalanced up/down split is not interpretable.")

    # ------------------------------------------------------------------ #
    # 5. Reproducibility
    # ------------------------------------------------------------------ #
    banner("5. REPRODUCIBILITY (same seed, two independent fits)")
    train, test = bt.split(df, folds[-1])
    a = bt.predict_recursive(fit(BEST_PARAMS, train, SELECTED_FEATURES, USE_CBRT),
                             train, test, SELECTED_FEATURES, USE_CBRT)
    b = bt.predict_recursive(fit(BEST_PARAMS, train, SELECTED_FEATURES, USE_CBRT),
                             train, test, SELECTED_FEATURES, USE_CBRT)
    print(f"bit-identical predictions across two fits: {bool(np.array_equal(a, b))}")
    print(f"max abs diff: {np.max(np.abs(a - b)):.3e}   seed={SEED}")

    # ------------------------------------------------------------------ #
    # 6. Final fit + SHAP
    # ------------------------------------------------------------------ #
    banner("6. FINAL MODEL + SHAP ATTRIBUTION")
    # The final artefact is fitted on ALL data with a FIXED tree count, not with
    # early stopping. Reason, measured: with early stopping the holdout is the last
    # 30 days of the full dataset -- December, the demand spike -- and validation
    # RMSE bottoms out after a handful of trees, producing a 5-tree underfit model.
    # Instead we take the median stopping point the backtest folds actually chose.
    n_trees = int(np.median(best_iters)) + 1
    print(f"per-fold early-stopping iterations: {best_iters} -> median+1 = {n_trees}")
    FINAL_PARAMS = {**BEST_PARAMS, "n_estimators": n_trees}
    tcol = TARGET_CBRT if USE_CBRT else TARGET_RAW
    final = make_model(FINAL_PARAMS, early_stop=None)
    final.fit(df[SELECTED_FEATURES], df[tcol], verbose=False)

    shap_sentences, shap_table = [], None
    try:
        import shap as shap_lib

        expl = shap_lib.TreeExplainer(final)
        window = test[SELECTED_FEATURES]
        sv = expl.shap_values(window)
        contrib = pd.DataFrame(
            {"feature": SELECTED_FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0),
             "mean_signed_shap": sv.mean(axis=0)}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_table = contrib
        print(f"shap {shap_lib.__version__} TreeExplainer on the last fold's "
              f"{len(window)} forecast rows")
        print(contrib.round(4).to_string(index=False))

        # Templated sentences: feature name + sign + magnitude bucket. The ORDER is
        # SHAP's, never re-ranked to make a nicer story.
        NICE = {
            "units_rolling_7d_avg": "recent 7-day average demand",
            "units_lag_7": "demand 7 sales-days ago",
            "sku": "which product it is",
            "category": "the product category",
            "day_of_week": "the day of the week",
            "month": "the time of year",
            "day_of_month": "the point in the month",
            "time_idx": "the long-run trend",
            "is_promo_active": "whether a promotion is running",
            "base_price": "the product's list price",
        }
        top = contrib.head(5)
        share = top["mean_abs_shap"] / contrib["mean_abs_shap"].sum()
        for (_, r), s in zip(top.iterrows(), share):
            bucket = "dominant" if s > 0.4 else "major" if s > 0.2 else \
                "moderate" if s > 0.08 else "minor"
            direction = "pushing the forecast up" if r["mean_signed_shap"] > 0 else \
                "pulling the forecast down"
            shap_sentences.append(
                f"{NICE.get(r['feature'], r['feature'])} is a {bucket} driver "
                f"({s*100:.0f}% of total attribution), on average "
                f"{direction} across this window."
            )
        print("\ntemplated explanation (order is SHAP's, unmodified):")
        for s in shap_sentences:
            print("  - " + s)
    except Exception as e:  # noqa: BLE001
        print(f"SHAP FAILED: {type(e).__name__}: {e}")
        print("falling back to gain importance (NOT a substitute -- gain is global "
              "and unsigned, SHAP is per-forecast and signed):")
        print(pd.Series(final.get_booster().get_score(importance_type="gain"))
              .sort_values(ascending=False).round(3).to_string())

    # ------------------------------------------------------------------ #
    # 7. Artefacts
    # ------------------------------------------------------------------ #
    banner("7. ARTEFACTS")
    final.get_booster().save_model(str(ART / "model.json"))
    meta = {
        "seed": SEED,
        "xgboost": xgb.__version__,
        "pandas": pd.__version__,
        "trained_on": str(ROOT / "data/processed/master_df_cbrt.csv"),
        "rows": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "target": TARGET_CBRT if USE_CBRT else TARGET_RAW,
        "inverse_transform": "cube (x**3)" if USE_CBRT else "identity",
        "features": SELECTED_FEATURES,
        "feature_set_name": SELECTED_FS_NAME,
        "features_considered": FEATURES,
        "categorical_features": [c for c in CATEGORICAL if c in SELECTED_FEATURES],
        "categorical_encoding": "pandas category dtype + enable_categorical=True",
        "category_levels": {c: [str(x) for x in df[c].cat.categories] for c in CATEGORICAL},
        "excluded_as_leakage": BANNED_FEATURES,
        "params": FINAL_PARAMS,
        "params_searched_over": {"max_depth": [3, 4, 5, 6, 8],
                                 "learning_rate": [0.05, 0.1]},
        "per_fold_early_stopping_iterations": best_iters,
        "horizon_strategy": "recursive (30-day, lags rebuilt from own predictions)",
        "backtest": "expanding-window rolling origin, 5 folds x 30 calendar days",
    }
    (ART / "feature_metadata.json").write_text(json.dumps(meta, indent=2))
    sweep.to_csv(ART / "hyperparameter_sweep.csv", index=False)
    abl.to_csv(ART / "feature_set_ablation.csv", index=False)
    ab.to_csv(ART / "ab_target_transform.csv", index=False)
    res.to_csv(ART / "backtest_folds.csv", index=False)
    hzdf.to_csv(ART / "horizon_degradation.csv", index=False)
    ddf.to_csv(ART / "directional.csv", index=False)
    dec.to_csv(ART / "decile_mae.csv")
    if shap_table is not None:
        shap_table.to_csv(ART / "shap_attribution.csv", index=False)
    for f in sorted(ART.iterdir()):
        print(f"  wrote {f}")
    print(f"\ntotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
