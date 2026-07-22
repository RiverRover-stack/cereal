"""The numbered training/evaluation stages used by `train.py`'s `main()`.

Each `stepN_*` function is one banner section from the original monolithic
`main()`, extracted verbatim (same computation, same prints) so the entry
script can read as a short, ordered list of stage calls. Split out purely for
readability -- `train.py` still runs everything in this exact order, in one
process, with the SEED and folds it constructs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
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
)


def make_model(seed: int, params: dict, early_stop: int | None = 50) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=4,
        early_stopping_rounds=early_stop,
        eval_metric="rmse",
        **params,
    )


def fit(seed: int, params: dict, train: pd.DataFrame, features: list[str], cbrt: bool):
    """Fit with a TIME-ORDERED early-stopping holdout carved off the end of train."""
    inner, valid = bt.time_ordered_valid(train, valid_days=30)
    tcol = TARGET_CBRT if cbrt else TARGET_RAW
    model = make_model(seed, params)
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


def step0_data_and_leakage(
    n_folds: int, horizon: int, n_select_folds: int,
) -> tuple[pd.DataFrame, list[bt.Fold], list[bt.Fold]]:
    df = load_frame()
    folds = bt.make_folds(df, n_folds, horizon)
    sel_folds = bt.make_folds(
        df, n_select_folds, horizon,
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

    return df, folds, sel_folds


def step1_hyperparameter_sweep(
    df: pd.DataFrame, sel_folds: list[bt.Fold], seed: int,
) -> tuple[pd.DataFrame, dict]:
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
            m = fit(seed, params, train, FEATURES, cbrt=True)
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
    best_params = dict(
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        n_estimators=600,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
    )
    print(f"\nselected: {best_params}")
    return sweep, best_params


def step1b_feature_ablation(
    df: pd.DataFrame,
    sel_folds: list[bt.Fold],
    best_params: dict,
    known_only_features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, str, list[str], dict[str, list[str]]]:
    # `time_idx` is a monotone counter. A tree cannot extrapolate: every future date
    # has a time_idx larger than anything in training, so every test row falls into
    # the single right-most leaf of every split on it. That can act as a constant
    # offset learned from the tail of the training window rather than a trend. This
    # is measured, not assumed.
    banner("1b. FEATURE-SET ABLATION (recursive, SELECTION folds, cbrt target)")
    feature_sets = {
        "full": FEATURES,
        "no_time_idx": [f for f in FEATURES if f != "time_idx"],
        "known_covariates_only": known_only_features,
    }
    abl_rows = []
    for nm, fs in feature_sets.items():
        maes, rmses, mapes = [], [], []
        for f in sel_folds:
            train, test = bt.split(df, f)
            m = fit(seed, best_params, train, fs, cbrt=True)
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
    selected_fs_name = abl.iloc[0]["feature_set"]
    selected_features = feature_sets[selected_fs_name]
    print(f"\nselected feature set: {selected_fs_name} -> {selected_features}")
    return abl, selected_fs_name, selected_features, feature_sets


def step2_target_transform_ab(
    df: pd.DataFrame,
    folds: list[bt.Fold],
    best_params: dict,
    feature_sets: dict[str, list[str]],
    selected_fs_name: str,
    seed: int,
) -> tuple[pd.DataFrame, bool, pd.DataFrame]:
    banner("2. A/B TARGET TRANSFORM -- identical model, identical folds, "
           "metrics in ORIGINAL UNITS")
    # Run the A/B on BOTH feature sets, all 5 folds. The transform's effect is not
    # guaranteed to be independent of the feature set, so reporting it on only one
    # would be a cherry-pick.
    ab_rows, dec_store = [], {}
    for fs_name in ("full", selected_fs_name) if selected_fs_name != "full" else ("full",):
        fs = feature_sets[fs_name]
        for cbrt in (False, True):
            name = "cbrt" if cbrt else "raw"
            all_y, all_p = [], []
            for f in folds:
                train, test = bt.split(df, f)
                m = fit(seed, best_params, train, fs, cbrt=cbrt)
                pred = bt.predict_recursive(m, train, test, fs, cbrt=cbrt)
                r = bt.metrics(test[TARGET_RAW], pred)
                ab_rows.append({"feature_set": fs_name, "target": name,
                                "fold": f.idx, **r})
                all_y.append(test[TARGET_RAW].to_numpy(float))
                all_p.append(pred)
            y, p = np.concatenate(all_y), np.concatenate(all_p)
            ab_rows.append({"feature_set": fs_name, "target": name, "fold": "ALL",
                            **bt.metrics(y, p)})
            if fs_name == selected_fs_name:
                dec_store[name] = bt.decile_mae(y, p)

    ab = pd.DataFrame(ab_rows)
    print(ab.round(4).to_string(index=False))

    print(f"\nPer-decile MAE and bias, feature set '{selected_fs_name}' "
          "(deciles of ACTUAL units; bias>0 = over-predicts)")
    dec = dec_store["raw"].join(dec_store["cbrt"], lsuffix="_raw", rsuffix="_cbrt")
    print(dec[["actual_mean_raw", "MAE_raw", "bias_raw", "MAE_cbrt", "bias_cbrt",
               "n_raw"]].round(3).to_string())

    sel = ab[(ab.feature_set == selected_fs_name) & (ab.fold == "ALL")]
    raw_all = sel[sel.target == "raw"].iloc[0]
    cb_all = sel[sel.target == "cbrt"].iloc[0]
    # Decision rule, fixed in advance: pooled MAE in ORIGINAL UNITS on the selected
    # feature set. MAPE is reported alongside because the two can and do disagree.
    use_cbrt = bool(cb_all["MAE"] < raw_all["MAE"])
    print(f"\ndecision rule = pooled MAE in original units on '{selected_fs_name}'")
    print(f"  raw : MAE={raw_all['MAE']:.4f} RMSE={raw_all['RMSE']:.4f} "
          f"MAPE={raw_all['MAPE']:.2f}")
    print(f"  cbrt: MAE={cb_all['MAE']:.4f} RMSE={cb_all['RMSE']:.4f} "
          f"MAPE={cb_all['MAPE']:.2f}")
    print(f"  -> keeping {'CUBE-ROOT' if use_cbrt else 'RAW'} target "
          f"(MAE delta {cb_all['MAE'] - raw_all['MAE']:+.4f}, "
          f"{(cb_all['MAE'] - raw_all['MAE']) / raw_all['MAE'] * 100:+.1f}%)")
    return ab, use_cbrt, dec


def _run_backtest_folds(
    df: pd.DataFrame,
    folds: list[bt.Fold],
    best_params: dict,
    features: list[str],
    known_only_features: list[str],
    selected_fs_name: str,
    use_cbrt: bool,
    seed: int,
) -> tuple[list[dict], list[pd.DataFrame], list, list[int]]:
    rows, hz_rows, dir_store, best_iters = [], [], [], []
    for f in folds:
        train, test = bt.split(df, f)
        y = test[TARGET_RAW].to_numpy(float)

        # Model A: full feature set (has lags) -- run in BOTH horizon regimes so the
        # cost of recursion is visible rather than hidden by the ablation choice.
        m_full = fit(seed, best_params, train, features, cbrt=use_cbrt)
        p_rec = bt.predict_recursive(m_full, train, test, features, cbrt=use_cbrt)
        p_ora = bt.predict_oracle(m_full, test, features, cbrt=use_cbrt)

        # Model B: known-covariates only -- no lags, so recursive and oracle are the
        # same computation by construction. Zero compounding.
        m_known = fit(seed, best_params, train, known_only_features, cbrt=use_cbrt)
        p_known = bt.predict_oracle(m_known, test, known_only_features, cbrt=use_cbrt)

        m_sel = m_known if selected_fs_name == "known_covariates_only" else m_full
        p_sel = p_known if selected_fs_name == "known_covariates_only" else p_rec
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

    return rows, hz_rows, dir_store, best_iters


def _print_backtest_report(
    res: pd.DataFrame, hzdf: pd.DataFrame, selected_fs_name: str, use_cbrt: bool,
) -> None:
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

    print(f"\nSELECTED PRODUCTION MODEL: {selected_fs_name}, "
          f"target={'cube-root' if use_cbrt else 'raw'}")
    sel_name = ("xgb_known_covariates_only"
                if selected_fs_name == "known_covariates_only" else "xgb_full_recursive")
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
    print("MAE by bucket x fold, recursive model (lags rebuilt from own predictions)")
    print(hzdf[hzdf.model == "full_recursive"]
          .pivot_table(index="bucket", columns="fold", values="MAE", observed=True)
          .round(3).to_string())
    print("\npooled by bucket, both models "
          "(known-only has no lags, so it cannot compound)")
    print(hzdf.groupby(["model", "bucket"], observed=True)[["MAE", "RMSE", "MAPE"]]
          .mean().round(3).to_string())


def step3_backtest(
    df: pd.DataFrame,
    folds: list[bt.Fold],
    best_params: dict,
    features: list[str],
    known_only_features: list[str],
    selected_fs_name: str,
    use_cbrt: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    banner("3. ROLLING-ORIGIN BACKTEST -- winner vs baselines, per fold")
    rows, hz_rows, dir_store, best_iters = _run_backtest_folds(
        df, folds, best_params, features, known_only_features,
        selected_fs_name, use_cbrt, seed,
    )
    res = pd.DataFrame(rows)
    hzdf = pd.concat(hz_rows)
    _print_backtest_report(res, hzdf, selected_fs_name, use_cbrt)
    return res, hzdf, dir_store, best_iters


def step4_directional(folds: list[bt.Fold], dir_store: list) -> pd.DataFrame:
    banner("4. DIRECTIONAL ACCURACY / F1 (post-hoc from the regressor's output)")
    drows = []
    for f, (test, p) in zip(folds, dir_store):
        drows.append({"fold": f.idx, **bt.directional(test, p)})
    ddf = pd.DataFrame(drows)
    print(ddf.round(4).to_string(index=False))
    print("\nmean: " + ddf.drop(columns=["fold"]).mean().round(4).to_dict().__str__())
    print("class balance = actual_up_share; F1 is reported because accuracy alone on "
          "an imbalanced up/down split is not interpretable.")
    return ddf


def step5_reproducibility(
    train: pd.DataFrame,
    test: pd.DataFrame,
    best_params: dict,
    selected_features: list[str],
    use_cbrt: bool,
    seed: int,
) -> None:
    banner("5. REPRODUCIBILITY (same seed, two independent fits)")
    a = bt.predict_recursive(fit(seed, best_params, train, selected_features, use_cbrt),
                             train, test, selected_features, use_cbrt)
    b = bt.predict_recursive(fit(seed, best_params, train, selected_features, use_cbrt),
                             train, test, selected_features, use_cbrt)
    print(f"bit-identical predictions across two fits: {bool(np.array_equal(a, b))}")
    print(f"max abs diff: {np.max(np.abs(a - b)):.3e}   seed={seed}")


def _fit_final_model(
    df: pd.DataFrame,
    selected_features: list[str],
    use_cbrt: bool,
    best_iters: list[int],
    best_params: dict,
    seed: int,
) -> tuple[xgb.XGBRegressor, dict]:
    # The final artefact is fitted on ALL data with a FIXED tree count, not with
    # early stopping. Reason, measured: with early stopping the holdout is the last
    # 30 days of the full dataset -- December, the demand spike -- and validation
    # RMSE bottoms out after a handful of trees, producing a 5-tree underfit model.
    # Instead we take the median stopping point the backtest folds actually chose.
    n_trees = int(np.median(best_iters)) + 1
    print(f"per-fold early-stopping iterations: {best_iters} -> median+1 = {n_trees}")
    final_params = {**best_params, "n_estimators": n_trees}
    tcol = TARGET_CBRT if use_cbrt else TARGET_RAW
    final = make_model(seed, final_params, early_stop=None)
    final.fit(df[selected_features], df[tcol], verbose=False)
    return final, final_params


def _explain_with_shap(
    final: xgb.XGBRegressor, test: pd.DataFrame, selected_features: list[str],
) -> pd.DataFrame | None:
    try:
        import shap as shap_lib

        expl = shap_lib.TreeExplainer(final)
        window = test[selected_features]
        sv = expl.shap_values(window)
        contrib = pd.DataFrame(
            {"feature": selected_features, "mean_abs_shap": np.abs(sv).mean(axis=0),
             "mean_signed_shap": sv.mean(axis=0)}
        ).sort_values("mean_abs_shap", ascending=False)
        print(f"shap {shap_lib.__version__} TreeExplainer on the last fold's "
              f"{len(window)} forecast rows")
        print(contrib.round(4).to_string(index=False))

        # Templated sentences: feature name + sign + magnitude bucket. The ORDER is
        # SHAP's, never re-ranked to make a nicer story.
        nice = {
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
        shap_sentences = []
        for (_, r), s in zip(top.iterrows(), share):
            bucket = "dominant" if s > 0.4 else "major" if s > 0.2 else \
                "moderate" if s > 0.08 else "minor"
            direction = "pushing the forecast up" if r["mean_signed_shap"] > 0 else \
                "pulling the forecast down"
            shap_sentences.append(
                f"{nice.get(r['feature'], r['feature'])} is a {bucket} driver "
                f"({s*100:.0f}% of total attribution), on average "
                f"{direction} across this window."
            )
        print("\ntemplated explanation (order is SHAP's, unmodified):")
        for s in shap_sentences:
            print("  - " + s)
        return contrib
    except Exception as e:  # noqa: BLE001
        print(f"SHAP FAILED: {type(e).__name__}: {e}")
        print("falling back to gain importance (NOT a substitute -- gain is global "
              "and unsigned, SHAP is per-forecast and signed):")
        print(pd.Series(final.get_booster().get_score(importance_type="gain"))
              .sort_values(ascending=False).round(3).to_string())
        return None


def step6_final_model_and_shap(
    df: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
    use_cbrt: bool,
    best_iters: list[int],
    best_params: dict,
    seed: int,
) -> tuple[xgb.XGBRegressor, dict, pd.DataFrame | None]:
    banner("6. FINAL MODEL + SHAP ATTRIBUTION")
    final, final_params = _fit_final_model(
        df, selected_features, use_cbrt, best_iters, best_params, seed
    )
    shap_table = _explain_with_shap(final, test, selected_features)
    return final, final_params, shap_table


def step7_save_artifacts(
    art_dir,
    root,
    df: pd.DataFrame,
    final: xgb.XGBRegressor,
    seed: int,
    use_cbrt: bool,
    selected_features: list[str],
    selected_fs_name: str,
    features: list[str],
    final_params: dict,
    best_iters: list[int],
    sweep: pd.DataFrame,
    abl: pd.DataFrame,
    ab: pd.DataFrame,
    res: pd.DataFrame,
    hzdf: pd.DataFrame,
    ddf: pd.DataFrame,
    dec: pd.DataFrame,
    shap_table: pd.DataFrame | None,
) -> None:
    banner("7. ARTEFACTS")
    final.get_booster().save_model(str(art_dir / "model.json"))
    meta = {
        "seed": seed,
        "xgboost": xgb.__version__,
        "pandas": pd.__version__,
        "trained_on": str(root / "data/processed/master_df_cbrt.csv"),
        "rows": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "target": TARGET_CBRT if use_cbrt else TARGET_RAW,
        "inverse_transform": "cube (x**3)" if use_cbrt else "identity",
        "features": selected_features,
        "feature_set_name": selected_fs_name,
        "features_considered": features,
        "categorical_features": [c for c in CATEGORICAL if c in selected_features],
        "categorical_encoding": "pandas category dtype + enable_categorical=True",
        "category_levels": {c: [str(x) for x in df[c].cat.categories] for c in CATEGORICAL},
        "excluded_as_leakage": BANNED_FEATURES,
        "params": final_params,
        "params_searched_over": {"max_depth": [3, 4, 5, 6, 8],
                                 "learning_rate": [0.05, 0.1]},
        "per_fold_early_stopping_iterations": best_iters,
        "horizon_strategy": "recursive (30-day, lags rebuilt from own predictions)",
        "backtest": "expanding-window rolling origin, 5 folds x 30 calendar days",
    }
    (art_dir / "feature_metadata.json").write_text(json.dumps(meta, indent=2))
    sweep.to_csv(art_dir / "hyperparameter_sweep.csv", index=False)
    abl.to_csv(art_dir / "feature_set_ablation.csv", index=False)
    ab.to_csv(art_dir / "ab_target_transform.csv", index=False)
    res.to_csv(art_dir / "backtest_folds.csv", index=False)
    hzdf.to_csv(art_dir / "horizon_degradation.csv", index=False)
    ddf.to_csv(art_dir / "directional.csv", index=False)
    dec.to_csv(art_dir / "decile_mae.csv")
    if shap_table is not None:
        shap_table.to_csv(art_dir / "shap_attribution.csv", index=False)
    for f in sorted(art_dir.iterdir()):
        print(f"  wrote {f}")
