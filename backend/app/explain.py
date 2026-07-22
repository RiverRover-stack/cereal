"""SHAP explainability (design-doc Phase 4, ADR-0006).

Ports the offline pipeline's SHAP + templating pattern (`models/train.py:416-460`)
to the live per-request fitted model. Explains the exact `feature_frame` rows
`forecasting.py` already carries on `Forecast` for this purpose
(`forecasting.py:126-128`) — same fitted model, same inputs, no refit.

`forecasting.py`'s `DEFAULT_TARGET` is "log" (`forecasting.py:82-85`), so raw SHAP
values come out in log-revenue units, not dollars. Sign and relative rank survive
the log transform; magnitude does not translate to a dollar figure. Per ADR-0006,
these templates only ever report "up"/"down" and a share-of-attribution
percentage — never a dollar amount.
"""

from __future__ import annotations

import numpy as np
import shap

from app.forecasting import Forecast

# Human-readable phrasing for each feature `forecasting.py` can produce
# (RECURSIVE_FEATURES / KNOWN_FEATURES / DIRECT_FEATURES). Anything unlisted falls
# back to its raw column name rather than erroring, since a new feature showing up
# here shouldn't break explanations.
_FEATURE_LABELS: dict[str, str] = {
    "day_of_week": "which days of the week fall in this window",
    "day_of_month": "where these dates fall in the month",
    "time_index": "your long-run sales trend",
    "rolling_7": "your recent 7-day sales trend",
    "horizon": "how far ahead in the 30 days this is",
}

_MAX_SENTENCES = 5

# Same thresholds as the offline pipeline's templating (models/train.py:449-450),
# applied to each feature's share of total mean absolute SHAP.
_BUCKETS = (
    (0.4, "dominant"),
    (0.2, "major"),
    (0.08, "moderate"),
)


def _bucket(share: float) -> str:
    for threshold, name in _BUCKETS:
        if share > threshold:
            return name
    return "minor"


def explain(forecast: Forecast) -> list[str]:
    """3-5 plain-English sentences ranking what drove this forecast, by SHAP.

    Order is SHAP's own ranking by mean absolute contribution, never re-ranked to
    make a nicer story (matches `models/train.py:433`). Returns an empty list if
    no feature has any measurable contribution (e.g. a degenerate all-constant
    forecast) rather than fabricating a sentence.
    """
    features = forecast.feature_frame[list(forecast.feature_names)]

    explainer = shap.TreeExplainer(forecast.model)
    shap_values = np.asarray(explainer.shap_values(features))

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    total = mean_abs.sum()
    if total <= 0:
        return []

    order = np.argsort(mean_abs)[::-1]

    sentences: list[str] = []
    for idx in order[:_MAX_SENTENCES]:
        share = float(mean_abs[idx] / total)
        if share <= 0:
            continue
        name = forecast.feature_names[idx]
        label = _FEATURE_LABELS.get(name, name)
        direction = (
            "pushing this forecast up"
            if mean_signed[idx] > 0
            else "pulling this forecast down"
        )
        sentences.append(
            f"{label[0].upper()}{label[1:]} is a {_bucket(share)} factor "
            f"({share * 100:.0f}% of the explanation), on average {direction} "
            "across the next 30 days."
        )
    return sentences
