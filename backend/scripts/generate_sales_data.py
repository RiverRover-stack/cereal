"""Synthetic Shopify-style sales data with *known* trend and seasonality.

Why this exists
---------------
`app/forecasting.py` is supposed to learn trend and seasonality from a seller's
history. To know whether it actually does, we need a history where the true
signal is not a guess — it is written down. This script generates one.

Everything the model is meant to discover is built from an explicit formula, and
every component is written to a ground-truth CSV alongside the data. If the
forecast misses the Black Friday spike, you can point at the exact multiplier it
failed to recover instead of arguing about it.

The model
---------
Daily demand intensity is *multiplicative* — each effect scales the ones before
it rather than adding to them. Retail works this way: a Saturday is "+20%", not
"+300 dollars", and that percentage grows with the business.

    lambda(t) = base x G(t) x W(t) x Y(t) x H(t) x P(t) x E(t)

    G(t)  trend        exponential compound growth
    W(t)  weekly       Fourier series over a 7-day period (peaks Saturday)
    Y(t)  annual       Fourier series over a 365.25-day period (Q4 peak, Feb trough)
    H(t)  calendar     Gaussian bumps on fixed events (Black Friday, Christmas, ...)
    P(t)  promotions   randomly arriving, exponentially decaying pulses
    E(t)  noise        AR(1) lognormal — autocorrelated, mean exactly 1

W, Y, H and P are each normalised to mean 1 over the generated window, so `base`
keeps its plain meaning ("average line items per day") and no component can
quietly absorb another's level. Taking logs turns the product into a sum, which
is why a log-target model (`forecasting.DEFAULT_TARGET == "log"`) is the right
shape for this data — the components become additive there.

From intensity to rows
----------------------
`lambda(t)` is a *rate*, not a row count. The requested total (default 100,000
line items) is apportioned across days in proportion to lambda using the largest-
remainder method, so the row count is exact while the daily shape is preserved.

Each line item then gets a SKU drawn from a day-varying mixture — every SKU
carries its own annual phase, so some sell in summer and some in winter, and the
store-level seasonality is the sum of products with genuinely different cycles.
Price drifts with inflation and drops during promotions, so revenue seasonality
is *not* a copy of unit seasonality. That distinction matters: a model that only
learns volume will miss the discount drag on Black Friday revenue.

Dependencies: NumPy, pandas, and the standard library. Nothing else is needed —
the point is that the signal comes from the math, not from a simulation package.

Usage
-----
    python backend/scripts/generate_sales_data.py
    python backend/scripts/generate_sales_data.py --rows 250000 --days 1460
    python backend/scripts/generate_sales_data.py --verify
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25

DEFAULT_ROWS = 100_000
DEFAULT_DAYS = 1095  # three years
DEFAULT_START = "2023-01-01"
DEFAULT_SEED = 42


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Harmonic:
    """One term of a Fourier series, in amplitude/phase form.

    `k * cos(2*pi*h*(x - phase))` is the same thing as the usual
    `a*cos + b*sin` pair, but written so the numbers mean something: `amplitude`
    is how strong the cycle is (in log space, so 0.18 is roughly +/-20%) and
    `phase` says where its peak sits, as a fraction of the period.
    """

    harmonic: int
    amplitude: float
    phase: float


@dataclass(frozen=True)
class Event:
    """A calendar event as a Gaussian bump on the demand curve.

    Real sales events are not single-day step changes — anticipation builds and
    demand decays afterwards. A Gaussian kernel of width `sigma_days` captures
    that shape with one parameter.

    `amplitude` is added to a baseline of 1, so 4.0 means "peak day is ~5x
    normal". It may be negative: Christmas Day itself is a trough, not a peak.

    `discount` is the extra markdown applied at the peak. Volume and price move
    in opposite directions during a sale, which is exactly the confounder a
    revenue forecaster has to handle.
    """

    name: str
    amplitude: float
    sigma_days: float
    discount: float = 0.0
    month: int | None = None
    day: int | None = None
    # For movable feasts: ("november", weekday, occurrence, offset_days).
    rule: tuple[str, int, int, int] | None = None


# Weekly cycle. Phase 5/7 puts the fundamental's peak on Saturday (Monday = 0).
# The second harmonic carves out the midweek dip so the shape is not a plain
# sine wave — real weeks have a Sat/Sun plateau and a Tue/Wed floor.
WEEKLY_HARMONICS = (
    Harmonic(harmonic=1, amplitude=0.20, phase=5 / 7),
    Harmonic(harmonic=2, amplitude=0.06, phase=5 / 7),
)

# Annual cycle. Fundamental peaks at phase 0.88 (mid-November, the run-up into
# Q4); the second harmonic adds a smaller summer shoulder so the year is not a
# single smooth hump.
ANNUAL_HARMONICS = (
    Harmonic(harmonic=1, amplitude=0.26, phase=0.88),
    Harmonic(harmonic=2, amplitude=0.09, phase=0.55),
)

EVENTS = (
    Event("new_year_sale", amplitude=0.55, sigma_days=3.0, discount=0.20, month=1, day=3),
    Event("valentines", amplitude=0.60, sigma_days=2.0, discount=0.10, month=2, day=14),
    Event("spring_promo", amplitude=0.45, sigma_days=4.0, discount=0.15, month=4, day=20),
    # Mother's Day: 2nd Sunday of May. (month, weekday, occurrence, offset)
    Event("mothers_day", amplitude=0.50, sigma_days=2.5, discount=0.10,
          rule=("may", 6, 2, 0)),
    Event("summer_sale", amplitude=0.85, sigma_days=3.5, discount=0.25, month=7, day=15),
    Event("back_to_school", amplitude=0.60, sigma_days=6.0, discount=0.10, month=8, day=25),
    Event("halloween", amplitude=0.40, sigma_days=3.0, discount=0.05, month=10, day=31),
    # Black Friday: the Friday after the 4th Thursday of November.
    Event("black_friday", amplitude=4.20, sigma_days=1.2, discount=0.35,
          rule=("november", 3, 4, 1)),
    Event("cyber_monday", amplitude=2.80, sigma_days=1.0, discount=0.30,
          rule=("november", 3, 4, 4)),
    Event("christmas_runup", amplitude=1.70, sigma_days=6.0, discount=0.12, month=12, day=15),
    # Negative on purpose: almost nobody shops on Christmas Day itself.
    Event("christmas_day_lull", amplitude=-0.65, sigma_days=1.1, month=12, day=25),
    Event("boxing_day", amplitude=1.30, sigma_days=2.0, discount=0.30, month=12, day=26),
)

# Annual compound growth of the underlying business, before any seasonality.
# 0.24 means the store is ~27% bigger each year (exp(0.24) - 1).
TREND_LOG_GROWTH = 0.24

# A slow, shallow second trend component: a business rarely grows at a perfectly
# constant rate. Amplitude in log space, period in years.
TREND_WOBBLE_AMPLITUDE = 0.06
TREND_WOBBLE_YEARS = 2.5

# Unplanned promotions: Poisson arrivals, then exponential decay. These are the
# part of the series that is genuinely unforecastable, and they exist so the
# backtest cannot score a suspiciously perfect MAPE.
PROMO_MEAN_GAP_DAYS = 45.0
PROMO_AMPLITUDE_RANGE = (0.30, 1.00)
PROMO_DECAY_RANGE = (2.0, 6.0)
PROMO_DISCOUNT = 0.18

# Day-to-day noise. `phi` is AR(1) persistence: a good week tends to stay good,
# which is what makes the 7-day rolling feature in forecasting.py worth having.
NOISE_SIGMA = 0.16
NOISE_PHI = 0.35

# Baseline markdown outside any event.
BASE_DISCOUNT = 0.02
MAX_DISCOUNT = 0.55

# Annual price inflation, in log space.
PRICE_INFLATION = 0.035

CATEGORIES = (
    # name, n_skus, price range, mean extra units per line, annual phase, amplitude
    ("apparel", 10, (24.0, 95.0), 0.9, 0.88, 0.35),
    ("outerwear", 5, (85.0, 260.0), 0.4, 0.95, 0.75),
    ("accessories", 8, (12.0, 48.0), 1.6, 0.90, 0.25),
    ("footwear", 6, (55.0, 180.0), 0.5, 0.60, 0.30),
    ("summer", 5, (18.0, 70.0), 1.1, 0.50, 0.85),
    ("home", 6, (30.0, 140.0), 0.7, 0.92, 0.20),
)

CHANNELS = ("online", "mobile_app", "marketplace", "retail_pos")
CHANNEL_WEIGHTS = (0.44, 0.31, 0.17, 0.08)

# Probability that a line item starts a new order. ~1/0.62 = 1.6 lines per order.
NEW_ORDER_PROBABILITY = 0.62


@dataclass
class Config:
    rows: int = DEFAULT_ROWS
    days: int = DEFAULT_DAYS
    start: str = DEFAULT_START
    seed: int = DEFAULT_SEED
    outdir: Path = field(default_factory=lambda: Path("backend/data"))


# --------------------------------------------------------------------------
# Component builders — each returns one factor of lambda(t)
# --------------------------------------------------------------------------


def _normalise(values: np.ndarray) -> np.ndarray:
    """Rescale a positive factor to mean 1 over the window.

    Without this, every component would carry part of the overall level and
    `base` would stop meaning anything. It also makes the printed component
    summaries directly comparable: each one reads as "percent above or below
    typical".
    """
    return values / values.mean()


def fourier_factor(
    position: np.ndarray, harmonics: tuple[Harmonic, ...]
) -> np.ndarray:
    """exp(sum_h  A_h * cos(2*pi*h*(position - phase_h))).

    `position` is the phase within the cycle, in [0, 1). The exponential keeps
    the factor positive and makes the effect multiplicative — in log space this
    is a plain sum of cosines, which is the form a model sees when it is fitted
    on log revenue.
    """
    log_factor = np.zeros_like(position, dtype=float)
    for term in harmonics:
        log_factor += term.amplitude * np.cos(
            2 * np.pi * term.harmonic * (position - term.phase)
        )
    return _normalise(np.exp(log_factor))


def trend_factor(t: np.ndarray) -> np.ndarray:
    """Compound growth plus a slow wobble.

    Pure `exp(r*t)` is too clean — a real store accelerates and stalls. The
    wobble is a single long-period cosine, small enough that the growth rate
    stays positive throughout.
    """
    years = t / DAYS_PER_YEAR
    growth = TREND_LOG_GROWTH * years
    wobble = TREND_WOBBLE_AMPLITUDE * np.sin(2 * np.pi * years / TREND_WOBBLE_YEARS)
    # Deliberately NOT normalised: the trend is supposed to carry the level.
    return np.exp(growth + wobble)


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> pd.Timestamp:
    """The `occurrence`-th `weekday` of a month (0-indexed occurrence, Mon=0)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset + 7 * occurrence)


def _event_dates(event: Event, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Every occurrence of an event inside the window (plus a year either side).

    The margin matters: Black Friday sits in late November, and a window ending
    on 31 December still needs the *tail* of a bump whose centre is outside it.
    """
    years = range(dates[0].year - 1, dates[-1].year + 2)
    if event.rule is not None:
        month_name, weekday, occurrence, offset = event.rule
        month = {"may": 5, "november": 11}[month_name]
        return [
            _nth_weekday(year, month, weekday, occurrence - 1)
            + pd.Timedelta(days=offset)
            for year in years
        ]
    return [pd.Timestamp(year=year, month=event.month, day=event.day) for year in years]


def event_factors(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Gaussian calendar bumps -> (demand factor, discount schedule, per-event log).

    Each occurrence contributes `A * exp(-d^2 / (2*sigma^2))` where `d` is days
    from the event's centre. Bumps add before the baseline of 1 is applied, so
    overlapping events (Black Friday into Cyber Monday) compound the way real
    ones do.
    """
    ordinal = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    bumps = np.zeros(len(dates))
    discounts = np.zeros(len(dates))
    log: list[dict[str, object]] = []

    for event in EVENTS:
        for centre in _event_dates(event, dates):
            distance = ordinal - float(centre.toordinal())
            kernel = np.exp(-(distance**2) / (2 * event.sigma_days**2))
            bumps += event.amplitude * kernel
            discounts += event.discount * kernel
            if dates[0] <= centre <= dates[-1]:
                log.append(
                    {
                        "event": event.name,
                        "date": centre.date().isoformat(),
                        "peak_multiplier": round(1 + event.amplitude, 2),
                        "sigma_days": event.sigma_days,
                        "peak_discount": event.discount,
                    }
                )

    # Floor at 0.15 so a stacked negative event can never drive demand to zero
    # (or negative) and produce a day the model cannot represent in log space.
    factor = np.clip(1.0 + bumps, 0.15, None)
    return factor, discounts, pd.DataFrame(log).sort_values("date", ignore_index=True)


def promo_factors(
    t: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly arriving promotions with exponential decay.

    Start days come from a Poisson process (gaps ~ Exponential(mean gap)). Each
    promotion jumps on its start day and decays as `exp(-d/theta)`, so it has a
    sharp onset and a soft tail — the asymmetric shape a flash sale actually has,
    unlike the symmetric Gaussians used for known calendar events.

    These are unpredictable by construction. They are the error floor: no
    forecaster should score near-zero MAPE on this data, and if one does, look
    for leakage.
    """
    factor = np.ones(len(t))
    discount = np.zeros(len(t))

    day = float(rng.exponential(PROMO_MEAN_GAP_DAYS))
    while day < len(t):
        start = int(day)
        amplitude = rng.uniform(*PROMO_AMPLITUDE_RANGE)
        theta = rng.uniform(*PROMO_DECAY_RANGE)
        elapsed = t[start:] - start
        pulse = amplitude * np.exp(-elapsed / theta)
        factor[start:] += pulse
        discount[start:] += PROMO_DISCOUNT * (pulse / amplitude)
        day += rng.exponential(PROMO_MEAN_GAP_DAYS)

    return _normalise(factor), discount


def noise_factor(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) lognormal noise with mean exactly 1.

    Independent noise would be wrong here: real stores have good weeks and bad
    weeks. `phi = 0.35` gives short-range persistence, which is precisely the
    thing `forecasting.rolling_7` exists to exploit.

    The `-sigma^2/2` term is the lognormal bias correction — without it,
    `exp(z)` has mean `exp(sigma^2/2)` and the noise would silently inflate the
    level. `_normalise` then removes the residual sampling error.
    """
    innovations = rng.standard_normal(n_days)
    z = np.empty(n_days)
    z[0] = innovations[0] * NOISE_SIGMA
    scale = NOISE_SIGMA * np.sqrt(1 - NOISE_PHI**2)
    for i in range(1, n_days):
        z[i] = NOISE_PHI * z[i - 1] + scale * innovations[i]
    return _normalise(np.exp(z - NOISE_SIGMA**2 / 2))


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------


def build_catalogue(rng: np.random.Generator) -> pd.DataFrame:
    """One row per SKU, each with its own price, basket size and annual cycle.

    The per-SKU `phase`/`amplitude` are what make store-level seasonality an
    emergent property rather than a stamped-on curve: summer SKUs peak in July
    while outerwear peaks in December, and the store total is their sum.
    """
    rows: list[dict[str, object]] = []
    for category, count, (low, high), extra_units, phase, amplitude in CATEGORIES:
        for index in range(count):
            rows.append(
                {
                    "sku": f"{category[:3].upper()}-{index + 1:03d}",
                    "category": category,
                    # Log-uniform: catalogues have many cheap items and a few
                    # expensive ones, not a flat spread of prices.
                    "base_price": float(
                        np.exp(rng.uniform(np.log(low), np.log(high)))
                    ),
                    "extra_units_mean": extra_units * rng.uniform(0.7, 1.3),
                    "annual_phase": (phase + rng.normal(0, 0.03)) % 1.0,
                    "annual_amplitude": amplitude * rng.uniform(0.8, 1.2),
                    # Long-run popularity, before seasonality. Lognormal, so a
                    # handful of SKUs carry most of the volume — as in any real
                    # catalogue.
                    "popularity": float(np.exp(rng.normal(0, 0.55))),
                    # How hard this SKU is discounted during events.
                    "promo_sensitivity": float(rng.uniform(0.6, 1.4)),
                }
            )
    catalogue = pd.DataFrame(rows)
    catalogue["base_price"] = catalogue["base_price"].round(2)
    return catalogue


def sku_day_weights(
    catalogue: pd.DataFrame, annual_position: np.ndarray
) -> np.ndarray:
    """(n_days, n_sku) mixture weights, each row summing to 1.

    Weight is `popularity * exp(A_j * cos(2*pi*(tau - phase_j)))` — the same
    Fourier form as the store-level cycle, but with per-SKU phase. Normalising
    each row makes this a *share* of the day's volume; the day's total volume is
    set by lambda(t), so the two decisions stay separate.
    """
    phase = catalogue["annual_phase"].to_numpy()
    amplitude = catalogue["annual_amplitude"].to_numpy()
    popularity = catalogue["popularity"].to_numpy()

    seasonal = np.exp(
        amplitude[None, :] * np.cos(2 * np.pi * (annual_position[:, None] - phase[None, :]))
    )
    weights = popularity[None, :] * seasonal
    return weights / weights.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Intensity -> exact row counts
# --------------------------------------------------------------------------


def apportion(total: int, weights: np.ndarray) -> np.ndarray:
    """Split `total` integer rows across days proportionally (largest remainder).

    Rounding each day independently would miss the requested total by tens of
    rows. The largest-remainder (Hamilton) method assigns every floor first, then
    hands the leftover rows to the days with the largest discarded fractions — so
    the total is exact and the daily shape is disturbed by at most one row.
    """
    shares = weights / weights.sum()
    exact = shares * total
    counts = np.floor(exact).astype(np.int64)
    remaining = total - int(counts.sum())
    if remaining > 0:
        order = np.argsort(-(exact - counts))
        counts[order[:remaining]] += 1
    return counts


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate(config: Config) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)

    dates = pd.date_range(start=config.start, periods=config.days, freq="D")
    t = np.arange(config.days, dtype=float)
    annual_position = (dates.dayofyear.to_numpy() - 1) / DAYS_PER_YEAR
    weekly_position = dates.dayofweek.to_numpy() / 7.0

    trend = trend_factor(t)
    weekly = fourier_factor(weekly_position, WEEKLY_HARMONICS)
    annual = fourier_factor(annual_position, ANNUAL_HARMONICS)
    events, event_discount, event_log = event_factors(dates)
    events = _normalise(events)
    promo, promo_discount = promo_factors(t, rng)
    noise = noise_factor(config.days, rng)

    intensity = trend * weekly * annual * events * promo * noise

    discount = np.clip(
        BASE_DISCOUNT + event_discount + promo_discount, 0.0, MAX_DISCOUNT
    )
    price_index = np.exp(PRICE_INFLATION * t / DAYS_PER_YEAR)

    counts = apportion(config.rows, intensity)

    # ---- expand days into line items ------------------------------------
    day_index = np.repeat(np.arange(config.days), counts)
    n_rows = len(day_index)

    catalogue = build_catalogue(rng)
    weights = sku_day_weights(catalogue, annual_position)

    # Inverse-CDF sampling: one uniform per row, compared against that day's
    # cumulative SKU distribution. Vectorised over all 100k rows at once — a
    # per-day multinomial loop would be ~1000x slower for no benefit.
    cdf = np.cumsum(weights, axis=1)
    draw = rng.random(n_rows)
    sku_index = (draw[:, None] > cdf[day_index]).sum(axis=1)
    sku_index = np.clip(sku_index, 0, len(catalogue) - 1)

    # Units: 1 + Poisson, so every line item sells at least one unit and the
    # tail is right-skewed (most lines are 1-2 units, a few are bulk).
    extra_mean = catalogue["extra_units_mean"].to_numpy()[sku_index]
    units = 1 + rng.poisson(extra_mean)

    sensitivity = catalogue["promo_sensitivity"].to_numpy()[sku_index]
    effective_discount = np.clip(discount[day_index] * sensitivity, 0.0, MAX_DISCOUNT)
    unit_price = (
        catalogue["base_price"].to_numpy()[sku_index]
        * price_index[day_index]
        * (1.0 - effective_discount)
        # Small per-line jitter: rounding rules, coupons, regional pricing.
        * rng.lognormal(mean=0.0, sigma=0.03, size=n_rows)
    )
    unit_price = np.round(unit_price, 2)
    revenue = np.round(units * unit_price, 2)

    # Orders: a Bernoulli "starts a new order" flag gives geometric basket
    # sizes. Forced True at every day boundary so no order straddles midnight.
    starts = rng.random(n_rows) < NEW_ORDER_PROBABILITY
    starts[0] = True
    starts[np.flatnonzero(np.diff(day_index) != 0) + 1] = True
    order_number = np.cumsum(starts)

    channel_index = rng.choice(len(CHANNELS), size=n_rows, p=CHANNEL_WEIGHTS)

    line_items = pd.DataFrame(
        {
            "date": dates[day_index].strftime("%Y-%m-%d"),
            "order_id": [f"ORD-{n:07d}" for n in order_number],
            "sku": catalogue["sku"].to_numpy()[sku_index],
            "category": catalogue["category"].to_numpy()[sku_index],
            "channel": np.array(CHANNELS)[channel_index],
            "units_sold": units,
            "unit_price": unit_price,
            "revenue": revenue,
        }
    )

    daily = (
        line_items.groupby("date", as_index=False)[["revenue", "units_sold"]]
        .sum()
        .sort_values("date", ignore_index=True)
    )
    daily["revenue"] = daily["revenue"].round(2)

    components = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "trend": trend,
            "weekly": weekly,
            "annual": annual,
            "events": events,
            "promo": promo,
            "noise": noise,
            "intensity": intensity,
            "discount": discount,
            "price_index": price_index,
            "line_items": counts,
        }
    ).round(6)
    components = components.merge(
        daily.rename(columns={"revenue": "actual_revenue", "units_sold": "actual_units"}),
        on="date",
        how="left",
    )

    return {
        "line_items": line_items,
        "daily": daily,
        "components": components,
        "catalogue": catalogue.round(4),
        "events": event_log,
    }


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify(frames: dict[str, pd.DataFrame]) -> str:
    """Measure the injected signal back out of the generated rows.

    This is not a test of the generator's arithmetic — it is a demonstration
    that the signal is *recoverable* from the CSV alone, which is the only
    property that matters for anything downstream.
    """
    daily = frames["daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dow"] = daily["date"].dt.day_name()
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year

    lines: list[str] = []
    add = lines.append

    add(f"rows (line items) : {len(frames['line_items']):,}")
    add(f"days              : {len(daily):,}")
    add(f"date range        : {daily['date'].min().date()} -> {daily['date'].max().date()}")
    add(f"total revenue     : ${frames['line_items']['revenue'].sum():,.2f}")
    add(f"total units       : {int(frames['line_items']['units_sold'].sum()):,}")
    add(f"orders            : {frames['line_items']['order_id'].nunique():,}")
    add(f"skus              : {frames['line_items']['sku'].nunique()}")
    add("")

    add("TREND — mean daily revenue by year (growth is compounded, not linear)")
    yearly = daily.groupby("year")["revenue"].mean()
    previous = None
    for year, value in yearly.items():
        change = "" if previous is None else f"   {value / previous - 1:+7.1%} YoY"
        add(f"  {year}   ${value:>12,.0f}{change}")
        previous = value
    add(f"  configured annual growth: {np.expm1(TREND_LOG_GROWTH):+.1%}")
    add("")

    add("WEEKLY SEASONALITY — mean revenue by weekday, indexed to 100")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = daily.groupby("dow")["revenue"].mean().reindex(order)
    index = 100 * weekday / weekday.mean()
    for name, value in index.items():
        bar = "#" * int(round(value / 4))
        add(f"  {name:<10} {value:6.1f}  {bar}")
    add("")

    add("ANNUAL SEASONALITY — mean revenue by month, indexed to 100")
    monthly = daily.groupby("month")["revenue"].mean()
    index = 100 * monthly / monthly.mean()
    for month, value in index.items():
        bar = "#" * int(round(value / 4))
        add(f"  {pd.Timestamp(2024, month, 1):%b}        {value:6.1f}  {bar}")
    add("")

    add("EVENT SPIKES — event-day revenue vs the surrounding 30-day median")
    revenue_by_date = daily.set_index("date")["revenue"]
    for _, row in frames["events"].iterrows():
        when = pd.Timestamp(row["date"])
        if when not in revenue_by_date.index:
            continue
        window = revenue_by_date.loc[when - pd.Timedelta(days=21) : when + pd.Timedelta(days=21)]
        baseline = window.median()
        add(
            f"  {row['date']}  {row['event']:<20} "
            f"${revenue_by_date[when]:>11,.0f}   {revenue_by_date[when] / baseline:5.2f}x baseline"
        )
    add("")

    add("SKU-LEVEL SEASONALITY — peak month per category (they differ on purpose)")
    items = frames["line_items"].copy()
    items["month"] = pd.to_datetime(items["date"]).dt.month
    by_category = items.groupby(["category", "month"])["units_sold"].sum().unstack()
    share = by_category.div(by_category.sum(axis=1), axis=0)
    for category, row in share.iterrows():
        peak = int(row.idxmax())
        trough = int(row.idxmin())
        add(
            f"  {category:<12} peak {pd.Timestamp(2024, peak, 1):%b}  "
            f"trough {pd.Timestamp(2024, trough, 1):%b}  "
            f"(peak share {row.max():.1%} vs trough {row.min():.1%})"
        )
    add("")

    add("NOISE FLOOR — how much of log revenue the deterministic parts explain")
    components = frames["components"].copy()
    deterministic = np.log(
        components["trend"] * components["weekly"] * components["annual"]
        * components["events"] * components["promo"]
    )
    observed = np.log(daily["revenue"].to_numpy())
    residual = observed - deterministic
    r2 = 1 - residual.var() / observed.var()
    add(f"  R^2 of log revenue on the known components: {r2:.3f}")
    add(f"  residual sd (log): {residual.std():.4f}  ~= {np.expm1(residual.std()):.1%} day-to-day")
    add("  (the remainder is the AR(1) noise plus Poisson basket variation —")
    add("   a forecaster scoring far below this is leaking, not learning)")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sales data with known trend and seasonality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            outputs (in --outdir):
              sales_line_items.csv   one row per line item (the --rows count)
              sales_daily.csv        aggregated per day — upload this to /forecast
              components.csv         ground truth: every factor of lambda(t) per day
              catalogue.csv          per-SKU price, popularity and annual phase
              events.csv             every calendar event occurrence in the window
            """
        ),
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help=f"line items to generate (default {DEFAULT_ROWS:,})")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"days of history (default {DEFAULT_DAYS})")
    parser.add_argument("--start", default=DEFAULT_START, help="first date, YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed — same seed gives byte-identical output")
    parser.add_argument("--outdir", type=Path, default=Path("backend/data"))
    parser.add_argument("--verify", action="store_true",
                        help="measure the injected signal back out and print it")
    args = parser.parse_args()

    config = Config(
        rows=args.rows, days=args.days, start=args.start,
        seed=args.seed, outdir=args.outdir,
    )

    frames = generate(config)

    config.outdir.mkdir(parents=True, exist_ok=True)
    targets = {
        "sales_line_items.csv": frames["line_items"],
        "sales_daily.csv": frames["daily"],
        "components.csv": frames["components"],
        "catalogue.csv": frames["catalogue"],
        "events.csv": frames["events"],
    }
    for name, frame in targets.items():
        path = config.outdir / name
        frame.to_csv(path, index=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"wrote {path}  ({len(frame):,} rows, {size_mb:.2f} MB)")

    if args.verify:
        print()
        print(verify(frames))


if __name__ == "__main__":
    main()
