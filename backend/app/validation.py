"""CSV upload validation for `POST /forecast` (design-doc Phase 2).

Every rejection here surfaces its message straight to the user — the frontend
renders the backend's text rather than a generic failure (design-doc.md:237-238),
so these strings are user-facing copy, not debug output. Say what is wrong and
what to do about it.
"""

from __future__ import annotations

import io

import pandas as pd

# design-doc.md:153. Checked while streaming, so an oversized upload is abandoned
# mid-read instead of being buffered in full first.
MAX_FILE_BYTES = 5 * 1024 * 1024
CHUNK_BYTES = 64 * 1024

REQUIRED_COLUMNS = ("date", "revenue", "units_sold")

# Dates are parsed strict-ISO. The looser `06/05/2026` is *deliberately* rejected:
# it means 5 June or 6 May depending on locale, and silently guessing would shift
# a seller's history by a month. Refusing with a stated format is the honest call.
DATE_FORMAT = "%Y-%m-%d"

# How many offending values to name in an error. Enough to spot the pattern,
# not so many that the message becomes a data dump.
MAX_REPORTED = 3


class CsvValidationError(ValueError):
    """Rejected content — becomes a 422."""


class FileTooLargeError(ValueError):
    """Exceeds MAX_FILE_BYTES — becomes a 413."""


def _describe(value: object) -> str:
    """Render one offending cell for a user-facing message.

    Blanks and `N/A` are already NaN by the time read_csv hands them over, so the
    original text is gone — showing the literal `'nan'` would name something the
    user cannot find in their file. Describe the state instead.
    """
    if pd.isna(value):
        return "a blank or N/A cell"
    return repr(str(value))


def _quote(values: list[object]) -> str:
    shown = ", ".join(_describe(v) for v in values[:MAX_REPORTED])
    extra = len(values) - MAX_REPORTED
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def _csv_line(index: int) -> int:
    """DataFrame row index → line number in the file (1 header + 0-based index)."""
    return index + 2


async def read_upload(file) -> bytes:
    """Read an UploadFile, aborting as soon as it passes the size cap.

    Starlette does not expose a trustworthy size before the body is consumed, so
    the limit is enforced against bytes actually read rather than a header value
    a client controls.
    """
    chunks: list[bytes] = []
    total = 0

    while chunk := await file.read(CHUNK_BYTES):
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            limit_mb = MAX_FILE_BYTES // (1024 * 1024)
            raise FileTooLargeError(
                f"File is larger than the {limit_mb}MB limit. "
                "Export a shorter date range and try again."
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _parse(raw: bytes) -> pd.DataFrame:
    if not raw.strip():
        raise CsvValidationError("The uploaded file is empty.")

    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except pd.errors.EmptyDataError as cause:
        raise CsvValidationError("The uploaded file is empty.") from cause
    except UnicodeDecodeError as cause:
        raise CsvValidationError(
            "The file could not be read as text. Upload a plain CSV export, "
            "not a spreadsheet file (.xlsx) or an archive."
        ) from cause
    except pd.errors.ParserError as cause:
        raise CsvValidationError(
            f"The file could not be parsed as CSV: {cause}"
        ) from cause

    # Real exports arrive with padded or capitalised headers ("Date", "Revenue ").
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise CsvValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"The file needs {', '.join(REQUIRED_COLUMNS)} "
            "(a sku column is optional)."
        )


def _parse_dates(frame: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(frame["date"], format=DATE_FORMAT, errors="coerce")

    if parsed.isna().any():
        bad = frame.loc[parsed.isna(), "date"]
        lines = [str(_csv_line(i)) for i in bad.index]
        raise CsvValidationError(
            f"Unparseable date(s) on line(s) {', '.join(lines[:MAX_REPORTED])}: "
            f"{_quote(list(bad))}. "
            "Dates must be in YYYY-MM-DD format."
        )

    return parsed


def _parse_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    # `N/A` is already NaN after read_csv; `$1710.20` survives as a string and is
    # coerced here. Both land as NaN, so one check covers both.
    parsed = pd.to_numeric(frame[column], errors="coerce")

    if parsed.isna().any():
        bad = frame.loc[parsed.isna(), column]
        lines = [str(_csv_line(i)) for i in bad.index]
        raise CsvValidationError(
            f"Non-numeric {column} on line(s) {', '.join(lines[:MAX_REPORTED])}: "
            f"{_quote(list(bad))}. "
            "Remove currency symbols, thousands separators and blank cells."
        )

    return parsed


def validate_csv(raw: bytes) -> tuple[pd.DataFrame, int]:
    """Validate an uploaded CSV and return `(daily_frame, raw_row_count)`.

    The returned frame is aggregated to **one row per date** — a real Shopify
    export is one row per line item, so several rows can share a date. Summing
    them is what makes the per-date series the model expects, and it is the
    groundwork for per-SKU forecasting (design-doc.md:73).

    The <30-row minimum is *not* enforced here: that is a forecasting constraint
    (design-doc.md:187) and belongs with Phase 3, not with file validation.
    """
    frame = _parse(raw)
    _require_columns(frame)

    if frame.empty:
        raise CsvValidationError(
            "The file has column headers but no data rows."
        )

    raw_rows = len(frame)

    clean = pd.DataFrame(
        {
            "date": _parse_dates(frame),
            "revenue": _parse_numeric(frame, "revenue"),
            "units_sold": _parse_numeric(frame, "units_sold"),
        }
    )

    daily = (
        clean.groupby("date", as_index=False)[["revenue", "units_sold"]]
        .sum()
        .sort_values("date", ignore_index=True)
    )

    return daily, raw_rows
