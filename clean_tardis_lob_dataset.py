from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("data") / "Tardis_sample_data" / "BTC20230120.csv"
DEFAULT_OUTPUT_PATH = Path("data") / "BTC20230120_clean_1s.pkl"
DEFAULT_DEPTH_LEVELS = 10
DEFAULT_CHUNK_SIZE = 200_000


@dataclass(frozen=True)
class TimestampProfile:
    total_rows: int
    unique_ms_rows: int
    duplicate_ms_rows: int
    seconds_covered: int
    avg_rows_per_second: float
    median_rows_per_second: float
    p95_rows_per_second: float
    max_rows_per_second: int
    recommended_frequency: str


@dataclass(frozen=True)
class InputFormatSpec:
    name: str
    usecols: list[str]
    timestamp_column: str
    timestamp_divisor: int
    exchange_value: str | None = None
    symbol_value: str | None = None
    has_datetime_column: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean a Tardis-style LOB CSV, convert timestamps to UTC, resample to a practical "
            "frequency, and save a FI-2010-style pickle bundle."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output pickle path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--freq",
        choices=("auto", "1ms", "1s"),
        default="auto",
        help="Resample frequency. Default: auto",
    )
    parser.add_argument(
        "--depth-levels",
        type=int,
        default=DEFAULT_DEPTH_LEVELS,
        help="Number of book levels to keep for FI-2010-style features. Default: 10",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="CSV chunk size. Default: 200000",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to print from the cleaned dataframe. Default: 5",
    )
    return parser.parse_args()


def fi2010_feature_columns(depth_levels: int = DEFAULT_DEPTH_LEVELS) -> list[str]:
    columns: list[str] = []
    for level in range(1, depth_levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def tardis_usecols(depth_levels: int) -> list[str]:
    columns = ["exchange", "symbol", "timestamp", "local_timestamp"]
    for level in range(depth_levels):
        columns.extend(
            [
                f"asks[{level}].price",
                f"asks[{level}].amount",
                f"bids[{level}].price",
                f"bids[{level}].amount",
            ]
        )
    return columns


def flat_export_usecols(depth_levels: int) -> list[str]:
    feature_columns = [str(index) for index in range(2, 2 + 4 * depth_levels)]
    return ["0", "1", *feature_columns]


def detect_input_format(input_path: Path, depth_levels: int) -> InputFormatSpec:
    header = pd.read_csv(input_path, nrows=0)
    columns = list(header.columns)

    if {"exchange", "symbol", "timestamp"}.issubset(columns):
        return InputFormatSpec(
            name="tardis_raw",
            usecols=tardis_usecols(depth_levels),
            timestamp_column="timestamp",
            timestamp_divisor=1000,
        )

    flat_required = set(flat_export_usecols(depth_levels))
    if flat_required.issubset(columns):
        return InputFormatSpec(
            name="flat_fi2010_export",
            usecols=[column for column in ["Unnamed: 0", *flat_export_usecols(depth_levels)] if column in columns],
            timestamp_column="0",
            timestamp_divisor=1,
            exchange_value="flat_csv_export",
            symbol_value=input_path.stem,
            has_datetime_column=True,
        )

    raise ValueError(
        "Unsupported CSV schema. Expected either raw Tardis columns "
        "('exchange', 'symbol', 'timestamp', ...) or a flat export with columns '0'...'41'."
    )


def normalize_timestamp_ms(series: pd.Series, timestamp_divisor: int) -> pd.Series:
    timestamp = series.astype("int64")
    if timestamp_divisor != 1:
        timestamp = (timestamp // timestamp_divisor).astype("int64")
    return timestamp.astype("int64")


def recommend_frequency(
    avg_rows_per_second: float,
    max_rows_per_second: int,
    unique_ms_rows: int,
) -> str:
    if avg_rows_per_second > 10 or max_rows_per_second > 100 or unique_ms_rows > 1_000_000:
        return "1s"
    return "1ms"


def profile_timestamps(input_path: Path, chunk_size: int, input_format: InputFormatSpec) -> TimestampProfile:
    second_counts: dict[int, int] = {}
    ms_counts: dict[int, int] = {}
    total_rows = 0

    for chunk in pd.read_csv(input_path, usecols=[input_format.timestamp_column], chunksize=chunk_size):
        timestamp_ms = normalize_timestamp_ms(chunk[input_format.timestamp_column], input_format.timestamp_divisor)
        total_rows += len(timestamp_ms)

        ms_value_counts = timestamp_ms.value_counts()
        for bucket_ms, count in ms_value_counts.items():
            ms_counts[int(bucket_ms)] = ms_counts.get(int(bucket_ms), 0) + int(count)

        counts = (timestamp_ms // 1000).value_counts()
        for second_bucket, count in counts.items():
            second_counts[int(second_bucket)] = second_counts.get(int(second_bucket), 0) + int(count)

    second_series = pd.Series(second_counts, dtype="int64")
    unique_ms_rows = len(ms_counts)
    recommended = recommend_frequency(
        avg_rows_per_second=float(second_series.mean()),
        max_rows_per_second=int(second_series.max()),
        unique_ms_rows=unique_ms_rows,
    )

    return TimestampProfile(
        total_rows=total_rows,
        unique_ms_rows=unique_ms_rows,
        duplicate_ms_rows=total_rows - unique_ms_rows,
        seconds_covered=int(len(second_series)),
        avg_rows_per_second=float(second_series.mean()),
        median_rows_per_second=float(second_series.median()),
        p95_rows_per_second=float(second_series.quantile(0.95)),
        max_rows_per_second=int(second_series.max()),
        recommended_frequency=recommended,
    )


def normalize_raw_tardis_chunk(chunk: pd.DataFrame, depth_levels: int) -> pd.DataFrame:
    cleaned = pd.DataFrame(
        {
            "exchange": chunk["exchange"],
            "symbol": chunk["symbol"],
            "event_timestamp_ms": normalize_timestamp_ms(chunk["timestamp"], timestamp_divisor=1000),
        }
    )
    cleaned["event_timestamp_utc"] = pd.to_datetime(cleaned["event_timestamp_ms"], unit="ms", utc=True)

    for level in range(depth_levels):
        idx = level + 1
        cleaned[f"ask_price_{idx}"] = chunk[f"asks[{level}].price"].astype(float)
        cleaned[f"ask_size_{idx}"] = chunk[f"asks[{level}].amount"].astype(float)
        cleaned[f"bid_price_{idx}"] = chunk[f"bids[{level}].price"].astype(float)
        cleaned[f"bid_size_{idx}"] = chunk[f"bids[{level}].amount"].astype(float)

    cleaned["mid_price"] = (cleaned["ask_price_1"] + cleaned["bid_price_1"]) / 2.0
    cleaned["spread"] = cleaned["ask_price_1"] - cleaned["bid_price_1"]
    return cleaned


def normalize_flat_export_chunk(chunk: pd.DataFrame, depth_levels: int, input_format: InputFormatSpec) -> pd.DataFrame:
    chunk = chunk.drop(columns=["Unnamed: 0"], errors="ignore")
    cleaned = pd.DataFrame(
        {
            "exchange": input_format.exchange_value,
            "symbol": input_format.symbol_value,
            "event_timestamp_ms": normalize_timestamp_ms(chunk[input_format.timestamp_column], timestamp_divisor=1),
        }
    )
    cleaned["event_timestamp_utc"] = pd.to_datetime(cleaned["event_timestamp_ms"], unit="ms", utc=True)

    for level in range(depth_levels):
        idx = level + 1
        bid_price_col = str(2 + 2 * level)
        bid_size_col = str(3 + 2 * level)
        ask_price_col = str(2 + 2 * depth_levels + 2 * level)
        ask_size_col = str(3 + 2 * depth_levels + 2 * level)
        cleaned[f"ask_price_{idx}"] = chunk[ask_price_col].astype(float)
        cleaned[f"ask_size_{idx}"] = chunk[ask_size_col].astype(float)
        cleaned[f"bid_price_{idx}"] = chunk[bid_price_col].astype(float)
        cleaned[f"bid_size_{idx}"] = chunk[bid_size_col].astype(float)

    cleaned["mid_price"] = (cleaned["ask_price_1"] + cleaned["bid_price_1"]) / 2.0
    cleaned["spread"] = cleaned["ask_price_1"] - cleaned["bid_price_1"]
    return cleaned


def normalize_book_columns(chunk: pd.DataFrame, depth_levels: int, input_format: InputFormatSpec) -> pd.DataFrame:
    if input_format.name == "tardis_raw":
        return normalize_raw_tardis_chunk(chunk, depth_levels=depth_levels)
    if input_format.name == "flat_fi2010_export":
        return normalize_flat_export_chunk(chunk, depth_levels=depth_levels, input_format=input_format)
    raise ValueError(f"Unsupported input format: {input_format.name}")


def add_bucket_columns(cleaned: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "1ms":
        cleaned["timestamp_bucket_ms"] = cleaned["event_timestamp_ms"]
    elif freq == "1s":
        cleaned["timestamp_bucket_ms"] = (cleaned["event_timestamp_ms"] // 1000) * 1000
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    cleaned["timestamp_utc"] = pd.to_datetime(cleaned["timestamp_bucket_ms"], unit="ms", utc=True)
    return cleaned


def reduce_chunk_to_last_snapshot_per_bucket(
    cleaned_chunk: pd.DataFrame,
    carry: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if carry is not None and not carry.empty:
        cleaned_chunk = pd.concat([carry, cleaned_chunk], ignore_index=True)

    cleaned_chunk = cleaned_chunk.sort_values("timestamp_bucket_ms").reset_index(drop=True)
    deduped = cleaned_chunk.drop_duplicates(subset=["timestamp_bucket_ms"], keep="last").reset_index(drop=True)

    if deduped.empty:
        return deduped, None
    if len(deduped) == 1:
        return deduped.iloc[0:0].copy(), deduped.copy()

    return deduped.iloc[:-1].copy(), deduped.iloc[[-1]].copy()


def clean_tardis_csv(
    input_path: Path,
    depth_levels: int,
    freq: str,
    chunk_size: int,
    input_format: InputFormatSpec | None = None,
) -> pd.DataFrame:
    detected_format = input_format or detect_input_format(input_path, depth_levels)
    parts: list[pd.DataFrame] = []
    carry: pd.DataFrame | None = None

    for chunk in pd.read_csv(input_path, usecols=detected_format.usecols, chunksize=chunk_size):
        cleaned = normalize_book_columns(chunk, depth_levels=depth_levels, input_format=detected_format)
        cleaned = add_bucket_columns(cleaned, freq=freq)
        finalized, carry = reduce_chunk_to_last_snapshot_per_bucket(cleaned, carry=carry)
        if not finalized.empty:
            parts.append(finalized)

    if carry is not None and not carry.empty:
        parts.append(carry)

    if not parts:
        raise RuntimeError("No cleaned rows were produced from the input CSV.")

    cleaned_frame = pd.concat(parts, ignore_index=True)
    cleaned_frame = cleaned_frame.sort_values("timestamp_bucket_ms").reset_index(drop=True)
    cleaned_frame = cleaned_frame.drop_duplicates(subset=["timestamp_bucket_ms"], keep="last").reset_index(drop=True)
    cleaned_frame = cleaned_frame.drop(columns=["timestamp_bucket_ms"])
    return cleaned_frame


def build_output_bundle(
    cleaned_frame: pd.DataFrame,
    profile: TimestampProfile,
    chosen_frequency: str,
    input_path: Path,
    depth_levels: int,
    input_format: InputFormatSpec,
) -> dict[str, object]:
    feature_columns = fi2010_feature_columns(depth_levels)
    fi2010_frame = cleaned_frame[feature_columns].copy()
    fi2010_matrix = fi2010_frame.to_numpy(dtype=float).T

    timestamp_unit_note = (
        "microseconds with millisecond precision"
        if input_format.timestamp_divisor == 1000
        else "milliseconds"
    )

    metadata = {
        "source_file": str(input_path.resolve()),
        "source_format": input_format.name,
        "exchange": cleaned_frame["exchange"].iloc[0],
        "symbol": cleaned_frame["symbol"].iloc[0],
        "time_zone": "UTC",
        "input_timestamp_unit": timestamp_unit_note,
        "chosen_frequency": chosen_frequency,
        "recommended_frequency": profile.recommended_frequency,
        "depth_levels": depth_levels,
        "feature_columns": feature_columns,
        "clean_start_utc": cleaned_frame["timestamp_utc"].iloc[0].isoformat(),
        "clean_end_utc": cleaned_frame["timestamp_utc"].iloc[-1].isoformat(),
        "profile": asdict(profile),
        "note": (
            "This pickle stores a cleaned UTC dataframe and a FI-2010-style 40-feature matrix. "
            "Each time bucket keeps the last available snapshot."
        ),
    }

    return {
        "clean_lob": cleaned_frame,
        "fi2010_features": fi2010_frame,
        "fi2010_matrix": fi2010_matrix,
        "metadata": metadata,
    }


def save_bundle(bundle: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(bundle, output_path)


def resolve_output_path(output_path: Path, chosen_frequency: str, input_path: Path) -> Path:
    if output_path != DEFAULT_OUTPUT_PATH:
        return output_path
    return DEFAULT_OUTPUT_PATH.with_name(f"{input_path.stem}_clean_{chosen_frequency}{DEFAULT_OUTPUT_PATH.suffix}")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing input CSV: {args.input}")

    input_format = detect_input_format(args.input, depth_levels=args.depth_levels)
    profile = profile_timestamps(args.input, chunk_size=args.chunk_size, input_format=input_format)
    chosen_frequency = profile.recommended_frequency if args.freq == "auto" else args.freq
    output_path = resolve_output_path(args.output, chosen_frequency, input_path=args.input)

    print("Detected input format")
    print(asdict(input_format))
    print()
    print("Timestamp profile")
    print(asdict(profile))
    print(f"Chosen frequency: {chosen_frequency}")

    cleaned_frame = clean_tardis_csv(
        input_path=args.input,
        depth_levels=args.depth_levels,
        freq=chosen_frequency,
        chunk_size=args.chunk_size,
        input_format=input_format,
    )
    bundle = build_output_bundle(
        cleaned_frame=cleaned_frame,
        profile=profile,
        chosen_frequency=chosen_frequency,
        input_path=args.input,
        depth_levels=args.depth_levels,
        input_format=input_format,
    )
    save_bundle(bundle=bundle, output_path=output_path)

    print()
    print("Cleaned head")
    print(cleaned_frame.head(args.head))
    print()
    print("Shapes")
    print("clean_lob:", bundle["clean_lob"].shape)
    print("fi2010_features:", bundle["fi2010_features"].shape)
    print("fi2010_matrix:", bundle["fi2010_matrix"].shape)
    print()
    print(f"Saved cleaned pickle to {output_path.resolve()}")


if __name__ == "__main__":
    main()
