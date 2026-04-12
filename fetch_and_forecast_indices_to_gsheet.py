#!/usr/bin/env python3
"""Pipeline reproducible para generar insumos de índices y proyecciones.

Objetivo:
1) Descargar histórico de índices (Stooq con fallback Yahoo Finance).
2) Normalizar/depurar y consolidar histórico.
3) Generar proyecciones de corto plazo (30 días hábiles por defecto).
4) Escribir resultados en Google Sheets (opcional) y exportar artefactos locales
   reutilizables (CSV/Parquet + metadata JSON) para otros proyectos.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import logging
import os
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gspread
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from google.auth.exceptions import RefreshError as _RefreshError
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
from statsmodels.tsa.arima.model import ARIMA


DEFAULT_SYMBOLS = ("^GSPC", "^IXIC", "^DJI", "^FTSE", "^IBEX", "^BVSP", "^MERV", "^VIX")
DEFAULT_SHEET_ID = "1UQCSPaCtBA_v8aTU1xL6W4xtDOl8PpaP-240gJtLn-0"
DEFAULT_WS_DATA = "indices"
DEFAULT_WS_FORECAST = "proyecciones_30d"
DEFAULT_START_DATE = dt.date(1980, 1, 1)
DEFAULT_HORIZON = 30
DEFAULT_FALLBACK_WINDOW = 252
DEFAULT_OUTPUT_DIR = Path("outputs/latest")


@dataclass(frozen=True)
class AppConfig:
    sheet_id: str
    ws_data: str
    ws_forecast: str
    symbols: tuple[str, ...]
    start_date: dt.date
    horizon: int
    fallback_window: int
    max_p: int
    max_q: int
    write_gsheet: bool
    output_dir: Path


def _canon(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _find_col_by_prefix(columns: list[str], aliases: Sequence[str]) -> str | None:
    canon_cols = {c: _canon(c) for c in columns}
    canon_aliases = [_canon(a) for a in aliases]
    for col, col_canon in canon_cols.items():
        for alias in canon_aliases:
            if col_canon.startswith(alias):
                return col
    return None


def _next_business_days(start_date: dt.date, n: int) -> list[dt.date]:
    out: list[dt.date] = []
    current = start_date
    while len(out) < n:
        current = current + dt.timedelta(days=1)
        if current.weekday() < 5:
            out.append(current)
    return out


def _safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "si", "sí"}


_STOOQ_MAP = {
    "^GSPC": "^spx",
    "^IXIC": "^ixic",
    "^DJI": "^dji",
    "^FTSE": "^ftse",
    "^IBEX": "^ibex",
    "^BVSP": "^bvsp",
    "^MERV": "^merv",
    "^VIX": "^vix",
}


def _stooq_fetch(symbol: str) -> pl.DataFrame | None:
    stooq_symbol = _STOOQ_MAP.get(symbol)
    if not stooq_symbol:
        return None

    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
    except Exception:
        return None

    if not raw:
        return None

    nulls = ["", "NA", "NaN", "null", "NULL", "N/A", "2290404134.7576"]
    try:
        raw_df = pl.read_csv(
            io.BytesIO(raw),
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=nulls,
            schema_overrides={
                "Date": pl.Utf8,
                "Open": pl.Utf8,
                "High": pl.Utf8,
                "Low": pl.Utf8,
                "Close": pl.Utf8,
                "Volume": pl.Utf8,
            },
            try_parse_dates=False,
            encoding="utf8-lossy",
        )
    except Exception:
        return None

    if "Date" not in raw_df.columns or "Close" not in raw_df.columns:
        return None

    df = raw_df.with_columns(
        [
            pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
            pl.col("Open").str.replace(",", "").cast(pl.Float64, strict=False).alias("open"),
            pl.col("High").str.replace(",", "").cast(pl.Float64, strict=False).alias("high"),
            pl.col("Low").str.replace(",", "").cast(pl.Float64, strict=False).alias("low"),
            pl.col("Close").str.replace(",", "").cast(pl.Float64, strict=False).alias("close"),
            pl.col("Volume").str.replace(",", "").cast(pl.Int64, strict=False).alias("volume"),
        ]
    ).select(["date", "open", "high", "low", "close", "volume"])

    df = (
        df.with_columns(
            [
                pl.lit(symbol).alias("symbol"),
                pl.col("close").alias("adj_close"),
                pl.lit("stooq").alias("source"),
            ]
        )
        .select(["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"])
    )

    price_any = pl.any_horizontal([pl.col(c).is_not_null() for c in ["close", "open", "high", "low"]])
    df = df.filter(pl.col("date").is_not_null() & price_any)
    return df if df.height > 0 else None


def _yahoo_fetch(symbol: str, start_date: dt.date) -> pl.DataFrame | None:
    try:
        data = yf.download(
            symbol,
            start=start_date.isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="column",
        )
    except Exception:
        return None

    if data is None or data.empty:
        return None

    pdf = data.reset_index()
    if isinstance(pdf.columns, pd.MultiIndex):
        pdf.columns = [
            "_".join([str(x) for x in tup if x is not None and str(x) != ""]).strip()
            for tup in pdf.columns.to_list()
        ]
    else:
        pdf.columns = [str(c) for c in pdf.columns]

    cols = list(pdf.columns)
    date_col = _find_col_by_prefix(cols, ["Date", "Datetime", "date", "datetime"]) or cols[0]

    rename_map: dict[str, str] = {}
    if date_col in pdf.columns:
        rename_map[date_col] = "date"

    mapping = {
        "open": ["Open"],
        "high": ["High"],
        "low": ["Low"],
        "close": ["Close"],
        "adj_close": ["Adj Close", "AdjClose", "Adjusted Close"],
        "volume": ["Volume", "Vol"],
    }
    for std_name, aliases in mapping.items():
        csrc = _find_col_by_prefix(cols, aliases)
        if csrc is not None:
            rename_map[csrc] = std_name

    df = pl.from_pandas(pdf).rename(rename_map)

    exprs: list[pl.Expr] = []
    if "date" in df.columns:
        exprs.append(pl.col("date").cast(pl.Date, strict=False).alias("date"))
    for col in ("open", "high", "low", "close", "adj_close"):
        if col in df.columns:
            exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    if "volume" in df.columns:
        exprs.append(pl.col("volume").cast(pl.Int64, strict=False).alias("volume"))

    if exprs:
        df = df.with_columns(exprs)

    if "adj_close" not in df.columns and "close" in df.columns:
        df = df.with_columns(pl.col("close").alias("adj_close"))
    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.with_columns(pl.col("adj_close").alias("close"))

    df = (
        df.with_columns([pl.lit(symbol).alias("symbol"), pl.lit("yahoo").alias("source")])
        .select(["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"])
    )

    price_any = pl.any_horizontal(
        [pl.col(c).is_not_null() for c in ["close", "adj_close", "open", "high", "low"] if c in df.columns]
    )
    df = df.filter(pl.col("date").is_not_null() & price_any)
    return df if df.height > 0 else None


def fetch_symbol(symbol: str, start_date: dt.date) -> pl.DataFrame | None:
    stooq = _stooq_fetch(symbol)
    if stooq is not None and stooq.height > 0:
        return stooq
    return _yahoo_fetch(symbol, start_date)


def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    if "adj_close" not in df.columns and "close" in df.columns:
        df = df.with_columns(pl.col("close").alias("adj_close"))
    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.with_columns(pl.col("adj_close").alias("close"))
    wanted = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]
    return df.select([c for c in wanted if c in df.columns])


def dedupe_sort(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
    )


def _get_credentials_from_env() -> Credentials:
    raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        b64 = os.environ.get("GCP_SERVICE_ACCOUNT_JSON_B64")
        if b64:
            raw_json = base64.b64decode(b64).decode("utf-8")

    if raw_json:
        info = json.loads(raw_json)
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    else:
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not path:
            raise RuntimeError(
                "Faltan credenciales: define GOOGLE_SERVICE_ACCOUNT_JSON, "
                "GCP_SERVICE_ACCOUNT_JSON_B64 o GOOGLE_APPLICATION_CREDENTIALS."
            )
        creds = Credentials.from_service_account_file(path, scopes=["https://www.googleapis.com/auth/spreadsheets"])

    try:
        creds.refresh(Request())
    except _RefreshError as exc:
        raise RuntimeError(f"No se pudo refrescar el token del Service Account: {exc}") from exc
    return creds


def _open_or_create_worksheet(
    gc: gspread.Client,
    sheet_id: str,
    worksheet_name: str,
    header_cols: list[str],
) -> gspread.Worksheet:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=max(len(header_cols), 10))
        set_with_dataframe(
            ws,
            pd.DataFrame(columns=header_cols),
            include_index=False,
            include_column_header=True,
            resize=True,
        )
    return ws


def _read_existing(ws: gspread.Worksheet, cols: list[str]) -> pd.DataFrame:
    values = ws.get_all_records()
    if not values:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(values)
    for col in cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "last_obs_date" in df.columns:
        df["last_obs_date"] = pd.to_datetime(df["last_obs_date"], errors="coerce").dt.date
    if "ingestion_ts" in df.columns:
        df["ingestion_ts"] = pd.to_datetime(df["ingestion_ts"], errors="coerce")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "fc_h",
        "yhat",
        "yhat_lo",
        "yhat_hi",
        "last_obs_value",
        "aicc",
        "insample_n",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _write_full(ws: gspread.Worksheet, pdf: pd.DataFrame, required_non_null_cols: list[str] | None = None) -> None:
    if required_non_null_cols:
        existing = [c for c in required_non_null_cols if c in pdf.columns]
        if existing:
            pdf = pdf.loc[~pdf[existing].isna().all(axis=1)].copy()
    set_with_dataframe(ws, pdf, include_index=False, include_column_header=True, resize=True)


def _coerce_hist_types(pdf: pd.DataFrame) -> pd.DataFrame:
    if "symbol" in pdf.columns:
        pdf["symbol"] = pdf["symbol"].astype("string")
    if "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce").dt.date
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    if "volume" in pdf.columns:
        pdf["volume"] = pd.to_numeric(pdf["volume"], errors="coerce").astype("Int64")
    if "ingestion_ts" in pdf.columns:
        pdf["ingestion_ts"] = pd.to_datetime(pdf["ingestion_ts"], errors="coerce")
    if "run_id" in pdf.columns:
        pdf["run_id"] = pdf["run_id"].astype("string")
    return pdf


def _coerce_fc_types(pdf: pd.DataFrame) -> pd.DataFrame:
    if "symbol" in pdf.columns:
        pdf["symbol"] = pdf["symbol"].astype("string")
    if "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce").dt.date
    if "last_obs_date" in pdf.columns:
        pdf["last_obs_date"] = pd.to_datetime(pdf["last_obs_date"], errors="coerce").dt.date
    if "ingestion_ts" in pdf.columns:
        pdf["ingestion_ts"] = pd.to_datetime(pdf["ingestion_ts"], errors="coerce")
    for col in ["fc_h", "yhat", "yhat_lo", "yhat_hi", "last_obs_value", "aicc", "insample_n"]:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    if "run_id" in pdf.columns:
        pdf["run_id"] = pdf["run_id"].astype("string")
    return pdf


def _aicc(llf: float, n_obs: int, k_params: int) -> float:
    if n_obs - k_params - 1 <= 0:
        return np.inf
    return -2.0 * llf + 2.0 * k_params + (2.0 * k_params * (k_params + 1)) / (n_obs - k_params - 1)


def _fit_arima_best_aicc(returns: np.ndarray, max_p: int = 3, max_q: int = 3):
    best = {"aicc": np.inf, "order": None, "res": None}
    y = np.asarray(returns, dtype=float)
    n_obs = y.shape[0]

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = ARIMA(
                    y,
                    order=(p, 0, q),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(method="statespace")
                aicc_val = _aicc(res.llf, n_obs, res.params.shape[0])
                if np.isfinite(aicc_val) and aicc_val < best["aicc"]:
                    best = {"aicc": aicc_val, "order": (p, 0, q), "res": res}
            except Exception:
                continue

    return best["res"], best["order"], float(best["aicc"]) if np.isfinite(best["aicc"]) else np.nan


def _winsorize(arr: np.ndarray, p: float = 0.01) -> np.ndarray:
    clean = np.asarray(arr, dtype=float)
    if clean.size == 0:
        return clean
    lo, hi = np.nanquantile(clean, [p, 1 - p])
    return np.clip(clean, lo, hi)


def _forecast_with_drift(last_val: float, last_date: dt.date, returns: np.ndarray, horizon: int, mu_zero: bool) -> pd.DataFrame:
    tail = np.asarray(returns, dtype=float)
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        tail = np.array([0.0], dtype=float)

    mu = 0.0 if mu_zero else float(np.nanmean(tail))
    sigma = float(np.nanstd(tail, ddof=1)) if tail.shape[0] > 1 else 0.0
    z = 1.96
    h = np.arange(1, horizon + 1, dtype=float)

    cum_mu = mu * h
    cum_sd = sigma * np.sqrt(h)

    yhat = last_val * np.exp(cum_mu)
    lo = last_val * np.exp(cum_mu - z * cum_sd)
    hi = last_val * np.exp(cum_mu + z * cum_sd)

    return pd.DataFrame(
        {
            "date": _next_business_days(last_date, horizon),
            "fc_h": list(range(1, horizon + 1)),
            "method": ["naive_drift_log"] * horizon,
            "yhat": yhat,
            "yhat_lo": lo,
            "yhat_hi": hi,
            "model_family": ["naive-drift"] * horizon,
            "model_spec": [f"drift_log(mu={mu:.6g},sigma={sigma:.6g})"] * horizon,
            "aicc": [np.nan] * horizon,
        }
    )


def forecast_one_symbol(
    pdf: pd.DataFrame,
    symbol: str,
    horizon: int,
    fallback_window: int,
    max_p: int,
    max_q: int,
) -> pd.DataFrame | None:
    data = pdf[(pdf["symbol"] == symbol) & pd.notna(pdf["adj_close"])].copy()
    if data.empty:
        return None

    data = data.sort_values("date")
    y = data["adj_close"].astype(float)

    log_price = _safe_log(y)
    returns = log_price.diff().dropna()

    if returns.shape[0] < 2:
        return None

    last_date = data["date"].iloc[-1]
    last_value = float(y.iloc[-1])
    insample_n = int(returns.shape[0])

    mu_zero = symbol.upper() in {"^VIX"}
    winsorized = _winsorize(returns.values.astype(float), p=0.01)

    if winsorized.shape[0] < 50:
        tail = winsorized[-min(fallback_window, winsorized.shape[0]) :]
        fc = _forecast_with_drift(last_value, last_date, tail, horizon, mu_zero)
        fc["model_desc"] = f"Naive-drift en log-retornos (n={tail.shape[0]}, winsor=1%)."
        fc["insample_n"] = insample_n
    else:
        res, order, best_aicc = _fit_arima_best_aicc(winsorized, max_p=max_p, max_q=max_q)
        if res is None:
            tail = winsorized[-min(fallback_window, winsorized.shape[0]) :]
            fc = _forecast_with_drift(last_value, last_date, tail, horizon, mu_zero)
            fc["model_desc"] = "Fallback naive-drift en log-retornos (sin convergencia ARIMA)."
            fc["insample_n"] = insample_n
        else:
            pred = res.get_forecast(steps=horizon)
            mean_step = np.asarray(pred.predicted_mean, dtype=float)
            se_step = np.asarray(pred.se_mean, dtype=float)
            cum_ret_mean = np.cumsum(mean_step)
            cum_ret_sd = np.sqrt(np.cumsum(se_step**2))

            z = 1.96
            yhat = last_value * np.exp(cum_ret_mean)
            lo = last_value * np.exp(cum_ret_mean - z * cum_ret_sd)
            hi = last_value * np.exp(cum_ret_mean + z * cum_ret_sd)

            fc = pd.DataFrame(
                {
                    "date": _next_business_days(last_date, horizon),
                    "fc_h": list(range(1, horizon + 1)),
                    "method": ["arima_logret_aicc"] * horizon,
                    "yhat": yhat,
                    "yhat_lo": lo,
                    "yhat_hi": hi,
                    "model_family": ["ARIMA"] * horizon,
                    "model_spec": [f"ARIMA{order}"] * horizon,
                    "aicc": [best_aicc] * horizon,
                    "insample_n": [insample_n] * horizon,
                    "model_desc": [
                        "ARIMA(p,0,q) en log-retornos con selección por AICc y winsorización 1%."
                    ]
                    * horizon,
                }
            )

    fc.insert(0, "symbol", symbol)
    fc["last_obs_date"] = last_date
    fc["last_obs_value"] = last_value
    return fc


def build_historical_dataset(config: AppConfig) -> pd.DataFrame:
    frames: list[pl.DataFrame] = []
    for symbol in config.symbols:
        df = fetch_symbol(symbol, config.start_date)
        if df is None or df.height == 0:
            logging.warning("Sin datos para %s", symbol)
            continue
        frames.append(normalize_schema(df))

    if not frames:
        raise RuntimeError("No se pudieron descargar datos de ningún símbolo.")

    hist = pl.concat(frames, how="vertical_relaxed")
    hist = dedupe_sort(hist)
    return hist.to_pandas()


def _hist_columns() -> list[str]:
    return [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
        "ingestion_ts",
        "run_id",
    ]


def _forecast_columns() -> list[str]:
    return [
        "symbol",
        "date",
        "fc_h",
        "method",
        "yhat",
        "yhat_lo",
        "yhat_hi",
        "last_obs_date",
        "last_obs_value",
        "model_desc",
        "model_family",
        "model_spec",
        "aicc",
        "insample_n",
        "ingestion_ts",
        "run_id",
    ]


def merge_hist(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    existing = _coerce_hist_types(existing)
    incoming = _coerce_hist_types(incoming)

    merged = pd.concat([existing, incoming], axis=0, ignore_index=True)
    if "symbol" in merged.columns:
        symbol_norm = merged["symbol"].astype("string").str.strip()
        merged = merged.loc[symbol_norm.notna() & symbol_norm.ne("")]
    if "date" in merged.columns:
        merged = merged.loc[merged["date"].notna()]

    merged = (
        merged.sort_values(["symbol", "date", "ingestion_ts"], kind="mergesort")
        .drop_duplicates(subset=["symbol", "date"], keep="last")
    )
    for col in _hist_columns():
        if col not in merged.columns:
            merged[col] = pd.Series(dtype="object")
    return merged[_hist_columns()]


def merge_forecast(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    existing = _coerce_fc_types(existing)
    incoming = _coerce_fc_types(incoming)

    # Conserva solo la última corrida por símbolo y horizonte (h=1..H),
    # evitando acumular bloques históricos de proyecciones obsoletas.
    merged = pd.concat([existing, incoming], axis=0, ignore_index=True)
    if "symbol" in merged.columns:
        symbol_norm = merged["symbol"].astype("string").str.strip()
        merged = merged.loc[symbol_norm.notna() & symbol_norm.ne("")]
    for required in ["fc_h", "date"]:
        if required in merged.columns:
            merged = merged.loc[merged[required].notna()]

    merged = (
        merged.sort_values(["symbol", "fc_h", "date", "ingestion_ts"], kind="mergesort")
        .drop_duplicates(subset=["symbol", "fc_h"], keep="last")
    )
    for col in _forecast_columns():
        if col not in merged.columns:
            merged[col] = pd.Series(dtype="object")
    return merged[_forecast_columns()]


def generate_forecasts(config: AppConfig, merged_hist: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for symbol in config.symbols:
        fc = forecast_one_symbol(
            merged_hist,
            symbol=symbol,
            horizon=config.horizon,
            fallback_window=config.fallback_window,
            max_p=config.max_p,
            max_q=config.max_q,
        )
        if fc is None or fc.empty:
            logging.warning("Sin proyección para %s", symbol)
            continue
        pieces.append(fc)

    if not pieces:
        raise RuntimeError("No se pudo generar ninguna proyección.")

    out = pd.concat(pieces, axis=0, ignore_index=True)
    out = _coerce_fc_types(out)
    for col in _forecast_columns():
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")
    return out[_forecast_columns()]


def export_outputs(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_csv = output_dir / "indices_historicos.csv"
    forecast_csv = output_dir / "proyecciones_30d.csv"
    metadata_json = output_dir / "run_metadata.json"

    historical.to_csv(historical_csv, index=False)
    forecast.to_csv(forecast_csv, index=False)

    try:
        historical.to_parquet(output_dir / "indices_historicos.parquet", index=False)
        forecast.to_parquet(output_dir / "proyecciones_30d.parquet", index=False)
    except Exception as exc:
        logging.warning("No se pudieron exportar Parquet (continuamos con CSV): %s", exc)

    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(config: AppConfig) -> None:
    run_id = uuid.uuid4().hex[:12]
    ingestion_ts = dt.datetime.utcnow().replace(microsecond=0)

    logging.info("Iniciando run_id=%s", run_id)
    logging.info("Símbolos: %s", ", ".join(config.symbols))

    hist = build_historical_dataset(config)
    hist["ingestion_ts"] = ingestion_ts
    hist["run_id"] = run_id
    hist = _coerce_hist_types(hist)

    merged_hist = hist
    merged_forecast: pd.DataFrame | None = None

    if config.write_gsheet:
        creds = _get_credentials_from_env()
        gc = gspread.authorize(creds)

        ws_hist = _open_or_create_worksheet(gc, config.sheet_id, config.ws_data, _hist_columns())
        existing_hist = _read_existing(ws_hist, _hist_columns())
        merged_hist = merge_hist(existing_hist, hist)
        _write_full(ws_hist, merged_hist, required_non_null_cols=["adj_close", "close", "open", "high", "low"])

        forecast = generate_forecasts(config, merged_hist)
        forecast["ingestion_ts"] = ingestion_ts
        forecast["run_id"] = run_id
        forecast = _coerce_fc_types(forecast)

        ws_fc = _open_or_create_worksheet(gc, config.sheet_id, config.ws_forecast, _forecast_columns())
        existing_fc = _read_existing(ws_fc, _forecast_columns())
        merged_forecast = merge_forecast(existing_fc, forecast)
        _write_full(ws_fc, merged_forecast, required_non_null_cols=["yhat"])
    else:
        merged_hist = merge_hist(pd.DataFrame(columns=_hist_columns()), hist)
        forecast = generate_forecasts(config, merged_hist)
        forecast["ingestion_ts"] = ingestion_ts
        forecast["run_id"] = run_id
        merged_forecast = merge_forecast(pd.DataFrame(columns=_forecast_columns()), forecast)

    symbol_coverage = (
        merged_hist.groupby("symbol", dropna=False)
        .agg(start_date=("date", "min"), end_date=("date", "max"), n_obs=("date", "count"))
        .reset_index()
        .to_dict(orient="records")
    )
    for row in symbol_coverage:
        for key in ("start_date", "end_date"):
            val = row.get(key)
            if isinstance(val, (dt.date, dt.datetime)):
                row[key] = val.isoformat()

    metadata = {
        "run_id": run_id,
        "generated_at_utc": ingestion_ts.isoformat() + "Z",
        "pipeline_version": "2.0.0",
        "symbols": list(config.symbols),
        "horizon_business_days": config.horizon,
        "sources": ["stooq", "yahoo"],
        "write_gsheet": config.write_gsheet,
        "sheet_id": config.sheet_id if config.write_gsheet else None,
        "ws_data": config.ws_data if config.write_gsheet else None,
        "ws_forecast": config.ws_forecast if config.write_gsheet else None,
        "historical_rows": int(len(merged_hist)),
        "forecast_rows": int(len(merged_forecast)),
        "symbol_coverage": symbol_coverage,
        "notes": [
            "Modelo principal: ARIMA(p,0,q) sobre retornos logarítmicos con selección por AICc.",
            "Fallback: naive-drift log-normal cuando la serie es corta o falla ARIMA.",
            "IC95% por aproximación gaussiana sobre retorno acumulado.",
        ],
    }

    export_outputs(merged_hist, merged_forecast, metadata, config.output_dir)
    logging.info("Pipeline finalizado. Salidas en %s", config.output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera histórico + proyección de índices y exporta insumos reutilizables.")
    parser.add_argument("--sheet-id", default=os.environ.get("SHEET_ID", DEFAULT_SHEET_ID))
    parser.add_argument("--ws-data", default=os.environ.get("WS_DATA", DEFAULT_WS_DATA))
    parser.add_argument("--ws-forecast", default=os.environ.get("WS_FC", DEFAULT_WS_FORECAST))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Lista CSV de símbolos (ej: ^GSPC,^IXIC)")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat(), help="Fecha inicial ISO (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Horizonte de proyección en días hábiles")
    parser.add_argument("--fallback-window", type=int, default=DEFAULT_FALLBACK_WINDOW)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-gsheet", action="store_true", help="No escribe en Google Sheets; solo genera insumos locales")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> AppConfig:
    symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        raise ValueError("Debes indicar al menos un símbolo.")

    start_date = dt.date.fromisoformat(args.start_date)
    if args.horizon <= 0:
        raise ValueError("--horizon debe ser mayor que 0.")

    write_gsheet = not args.no_gsheet
    env_override = os.environ.get("WRITE_GSHEET")
    if env_override is not None:
        write_gsheet = _to_bool(env_override, write_gsheet)

    return AppConfig(
        sheet_id=args.sheet_id,
        ws_data=args.ws_data,
        ws_forecast=args.ws_forecast,
        symbols=symbols,
        start_date=start_date,
        horizon=args.horizon,
        fallback_window=max(2, int(args.fallback_window)),
        max_p=max(0, int(args.max_p)),
        max_q=max(0, int(args.max_q)),
        write_gsheet=write_gsheet,
        output_dir=Path(args.output_dir),
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = _build_config(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
