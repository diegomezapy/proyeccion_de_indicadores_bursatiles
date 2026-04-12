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
import math
import os
import re
import time
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Sequence
from urllib.parse import quote_plus

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

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


DEFAULT_SYMBOLS = ("^GSPC", "^IXIC", "^DJI", "^FTSE", "^IBEX", "^BVSP", "^MERV", "^VIX")
DEFAULT_SHEET_ID = "1UQCSPaCtBA_v8aTU1xL6W4xtDOl8PpaP-240gJtLn-0"
DEFAULT_WS_DATA = "indices"
DEFAULT_WS_FORECAST = "proyecciones_30d"
DEFAULT_WS_SCRAPING_QA = "scraping_qa"
DEFAULT_WS_SENTIMENT = "sentimiento_features"
DEFAULT_WS_BENCHMARK = "benchmark_modelos"
DEFAULT_START_DATE = dt.date(1980, 1, 1)
DEFAULT_HORIZON = 30
DEFAULT_FALLBACK_WINDOW = 252
DEFAULT_OUTPUT_DIR = Path("outputs/latest")
DEFAULT_SENTIMENT_MAX_ITEMS = 40


@dataclass(frozen=True)
class AppConfig:
    sheet_id: str
    ws_data: str
    ws_forecast: str
    ws_scraping_qa: str
    ws_sentiment: str
    ws_benchmark: str
    symbols: tuple[str, ...]
    start_date: dt.date
    horizon: int
    fallback_window: int
    max_p: int
    max_q: int
    sentiment_max_items: int
    write_gsheet: bool
    with_sentiment: bool
    with_benchmark: bool
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

_SYMBOL_QUERIES = {
    "^GSPC": "S&P 500 OR Standard and Poor 500 OR SP500 stock market",
    "^DJI": "Dow Jones OR DJIA stock market",
}

_POS_WORDS = {
    "gain",
    "gains",
    "surge",
    "surges",
    "rally",
    "bullish",
    "beat",
    "beats",
    "upgrade",
    "upside",
    "growth",
    "expansion",
    "strong",
    "record",
    "optimism",
    "recovery",
    "outperform",
}

_NEG_WORDS = {
    "loss",
    "losses",
    "drop",
    "drops",
    "fall",
    "falls",
    "bearish",
    "miss",
    "misses",
    "downgrade",
    "risk",
    "crash",
    "weak",
    "recession",
    "inflation",
    "selloff",
    "volatility",
    "uncertainty",
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


def _fetch_attempt(symbol: str, source: str, fn, *args) -> tuple[pl.DataFrame | None, dict]:
    started = dt.datetime.utcnow().replace(microsecond=0)
    t0 = time.perf_counter()
    ok = False
    rows = 0
    error = ""
    df: pl.DataFrame | None = None
    try:
        df = fn(*args)
        rows = int(df.height) if df is not None else 0
        ok = rows > 0
        if not ok:
            error = "sin datos"
    except Exception as exc:
        error = str(exc)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    ended = dt.datetime.utcnow().replace(microsecond=0)

    report = {
        "symbol": symbol,
        "source": source,
        "status": "ok" if ok else "error",
        "rows_raw": rows,
        "latency_ms": latency_ms,
        "error": error,
        "attempt_started_utc": started,
        "attempt_ended_utc": ended,
    }
    return (df if ok else None), report


def fetch_symbol(symbol: str, start_date: dt.date) -> tuple[pl.DataFrame | None, list[dict]]:
    attempts: list[dict] = []

    stooq_df, stooq_report = _fetch_attempt(symbol, "stooq", _stooq_fetch, symbol)
    attempts.append(stooq_report)
    if stooq_df is not None and stooq_df.height > 0:
        return stooq_df, attempts

    yahoo_df, yahoo_report = _fetch_attempt(symbol, "yahoo", _yahoo_fetch, symbol, start_date)
    attempts.append(yahoo_report)
    return yahoo_df, attempts


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


def _quality_metrics_for_hist(symbol_df: pl.DataFrame) -> tuple[str | None, str | None, float]:
    if symbol_df.height == 0:
        return None, None, np.nan
    start = symbol_df.select(pl.col("date").min()).item()
    end = symbol_df.select(pl.col("date").max()).item()
    miss_ratio = float(
        symbol_df.select(pl.col("adj_close").is_null().sum() / pl.len())
        .to_numpy()[0, 0]
    )
    start_iso = start.isoformat() if isinstance(start, dt.date) else None
    end_iso = end.isoformat() if isinstance(end, dt.date) else None
    return start_iso, end_iso, miss_ratio


def build_historical_dataset(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pl.DataFrame] = []
    qa_rows: list[dict] = []
    for symbol in config.symbols:
        df, attempts = fetch_symbol(symbol, config.start_date)
        qa_rows.extend(attempts)
        if df is None or df.height == 0:
            logging.warning("Sin datos para %s", symbol)
            continue
        clean = normalize_schema(df)
        start_iso, end_iso, miss_ratio = _quality_metrics_for_hist(clean)
        success_source = str(clean.select(pl.col("source").drop_nulls().first()).item())
        qa_rows.append(
            {
                "symbol": symbol,
                "source": success_source,
                "status": "accepted",
                "rows_raw": int(clean.height),
                "latency_ms": np.nan,
                "error": "",
                "attempt_started_utc": pd.NaT,
                "attempt_ended_utc": pd.NaT,
                "series_start_date": start_iso,
                "series_end_date": end_iso,
                "pct_missing_adj_close": miss_ratio,
            }
        )
        frames.append(clean)

    if not frames:
        raise RuntimeError("No se pudieron descargar datos de ningún símbolo.")

    hist = pl.concat(frames, how="vertical_relaxed")
    hist = dedupe_sort(hist)
    qa = pd.DataFrame(qa_rows)
    if "attempt_started_utc" in qa.columns:
        qa["attempt_started_utc"] = pd.to_datetime(qa["attempt_started_utc"], errors="coerce")
    if "attempt_ended_utc" in qa.columns:
        qa["attempt_ended_utc"] = pd.to_datetime(qa["attempt_ended_utc"], errors="coerce")
    return hist.to_pandas(), qa


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


def _scraping_columns() -> list[str]:
    return [
        "symbol",
        "source",
        "status",
        "rows_raw",
        "latency_ms",
        "error",
        "attempt_started_utc",
        "attempt_ended_utc",
        "series_start_date",
        "series_end_date",
        "pct_missing_adj_close",
        "run_id",
        "ingestion_ts",
    ]


def _sentiment_columns() -> list[str]:
    return [
        "symbol",
        "date",
        "n_news",
        "sent_mean",
        "sent_median",
        "sent_std",
        "sent_pos_ratio",
        "sent_neg_ratio",
        "last_headline_at",
        "extractor",
        "run_id",
        "ingestion_ts",
    ]


def _benchmark_columns() -> list[str]:
    return [
        "symbol",
        "model",
        "feature_set",
        "uses_sentiment",
        "train_n",
        "test_n",
        "mae",
        "rmse",
        "mape",
        "directional_accuracy",
        "run_id",
        "ingestion_ts",
    ]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ]+", str(text).lower())


def _score_sentiment(text: str) -> tuple[float, str, int]:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0, "neutral", 0
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    score = (pos - neg) / math.sqrt(len(tokens))
    label = "neutral"
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    return float(score), label, len(tokens)


def _google_news_items(query: str, max_items: int) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
    except Exception:
        return []

    if not raw:
        return []

    try:
        root = ET.fromstring(raw)
    except Exception:
        return []

    items: list[dict] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source = (item.findtext("source") or "").strip()

        published_at: dt.datetime | None = None
        if pub:
            try:
                parsed = parsedate_to_datetime(pub)
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
                published_at = parsed
            except Exception:
                published_at = None

        if not title or not link:
            continue
        items.append(
            {
                "title": title,
                "url": link,
                "source_name": source or "google_news_rss",
                "published_at": published_at,
            }
        )
        if len(items) >= max_items:
            break
    return items


def build_sentiment_features(config: AppConfig, run_id: str, ingestion_ts: dt.datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict] = []
    for symbol in config.symbols:
        query = _SYMBOL_QUERIES.get(symbol, f"{symbol} stock index market")
        for item in _google_news_items(query, max_items=config.sentiment_max_items):
            score, label, tok_n = _score_sentiment(item["title"])
            published_at = item["published_at"]
            raw_rows.append(
                {
                    "symbol": symbol,
                    "query": query,
                    "title": item["title"],
                    "url": item["url"],
                    "source_name": item["source_name"],
                    "published_at": published_at,
                    "date": published_at.date() if isinstance(published_at, dt.datetime) else None,
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "token_n": tok_n,
                    "extractor": "google_news_rss_lexicon_v1",
                    "run_id": run_id,
                    "ingestion_ts": ingestion_ts,
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "symbol",
                    "query",
                    "title",
                    "url",
                    "source_name",
                    "published_at",
                    "date",
                    "sentiment_score",
                    "sentiment_label",
                    "token_n",
                    "extractor",
                    "run_id",
                    "ingestion_ts",
                ]
            ),
            pd.DataFrame(columns=_sentiment_columns()),
        )

    raw_df["published_at"] = pd.to_datetime(raw_df["published_at"], errors="coerce")
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.date
    raw_df = raw_df.drop_duplicates(subset=["symbol", "url"], keep="first")

    grouped = raw_df.dropna(subset=["date"]).groupby(["symbol", "date"], dropna=False)
    feats = grouped.agg(
        n_news=("title", "count"),
        sent_mean=("sentiment_score", "mean"),
        sent_median=("sentiment_score", "median"),
        sent_std=("sentiment_score", "std"),
        sent_pos_ratio=("sentiment_label", lambda s: float((s == "positive").mean())),
        sent_neg_ratio=("sentiment_label", lambda s: float((s == "negative").mean())),
        last_headline_at=("published_at", "max"),
    ).reset_index()
    feats["extractor"] = "google_news_rss_lexicon_v1"
    feats["run_id"] = run_id
    feats["ingestion_ts"] = ingestion_ts
    feats["sent_std"] = pd.to_numeric(feats["sent_std"], errors="coerce").fillna(0.0)

    for col in _sentiment_columns():
        if col not in feats.columns:
            feats[col] = pd.Series(dtype="object")
    return raw_df, feats[_sentiment_columns()]


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-8)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mape = _safe_mape(yt, yp)
    da = float(np.mean(np.sign(yt) == np.sign(yp)))
    return mae, rmse, mape, da


def build_benchmark_table(
    config: AppConfig,
    merged_hist: pd.DataFrame,
    sentiment_features: pd.DataFrame,
    run_id: str,
    ingestion_ts: dt.datetime,
) -> pd.DataFrame:
    rows: list[dict] = []

    sent = sentiment_features.copy()
    if not sent.empty:
        sent["date"] = pd.to_datetime(sent["date"], errors="coerce").dt.date
        sent["sent_mean"] = pd.to_numeric(sent["sent_mean"], errors="coerce").fillna(0.0)
        sent["n_news"] = pd.to_numeric(sent["n_news"], errors="coerce").fillna(0.0)

    for symbol in config.symbols:
        s = merged_hist[merged_hist["symbol"] == symbol].copy()
        if s.empty:
            continue
        s = s.sort_values("date")
        s["adj_close"] = pd.to_numeric(s["adj_close"], errors="coerce")
        s = s.dropna(subset=["adj_close"])
        if s.shape[0] < 300:
            continue

        s["ret"] = np.log(s["adj_close"]).diff()
        for lag in range(1, 6):
            s[f"lag_{lag}"] = s["ret"].shift(lag)
        s["target"] = s["ret"].shift(-1)

        if not sent.empty:
            sent_s = sent[sent["symbol"] == symbol][["date", "sent_mean", "n_news"]].copy()
            s = s.merge(sent_s, on="date", how="left")
        else:
            s["sent_mean"] = 0.0
            s["n_news"] = 0.0
        s["sent_mean"] = pd.to_numeric(s["sent_mean"], errors="coerce").fillna(0.0)
        s["n_news"] = pd.to_numeric(s["n_news"], errors="coerce").fillna(0.0)

        base_cols = [f"lag_{lag}" for lag in range(1, 6)]
        s_base = s.dropna(subset=base_cols + ["target"]).copy()
        if s_base.shape[0] < 220:
            continue

        split_idx = max(int(s_base.shape[0] * 0.8), s_base.shape[0] - 252)
        split_idx = min(max(split_idx, 120), s_base.shape[0] - 40)
        train = s_base.iloc[:split_idx].copy()
        test = s_base.iloc[split_idx:].copy()
        if test.shape[0] < 30:
            continue

        y_train = train["target"].values
        y_test = test["target"].values

        baseline_pred = np.repeat(float(np.nanmean(y_train)), test.shape[0])
        mae, rmse, mape, da = _eval_metrics(y_test, baseline_pred)
        rows.append(
            {
                "symbol": symbol,
                "model": "drift_mean",
                "feature_set": "ret_lags_1_5",
                "uses_sentiment": False,
                "train_n": int(train.shape[0]),
                "test_n": int(test.shape[0]),
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "directional_accuracy": da,
                "run_id": run_id,
                "ingestion_ts": ingestion_ts,
            }
        )

        if not SKLEARN_AVAILABLE:
            rows.append(
                {
                    "symbol": symbol,
                    "model": "mlp_regressor",
                    "feature_set": "ret_lags_1_5",
                    "uses_sentiment": False,
                    "train_n": int(train.shape[0]),
                    "test_n": int(test.shape[0]),
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "directional_accuracy": np.nan,
                    "run_id": run_id,
                    "ingestion_ts": ingestion_ts,
                }
            )
            continue

        if SKLEARN_AVAILABLE:
            try:
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "mlp",
                            MLPRegressor(
                                hidden_layer_sizes=(32, 16),
                                activation="relu",
                                alpha=1e-4,
                                random_state=42,
                                max_iter=700,
                            ),
                        ),
                    ]
                )
                model.fit(train[base_cols].values, y_train)
                pred = model.predict(test[base_cols].values)
                mae, rmse, mape, da = _eval_metrics(y_test, pred)
                rows.append(
                    {
                        "symbol": symbol,
                        "model": "mlp_regressor",
                        "feature_set": "ret_lags_1_5",
                        "uses_sentiment": False,
                        "train_n": int(train.shape[0]),
                        "test_n": int(test.shape[0]),
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape,
                        "directional_accuracy": da,
                        "run_id": run_id,
                        "ingestion_ts": ingestion_ts,
                    }
                )
            except Exception as exc:
                logging.warning("Benchmark MLP (sin sentimiento) falló para %s: %s", symbol, exc)

            try:
                ext_cols = base_cols + ["sent_mean", "n_news"]
                s_ext = s.dropna(subset=ext_cols + ["target"]).copy()
                if s_ext.shape[0] >= 220:
                    split_ext = max(int(s_ext.shape[0] * 0.8), s_ext.shape[0] - 252)
                    split_ext = min(max(split_ext, 120), s_ext.shape[0] - 40)
                    tr_ext = s_ext.iloc[:split_ext].copy()
                    te_ext = s_ext.iloc[split_ext:].copy()
                    if te_ext.shape[0] >= 30:
                        model_ext = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                (
                                    "mlp",
                                    MLPRegressor(
                                        hidden_layer_sizes=(32, 16),
                                        activation="relu",
                                        alpha=1e-4,
                                        random_state=42,
                                        max_iter=700,
                                    ),
                                ),
                            ]
                        )
                        model_ext.fit(tr_ext[ext_cols].values, tr_ext["target"].values)
                        pred_ext = model_ext.predict(te_ext[ext_cols].values)
                        mae, rmse, mape, da = _eval_metrics(te_ext["target"].values, pred_ext)
                        rows.append(
                            {
                                "symbol": symbol,
                                "model": "mlp_regressor",
                                "feature_set": "ret_lags_1_5_plus_sentiment",
                                "uses_sentiment": True,
                                "train_n": int(tr_ext.shape[0]),
                                "test_n": int(te_ext.shape[0]),
                                "mae": mae,
                                "rmse": rmse,
                                "mape": mape,
                                "directional_accuracy": da,
                                "run_id": run_id,
                                "ingestion_ts": ingestion_ts,
                            }
                        )
            except Exception as exc:
                logging.warning("Benchmark MLP (con sentimiento) falló para %s: %s", symbol, exc)

    out = pd.DataFrame(rows)
    for col in _benchmark_columns():
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")
    return out[_benchmark_columns()]


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
    scraping_qa: pd.DataFrame,
    sentiment_raw: pd.DataFrame,
    sentiment_features: pd.DataFrame,
    benchmark: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_csv = output_dir / "indices_historicos.csv"
    forecast_csv = output_dir / "proyecciones_30d.csv"
    scraping_csv = output_dir / "scraping_qa.csv"
    sentiment_raw_csv = output_dir / "sentimiento_noticias_raw.csv"
    sentiment_feat_csv = output_dir / "sentimiento_features.csv"
    benchmark_csv = output_dir / "benchmark_modelos.csv"
    metadata_json = output_dir / "run_metadata.json"

    historical.to_csv(historical_csv, index=False)
    forecast.to_csv(forecast_csv, index=False)
    scraping_qa.to_csv(scraping_csv, index=False)
    sentiment_raw.to_csv(sentiment_raw_csv, index=False)
    sentiment_features.to_csv(sentiment_feat_csv, index=False)
    benchmark.to_csv(benchmark_csv, index=False)

    try:
        historical.to_parquet(output_dir / "indices_historicos.parquet", index=False)
        forecast.to_parquet(output_dir / "proyecciones_30d.parquet", index=False)
        scraping_qa.to_parquet(output_dir / "scraping_qa.parquet", index=False)
        sentiment_raw.to_parquet(output_dir / "sentimiento_noticias_raw.parquet", index=False)
        sentiment_features.to_parquet(output_dir / "sentimiento_features.parquet", index=False)
        benchmark.to_parquet(output_dir / "benchmark_modelos.parquet", index=False)
    except Exception as exc:
        logging.warning("No se pudieron exportar Parquet (continuamos con CSV): %s", exc)

    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(config: AppConfig) -> None:
    run_id = uuid.uuid4().hex[:12]
    ingestion_ts = dt.datetime.utcnow().replace(microsecond=0)

    logging.info("Iniciando run_id=%s", run_id)
    logging.info("Símbolos: %s", ", ".join(config.symbols))

    hist, scraping_qa = build_historical_dataset(config)
    hist["ingestion_ts"] = ingestion_ts
    hist["run_id"] = run_id
    hist = _coerce_hist_types(hist)

    scraping_qa = scraping_qa.copy()
    if scraping_qa.empty:
        scraping_qa = pd.DataFrame(columns=_scraping_columns())
    scraping_qa["run_id"] = run_id
    scraping_qa["ingestion_ts"] = ingestion_ts
    for col in _scraping_columns():
        if col not in scraping_qa.columns:
            scraping_qa[col] = pd.Series(dtype="object")
    scraping_qa = scraping_qa[_scraping_columns()]

    merged_hist = hist
    merged_forecast: pd.DataFrame | None = None
    sentiment_raw = pd.DataFrame(
        columns=[
            "symbol",
            "query",
            "title",
            "url",
            "source_name",
            "published_at",
            "date",
            "sentiment_score",
            "sentiment_label",
            "token_n",
            "extractor",
            "run_id",
            "ingestion_ts",
        ]
    )
    sentiment_features = pd.DataFrame(columns=_sentiment_columns())
    benchmark = pd.DataFrame(columns=_benchmark_columns())

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

    if config.with_sentiment:
        sentiment_raw, sentiment_features = build_sentiment_features(config, run_id, ingestion_ts)
        if sentiment_features.empty:
            logging.warning("No se pudo construir sentimiento (sin noticias o sin conexión).")

    if config.with_benchmark:
        benchmark = build_benchmark_table(config, merged_hist, sentiment_features, run_id, ingestion_ts)
        if benchmark.empty:
            logging.warning("No se pudo construir benchmark (muestra insuficiente o sklearn no disponible).")

    if config.write_gsheet:
        creds = _get_credentials_from_env()
        gc = gspread.authorize(creds)

        ws_qa = _open_or_create_worksheet(gc, config.sheet_id, config.ws_scraping_qa, _scraping_columns())
        _write_full(ws_qa, scraping_qa)

        if config.with_sentiment:
            ws_sent = _open_or_create_worksheet(gc, config.sheet_id, config.ws_sentiment, _sentiment_columns())
            _write_full(ws_sent, sentiment_features)

        if config.with_benchmark:
            ws_bm = _open_or_create_worksheet(gc, config.sheet_id, config.ws_benchmark, _benchmark_columns())
            _write_full(ws_bm, benchmark)

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
        "ws_scraping_qa": config.ws_scraping_qa if config.write_gsheet else None,
        "ws_sentiment": config.ws_sentiment if (config.write_gsheet and config.with_sentiment) else None,
        "ws_benchmark": config.ws_benchmark if (config.write_gsheet and config.with_benchmark) else None,
        "historical_rows": int(len(merged_hist)),
        "forecast_rows": int(len(merged_forecast)),
        "scraping_qa_rows": int(len(scraping_qa)),
        "sentiment_rows": int(len(sentiment_features)),
        "benchmark_rows": int(len(benchmark)),
        "with_sentiment": config.with_sentiment,
        "with_benchmark": config.with_benchmark,
        "sklearn_available": SKLEARN_AVAILABLE,
        "symbol_coverage": symbol_coverage,
        "notes": [
            "Modelo principal: ARIMA(p,0,q) sobre retornos logarítmicos con selección por AICc.",
            "Fallback: naive-drift log-normal cuando la serie es corta o falla ARIMA.",
            "IC95% por aproximación gaussiana sobre retorno acumulado.",
            "Capa de scraping_qa: auditoría por símbolo/fuente con latencia, estado y cobertura.",
            "Capa de sentimiento: headlines de Google News RSS + léxico financiero simple (v1).",
            "Benchmark: drift_mean vs MLPRegressor con y sin features de sentimiento.",
        ],
    }

    export_outputs(
        merged_hist,
        merged_forecast,
        scraping_qa,
        sentiment_raw,
        sentiment_features,
        benchmark,
        metadata,
        config.output_dir,
    )
    logging.info("Pipeline finalizado. Salidas en %s", config.output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera histórico + proyección de índices y exporta insumos reutilizables.")
    parser.add_argument("--sheet-id", default=os.environ.get("SHEET_ID", DEFAULT_SHEET_ID))
    parser.add_argument("--ws-data", default=os.environ.get("WS_DATA", DEFAULT_WS_DATA))
    parser.add_argument("--ws-forecast", default=os.environ.get("WS_FC", DEFAULT_WS_FORECAST))
    parser.add_argument("--ws-scraping-qa", default=os.environ.get("WS_SCRAPING_QA", DEFAULT_WS_SCRAPING_QA))
    parser.add_argument("--ws-sentiment", default=os.environ.get("WS_SENTIMENT", DEFAULT_WS_SENTIMENT))
    parser.add_argument("--ws-benchmark", default=os.environ.get("WS_BENCHMARK", DEFAULT_WS_BENCHMARK))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Lista CSV de símbolos (ej: ^GSPC,^IXIC)")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat(), help="Fecha inicial ISO (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Horizonte de proyección en días hábiles")
    parser.add_argument("--fallback-window", type=int, default=DEFAULT_FALLBACK_WINDOW)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--sentiment-max-items", type=int, default=DEFAULT_SENTIMENT_MAX_ITEMS)
    parser.add_argument("--no-sentiment", action="store_true", help="Deshabilita la capa de sentimiento.")
    parser.add_argument("--no-benchmark", action="store_true", help="Deshabilita benchmark drift vs ANN.")
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
        ws_scraping_qa=args.ws_scraping_qa,
        ws_sentiment=args.ws_sentiment,
        ws_benchmark=args.ws_benchmark,
        symbols=symbols,
        start_date=start_date,
        horizon=args.horizon,
        fallback_window=max(2, int(args.fallback_window)),
        max_p=max(0, int(args.max_p)),
        max_q=max(0, int(args.max_q)),
        sentiment_max_items=max(5, int(args.sentiment_max_items)),
        write_gsheet=write_gsheet,
        with_sentiment=not bool(args.no_sentiment),
        with_benchmark=not bool(args.no_benchmark),
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
