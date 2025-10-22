# fetch_and_forecast_indices_to_gsheet.py
from __future__ import annotations
import os, io, base64, datetime as dt, urllib.request, json
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
import gspread
from gspread_dataframe import set_with_dataframe
from statsmodels.tsa.arima.model import ARIMA
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError as _RefreshError

# =========================
# Configuración principal
# =========================
SHEET_ID   = "1UQCSPaCtBA_v8aTU1xL6W4xtDOl8PpaP-240gJtLn-0"
WS_DATA    = "indices"
WS_FC      = "proyecciones_30d"
SYMBOLS    = ["^GSPC","^IXIC","^DJI","^FTSE","^IBEX","^BVSP","^MERV","^VIX"]
START_DATE = dt.date(1980, 1, 1)
HORIZON    = 30  # días hábiles

# =========================
# Utilidades generales
# =========================
def _canon(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

def _find_col_by_prefix(pdf_cols: list[str], aliases: list[str]) -> str | None:
    cols_canon = {c: _canon(c) for c in pdf_cols}
    alias_canon = [_canon(a) for a in aliases]
    for c, cc in cols_canon.items():
        for ac in alias_canon:
            if cc.startswith(ac):
                return c
    return None

def _next_business_days(start_date: dt.date, n: int) -> list[dt.date]:
    out = []
    d = start_date
    while len(out) < n:
        d = d + dt.timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out

def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

# =========================
# Descarga de datos
# =========================
# Fuente primaria: Stooq (más estable en Actions para índices con '^')
_STOOQ_MAP = {
    "^GSPC": "^spx", "^IXIC": "^ixic", "^DJI": "^dji", "^FTSE": "^ftse",
    "^IBEX": "^ibex", "^BVSP": "^bvsp", "^MERV": "^merv", "^VIX": "^vix"
}

def _stooq_fetch(symbol: str) -> pl.DataFrame | None:
    s = _STOOQ_MAP.get(symbol)
    if not s:
        return None
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
    except Exception:
        return None
    if not raw:
        return None

    nulls = ["", "NA", "NaN", "null", "NULL", "N/A", "2290404134.7576"]
    try:
        df_txt = pl.read_csv(
            io.BytesIO(raw),
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=nulls,
            schema_overrides={
                "Date": pl.Utf8, "Open": pl.Utf8, "High": pl.Utf8,
                "Low": pl.Utf8, "Close": pl.Utf8, "Volume": pl.Utf8
            },
            try_parse_dates=False,
            encoding="utf8-lossy"
        )
    except Exception:
        return None

    if "Date" not in df_txt.columns or "Close" not in df_txt.columns:
        return None

    df = df_txt.with_columns([
        pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
        pl.col("Open").str.replace(",", "").cast(pl.Float64, strict=False).alias("open"),
        pl.col("High").str.replace(",", "").cast(pl.Float64, strict=False).alias("high"),
        pl.col("Low"). str.replace(",", "").cast(pl.Float64, strict=False).alias("low"),
        pl.col("Close").str.replace(",", "").cast(pl.Float64, strict=False).alias("close"),
        pl.col("Volume").str.replace(",", "").cast(pl.Int64,  strict=False).alias("volume"),
    ]).select(["date","open","high","low","close","volume"])

    df = df.with_columns([
        pl.lit(symbol).alias("symbol"),
        pl.lit("stooq").alias("source")
    ]).select(["symbol","date","open","high","low","close","volume","source"])

    # descartar filas sin fecha o sin precio
    price_any = pl.any_horizontal(
        [pl.col(c).is_not_null() for c in ["close","open","high","low"] if c in df.columns]
    )
    df = df.filter(pl.col("date").is_not_null() & price_any)

    # asegurar adj_close
    if "adj_close" not in df.columns:
        df = df.with_columns(pl.col("close").alias("adj_close"))
    df = df.select(["symbol","date","open","high","low","close","adj_close","volume","source"])

    return df if df.height > 0 else None

def _yahoo_fetch(symbol: str) -> pl.DataFrame | None:
    try:
        data = yf.download(
            symbol,
            start=START_DATE.isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="column"
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

    rename_map = {}
    if date_col in pdf.columns:
        rename_map[date_col] = "date"
    mapping = {
        "open":      ["Open"],
        "high":      ["High"],
        "low":       ["Low"],
        "close":     ["Close"],
        "adj_close": ["Adj Close", "AdjClose", "Adjusted Close"],
        "volume":    ["Volume", "Vol"],
    }
    for std_name, aliases in mapping.items():
        csrc = _find_col_by_prefix(cols, aliases)
        if csrc is not None:
            rename_map[csrc] = std_name

    df = pl.from_pandas(pdf).rename(rename_map)

    exprs = []
    if "date" in df.columns:
        exprs.append(pl.col("date").cast(pl.Date, strict=False).alias("date"))
    for c in ("open","high","low","close","adj_close"):
        if c in df.columns:
            exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
    if "volume" in df.columns:
        exprs.append(pl.col("volume").cast(pl.Int64, strict=False).alias("volume"))
    if exprs:
        df = df.with_columns(exprs)

    if "adj_close" not in df.columns and "close" in df.columns:
        df = df.with_columns(pl.col("close").alias("adj_close"))
    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.with_columns(pl.col("adj_close").alias("close"))

    df = df.with_columns([pl.lit(symbol).alias("symbol"), pl.lit("yahoo").alias("source")])

    wanted = ["symbol","date","open","high","low","close","adj_close","volume","source"]
    df = df.select([c for c in wanted if c in df.columns])

    # filtrar filas sin fecha o sin precio
    price_any = pl.any_horizontal(
        [pl.col(c).is_not_null() for c in ["close","adj_close","open","high","low"] if c in df.columns]
    )
    df = df.filter(pl.col("date").is_not_null() & price_any)

    return df if df.height > 0 else None

def fetch_symbol(symbol: str) -> pl.DataFrame | None:
    # Stooq primero (más confiable en Actions para índices con “^”)
    df = _stooq_fetch(symbol)
    if df is not None and df.height > 0:
        return df
    # Respaldo Yahoo
    return _yahoo_fetch(symbol)

def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    if "adj_close" not in df.columns and "close" in df.columns:
        df = df.with_columns(pl.col("close").alias("adj_close"))
    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.with_columns(pl.col("adj_close").alias("close"))
    wanted = ["symbol","date","open","high","low","close","adj_close","volume","source"]
    return df.select([c for c in wanted if c in df.columns])

def dedupe_sort(df: pl.DataFrame) -> pl.DataFrame:
    return (df.unique(subset=["symbol","date"], keep="last")
              .sort(["symbol","date"])
              .with_columns(pl.col("date").cast(pl.Date, strict=False)))

# =========================
# Autenticación Sheets (Secrets)
# =========================
def _get_credentials_from_secrets() -> Credentials:
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        b64 = os.environ.get("GCP_SERVICE_ACCOUNT_JSON_B64")
        if b64:
            raw = base64.b64decode(b64).decode("utf-8")
    if not raw:
        raise RuntimeError("Falta definir GOOGLE_SERVICE_ACCOUNT_JSON (JSON crudo) o GCP_SERVICE_ACCOUNT_JSON_B64 (Base64).")

    try:
        info = json.loads(raw)
    except Exception as e:
        raise RuntimeError("El contenido de las credenciales no es un JSON válido.") from e

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    try:
        creds.refresh(Request())
    except _RefreshError as e:
        raise RuntimeError(
            "No se pudo refrescar el token del Service Account. Revise desfase de reloj o validez del JWT."
        ) from e
    return creds

def _open_or_create_worksheet(gc: gspread.Client, sheet_id: str, worksheet_name: str, header_cols: list[str]) -> gspread.Worksheet:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=len(header_cols))
        set_with_dataframe(ws, pd.DataFrame(columns=header_cols), include_index=False, include_column_header=True, resize=True)
    return ws

def _read_existing(ws: gspread.Worksheet, cols: list[str]) -> pd.DataFrame:
    values = ws.get_all_records()
    if not values:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(values)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="float64")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    for c in ["open","high","low","close","adj_close","yhat","yhat_lo","yhat_hi","last_obs_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ingestion_ts" in df.columns:
        df["ingestion_ts"] = pd.to_datetime(df["ingestion_ts"], errors="coerce")
    if "last_obs_date" in df.columns:
        df["last_obs_date"] = pd.to_datetime(df["last_obs_date"], errors="coerce").dt.date
    if "fc_h" in df.columns:
        df["fc_h"] = pd.to_numeric(df["fc_h"], errors="coerce").astype("Int64")
    return df

def _write_full(ws: gspread.Worksheet, pdf: pd.DataFrame) -> None:
    price_cols = [c for c in ["open","high","low","close","adj_close"] if c in pdf.columns]
    if price_cols:
        mask_any = ~pdf[price_cols].isna().all(axis=1)
        pdf = pdf.loc[mask_any].copy()
    set_with_dataframe(ws, pdf, include_index=False, include_column_header=True, resize=True)

def _coerce_hist_types(pdf: pd.DataFrame) -> pd.DataFrame:
    if "symbol" in pdf.columns:
        pdf["symbol"] = pdf["symbol"].astype("string")
    if "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce").dt.date
    for c in ["open","high","low","close","adj_close"]:
        if c in pdf.columns:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    if "volume" in pdf.columns:
        pdf["volume"] = pd.to_numeric(pdf["volume"], errors="coerce").astype("Int64")
    if "ingestion_ts" in pdf.columns:
        pdf["ingestion_ts"] = pd.to_datetime(pdf["ingestion_ts"], errors="coerce")
    return pdf

def _coerce_fc_types(pdf: pd.DataFrame) -> pd.DataFrame:
    if "symbol" in pdf.columns:
        pdf["symbol"] = pdf["symbol"].astype("string")
    if "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce").dt.date
    for c in ["yhat","yhat_lo","yhat_hi","last_obs_value"]:
        if c in pdf.columns:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    if "last_obs_date" in pdf.columns:
        pdf["last_obs_date"] = pd.to_datetime(pdf["last_obs_date"], errors="coerce").dt.date
    if "fc_h" in df.columns if (df:=pdf) is not None else []:  # guardia sintáctica
        pdf["fc_h"] = pd.to_numeric(pdf["fc_h"], errors="coerce").astype("Int64")
    if "ingestion_ts" in pdf.columns:
        pdf["ingestion_ts"] = pd.to_datetime(pdf["ingestion_ts"], errors="coerce")
    return pdf

# =========================
# Modelado y pronóstico
# =========================
def _aicc(llf: float, n_obs: int, k_params: int) -> float:
    if n_obs - k_params - 1 <= 0:
        return np.inf
    return -2.0 * llf + 2.0 * k_params + (2.0 * k_params * (k_params + 1)) / (n_obs - k_params - 1)

def _fit_arima_best_aicc(returns: np.ndarray, max_p: int = 3, max_q: int = 3):
    best = {"aicc": np.inf, "order": None, "res": None}
    y = np.asarray(returns, dtype=float)
    n = y.shape[0]
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = ARIMA(y, order=(p, 0, q), trend="c",
                              enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(method="statespace", disp=0)
                k = res.params.shape[0]
                val = _aicc(res.llf, n, k)
                if np.isfinite(val) and val < best["aicc"]:
                    best = {"aicc": val, "order": (p, 0, q), "res": res}
            except Exception:
                pass
    return best["res"], best["order"]

def forecast_one_symbol(pdf: pd.DataFrame, symbol: str, horizon: int) -> pd.DataFrame | None:
    d = pdf[(pdf["symbol"] == symbol) & pd.notna(pdf["adj_close"])].copy()
    if d.empty:
        return None
    d = d.sort_values("date")
    y = d["adj_close"].astype(float)

    if y.notna().sum() < 50:
        last_date = d["date"].iloc[-1]; last_val = y.iloc[-1]
        fc_dates = _next_business_days(last_date, horizon)
        ky = min(60, max(2, y.notna().sum() - 1))
        r = _safe_log(y).diff()
        mu = np.nanmean(r.tail(ky))
        sigma = np.nanstd(r.tail(ky))
        z = 1.96
        yhat = last_val * np.exp(np.cumsum(np.repeat(mu, horizon)))
        lo = last_val * np.exp(np.cumsum(np.repeat(mu - z * sigma, horizon)))
        hi = last_val * np.exp(np.cumsum(np.repeat(mu + z * sigma, horizon)))
        return pd.DataFrame({
            "symbol": symbol, "date": fc_dates, "fc_h": list(range(1, horizon + 1)),
            "method": ["naive_drift_log"] * horizon, "yhat": yhat, "yhat_lo": lo, "yhat_hi": hi,
            "last_obs_date": last_date, "last_obs_value": last_val,
            "model_desc": [f"Naive-drift en log con ventana={ky}"] * horizon
        })

    logp = _safe_log(y)
    returns = logp.diff().dropna().values
    res, order = _fit_arima_best_aicc(returns, max_p=3, max_q=3)

    last_date = d["date"].iloc[-1]
    last_val  = d["adj_close"].iloc[-1]
    fc_dates  = _next_business_days(last_date, horizon)

    if res is None:
        ky = min(60, max(2, len(returns)))
        mu = float(np.nanmean(returns[-ky:])) if ky > 0 else 0.0
        sigma = float(np.nanstd(returns[-ky:])) if ky > 0 else 0.0
        z = 1.96
        yhat = last_val * np.exp(np.cumsum(np.repeat(mu, horizon)))
        lo = last_val * np.exp(np.cumsum(np.repeat(mu - z * sigma, horizon)))
        hi = last_val * np.exp(np.cumsum(np.repeat(mu + z * sigma, horizon)))
        return pd.DataFrame({
            "symbol": symbol, "date": fc_dates, "fc_h": list(range(1, horizon + 1)),
            "method": ["naive_drift_log"] * horizon, "yhat": yhat, "yhat_lo": lo, "yhat_hi": hi,
            "last_obs_date": last_date, "last_obs_value": last_val,
            "model_desc": ["Fallback naive-drift (sin ARIMA)"] * horizon
        })

    fc = res.get_forecast(steps=horizon)
    mean_ret = fc.predicted_mean
    conf     = fc.conf_int(alpha=0.05)

    cum_ret = np.cumsum(np.asarray(mean_ret))
    yhat = last_val * np.exp(cum_ret)
    lo = last_val * np.exp(np.cumsum(conf.iloc[:, 0].values))
    hi = last_val * np.exp(np.cumsum(conf.iloc[:, 1].values))
    desc = f"ARIMA en retornos log (p,d,q)={order}, selección por AICc"
    return pd.DataFrame({
        "symbol": symbol, "date": fc_dates, "fc_h": list(range(1, horizon + 1)),
        "method": ["arima_logret_aicc"] * horizon, "yhat": yhat, "yhat_lo": lo, "yhat_hi": hi,
        "last_obs_date": last_date, "last_obs_value": last_val,
        "model_desc": [desc] * horizon
    })

# =========================
# Flujo principal
# =========================
def main() -> None:
    # 1) Descarga y normalización
    frames = []
    for s in SYMBOLS:
        df = fetch_symbol(s)
        if df is not None and df.height > 0:
            frames.append(normalize_schema(df))
    if not frames:
        return

    hist = pl.concat(frames, how="vertical_relaxed")
    hist = dedupe_sort(hist).with_columns(pl.lit(dt.datetime.utcnow()).alias("ingestion_ts"))
    pdf_hist = hist.to_pandas()

    # 2) Autenticación GSheet desde secrets (sin archivos intermedios)
    credentials = _get_credentials_from_secrets()
    gc = gspread.authorize(credentials)

    # 3) Upsert de histórico en 'indices'
    ws_hist = _open_or_create_worksheet(
        gc, SHEET_ID, WS_DATA,
        ["symbol","date","open","high","low","close","adj_close","volume","source","ingestion_ts"]
    )
    exist_hist = _read_existing(
        ws_hist, ["symbol","date","open","high","low","close","adj_close","volume","source","ingestion_ts"]
    )
    pdf_hist   = _coerce_hist_types(pdf_hist)
    exist_hist = _coerce_hist_types(exist_hist)

    merged_hist = (
        pd.concat([exist_hist, pdf_hist], axis=0, ignore_index=True)
          .sort_values(["symbol","date","ingestion_ts"], kind="mergesort")
          .drop_duplicates(subset=["symbol","date"], keep="last")
    )
    merged_hist = merged_hist[["symbol","date","open","high","low","close","adj_close","volume","source","ingestion_ts"]]
    _write_full(ws_hist, merged_hist)

    # 4) Pronósticos y upsert en 'proyecciones_30d'
    fc_all = []
    for s in SYMBOLS:
        fc = forecast_one_symbol(merged_hist, s, HORIZON)
        if fc is not None and len(fc) > 0:
            fc_all.append(fc)
    if not fc_all:
        return
    pdf_fc = pd.concat(fc_all, axis=0, ignore_index=True)
    pdf_fc["ingestion_ts"] = dt.datetime.utcnow()
    pdf_fc = _coerce_fc_types(pdf_fc)

    ws_fc = _open_or_create_worksheet(
        gc, SHEET_ID, WS_FC,
        ["symbol","date","fc_h","method","yhat","yhat_lo","yhat_hi",
         "last_obs_date","last_obs_value","model_desc","ingestion_ts"]
    )
    exist_fc = _read_existing(
        ws_fc, ["symbol","date","fc_h","method","yhat","yhat_lo","yhat_hi",
                "last_obs_date","last_obs_value","model_desc","ingestion_ts"]
    )
    exist_fc = _coerce_fc_types(exist_fc)

    merged_fc = (
        pd.concat([exist_fc, pdf_fc], axis=0, ignore_index=True)
          .sort_values(["symbol","date","fc_h","ingestion_ts"], kind="mergesort")
          .drop_duplicates(subset=["symbol","date","fc_h"], keep="last")
    )
    merged_fc = merged_fc[[
        "symbol","date","fc_h","method","yhat","yhat_lo","yhat_hi",
        "last_obs_date","last_obs_value","model_desc","ingestion_ts"
    ]]
    _write_full(ws_fc, merged_fc)

if __name__ == "__main__":
    main()
