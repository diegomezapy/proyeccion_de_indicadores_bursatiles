# fetch_and_forecast_indices_to_gsheet.py
from __future__ import annotations
import os, io, base64, datetime as dt, tempfile, urllib.request
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from pmdarima import auto_arima

SHEET_ID = "1UQCSPaCtBA_v8aTU1xL6W4xtDOl8PpaP-240gJtLn-0"
WS_DATA = "indices"
WS_FC   = "proyecciones_30d"
SYMBOLS = ["^GSPC","^IXIC","^DJI","^FTSE","^IBEX","^BVSP","^MERV","^VIX"]
START_DATE = dt.date(1980, 1, 1)
HORIZON = 30

def _stooq_fallback(symbol: str) -> pl.DataFrame | None:
    mapping = {"^GSPC":"^spx","^IXIC":"^ixic","^DJI":"^dji","^FTSE":"^ftse","^IBEX":"^ibex","^BVSP":"^bvsp","^MERV":"^merv","^VIX":"^vix"}
    s = mapping.get(symbol)
    if not s: return None
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    if not raw: return None
    df = pl.read_csv(io.BytesIO(raw))
    df = df.rename({"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df = df.with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col(["open","high","low","close"]).cast(pl.Float64, strict=False),
        pl.col("volume").cast(pl.Int64, strict=False),
        pl.lit(symbol).alias("symbol"),
        pl.lit("stooq").alias("source")
    ]).select(["symbol","date","open","high","low","close","volume","source"])
    return df

def _yahoo_fetch(symbol: str, start: dt.date) -> pl.DataFrame | None:
    data = yf.download(symbol, start=start.isoformat(), progress=False, auto_adjust=False, threads=True)
    if data is None or data.empty: return None
    pdf = data.reset_index()
    df = pl.from_pandas(pdf).rename({
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close",
        "Adj Close":"adj_close","Volume":"volume"
    }).with_columns([pl.lit(symbol).alias("symbol"), pl.lit("yahoo").alias("source")]).select(
        ["symbol","date","open","high","low","close","adj_close","volume","source"]
    )
    return df

def fetch_symbol(symbol: str) -> pl.DataFrame | None:
    df = _yahoo_fetch(symbol, START_DATE)
    if df is not None and df.height > 0: return df
    return _stooq_fallback(symbol)

def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    if "adj_close" not in df.columns: df = df.with_columns(pl.col("close").alias("adj_close"))
    wanted = ["symbol","date","open","high","low","close","adj_close","volume","source"]
    return df.select([c for c in wanted if c in df.columns])

def dedupe_sort(df: pl.DataFrame) -> pl.DataFrame:
    return (df.unique(subset=["symbol","date"], keep="last").sort(["symbol","date"]).with_columns(pl.col("date").cast(pl.Date)))

def _ensure_credentials_file() -> str:
    p = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if p and os.path.isfile(p): return p
    b64 = os.environ.get("GCP_SERVICE_ACCOUNT_JSON_B64")
    if b64:
        data = base64.b64decode(b64)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json"); tmp.write(data); tmp.flush(); tmp.close()
        return tmp.name
    raise RuntimeError("Credenciales no encontradas. Configure GOOGLE_APPLICATION_CREDENTIALS o GCP_SERVICE_ACCOUNT_JSON_B64.")

def _open_or_create_worksheet(gc: gspread.Client, sheet_id: str, worksheet_name: str, header_cols: list[str]) -> gspread.Worksheet:
    sh = gc.open_by_key(sheet_id)
    try: ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=len(header_cols))
        set_with_dataframe(ws, pd.DataFrame(columns=header_cols), include_index=False, include_column_header=True, resize=True)
    return ws

def _read_existing(ws: gspread.Worksheet, cols: list[str]) -> pd.DataFrame:
    values = ws.get_all_records()
    if not values: return pd.DataFrame(columns=cols)
    df = pd.DataFrame(values)
    for c in cols:
        if c not in df.columns: df[c] = pd.Series(dtype="float64")
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"]).dt.date
    if "volume" in df.columns: df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    for c in ["open","high","low","close","adj_close","yhat","yhat_lo","yhat_hi","last_obs_value"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _write_full(ws: gspread.Worksheet, pdf: pd.DataFrame) -> None:
    set_with_dataframe(ws, pdf, include_index=False, include_column_header=True, resize=True)

def _next_business_days(start_date: dt.date, n: int) -> list[dt.date]:
    out = []; d = start_date
    while len(out) < n:
        d = d + dt.timedelta(days=1)
        if d.weekday() < 5: out.append(d)
    return out

def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def forecast_one_symbol(pdf: pd.DataFrame, symbol: str, horizon: int) -> pd.DataFrame | None:
    d = pdf[(pdf["symbol"] == symbol) & pd.notna(pdf["adj_close"])].copy()
    if d.empty: return None
    d = d.sort_values("date")
    y = d["adj_close"].astype(float)
    if y.notna().sum() < 50:
        last_date = d["date"].iloc[-1]; last_val = y.iloc[-1]
        fc_dates = _next_business_days(last_date, horizon)
        ky = min(60, max(2, y.notna().sum() - 1))
        r = _safe_log(y).diff(); mu = np.nanmean(r.tail(ky)); sigma = np.nanstd(r.tail(ky)); z = 1.96
        yhat = last_val * np.exp(np.cumsum(np.repeat(mu, horizon)))
        lo = last_val * np.exp(np.cumsum(np.repeat(mu - z*sigma, horizon)))
        hi = last_val * np.exp(np.cumsum(np.repeat(mu + z*sigma, horizon)))
        return pd.DataFrame({"symbol":symbol,"date":fc_dates,"fc_h":list(range(1,horizon+1)),"method":["naive_drift_log"]*horizon,
                             "yhat":yhat,"yhat_lo":lo,"yhat_hi":hi,"last_obs_date":last_date,"last_obs_value":last_val,
                             "model_desc":[f"Naive-drift en log con ventana={ky}"]*horizon})
    logp = _safe_log(y); returns = logp.diff().dropna()
    model = auto_arima(returns.values, start_p=0, start_q=0, max_p=5, max_q=5, seasonal=False, information_criterion="aicc",
                       stepwise=True, suppress_warnings=True, error_action="ignore", maxiter=200, with_intercept=True)
    fc_ret, conf = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
    last_date = d["date"].iloc[-1]; last_val = d["adj_close"].iloc[-1]
    fc_dates = _next_business_days(last_date, horizon)
    cum_ret = np.cumsum(fc_ret); yhat = last_val * np.exp(cum_ret)
    lo = last_val * np.exp(np.cumsum(conf[:, 0])); hi = last_val * np.exp(np.cumsum(conf[:, 1]))
    desc = f"ARIMA auto en retornos log (p,d,q)={model.order}, ic=aicc"
    return pd.DataFrame({"symbol":symbol,"date":fc_dates,"fc_h":list(range(1,horizon+1)),"method":["auto_arima_logret"]*horizon,
                         "yhat":yhat,"yhat_lo":lo,"yhat_hi":hi,"last_obs_date":last_date,"last_obs_value":last_val,
                         "model_desc":[desc]*horizon})

def main() -> None:
    frames = []
    for s in SYMBOLS:
        df = fetch_symbol(s)
        if df is not None and df.height > 0: frames.append(normalize_schema(df))
    if not frames: return
    hist = pl.concat(frames, how="vertical_relaxed")
    hist = dedupe_sort(hist).with_columns(pl.lit(dt.datetime.utcnow()).alias("ingestion_ts"))
    pdf_hist = hist.to_pandas()

    creds_path = _ensure_credentials_file()
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc = gspread.authorize(credentials)

    ws_hist = _open_or_create_worksheet(gc, SHEET_ID, WS_DATA,
        ["symbol","date","open","high","low","close","adj_close","volume","source","ingestion_ts"])
    exist_hist = _read_existing(ws_hist, ["symbol","date","open","high","low","close","adj_close","volume","source","ingestion_ts"])
    merged_hist = (pd.concat([exist_hist, pdf_hist], axis=0, ignore_index=True)
                     .sort_values(["symbol","date","ingestion_ts"])
                     .drop_duplicates(subset=["symbol","date"], keep="last"))
    merged_hist = merged_hist[["symbol","date","open","high","low","close","adj_close","volume","source","ingestion_ts"]]
    _write_full(ws_hist, merged_hist)

    fc_all = []
    for s in SYMBOLS:
        fc = forecast_one_symbol(merged_hist, s, HORIZON)
        if fc is not None and len(fc) > 0: fc_all.append(fc)
    if not fc_all: return
    pdf_fc = pd.concat(fc_all, axis=0, ignore_index=True)
    pdf_fc["ingestion_ts"] = dt.datetime.utcnow()

    ws_fc = _open_or_create_worksheet(gc, SHEET_ID, WS_FC,
        ["symbol","date","fc_h","method","yhat","yhat_lo","yhat_hi","last_obs_date","last_obs_value","model_desc","ingestion_ts"])
    exist_fc = _read_existing(ws_fc, ["symbol","date","fc_h","method","yhat","yhat_lo","yhat_hi","last_obs_date","last_obs_value","model_desc","ingestion_ts"])
    merged_fc = (pd.concat([exist_fc, pdf_fc], axis=0, ignore_index=True)
                   .sort_values(["symbol","date","fc_h","ingestion_ts"])
                   .drop_duplicates(subset=["symbol","date","fc_h"], keep="last"))
    merged_fc = merged_fc[["symbol","date","fc_h","method","yhat","yhat_lo","yhat_hi","last_obs_date","last_obs_value","model_desc","ingestion_ts"]]
    _write_full(ws_fc, merged_fc)

if __name__ == "__main__":
    main()
