# Proyección de indicadores bursátiles

Pipeline reproducible para descargar series de índices bursátiles, escribirlas en Google Sheets y generar proyecciones a 30 días hábiles. Incluye tablero estático que lee en vivo desde la hoja usando **GViz JSON** (sin CSV).

## Estructura
```
proyeccion_de_indicadores_bursatiles/
├─ fetch_and_forecast_indices_to_gsheet.py
├─ index.html
└─ .github/
   └─ workflows/
      └─ update_indices.yml
```

## Google Sheet
- ID: `1UQCSPaCtBA_v8aTU1xL6W4xtDOl8PpaP-240gJtLn-0`
- Pestañas esperadas: `indices` y `proyecciones_30d`
- Comparte con el **Service Account** como **Editor**.
- Para que el tablero pueda leer sin autenticación, habilita acceso de lectura anónimo mediante **Publicar en la web** o “Cualquier usuario con el vínculo, lector”. El `index.html` usa:
  - `https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:json&sheet=indices`
  - `https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:json&sheet=proyecciones_30d`

## Autenticación (NO subir credenciales al repo)
Usar **GitHub Secrets**:
- `GOOGLE_SERVICE_ACCOUNT_JSON` con el **JSON crudo** del SA, o
- `GCP_SERVICE_ACCOUNT_JSON_B64` con el JSON en **Base64**.

El workflow detecta cualquiera de las dos y crea `creds.json` temporalmente para la ejecución.

## Uso local
```
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install polars pyarrow yfinance gspread gspread-dataframe pandas google-auth pmdarima numpy

# Autenticación local (elige UNA de las dos)
export GOOGLE_APPLICATION_CREDENTIALS="/ruta/service-account.json"
# o
export GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 -w0 service-account.json)"

python fetch_and_forecast_indices_to_gsheet.py
```

## GitHub Actions
- Workflow programado a las **06:00 America/Guayaquil** (11:00 UTC). También puedes dispararlo con **Run workflow**.
- Si el run queda en **Queued**, revisa permisos de Actions, minutos disponibles y que GitHub-hosted runners estén habilitados.

## Notas de modelado
- Modelado ARIMA automático sobre **retornos logarítmicos** con selección por AICc (`pmdarima.auto_arima`).
- Reconstrucción al nivel multiplicativo, bandas aproximadas por transformación log-normal.
- *Fallback* `naive drift` en log-precio si la historia es corta.
- Dedupe de proyecciones por clave `(symbol, date, fc_h)` para conservar los 30 horizontes.
