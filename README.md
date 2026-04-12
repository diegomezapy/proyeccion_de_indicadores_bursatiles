# Proyección de Indicadores Bursátiles

Pipeline reproducible para **descargar índices**, **auditar scraping**, **agregar señales de sentimiento**, **comparar modelos** y **publicar insumos reutilizables** para investigación, reportes y modelos downstream.

## Qué resuelve este repositorio
1. Extrae histórico diario de índices (Stooq con fallback Yahoo).
2. Genera auditoría de extracción (`scraping_qa`) con estado, latencia y cobertura.
3. Normaliza y deduplica la serie histórica.
4. Proyecta hasta 30 días hábiles con ARIMA sobre retornos log.
5. Construye features diarias de sentimiento a partir de titulares RSS de Google News.
6. Calcula benchmark de modelos (`drift_mean` vs `MLPRegressor` cuando está disponible).
7. Exporta artefactos de datos (`CSV`, `Parquet`, `JSON`) para reutilización.
8. (Opcional) sincroniza resultados a Google Sheets para consumo por dashboard.

## Estructura
```text
.
├─ fetch_and_forecast_indices_to_gsheet.py
├─ index.html
├─ requirements.txt
├─ docs/
│  ├─ METODOLOGIA.md
│  └─ DICCIONARIO_DATOS.md
└─ .github/workflows/
   └─ update_indices.yml
```

## Salidas de datos (insumos)
Cada corrida genera en `outputs/latest/`:
- `indices_historicos.csv`
- `indices_historicos.parquet`
- `proyecciones_30d.csv`
- `proyecciones_30d.parquet`
- `scraping_qa.csv`
- `scraping_qa.parquet`
- `sentimiento_noticias_raw.csv`
- `sentimiento_noticias_raw.parquet`
- `sentimiento_features.csv`
- `sentimiento_features.parquet`
- `benchmark_modelos.csv`
- `benchmark_modelos.parquet`
- `run_metadata.json`

Estas salidas están diseñadas para que otros proyectos puedan consumirlas sin depender del dashboard.

## Configuración de Google Sheets
- Hoja objetivo por defecto: `1UQCSPaCtBA_v8aTU1xL6W4xtDOl8PpaP-240gJtLn-0`
- Pestañas recomendadas: `indices`, `proyecciones_30d`, `scraping_qa`, `sentimiento_features`, `benchmark_modelos`
- Compartir la hoja con el Service Account como **Editor**.
- Para lectura pública del tablero, habilitar acceso de lectura (publicar en web o vínculo público).

## Credenciales (no subir al repositorio)
El pipeline soporta cualquiera de estas opciones:
- `GOOGLE_SERVICE_ACCOUNT_JSON` (JSON crudo en variable de entorno)
- `GCP_SERVICE_ACCOUNT_JSON_B64` (JSON en base64)
- `GOOGLE_APPLICATION_CREDENTIALS` (ruta local al JSON)

## Uso local
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Modo completo (escribe en Google Sheets + exporta insumos locales)
python fetch_and_forecast_indices_to_gsheet.py --output-dir outputs/latest

# Modo solo insumos locales (sin Google Sheets)
python fetch_and_forecast_indices_to_gsheet.py --no-gsheet --output-dir outputs/latest
```

### Parámetros útiles
```bash
python fetch_and_forecast_indices_to_gsheet.py \
  --symbols ^GSPC,^DJI \
  --horizon 20 \
  --max-p 4 --max-q 4 \
  --sentiment-max-items 60 \
  --output-dir outputs/run_2026_04_11
```

Para desactivar capas específicas:
```bash
python fetch_and_forecast_indices_to_gsheet.py --no-sentiment
python fetch_and_forecast_indices_to_gsheet.py --no-benchmark
```

## GitHub Actions
Workflow: `.github/workflows/update_indices.yml`
- Ejecución manual: **Run workflow**.
- Programación: lunes a viernes a las `11:00 UTC` (aprox. `06:00 America/Guayaquil`).
- El workflow instala dependencias, ejecuta el pipeline y sube `outputs/latest` como artefacto.

## Metodología
La explicación formal está en `docs/METODOLOGIA.md`.

Resumen:
- Modelo principal: ARIMA\((p,0,q)\) sobre retornos logarítmicos con selección por AICc.
- Fallback: naive drift en log-retornos para robustez ante baja muestra o no convergencia.
- Bandas 95% en nivel por aproximación log-normal.
- Sentimiento: scoring léxico simple sobre titulares RSS por fecha/símbolo.
- Benchmark: `drift_mean` y `MLPRegressor` sobre retornos (si `scikit-learn` está disponible).

## Dashboard
`index.html` consume GViz JSON desde Google Sheets para visualizar:
- histórico,
- proyección + bandas,
- retornos y distribución,
- KPIs y métricas de riesgo.
- panel de calidad de extracción.
- panel de sentimiento diario.
- panel de benchmark comparativo.

## Referencias
- Box et al. (2015), *Time Series Analysis: Forecasting and Control*.
- Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*.
- Burnham & Anderson (2002), *Model Selection and Multimodel Inference*.
- Documentación de `statsmodels` y `yfinance`.
