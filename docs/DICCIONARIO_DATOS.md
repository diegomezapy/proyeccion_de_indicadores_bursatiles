# Diccionario de datos

## Hoja / archivo: `indices`

| Campo | Tipo | Descripción |
|---|---|---|
| `symbol` | string | Ticker del índice (ej. `^GSPC`). |
| `date` | date | Fecha de observación (día hábil). |
| `open` | float | Precio de apertura. |
| `high` | float | Precio máximo diario. |
| `low` | float | Precio mínimo diario. |
| `close` | float | Precio de cierre. |
| `adj_close` | float | Cierre ajustado (preferido para modelado). |
| `volume` | int | Volumen reportado por la fuente (si existe). |
| `source` | string | Fuente efectiva (`stooq` o `yahoo`). |
| `ingestion_ts` | datetime UTC | Timestamp de ingestión de la corrida. |
| `run_id` | string | Identificador único de corrida. |

## Hoja / archivo: `proyecciones_30d`

| Campo | Tipo | Descripción |
|---|---|---|
| `symbol` | string | Ticker del índice proyectado. |
| `date` | date | Fecha objetivo proyectada. |
| `fc_h` | int | Horizonte (1..H) en días hábiles. |
| `method` | string | Método efectivo (`arima_logret_aicc` o `naive_drift_log`). |
| `yhat` | float | Proyección puntual en nivel. |
| `yhat_lo` | float | Límite inferior IC95%. |
| `yhat_hi` | float | Límite superior IC95%. |
| `last_obs_date` | date | Última fecha observada usada como base. |
| `last_obs_value` | float | Último nivel observado usado como base. |
| `model_desc` | string | Descripción corta del modelo aplicado. |
| `model_family` | string | Familia de modelo (`ARIMA` o `naive-drift`). |
| `model_spec` | string | Especificación (`ARIMA(p,0,q)` o parámetros drift). |
| `aicc` | float | AICc del modelo seleccionado (si aplica). |
| `insample_n` | int | Número de retornos usados en ajuste. |
| `ingestion_ts` | datetime UTC | Timestamp de ejecución. |
| `run_id` | string | Identificador de corrida para trazabilidad. |

## Hoja / archivo: `scraping_qa`

| Campo | Tipo | Descripción |
|---|---|---|
| `symbol` | string | Símbolo evaluado. |
| `source` | string | Fuente intentada (`stooq` o `yahoo`). |
| `status` | string | Estado del intento (`ok`, `error`, `accepted`). |
| `rows_raw` | int | Filas obtenidas en el intento/fuente. |
| `latency_ms` | int | Tiempo del intento en milisegundos. |
| `error` | string | Mensaje de error (si aplica). |
| `attempt_started_utc` | datetime UTC | Inicio del intento. |
| `attempt_ended_utc` | datetime UTC | Fin del intento. |
| `series_start_date` | date | Inicio de cobertura final aceptada. |
| `series_end_date` | date | Fin de cobertura final aceptada. |
| `pct_missing_adj_close` | float | Proporción de faltantes en `adj_close`. |
| `run_id` | string | Identificador de corrida. |
| `ingestion_ts` | datetime UTC | Timestamp de corrida. |

## Hoja / archivo: `sentimiento_features`

| Campo | Tipo | Descripción |
|---|---|---|
| `symbol` | string | Símbolo del índice. |
| `date` | date | Fecha de agregación diaria del sentimiento. |
| `n_news` | int | Cantidad de titulares procesados ese día. |
| `sent_mean` | float | Media del score de sentimiento diario. |
| `sent_median` | float | Mediana del score diario. |
| `sent_std` | float | Desvío del score diario. |
| `sent_pos_ratio` | float | Proporción de titulares positivos. |
| `sent_neg_ratio` | float | Proporción de titulares negativos. |
| `last_headline_at` | datetime UTC | Hora del último titular del día. |
| `extractor` | string | Versión del extractor/scoring de sentimiento. |
| `run_id` | string | Identificador de corrida. |
| `ingestion_ts` | datetime UTC | Timestamp de corrida. |

## Hoja / archivo: `benchmark_modelos`

| Campo | Tipo | Descripción |
|---|---|---|
| `symbol` | string | Símbolo evaluado. |
| `model` | string | Modelo (`drift_mean`, `mlp_regressor`). |
| `feature_set` | string | Set de features usado. |
| `uses_sentiment` | bool | Indica si incluyó variables de sentimiento. |
| `train_n` | int | Tamaño de muestra de entrenamiento. |
| `test_n` | int | Tamaño de muestra de prueba. |
| `mae` | float | Error absoluto medio. |
| `rmse` | float | Raíz del error cuadrático medio. |
| `mape` | float | Error porcentual absoluto medio. |
| `directional_accuracy` | float | Precisión direccional del signo de retorno. |
| `run_id` | string | Identificador de corrida. |
| `ingestion_ts` | datetime UTC | Timestamp de corrida. |

## Artefactos locales
Cada ejecución exporta en `outputs/latest/`:
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
