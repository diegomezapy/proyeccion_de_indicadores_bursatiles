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

## Artefactos locales
Cada ejecución exporta en `outputs/latest/`:
- `indices_historicos.csv`
- `indices_historicos.parquet`
- `proyecciones_30d.csv`
- `proyecciones_30d.parquet`
- `run_metadata.json`
