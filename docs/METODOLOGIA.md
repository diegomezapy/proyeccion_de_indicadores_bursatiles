# Metodología de generación de insumos

## 1) Propósito
Este repositorio produce **insumos reutilizables** para análisis financiero cuantitativo:
- Serie histórica normalizada por símbolo.
- Proyecciones a corto plazo (30 días hábiles por defecto).
- Metadatos de corrida para trazabilidad y auditoría.

El enfoque prioriza **reproducibilidad, claridad metodológica y utilidad operativa** para proyectos que necesiten una capa de datos base.
Desde abril de 2026 también incorpora una capa explícita para evaluación de extracción/scraping, sentimiento y comparación de modelos.

## 2) Universo y fuentes
Símbolos por defecto:
- `^GSPC`, `^IXIC`, `^DJI`, `^FTSE`, `^IBEX`, `^BVSP`, `^MERV`, `^VIX`

Fuentes:
- Primaria: **Stooq** (descarga CSV diaria).
- Fallback: **Yahoo Finance** vía `yfinance`.

Regla de adquisición:
1. Intentar Stooq.
2. Si falla o no hay datos, usar Yahoo.

## 3) Estandarización de datos
La salida histórica se normaliza al esquema:
- `symbol, date, open, high, low, close, adj_close, volume, source`

Criterios de limpieza:
- Conversión de tipos a fecha/número.
- Eliminación de filas sin fecha o sin ningún precio válido.
- Dedupe por clave `(symbol, date)`, preservando la observación más reciente.

## 3.1) Auditoría de extracción (`scraping_qa`)
Cada corrida registra evidencia de calidad de extracción por símbolo/fuente:
- Estado por intento (`ok/error`) y mensaje de error.
- Latencia por intento (`latency_ms`).
- Filas recuperadas (`rows_raw`).
- Cobertura final aceptada (`series_start_date`, `series_end_date`).
- Razón de faltantes en `adj_close` (`pct_missing_adj_close`).

Esto permite validar robustez y eficiencia del proceso de extracción automatizada.

## 4) Modelo de proyección
Se modela sobre retornos logarítmicos:
\[
r_t = \log(P_t) - \log(P_{t-1})
\]
con \(P_t\) como `adj_close`.

### 4.1 Modelo principal
- ARIMA\((p,0,q)\) sobre \(r_t\).
- Búsqueda en grilla para \(p,q \in [0,3]\).
- Selección por **AICc** (criterio de información corregido).
- Winsorización al 1% para amortiguar outliers extremos en retornos.

### 4.2 Fallback robusto
Si la serie es corta o ARIMA no converge:
- Se usa **naive drift** sobre retornos log.
- Media y desvío estimados en ventana reciente (`fallback_window`, por defecto 252).
- Para `^VIX`, la deriva se fuerza a 0 (supuesto conservador para nivel de volatilidad).

### 4.3 Reconstrucción al nivel e intervalos
Con retorno acumulado esperado \(S_h\):
\[
\hat P_{t+h} = P_t \exp(\mathbb{E}[S_h])
\]
Bandas 95% (aprox. gaussiana):
\[
\hat P_{t+h}^{\pm} = P_t \exp\left(\mathbb{E}[S_h] \pm 1.96\cdot \sigma_{S_h}\right)
\]

## 5) Capa de sentimiento
Se incorpora una fuente de texto para enriquecer insumos:
- Fuente: RSS de Google News (consulta por símbolo/índice).
- Unidad de captura: titular + URL + fecha de publicación.
- Scoring: léxico financiero simple (positivo/negativo) sobre tokens del titular.
- Agregación diaria por símbolo:
  - `n_news`
  - `sent_mean`, `sent_median`, `sent_std`
  - `sent_pos_ratio`, `sent_neg_ratio`

Este módulo está diseñado como señal auxiliar para experimentos con redes neuronales y no reemplaza un modelo NLP entrenado.

## 6) Benchmark de modelos (`benchmark_modelos`)
Se genera una comparación cuantitativa por símbolo:
- `drift_mean` (baseline en retorno diario siguiente).
- `MLPRegressor` con lags de retorno (si `scikit-learn` está disponible).
- `MLPRegressor` con lags + features de sentimiento (si existe cobertura de sentimiento).

Métricas reportadas:
- MAE
- RMSE
- MAPE
- Directional accuracy

## 7) Supuestos y límites
- Horizonte corto (días hábiles), no diseñado para escenarios estructurales de largo plazo.
- Para ARIMA, la acumulación de varianza usa aproximación diagonal (sin covarianzas cruzadas entre pasos).
- No incorpora explícitamente calendario de feriados por mercado (usa lunes-viernes).
- No modela cambios de régimen, saltos extremos ni volatilidad condicional tipo GARCH.
- La capa de sentimiento usa un léxico simple; puede subestimar matices semánticos y contexto macro.
- El benchmark ANN depende de `scikit-learn`; si no está instalado, se reporta baseline.

## 8) Trazabilidad y reproducibilidad
Cada corrida genera:
- `run_id` único.
- `ingestion_ts` en UTC.
- `run_metadata.json` con cobertura por símbolo, tamaño de salida y parámetros.

Esto permite versionar insumos y auditar de forma simple qué datos/modelo alimentaron un análisis downstream.

## 9) Validación recomendada para producción
Antes de usar como insumo crítico:
1. Backtesting walk-forward por símbolo y horizonte.
2. Métricas: MAE, RMSE, MAPE y cobertura empírica del IC95%.
3. Pruebas de estabilidad por subperíodos (pre-crisis, crisis, post-crisis).
4. Comparación contra benchmark ingenuo (random walk / drift).

## 10) Referencias
1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*.
3. Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference*.
4. statsmodels documentation: ARIMA/SARIMAX.
5. yfinance documentation.
