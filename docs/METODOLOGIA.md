# Metodología Completa de Generación de Insumos

## 1) Propósito Científico y Alcance
Este repositorio implementa un flujo metodológico para producir insumos cuantitativos y trazables orientados a investigación aplicada en mercados financieros. El diseño actual cubre tres capas integradas:

1. Capa de extracción y control de calidad de datos de mercado.
2. Capa de modelado predictivo de corto plazo para índices bursátiles.
3. Capa de señales textuales (sentimiento) y benchmark contra modelos de red neuronal.

El objetivo operativo es doble:
- Generar insumos reutilizables para terceros (CSV/Parquet/hojas).
- Permitir auditoría metodológica de extremo a extremo (fuente, fecha, corrida, supuestos y limitaciones).

## 2) Pregunta Metodológica y Marco Teórico
### 2.1 Pregunta metodológica
Bajo información histórica de precios y señal textual agregada, ¿qué nivel de desempeño relativo se obtiene entre:
- un baseline parsimonioso (drift en retornos),
- un modelo ARIMA para proyección,
- y un modelo ANN (MLP) en esquema de benchmark?

### 2.2 Marco conceptual
El planteamiento se apoya en literatura clásica de series financieras y pronóstico:
- Series temporales Box-Jenkins para estructura ARIMA.
- Selección de modelos por criterio de información (AICc) para parsimonia y ajuste.
- Hechos estilizados de retornos financieros (no normalidad, asimetría, colas pesadas, heterocedasticidad), que justifican robustecimiento y evaluación cuidadosa.
- Integración de señales de texto financiero como variable exógena de apoyo.

## 3) Universo de Análisis y Fuentes
### 3.1 Símbolos (default del pipeline)
- `^GSPC`, `^IXIC`, `^DJI`, `^FTSE`, `^IBEX`, `^BVSP`, `^MERV`, `^VIX`.

### 3.2 Fuentes de precios
- Primaria: Stooq (descarga CSV).
- Fallback: Yahoo Finance vía `yfinance`.

Regla de adquisición implementada:
1. Intentar Stooq.
2. Si falla o no retorna filas válidas, usar Yahoo.

### 3.3 Fuente de texto para sentimiento
- Google News RSS por query específica de símbolo/índice.
- Unidad de captura: titular, URL, fecha de publicación y fuente declarada.

## 4) Protocolo de Extracción y Auditoría de Calidad
La capa `scraping_qa` materializa el control de calidad de la extracción y permite replicar diagnósticos.

### 4.1 Variables de auditoría
- `status` del intento (`ok`, `error`, `accepted`).
- `rows_raw` por intento/fuente.
- `latency_ms` por intento.
- `error` textual en fallas.
- `series_start_date` y `series_end_date` de la serie aceptada.
- `pct_missing_adj_close` como proxy de completitud.

### 4.2 Criterios de aceptación
Una extracción se considera aceptada cuando:
- existe fecha válida,
- existe símbolo válido,
- y existe al menos un precio utilizable (`close`, `adj_close`, `open`, `high`, `low`).

Este enfoque se alinea con prácticas de calidad de datos orientadas a completitud, consistencia y trazabilidad en pipelines analíticos.

## 5) Preparación y Estandarización de Datos
### 5.1 Esquema canónico de históricos
`symbol, date, open, high, low, close, adj_close, volume, source, ingestion_ts, run_id`

### 5.2 Limpieza
- Coerción robusta de tipos (`date`, numéricos, timestamps).
- Eliminación de filas inválidas.
- Dedupe por clave `(symbol, date)`, conservando el registro más reciente.

### 5.3 Consolidación incremental
- Históricos: merge + deduplicación por `(symbol, date)`.
- Proyecciones: se conserva la última corrida por `(symbol, fc_h)` para evitar acumulación de bloques obsoletos.

## 6) Especificación Formal del Modelo de Proyección
### 6.1 Transformación principal
Se modela sobre retornos logarítmicos:

\[
r_t = \log(P_t) - \log(P_{t-1})
\]

con \(P_t\) definido por `adj_close`.

### 6.2 Modelo principal
- ARIMA\((p,0,q)\) sobre \(r_t\).
- Búsqueda en grilla para \(p,q\in[0,3]\) (parametrizable).
- Selección por AICc:

\[
\mathrm{AICc} = -2\ell(\hat\theta) + 2k + \frac{2k(k+1)}{n-k-1}
\]

donde \(\ell\) es log-verosimilitud, \(k\) número de parámetros y \(n\) tamaño muestral.

### 6.3 Robustecimiento
- Winsorización al 1% de retornos para disminuir sensibilidad a outliers extremos.

### 6.4 Fallback
Cuando la muestra es corta o no converge ARIMA:
- `naive_drift_log` sobre retornos log.
- Para `^VIX`, deriva forzada a 0 como supuesto conservador de nivel de volatilidad.

### 6.5 Reconstrucción al nivel y bandas
Para horizonte \(h\), con retorno acumulado \(S_h\):

\[
\hat P_{t+h} = P_t\exp(\mathbb{E}[S_h])
\]

Bandas 95% por aproximación gaussiana sobre retorno acumulado:

\[
\hat P_{t+h}^{\pm} = P_t\exp\left(\mathbb{E}[S_h] \pm 1.96\sigma_{S_h}\right)
\]

## 7) Capa de Sentimiento Textual
### 7.1 Extracción
Se consulta RSS de noticias con términos ligados a cada índice.

### 7.2 Scoring
Se aplica un léxico financiero simple (positivo/negativo) sobre tokens del titular. El score normalizado se calcula como:

\[
\mathrm{score} = \frac{N_{pos}-N_{neg}}{\sqrt{N_{tokens}}}
\]

### 7.3 Agregación diaria por símbolo
- `n_news`
- `sent_mean`, `sent_median`, `sent_std`
- `sent_pos_ratio`, `sent_neg_ratio`
- `last_headline_at`

Nota: este módulo es una señal auxiliar de bajo costo computacional; no reemplaza un modelo NLP entrenado para finanzas.

## 8) Benchmark de Modelos
La hoja `benchmark_modelos` reporta evaluación sobre retorno diario siguiente (`target = ret_{t+1}`):

1. `drift_mean` (baseline): media de retornos de entrenamiento.
2. `mlp_regressor` con lags de retornos (`lag_1..lag_5`).
3. `mlp_regressor` con lags + sentimiento (`sent_mean`, `n_news`) cuando hay cobertura.

### 8.1 Partición de evaluación
- Split temporal train/test (sin barajar), con holdout final.
- El esquema implementado es benchmark operativo; para investigación formal se recomienda validación walk-forward adicional.

### 8.2 Métricas reportadas
- MAE
- RMSE
- MAPE
- Directional Accuracy

## 9) Trazabilidad, Reproducibilidad y Gobernanza
Cada ejecución registra:
- `run_id` único,
- `ingestion_ts` UTC,
- parámetros de ejecución,
- cobertura por símbolo,
- tamaño de salidas por capa.

Esto permite reproducibilidad técnica, auditoría forense y reutilización de insumos en trabajos externos.

## 10) Validez, Riesgos y Limitaciones
1. Horizonte de corto plazo: no está diseñado para inferencia estructural de largo plazo.
2. No incorpora calendario de feriados por mercado (solo días hábiles lunes-viernes).
3. ARIMA y MLP pueden degradarse en cambios de régimen.
4. Sentimiento léxico simple puede perder contexto semántico y tono financiero avanzado.
5. El benchmark actual no implica causalidad; refleja desempeño predictivo bajo una ventana y partición específicas.

## 11) Recomendaciones para Uso Académico
1. Ejecutar backtesting walk-forward por símbolo/horizonte.
2. Reportar intervalos de confianza de métricas por bootstrap temporal.
3. Comparar contra benchmarks adicionales (random walk, ARIMA estacional, modelos con volatilidad condicional).
4. En sentimiento, contrastar léxico con embeddings/modelos NLP especializados en finanzas.
5. Registrar versión de dependencias y hash de artefactos para replicación independiente.

## 12) Referencias Bibliográficas
1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
3. Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
4. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
5. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.). Springer.
6. Tsay, R. S. (2010). *Analysis of Financial Time Series* (3rd ed.). Wiley.
7. Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of UK Inflation. *Econometrica*, 50(4), 987-1007.
8. Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
9. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
10. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
11. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
12. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383-417.
13. Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35-65.
14. Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. *ICWSM*.
15. Documentación oficial de `statsmodels`, `yfinance`, `gspread` y `scikit-learn`.
