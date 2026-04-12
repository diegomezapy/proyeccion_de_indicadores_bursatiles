# Metodología Integral para la Generación de Insumos Analíticos en Investigación Aplicada a Mercados Financieros

## 1. Propósito científico, justificación y alcance metodológico

El presente repositorio implementa un flujo metodológico integral para la producción de insumos cuantitativos, auditables y reutilizables, orientados a investigación aplicada en mercados financieros. Su arquitectura responde a un enfoque de ciencia de datos reproducible, en el que cada etapa del proceso, desde la adquisición de datos hasta la construcción de salidas analíticas, queda explícitamente documentada, parametrizada y trazada.

El propósito central del sistema consiste en transformar datos heterogéneos de mercado y señales textuales en productos analíticos consistentes, adecuados tanto para exploración empírica como para evaluación metodológica comparada. En este sentido, el repositorio no se limita a generar predicciones puntuales, sino que constituye una infraestructura analítica completa para extraer, depurar, estandarizar, modelar y evaluar información financiera de corto plazo.

Desde una perspectiva funcional, la metodología cubre tres dimensiones articuladas:

1. la adquisición y validación de datos históricos de mercado,
2. el modelado predictivo de corto horizonte sobre series financieras,
3. la incorporación de señales textuales de sentimiento y su contraste frente a modelos estadísticos y de aprendizaje automático.

Este diseño permite satisfacer dos objetivos complementarios. En primer lugar, producir insumos listos para consumo por terceros, tales como archivos CSV, Parquet y tablas consolidadas. En segundo lugar, garantizar trazabilidad metodológica de extremo a extremo, de modo que cada salida pueda ser auditada en función de su fuente, fecha de extracción, parámetros utilizados, supuestos analíticos y limitaciones conocidas.

El alcance actual se concentra en predicción operativa de corto plazo, benchmarking entre modelos y generación de señales auxiliares. Por tanto, el sistema no debe interpretarse como un marco de inferencia estructural de largo horizonte, sino como una plataforma de análisis predictivo, monitoreo y experimentación metodológica aplicada.

---

## 2. Planteamiento metodológico y fundamentos conceptuales

### 2.1. Pregunta metodológica central

La pregunta que orienta esta arquitectura analítica puede formularse del siguiente modo:

> Dada una serie histórica de precios ajustados y una señal textual agregada derivada de noticias financieras, ¿qué desempeño relativo presentan distintos enfoques predictivos de corto plazo, específicamente un baseline parsimonioso, un modelo ARIMA y una red neuronal tipo MLP, cuando se evalúan sobre índices bursátiles mediante un esquema de validación temporal?

Esta formulación ubica el problema dentro del campo de la predicción financiera supervisada, reconociendo explícitamente que el objetivo es comparativo y operacional, no causal.

### 2.2. Fundamentos teóricos

La metodología se apoya en cuatro bloques conceptuales principales.

**Primero**, la tradición Box-Jenkins de modelado de series temporales, que proporciona un marco sistemático para representar dependencia serial, parsimonia y estructura dinámica en variables temporales. En este marco, los modelos ARIMA constituyen un estándar robusto para capturar patrones autoregresivos y de media móvil bajo supuestos razonables de estacionariedad o transformaciones apropiadas.

**Segundo**, la teoría de selección de modelos basada en criterios de información, particularmente el AICc, que permite balancear ajuste y complejidad en muestras finitas. Esta perspectiva es especialmente relevante en contextos financieros, donde la sobreparametrización puede inducir aparente buen ajuste in sample pero pobre capacidad predictiva out of sample.

**Tercero**, la literatura sobre hechos estilizados de retornos financieros, que documenta propiedades tales como asimetría, curtosis elevada, colas pesadas, agrupamiento de volatilidad y desviaciones respecto de la normalidad. Estas características justifican la adopción de estrategias de robustecimiento y una evaluación especialmente cuidadosa de los supuestos estadísticos.

**Cuarto**, la integración de información textual como señal auxiliar. La evidencia reciente en finanzas computacionales sugiere que el flujo de noticias puede contener información incremental sobre expectativas, percepción de riesgo y dirección de mercado. En esta implementación, la señal textual se utiliza como insumo complementario y de bajo costo computacional, sin pretender reemplazar modelos avanzados de procesamiento de lenguaje natural entrenados específicamente sobre corpus financieros.

---

## 3. Universo de análisis y cobertura empírica

### 3.1. Activos e índices considerados

La versión por defecto del pipeline contempla los siguientes símbolos representativos de distintos mercados y segmentos:

* `^GSPC`, S&P 500
* `^IXIC`, Nasdaq Composite
* `^DJI`, Dow Jones Industrial Average
* `^FTSE`, FTSE 100
* `^IBEX`, IBEX 35
* `^BVSP`, Bovespa
* `^MERV`, MERVAL
* `^VIX`, índice de volatilidad implícita

Esta selección permite combinar mercados desarrollados y emergentes, así como incorporar un indicador específico de incertidumbre financiera (`^VIX`) que presenta propiedades dinámicas diferenciadas respecto de los índices accionarios tradicionales.

### 3.2. Horizonte analítico

La metodología está diseñada para trabajo de corto plazo. En consecuencia, las predicciones y benchmarks deben interpretarse dentro de horizontes operativos reducidos, donde la estructura reciente de retornos y la información textual contemporánea pueden tener alguna utilidad predictiva relativa.

### 3.3. Unidad de observación

La unidad temporal básica es el día de negociación. Las series se construyen a partir de precios diarios, privilegiando el precio ajustado (`adj_close`) cuando está disponible, dado que esta variable corrige por eventos corporativos relevantes y mejora la comparabilidad temporal.

---

## 4. Fuentes de datos y estrategia de adquisición

### 4.1. Fuentes de precios de mercado

La estrategia de adquisición de datos de precios sigue una lógica jerárquica con redundancia de fuentes.

La fuente primaria es **Stooq**, mediante descarga de archivos CSV. Esta fuente se privilegia por su simplicidad de acceso y compatibilidad con un flujo automatizado de extracción.

Como mecanismo de respaldo se utiliza **Yahoo Finance**, accedido a través de la librería `yfinance`. Esta segunda capa actúa como fallback cuando la fuente primaria falla, retorna series vacías, presenta errores de conexión o no ofrece observaciones válidas para el símbolo solicitado.

La regla operativa implementada es la siguiente:

1. intentar la descarga desde Stooq,
2. verificar validez mínima de la respuesta,
3. si la extracción no supera los controles básicos, recurrir automáticamente a Yahoo Finance.

Este enfoque mejora la resiliencia del pipeline frente a fallos transitorios y reduce el riesgo de interrupciones en la generación de insumos.

### 4.2. Fuente de información textual

La señal de sentimiento se construye a partir de **Google News RSS**, consultado mediante expresiones específicas asociadas a cada índice o mercado. Para cada noticia capturada se registran, como mínimo:

* titular,
* URL,
* fecha de publicación,
* medio o fuente declarada.

La unidad mínima de análisis textual es el titular noticioso. Esta decisión obedece a criterios de costo computacional, disponibilidad homogénea y facilidad de automatización.

---

## 5. Protocolo de extracción, trazabilidad y control de calidad

La capa de extracción y aseguramiento de calidad constituye un componente esencial del repositorio. Su objetivo no es únicamente descargar datos, sino producir evidencia documentada de cómo se obtuvo cada serie y bajo qué condiciones fue aceptada o rechazada.

### 5.1. Registro de auditoría

Cada intento de extracción genera metadatos estructurados que permiten reconstruir el proceso ex post. Entre las variables registradas se incluyen:

* `status`, estado del intento (`ok`, `error`, `accepted`),
* `rows_raw`, cantidad de filas recuperadas,
* `latency_ms`, latencia de la consulta,
* `error`, descripción textual del fallo cuando ocurre,
* `series_start_date`, primera fecha observada,
* `series_end_date`, última fecha observada,
* `pct_missing_adj_close`, porcentaje de valores faltantes en `adj_close`.

Este esquema convierte a la capa de scraping en un módulo auditable y no en una simple rutina de descarga.

### 5.2. Criterios mínimos de aceptación de una serie

Una extracción se considera metodológicamente aceptable cuando cumple simultáneamente las siguientes condiciones:

1. existe una fecha válida interpretable,
2. existe un símbolo correctamente identificado,
3. existe al menos un precio utilizable entre `close`, `adj_close`, `open`, `high` o `low`.

Adicionalmente, la existencia de niveles excesivos de valores faltantes o incoherencias estructurales puede motivar rechazo o marcado especial para revisión.

### 5.3. Dimensiones de calidad cubiertas

Los controles implementados apuntan principalmente a cuatro dimensiones de calidad de datos:

* **completitud**, presencia efectiva de variables y observaciones relevantes,
* **consistencia**, integridad de formatos y tipos,
* **validez**, conformidad con reglas mínimas esperadas,
* **trazabilidad**, posibilidad de reconstrucción del origen y resultado del proceso.

---

## 6. Preparación, limpieza y estandarización de datos

### 6.1. Esquema canónico de almacenamiento

Las series históricas consolidadas se organizan en un formato estandarizado con la siguiente estructura:

[
\texttt{symbol, date, open, high, low, close, adj_close, volume, source, ingestion_ts, run_id}
]

Este esquema asegura interoperabilidad entre módulos, facilita la integración incremental y permite rastrear la procedencia exacta de cada registro.

### 6.2. Procedimientos de limpieza

La fase de preparación contempla, al menos, las siguientes operaciones:

* coerción robusta de tipos,
* normalización de fechas,
* conversión de columnas numéricas,
* eliminación de filas manifiestamente inválidas,
* depuración de observaciones sin precio utilizable,
* ordenamiento cronológico por símbolo y fecha.

### 6.3. Regla de deduplicación

Para evitar multiplicidad de registros equivalentes, se aplica una deduplicación por clave compuesta `(symbol, date)`, conservando el registro más reciente o más confiable según la lógica de ingestión. Esto resulta especialmente importante cuando el sistema ejecuta corridas sucesivas o combina múltiples fuentes.

### 6.4. Consolidación incremental

El diseño admite actualización acumulativa de históricos. Las reglas principales son:

* para históricos, se realiza unión incremental seguida de deduplicación por `(symbol, date)`,
* para proyecciones, se conserva la última corrida por `(symbol, fc_h)`, evitando acumular bloques obsoletos de pronósticos.

Este enfoque distingue claramente entre información histórica persistente e información prospectiva dependiente del momento de ejecución.

---

## 7. Especificación formal del componente predictivo

## 7.1. Variable de modelado

El modelado principal se realiza sobre retornos logarítmicos definidos como:

[
r_t = \log(P_t) - \log(P_{t-1}),
]

donde (P_t) representa el precio ajustado (`adj_close`) del activo en el período (t).

La elección de retornos logarítmicos se justifica por varias razones:

1. aproximan la aditividad temporal,
2. reducen problemas asociados al nivel nominal de los precios,
3. facilitan la interpretación porcentual aproximada para pequeñas variaciones,
4. son estándar en modelado financiero y evaluación comparativa.

### 7.2. Modelo principal, ARIMA sobre retornos

El modelo principal corresponde a un proceso ARIMA((p,0,q)) estimado sobre la serie de retornos. Dado que la transformación a retornos apunta a reducir no estacionariedad en nivel, no se incorpora diferenciación adicional en la especificación por defecto.

La selección del orden se realiza mediante búsqueda en grilla para (p,q \in [0,3]), aunque este rango es parametrizable según el contexto de aplicación.

El criterio de elección es el **AICc**, definido por:

[
\mathrm{AICc} = -2\ell(\hat{\theta}) + 2k + \frac{2k(k+1)}{n-k-1},
]

donde:

* (\ell(\hat{\theta})) es la log-verosimilitud evaluada en el estimador,
* (k) es el número de parámetros libres,
* (n) es el tamaño muestral efectivo.

El uso de AICc, en lugar de AIC simple, es especialmente apropiado cuando el tamaño de muestra no es muy grande respecto del número de parámetros candidatos.

### 7.3. Robustecimiento frente a observaciones extremas

Dado que los retornos financieros suelen exhibir colas pesadas y observaciones atípicas, se implementa una **winsorización al 1%** sobre los retornos antes de la estimación. Esta transformación no elimina observaciones, sino que acota valores extremos, reduciendo la influencia desproporcionada de shocks aislados sobre el ajuste del modelo.

Este procedimiento no pretende corregir completamente la no normalidad, pero sí mejorar estabilidad numérica y robustez operativa en un entorno automatizado.

### 7.4. Estrategias de fallback

Cuando el modelo ARIMA no puede estimarse adecuadamente, por ejemplo debido a longitud insuficiente de la serie, convergencia fallida o singularidad numérica, el sistema recurre a un baseline de menor complejidad denominado `naive_drift_log`, basado en la deriva promedio de los retornos logarítmicos.

En el caso particular de `^VIX`, la deriva se fija en cero como supuesto conservador. Esta decisión reconoce que el índice de volatilidad presenta comportamiento distinto al de los índices bursátiles convencionales y que imponer una deriva promedio inestable podría generar proyecciones poco plausibles.

### 7.5. Reconstrucción de pronósticos al nivel del precio

Una vez pronosticados los retornos acumulados, la proyección al nivel del precio se obtiene mediante:

[
\hat P_{t+h} = P_t \exp(\mathbb{E}[S_h]),
]

donde (S_h) representa el retorno logarítmico acumulado hasta el horizonte (h).

### 7.6. Intervalos o bandas de proyección

Las bandas de incertidumbre al 95% se construyen bajo aproximación gaussiana sobre el retorno acumulado:

[
\hat P_{t+h}^{\pm} = P_t \exp\left(\mathbb{E}[S_h] \pm 1.96 \sigma_{S_h}\right),
]

donde (\sigma_{S_h}) es la desviación estándar del retorno acumulado pronosticado.

Estas bandas no deben interpretarse como intervalos de confianza exactos en sentido estructural, sino como bandas operativas de incertidumbre sujetas a los supuestos del modelo y a la calidad de la aproximación.

---

## 8. Construcción de la señal textual de sentimiento

### 8.1. Motivación metodológica

Los mercados financieros reaccionan no solo a fundamentos observables, sino también a narrativas, expectativas y shocks informativos. En consecuencia, una señal textual derivada de noticias puede aportar información complementaria sobre el contexto reciente del mercado.

En esta implementación, la señal textual se concibe como una variable auxiliar de apoyo y no como sustituto de modelos especializados de NLP financiero.

### 8.2. Procedimiento de extracción

Para cada símbolo se ejecutan consultas RSS orientadas a recuperar titulares vinculados al índice o mercado correspondiente. Los elementos textuales son posteriormente limpiados y tokenizados para su análisis léxico.

### 8.3. Cálculo del score de sentimiento

Se aplica un léxico financiero simple, clasificando tokens como positivos o negativos según un diccionario predefinido. El score normalizado se expresa como:

[
\mathrm{score} = \frac{N_{pos} - N_{neg}}{\sqrt{N_{tokens}}},
]

donde:

* (N_{pos}) es el número de tokens positivos,
* (N_{neg}) es el número de tokens negativos,
* (N_{tokens}) es el número total de tokens relevantes.

La normalización por la raíz cuadrada del tamaño del texto busca estabilizar parcialmente el efecto de longitud del titular.

### 8.4. Agregación diaria por símbolo

La señal final se resume diariamente por símbolo mediante estadísticas tales como:

* `n_news`, cantidad de noticias capturadas,
* `sent_mean`, media del score,
* `sent_median`, mediana del score,
* `sent_std`, dispersión del score,
* `sent_pos_ratio`, proporción de noticias con tono positivo,
* `sent_neg_ratio`, proporción de noticias con tono negativo,
* `last_headline_at`, timestamp de la noticia más reciente.

### 8.5. Limitaciones del componente textual

Este módulo presenta limitaciones importantes que deben explicitarse:

1. opera sobre titulares y no sobre texto completo,
2. depende de un léxico simple, sin modelado contextual profundo,
3. puede perder matices semánticos, ironía o lenguaje técnico,
4. no distingue necesariamente entre noticias de alta y baja relevancia económica.

Por ello, su función metodológica es auxiliar y exploratoria.

---

## 9. Benchmark de modelos y diseño de evaluación

### 9.1. Variable objetivo

La evaluación comparativa se realiza sobre el retorno diario siguiente, definido como:

[
\text{target}*t = r*{t+1}.
]

Con ello, el problema se formula como predicción de un paso adelante en un entorno estrictamente temporal.

### 9.2. Modelos comparados

La hoja `benchmark_modelos` resume el rendimiento de tres estrategias principales:

1. **`drift_mean`**, baseline basado en la media histórica de retornos de entrenamiento,
2. **`mlp_regressor` con rezagos**, utilizando como predictores `lag_1` a `lag_5`,
3. **`mlp_regressor` con rezagos y sentimiento**, incorporando además variables como `sent_mean` y `n_news` cuando existe cobertura informativa.

Este diseño permite distinguir el valor relativo de:

* un baseline extremadamente parsimonioso,
* un modelo estadístico de series temporales,
* una arquitectura no lineal simple de aprendizaje supervisado.

### 9.3. Partición temporal de entrenamiento y prueba

La evaluación se implementa mediante un split temporal train/test, sin barajar observaciones. Esta decisión es metodológicamente obligatoria en series temporales, ya que mezclar observaciones rompería la estructura cronológica y generaría fuga de información.

El holdout final representa desempeño out of sample sobre un bloque posterior de la serie.

### 9.4. Métricas de desempeño

Las principales métricas reportadas son:

* **MAE**, error absoluto medio,
* **RMSE**, raíz del error cuadrático medio,
* **MAPE**, error porcentual absoluto medio,
* **Directional Accuracy**, proporción de veces en que el modelo acierta la dirección del retorno.

Cada métrica captura una dimensión distinta del error. Mientras MAE y RMSE se centran en magnitud, Directional Accuracy resulta particularmente relevante en finanzas, donde anticipar el signo del movimiento puede ser tan importante como aproximar su amplitud.

### 9.5. Alcance interpretativo del benchmark

El benchmark debe interpretarse como una comparación operativa bajo una ventana, partición temporal y conjunto específico de predictores. No autoriza afirmaciones de superioridad universal entre métodos ni implica que un modelo capture mecanismos causales del mercado.

---

## 10. Reproducibilidad, gobernanza y trazabilidad técnica

La reproducibilidad constituye un eje transversal de la metodología. Cada ejecución registra metadatos esenciales que permiten reconstruir el contexto exacto en que se generaron los insumos.

Entre los elementos de trazabilidad se incluyen:

* `run_id` único por corrida,
* `ingestion_ts` en UTC,
* parámetros relevantes de la ejecución,
* cobertura lograda por símbolo,
* volumen de salidas producidas por cada capa del pipeline.

Este esquema permite:

1. replicación técnica de resultados,
2. auditoría forense ante discrepancias,
3. control de versiones de artefactos,
4. integración con flujos de investigación externos.

Desde una perspectiva de gobernanza de datos, este enfoque reduce opacidad, mejora rendición de cuentas metodológica y favorece la reutilización responsable.

---

## 11. Supuestos, amenazas a la validez y limitaciones

Todo sistema predictivo automatizado requiere declarar explícitamente sus restricciones. En este caso, las principales son las siguientes.

### 11.1. Limitación de horizonte

La arquitectura está pensada para análisis de corto plazo. Por tanto, extrapolaciones a horizontes largos o interpretaciones estructurales deben evitarse.

### 11.2. Calendario de negociación simplificado

El pipeline utiliza una lógica de días hábiles tipo lunes a viernes, sin incorporar en forma exhaustiva calendarios de feriados específicos por mercado. Esto puede introducir pequeñas distorsiones en ciertos horizontes o comparaciones internacionales.

### 11.3. Cambios de régimen

Tanto ARIMA como MLP pueden deteriorar notablemente su desempeño cuando ocurren rupturas estructurales, shocks exógenos, crisis o cambios abruptos en la dinámica de volatilidad.

### 11.4. Aproximación simplificada del sentimiento

El módulo textual carece de comprensión contextual profunda. En consecuencia, su señal puede ser ruidosa o insuficiente en contextos de lenguaje financiero sofisticado.

### 11.5. Dependencia de la calidad de las fuentes

La calidad analítica final depende críticamente de la calidad, disponibilidad y consistencia de las fuentes externas consultadas. Aunque el sistema incorpora fallback y auditoría, ningún pipeline automatizado queda completamente exento de riesgo de errores de origen.

### 11.6. No causalidad

Los resultados del benchmark reflejan capacidad predictiva relativa en un diseño específico, no causalidad ni eficiencia económica explotable en términos operativos de trading real.

---

## 12. Recomendaciones para fortalecimiento metodológico en contexto académico

Para convertir este pipeline en una base aún más robusta de investigación formal, se recomienda avanzar en las siguientes líneas:

### 12.1. Validación walk forward

Sustituir o complementar el holdout simple por esquemas walk forward o rolling origin, que permiten estimar estabilidad temporal del desempeño predictivo.

### 12.2. Intervalos de incertidumbre para métricas

Calcular intervalos de confianza de MAE, RMSE y Directional Accuracy mediante bootstrap temporal o técnicas de remuestreo compatibles con dependencia serial.

### 12.3. Modelos adicionales de referencia

Incorporar benchmarks adicionales, tales como random walk, modelos ARIMA con componentes estacionales cuando corresponda, modelos GARCH para volatilidad condicional y enfoques híbridos.

### 12.4. Mejoras en la capa textual

Comparar el léxico simple con enfoques más avanzados basados en embeddings o modelos de lenguaje especializados en finanzas, por ejemplo FinBERT u otros clasificadores entrenados sobre corpus económico-financieros.

### 12.5. Gestión de dependencias y hash de artefactos

Registrar versión de paquetes, entorno de ejecución y hash criptográfico de archivos de salida para asegurar replicación independiente exacta.

### 12.6. Evaluación económica

Complementar las métricas estadísticas con criterios de utilidad económica, por ejemplo reglas simples de decisión, costos de error direccional o desempeño de estrategias simuladas bajo supuestos prudentes.

---

## 13. Conclusión metodológica

En síntesis, este repositorio constituye una plataforma metodológica integral para la generación de insumos analíticos en investigación aplicada a mercados financieros. Su principal fortaleza radica en la articulación de tres dimensiones frecuentemente tratadas por separado, adquisición robusta de datos, modelado predictivo reproducible y señalización textual complementaria.

Más que un conjunto de scripts aislados, la arquitectura conforma un sistema de trabajo científicamente defendible, porque documenta fuentes, explicita supuestos, registra cada corrida, estandariza las salidas y permite comparar metodologías bajo criterios transparentes. Esta característica lo vuelve especialmente útil para entornos académicos, proyectos de investigación aplicada y ejercicios de benchmarking metodológico donde la trazabilidad es tan importante como la predicción misma.

El valor científico del pipeline no reside únicamente en su capacidad de producir pronósticos, sino en su posibilidad de ser auditado, extendido, criticado y mejorado. Precisamente por ello, su diseño representa una base sólida para trabajos posteriores en series financieras, minería de texto, evaluación comparativa de modelos y desarrollo de sistemas analíticos reproducibles.

---

## 14. Referencias bibliográficas

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

---
