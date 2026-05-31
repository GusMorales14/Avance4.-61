# Avance 4 – Modelos alternativos
## Proyecto Integrador | Equipo 61 | Tecnológico de Monterrey – MNA

**Asignatura:** Proyecto de Inteligencia Artificial Aplicada  
**Entrega:** Avance 4  

| Integrante | Matrícula |
|------------|-----------|
| Gustavo Adolfo Morales García | A00828432 |
| Alejandro Jesús Mondragón Jiménez | A01795837 |
| Sebastián Ezequiel Coronado Rivera | A01212824 |

---

## Descripción general

Este avance forma parte de un proyecto de inferencia de propiedades físicas de galaxias a partir de imágenes astronómicas del survey **MaNGA (Mapping Nearby Galaxies at APO)** del Sloan Digital Sky Survey (SDSS). La tarea es un problema de **regresión supervisada** donde la variable objetivo es `LogMass` (logaritmo de la masa estelar), inferida directamente desde imágenes RGB de galaxias.

Partiendo del modelo CNN custom del Avance 3 (R² ≈ 0.79, MAE ≈ 0.34 dex), este avance explora 7 modelos individuales con enfoques variados, busca los dos de mejor desempeño, los ajusta mediante búsqueda de hiperparámetros y selecciona el modelo final.

---

## Estructura del repositorio

```
Avance4_61/
├── Avance4_61.ipynb       # Notebook principal (este avance)
├── Avance3_61.ipynb       # Notebook del avance anterior (referencia)
├── inferencia.csv         # Metadata tabulada de galaxias MaNGA
├── Imagenes/              # Directorio de imágenes .jpg (una por galaxia)
│   ├── manga-10001-12701.jpg
│   └── ...
└── README.md              # Este documento
```

> **Nota:** El directorio `Imagenes/` y `inferencia.csv` no se incluyen por su tamaño. Ajustar la variable `image_dir` en la celda correspondiente del notebook para apuntar a la ruta local correcta.

---

## Dataset

| Característica | Valor |
|----------------|-------|
| Fuente | MaNGA – SDSS DR17 |
| Total de galaxias | ~10,126 |
| Formato de imagen | JPEG RGB, 224×224 px |
| Variable objetivo | `LogMass` (masa estelar estandarizada) |
| Split | 70% entrenamiento / 15% validación / 15% prueba |
| Semilla aleatoria | 42 (idéntica al Avance 3 para comparabilidad) |

Las imágenes se relacionan con sus metadatos mediante el identificador `name` (ej. `manga-10001-12701`). Las variables tabulares fueron preprocesadas con pipelines de imputación, transformación Yeo-Johnson y estandarización — **exactamente iguales al Avance 3**.

---

## Modelos construidos

Se construyeron **7 modelos individuales** más el baseline de referencia:

| # | Identificador | Tipo | Descripción |
|---|---------------|------|-------------|
| 0 | `M0_Baseline_CNN` | Referencia | CNN 3 bloques (Avance 3, sin reentrenar) |
| 1 | `M1_CNN_GridSearch` | Desde cero | **Mismo baseline con GridSearch de HPs** |
| 2 | `M2_ResNet_Custom` | Desde cero | Bloques residuales con skip connections |
| 3 | `M3_CNN_Deep_BN` | Desde cero | CNN 4 bloques + Batch Normalization |
| 4 | `M4_EfficientNetB0` | Transfer Learning | Feature extraction + fine-tuning 30 capas |
| 5 | `M5_MobileNetV2` | Transfer Learning | Feature extraction + fine-tuning 20 capas |
| 6 | `M6_VGG16` | Transfer Learning | Feature extraction puro + L2 |
| 7 | `M7_CNN_Multimodal` | Fusión | Imagen + variables tabulares (rama dual) |

### Notas de diseño por modelo

**M1 – CNN + GridSearch:** El equipo definió que el Modelo 1 debe explorar los mejores hiperparámetros del modelo del Avance 3. Se reconstruye la misma arquitectura CNN de forma parametrizada y se ejecuta `keras-tuner` (RandomSearch) sobre `dropout_rate`, `dense_units` y `learning_rate`. Este modelo usa `GlobalAveragePooling2D` en lugar de `Flatten` para reducir parámetros.

**M2 – ResNet Custom:** Implementa bloques residuales desde cero con un stem Conv7×7, tres grupos de bloques (64→128→256 filtros) con strides progresivos y proyecciones 1×1. La instrucción del equipo era usar un "algoritmo diferente a CNN plana".

**M3 – CNN Profunda + BN:** Extiende el baseline añadiendo un cuarto bloque convolucional (256 filtros) y Batch Normalization después de cada bloque. La BN estabiliza el gradiente y actúa como regularizador implícito.

**M4–M6 – Transfer Learning:** Modelos preentrenados en ImageNet con estrategia de dos fases. EfficientNetB0 y MobileNetV2 usan fine-tuning selectivo; VGG16 usa feature extraction puro con regularización L2 en la cabeza.

**M7 – CNN Multimodal:** El único modelo que combina imagen y variables tabulares. Tiene una rama CNN para procesar la imagen y una rama MLP para las 13 variables tabulares preprocesadas. Las salidas se concatenan antes de la cabeza de regresión final.

---

## Búsqueda de hiperparámetros

### Modelo 1 (siempre activo)
El GridSearch del Modelo 1 se ejecuta automáticamente como parte del flujo normal. Usa `keras-tuner` (RandomSearch con 16 trials) si está disponible, o búsqueda manual con `itertools.product` como fallback.

### Modelos 2–7 (opcionales)
Cada modelo tiene una celda con un flag de activación:

```python
RUN_GRIDSEARCH_Mn = False  # <- Cambiar a True para activar
```

**Recomendación:** ejecutar todos los modelos con hiperparámetros por defecto primero, identificar los Top 2 en la tabla comparativa, y activar el GridSearch solo para esos dos antes de la sección de ajuste.

**Instalación de keras-tuner:**
```bash
pip install keras-tuner
```

| Modelo | Hiperparámetros explorados |
|--------|--------------------------|
| M1 (activo) | `dropout_rate`, `dense_units`, `learning_rate` |
| M2 | `dropout_rate`, `dense_units`, `learning_rate` |
| M3 | `dropout_rate`, `dense_units`, `learning_rate` |
| M4 | `dense_units`, `dropout_rate`, `freeze_layers` |
| M5 | `dense_units`, `dropout_rate`, `freeze_layers` |
| M6 | `dense1`, `dense2`, `dropout1`, `dropout2`, `l2_reg` |
| M7 | `img_dense`, `tab_dense`, `fusion_dense`, `dropout_rate` |

---

## Métricas de evaluación

Consistentes con el Avance 3, todas las métricas se calculan sobre el conjunto de **prueba (15%)**.

| Métrica | Descripción | Interpretación |
|---------|-------------|----------------|
| **R²** | Coeficiente de determinación | Varianza explicada (mayor es mejor) |
| **MAE (dex)** | Error absoluto medio | Error promedio (menor es mejor) |
| **RMSE (dex)** | Raíz del error cuadrático medio | Sensible a errores grandes (menor es mejor) |
| **Scatter (σ)** | Desv. estándar de residuales | Dispersión de predicciones |
| **Bias** | Media de residuales | Sesgo sistemático del modelo |
| **RMSE/MAE** | Ratio diagnóstico | >2.0 indica outliers severos |

**Categorización cualitativa:**

| R² | Categoría |
|----|-----------|
| ≥ 0.90 | EXCELENTE |
| ≥ 0.80 | MUY BUENO |
| ≥ 0.70 | BUENO |
| ≥ 0.50 | REGULAR |
| < 0.50 | INACEPTABLE |

---

## Requisitos del entorno

```
python          >= 3.10
tensorflow      >= 2.12
keras-tuner     >= 1.3    (opcional, para GridSearch)
scikit-learn    >= 1.2
pandas          >= 1.5
numpy           >= 1.24
matplotlib      >= 3.6
seaborn         >= 0.12
scipy           >= 1.10
```

**Instalación:**
```bash
pip install tensorflow keras-tuner scikit-learn pandas numpy matplotlib seaborn scipy
```

**Entorno recomendado:** GPU con al menos 8 GB VRAM. En CPU el entrenamiento de modelos con Transfer Learning puede tomar entre 30–90 minutos por modelo.

---

## Reproducibilidad

```python
tf.random.set_seed(42)
np.random.seed(42)
# train_test_split con random_state=42 — idéntico al Avance 3
```

---

## Tabla comparativa de resultados

> **Pendiente:** La tabla se completará una vez ejecutadas las celdas del notebook en el entorno local con acceso al dataset MaNGA.

| Modelo | R² | MAE (dex) | RMSE (dex) | Parámetros | Tiempo | Categoría |
|--------|----|-----------|------------|-----------|--------|-----------|
| M0 – Baseline CNN (ref.) | 0.7903 | 0.3367 | 0.4491 | 11.2M | — | MUY BUENO |
| M1 – CNN + GridSearch | — | — | — | — | — | — |
| M2 – ResNet Custom | — | — | — | — | — | — |
| M3 – CNN Deep BN | — | — | — | — | — | — |
| M4 – EfficientNetB0 | — | — | — | — | — | — |
| M5 – MobileNetV2 | — | — | — | — | — | — |
| M6 – VGG16 | — | — | — | — | — | — |
| M7 – CNN Multimodal | — | — | — | — | — | — |
| **Top 1 Ajustado** | — | — | — | — | — | — |
| **Top 2 Ajustado** | — | — | — | — | — | — |

---

## Modelo final seleccionado

> **Pendiente:** Se completará una vez obtenidos los resultados de la tabla comparativa.

**Modelo:** _por determinar_

**Justificación:** _por completar tras análisis de resultados_

| Métrica | Baseline (Avance 3) | Modelo Final | Mejora |
|---------|--------------------|-----------|----|
| R² | 0.7903 | — | — |
| MAE (dex) | 0.3367 | — | — |
| RMSE (dex) | 0.4491 | — | — |

---

## Conclusiones

> **Pendiente:** Se completará una vez ejecutadas las celdas del notebook.

Aspectos a analizar:
- Impacto del GridSearch sobre la arquitectura del Avance 3 (¿cuánto gana M1 vs M0?)
- Comparación bloques residuales vs CNN plana (M2 vs M0/M1)
- Transfer Learning vs entrenamiento desde cero
- Aporte de la información tabular en el modelo Multimodal (M7)
- Análisis de sesgo y distribución de residuales del modelo final
- Propuestas para el siguiente avance

---

## Referencias

- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks*. ICLR.
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
- Bundy, K. et al. (2015). *Overview of the SDSS-IV MaNGA Survey*. ApJ.

---

*Proyecto Integrador – Maestría en Inteligencia Artificial Aplicada – Tecnológico de Monterrey.*
