# Detección de Plagio en Código de Estudiantes — Features Tabulares vs. CodeBERT

**Notebook:** [`reto_v3.ipynb`](reto_v3.ipynb)
**Equipo (EquipoCompiladores):** Juan Pablo Chávez Leal · Mauricio Anguiano Juárez · Daniel Contreras Chávez — ITESM, Campus Querétaro (TC3002B).

---

## 1. Contexto y problemática

El plagio en código tiene dos formas principales en los cursos de programación:

1. **Plagio peer-to-peer:** un estudiante copia el código de otro compañero o de un repositorio en línea. Los detectores tradicionales como MOSS detectan esto comparando similitud estructural entre entregas.
2. **Plagio IA-asistido:** un estudiante entrega como propio un código generado por un LLM (ChatGPT, Copilot, Gemini). Este caso no puede detectarse con MOSS porque cada entrega generada es sintácticamente única — no existe una fuente "original" con la que comparar. Ramachandra et al. (2026) reportan que ~10–15% de los estudiantes en cursos CS1 incurren en esta forma de deshonestidad académica.

**El problema de este notebook:** clasificación binaria a nivel de **par de archivos** Python — plagio vs. no-plagio (`Label`) — comparando si añadir similitud semántica de CodeBERT mejora sobre un enfoque puramente tabular.

## 2. Dataset

- **293 pares** de archivos Python etiquetados.
- `cheating_features_dataset.csv` — 13 features tabulares pre-calculadas + `Label`.
- `cheating_dataset.csv` — pares `(File_1, File_2, Label)`.
- Carpeta `data/` — 174 archivos `.py` reales referenciados por los pares.
- **Balance:** 193 no-plagio / 100 plagio (~2:1), manejado con `class_weight` y `sample_weight` balanceados.
- **Split:** 234 train / 59 validación, con `StratifiedShuffleSplit` (`SEED=42`).

## 3. Arquitectura y metodología

Se comparan dos enfoques, inspirados en Ramachandra et al. (2026):

| Enfoque | Representación | Clasificadores |
|---|---|---|
| **A — Línea base** | 13 features tabulares | Random Forest · SVM · XGBoost · Voting Ensemble |
| **B — + CodeBERT** | 13 features tabulares + similitud coseno entre embeddings CLS de CodeBERT | Los mismos 4 clasificadores |

### Features tabulares (13)

| Feature | Qué mide |
|---|---|
| `AST Similarity` | Similitud entre los árboles sintácticos abstractos de ambos archivos |
| `Token Similarity` | Proporción de tokens compartidos |
| `Levenshtein Similarity` | Distancia de edición normalizada entre el texto de los dos archivos |
| `Length File 1/2` | Longitud en caracteres de cada archivo |
| `Function Count File 1/2` | Número de funciones definidas |
| `Variable Count File 1/2` | Número de variables definidas |
| `Comment Ratio File 1/2` | Proporción de líneas de comentario |
| `Cyclomatic Complexity File 1/2` | Complejidad ciclomática (ramificaciones del flujo) |

Estas features corresponden al tipo de representación que Ramachandra et al. llaman "style anomaly and style inconsistency metrics".

### Decisiones de diseño

**¿Por qué clasificadores clásicos y no redes neuronales (MLP)?** Con 293 muestras totales (~234 de entrenamiento), los clasificadores clásicos superan sistemáticamente a las redes neuronales: los métodos basados en árboles y kernels tienen menos parámetros y no requieren miles de ejemplos para generalizar. El paper de referencia también usa Random Forest, SVM y XGBoost como sus mejores modelos.

**¿Por qué similitud coseno y no embeddings concatenados?** Con 234 ejemplos de entrenamiento, pasar 1,536 dimensiones (768+768) a un clasificador generaría una matriz de features enormemente dispersa imposible de generalizar. En cambio, `cosine_similarity(emb_1, emb_2)` — una sola feature escalar — comprime la similitud semántica entre los dos archivos en un valor que el clasificador puede aprovechar con pocos datos.

## 4. Resultados

Métrica: **macro F1** sobre el conjunto de validación (59 pares), que trata por igual a ambas clases — crucial con el desbalance 2:1.

| Modelo | Enfoque | Macro F1 |
|--------|---------|----------|
| **XGBoost** | A | **0.7521** (mejor) |
| Voting/Ensemble | A | 0.7485 |
| XGBoost | B | 0.7360 |
| Voting/Ensemble | B | 0.7318 |
| Random Forest | A | 0.7190 |
| Random Forest | B | 0.7190 |
| SVM | A | 0.6947 |
| SVM | B | 0.6641 |

**Δ (mejor B − mejor A) = −0.0161** → añadir la similitud coseno de CodeBERT **no mejora** el desempeño en este dataset; el enfoque tabular (A) es suficiente.

La brecha frente al paper (F1 de 0.988–0.995) se explica principalmente por el tamaño del dataset (293 pares vs. 3,360 muestras) y el dominio (pares de archivos Python vs. snippets individuales de C++). El contraste metodológico replicado es el mismo: representación estadística vs. representación contextual de código.

### Salidas generadas

- `models/comparativa_final.png` — matrices de confusión y curvas ROC de ambos enfoques.
- `models/feature_importance.png` — importancia de features del Random Forest (Enfoque B), destacando la feature de CodeBERT.

## 5. Limitaciones

- Dataset pequeño (293 pares): los resultados son sensibles al split de validación (59 pares).
- La similitud coseno de CodeBERT entre pares resultó casi constante (media 0.995, desviación 0.011), lo que explica en parte que no aporte señal discriminativa adicional.
- Añadir CodeBERT no aporta mejora en este dataset; el enfoque tabular es suficiente.

## 6. Referencia

Ramachandra, S., Chaudhary, A., Tran, J., Desai, P., Pang, A., & Salloum, M. (2026). *Detecting AI-Generated Code in Introductory Programming Courses.* SIGCSE TS 2026, UC Riverside. https://doi.org/10.1145/3770762.3772522

Contexto del paper (no son resultados de este notebook): reporta F1 = 0.988 (TF-IDF + Ensemble) y F1 = 0.995 (CodeBERT + Ensemble) sobre 3,360 muestras de C++.
