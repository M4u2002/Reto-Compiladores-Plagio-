# Detección de Código Generado por IA — SemEval-2026 Task 13 (Subtask A)

**Notebook:** [`retov3.ipynb`](retov3.ipynb)
**Equipo (EquipoCompiladores):** Juan Pablo Chávez Leal · Mauricio Anguiano Juárez · Daniel Contreras Chávez — ITESM, Campus Querétaro (TC3002B).

---

## 1. Contexto y problemática

La adopción masiva de asistentes de programación basados en LLM (ChatGPT, Copilot, Gemini) ha hecho cada vez más difícil distinguir el código escrito por una persona del generado por una IA. En cursos introductorios de programación esto representa un nuevo tipo de deshonestidad académica que los detectores clásicos de similitud (MOSS, JPlag) no pueden cubrir, ya que no existe una fuente "original" con la cual comparar: cada fragmento generado por un LLM es sintácticamente único.

**El problema:** clasificación binaria a nivel de fragmento de código.

- `0` = humano
- `1` = máquina (generado por IA)

**Objetivo general:** replicar el enfoque de Ramachandra et al. (2026) para la detección de código generado por IA y evaluarlo de forma reproducible sobre el dataset oficial de la competencia SemEval-2026 Task 13, adaptando su arquitectura a este dataset.

## 2. Dataset

- **Fuente:** Kaggle [`daniilor/semeval-2026-task13`](https://www.kaggle.com/datasets/daniilor/semeval-2026-task13) (SemEval-2026 Task 13, Subtask A), versión 3, ~1.03 GB.
- **Columnas:** `code` (fragmento de código fuente), `label` (0 = humano, 1 = máquina), `language` (C++, Python, Java, entre otros), `generator` (modelo/origen del fragmento).
- **Tamaño:** 601,000 filas en total.
- **Splits:** se usan los **splits oficiales de la competencia** — el conjunto de entrenamiento (`task_a_training_set_1.parquet`) para entrenar y el de validación (`task_a_validation_set.parquet`) para evaluar.
- La descarga se realiza vía la API de Kaggle (requiere credenciales del usuario).

## 3. Arquitectura y metodología (replicación del paper)

Siguiendo a Ramachandra et al. (2026), en lugar de redes neuronales profundas se detecta el código generado por IA combinando **características extraídas del código fuente** con representaciones de tokens, clasificadas mediante **modelos de aprendizaje automático tradicional**.

### Features (11 estructurales/estilométricas)

Calculadas por fragmento: `n_lines`, `n_chars`, `n_blank`, `n_comment`, `avg_indent`, `avg_line_len`, `n_space`, `n_tab`, `ratio_ws`, `ratio_comment`, `ratio_blank`.

### TF-IDF sobre tokens del código

`lowercase=False` (preserva identificadores), `ngram_range=(1,2)`, `max_features=20000`, `min_df=2`.

Ambos bloques se concatenan en una matriz sparse: **20,000 columnas TF-IDF + 11 estructurales = 20,011 features**.

> Para evitar fuga de datos (*data leakage*), el vectorizador TF-IDF y el escalador (`StandardScaler`) se ajustan **únicamente con el conjunto de entrenamiento** y solo se aplican (transform) sobre el de validación.

### Modelos

- `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
- `XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9)`

La métrica de evaluación es **macro F1**, la métrica oficial de la Subtask A.

## 4. Extensión propia (NO forma parte del paper) — CodeBERT

Como aporte adicional del equipo, con fines puramente comparativos, se extraen embeddings contextuales del modelo `microsoft/codebert-base` (RoBERTa preentrenado sobre código en varios lenguajes). Para cada fragmento se toma el embedding del token `[CLS]` (768 dimensiones) y se entrena un Random Forest sobre esos vectores — mismo tipo de clasificador que el enfoque del paper, distinta representación, para que la comparación sea justa.

Por costo computacional, la extensión usa una **muestra** (5,000 fragmentos de entrenamiento y 1,000 de validación) en lugar del dataset completo. Esta sección puede eliminarse sin afectar la replicación del paper.

## 5. Resultados

Métrica: **macro F1** sobre el conjunto de evaluación.


| Modelo | Macro F1 | Origen |
|--------|----------|--------|
| **XGBoost** | **0.9909** | Replicación del paper |
| Random Forest | 0.9821 | Replicación del paper |
| CodeBERT + Random Forest | 0.9419 | Extensión propia (muestra de 5,000) |

Estos resultados son consistentes con los reportados por Ramachandra et al. (F1 = 0.988 con TF-IDF + Ensemble sobre su propio dataset): la representación estadística (features + TF-IDF) con clasificadores de árboles es suficiente para separar código humano de código generado.

## 6. Reproducción y persistencia

1. **Primera vez:** ejecutar el notebook completo (requiere credenciales de Kaggle y, para la extensión, GPU recomendada). Al final, la celda **GUARDAR (12)** serializa los modelos entrenados, los preprocesadores y las métricas en `artefactos_reto/` (~380 MB; excluida de git por su tamaño).
2. **Kernel nuevo, sin reentrenar:** ejecutar la CELDA 1 (imports) y luego la celda **CARGAR (13)**. Eso restaura todo y permite saltar las celdas 2 a 11 (descarga, features, entrenamiento y gráficas) e ir directo a la tabla comparativa y la predicción sobre fragmentos nuevos.
3. Los pesos preentrenados de CodeBERT no se guardan en el repo: quedan en la caché de HuggingFace y se reutilizan automáticamente.
4. El notebook también genera gráficas de evaluación en `figuras/`: matrices de confusión (`matrices_confusion.png`), curvas ROC (`curvas_roc.png`) e importancia de features (`importancia_features.png`).

## 7. Limitaciones

- La extensión CodeBERT se entrena y evalúa sobre una muestra reducida (5,000 / 1,000) por costo computacional, por lo que su comparación contra los modelos del enfoque principal (evaluados sobre todo el split de validación) es indicativa, no concluyente.
- El dataset es grande (601k filas), lo que hace el entrenamiento pesado (especialmente el Random Forest sin límite de profundidad).
- En fragmentos muy cortos (pocas líneas) hay poca señal estilométrica y el clasificador puede equivocarse, como muestra el ejemplo de la celda de predicción: una función trivial escrita a mano puede clasificarse como generada por IA.

## 8. Referencia

Ramachandra, S., Chaudhary, A., Tran, J., Desai, P., Pang, A., & Salloum, M. (2026). *Detecting AI-Generated Code in Introductory Programming Courses.* SIGCSE TS 2026, UC Riverside. https://doi.org/10.1145/3770762.3772522

Contexto del paper (no son resultados de este notebook): reporta F1 = 0.988 (TF-IDF + Ensemble) y F1 = 0.995 (CodeBERT + Ensemble) sobre 3,360 muestras de C++, y que ~10–15% de los estudiantes de CS1 cometen deshonestidad asistida por IA.
