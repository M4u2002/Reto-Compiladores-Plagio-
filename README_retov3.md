# Detección de Código Generado por IA — SemEval-2026 Task 13 (Subtask A)

**Notebook:** [`retov3.ipynb`](retov3.ipynb)
**Equipo (EquipoCompiladores):** Juan Pablo Chávez Leal · Mauricio Anguiano Juárez · Daniel Contreras Chávez — ITESM, Campus Querétaro (TC3002B).

---

## 1. Contexto y problemática

La adopción masiva de la inteligencia artificial en la industria ha hecho cada vez más difícil distinguir el código escrito por una persona del generado por una IA. En cursos introductorios de programación esto representa un nuevo tipo de deshonestidad académica que los detectores clásicos de similitud (MOSS, JPlag) no pueden cubrir, ya que no existe una fuente "original" con la cual comparar: cada fragmento generado por un LLM es sintácticamente único.

**Objetivo general:** replicar el enfoque de Ramachandra et al. (2026) para la detección de código generado por IA y evaluarlo de forma reproducible sobre el dataset oficial de la competencia SemEval-2026 Task 13, adaptando su arquitectura a este dataset.

## 2. Dataset

- **Fuente:** Kaggle [`daniilor/semeval-2026-task13`](https://www.kaggle.com/datasets/daniilor/semeval-2026-task13) (SemEval-2026 Task 13, Subtask A).
- **Columnas:** `code` (fragmento de código fuente), `label` (0 = humano, 1 = máquina), `language` (C++, Python, Java, entre otros), `generator` (modelo/origen del fragmento).
- **Tamaño:** 601,000 filas en total.

## 3. Arquitectura y metodología (replicación del paper)

Siguiendo a Ramachandra et al. (2026), en lugar de redes neuronales profundas se detecta el código generado por IA combinando características extraídas del código fuente con representaciones de tokens, clasificadas mediante modelos de aprendizaje automático tradicional.

### Features (11 estructurales)

Calculadas por fragmento: `n_lines`, `n_chars`, `n_blank`, `n_comment`, `avg_indent`, `avg_line_len`, `n_space`, `n_tab`, `ratio_ws`, `ratio_comment`, `ratio_blank`.

### TF-IDF sobre tokens del código

`lowercase=False`, `ngram_range=(1,2)`, `max_features=20000`, `min_df=2`.

### Modelos

- `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
- `XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9)`

La métrica de evaluación es macro F1, esta fue una métrica también utilizada por los investigadores.

## 5. Resultados
| Modelo | Macro F1 | Origen |
|--------|----------|--------|
| XGBoost | 0.9909 | Replicación del paper |
| Random Forest | 0.9821 | Replicación del paper |


## 6. Reproducción y persistencia

1. Primera vez: ejecutar el notebook completo. Al final, la celda de guardado almacena los modelos entrenados, los preprocesadores y las métricas en `artefactos_reto/`.
2. Carga de los modelos: ejecutar la primera celda, en la que se encuentran los imports del proyecto y luego la celda de la carga de datos. Eso permite revisar las métricas y utilizar el modelo lo antes posible.

## 7. Limitaciones

- El dataset es grande (601k filas), lo que hace el entrenamiento pesado (especialmente el Random Forest sin límite de profundidad).
- En fragmentos muy cortos de código hay pocas features estructurales y el clasificador tendría menor precisión a la hora de sus predicciones.

## 8. Referencia

Ramachandra, S., Chaudhary, A., Tran, J., Desai, P., Pang, A., & Salloum, M. (2026). *Detecting AI-Generated Code in Introductory Programming Courses.* SIGCSE TS 2026, UC Riverside. https://doi.org/10.1145/3770762.3772522

Contexto del paper (no son resultados de este notebook): reporta F1 = 0.988 (TF-IDF + Ensemble) y F1 = 0.995 (CodeBERT + Ensemble) sobre 3,360 muestras de C++, y que ~10–15% de los estudiantes de CS1 cometen deshonestidad asistida por IA.
