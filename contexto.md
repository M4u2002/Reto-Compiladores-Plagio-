# 📋 PROMPT PARA CLAUDE CODE (pégalo primero en VS Code)

Eres un asistente que va a generar el **marco de referencia** de un proyecto académico
(TC3002B, ITESM Querétaro) en archivos `README.md`.

Tengo DOS notebooks distintos y quiero DOS READMEs independientes:
- `README_retov3.md`  → basado en el BLOQUE 1
- `README_reto_v3.md` → basado en el BLOQUE 2

REGLAS IMPORTANTES:
1. NO mezcles los dos notebooks: usan datasets, tareas y métricas DISTINTAS.
2. NO inventes métricas ni datos: usa SOLO los valores reales que te paso aquí.
3. Cada README debe incluir: contexto/problema, dataset, arquitectura, features,
   modelos, resultados reales, referencia (paper) y limitaciones.
4. Escribe en español, formato Markdown limpio, con tablas para los resultados.
5. retov3 = "humano vs IA" sobre SemEval; reto_v3 = "plagio vs no-plagio" sobre 293 pares.

A continuación te paso el contexto completo de cada uno.

# 🟦 BLOQUE 1 — Contexto para `README_retov3.md`

## Proyecto
Detección de código generado por IA vs código humano, adaptando la arquitectura del
paper de Ramachandra et al. (2026) al dataset SemEval-2026 Task 13.

## Problema
Clasificación binaria a nivel de fragmento de código:
- `0` = humano
- `1` = máquina (generado por IA)

## Dataset
- Fuente: Kaggle `daniilor/semeval-2026-task13` (SemEval-2026 Task 13, Subtask A), versión 3, ~1.03 GB.
- Columnas: `code`, `label`, `language`, `generator`.
- Tamaño: **601,000 filas**.
- Split: 80/20 estratificado, `random_state=42`.
- Test: 120,200 filas (57,389 humano / 62,811 máquina).

## Features
**11 features estructurales** extraídas del código:
`n_lines`, `n_chars`, `n_blank`, `n_comment`, `avg_indent`, `avg_line_len`,
`n_space`, `n_tab`, `ratio_ws`, `ratio_comment`, `ratio_blank`.

**TF-IDF** sobre el código:
`lowercase=False`, `ngram_range=(1,2)`, `max_features=20000`, `min_df=2`.

Matriz combinada final: **(601000, 20011)** = 20000 TF-IDF + 11 estructurales.

## Modelos
- `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
- `XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9)`

## Extensión (comparativa)
CodeBERT: embeddings CLS (768 dimensiones) + RandomForest, sobre una muestra `SAMPLE=5000`.

## Resultados reales
Métrica: **macro F1**.

| Modelo | Macro F1 |
|--------|----------|
| **Random Forest** | **0.9817** |

(Random Forest es el mejor modelo en este notebook.)

## Limitaciones / notas
- La extensión CodeBERT corre sobre muestra reducida (5000) por costo computacional.
- Dataset grande (601k filas) → entrenamiento pesado.

# 🟩 BLOQUE 2 — Contexto para `README_reto_v3.md`

## Proyecto
Detección de plagio entre pares de archivos de código Python, comparando un enfoque
de features tabulares vs el mismo enfoque + embeddings de CodeBERT.

## Problema
Clasificación binaria a nivel de PAR de archivos:
- plagio vs no-plagio (`Label`)

## Dataset
- 293 pares de archivos Python.
- `cheating_features_dataset.csv`: 13 features + `Label`.
- `cheating_dataset.csv`: `File_1`, `File_2`, `Label`.
- Carpeta `data/`: 174 archivos `.py`.
- Balance: **193 no-plagio / 100 plagio** (~2:1).
- Split: 234 train / 59 val (`StratifiedShuffleSplit`, `SEED=42`).

## Features (13 tabulares)
`AST Similarity`, `Token Similarity`, `Levenshtein Similarity`,
`Length File 1`, `Length File 2`, `Function Count File 1`, `Function Count File 2`,
`Variable Count File 1`, `Variable Count File 2`, `Comment Ratio File 1`,
`Comment Ratio File 2`, `Cyclomatic Complexity File 1`, `Cyclomatic Complexity File 2`.

## Enfoques
- **Enfoque A**: 13 features → RandomForest / SVM / XGBoost / Voting.
- **Enfoque B**: 13 features + 1 feature de similitud coseno de embeddings CLS de CodeBERT → mismos 4 clasificadores.

## Resultados reales
Métrica: **macro F1**.

| Modelo | Enfoque | Macro F1 |
|--------|---------|----------|
| XGBoost | A | **0.7521** (mejor) |
| Voting/Ensemble | A | 0.7485 |
| XGBoost | B | 0.7360 |
| Voting/Ensemble | B | 0.7318 |
| Random Forest | A | 0.7190 |
| Random Forest | B | 0.7190 |
| SVM | A | 0.6947 |
| SVM | B | 0.6641 |

**Δ (mejor B − mejor A) = −0.0161** → CodeBERT NO mejora el desempeño en este dataset.

## Salidas generadas
- `models/comparativa_final.png`
- `models/feature_importance.png`

## Limitaciones / notas
- Dataset pequeño (293 pares) → resultados sensibles al split.
- Añadir CodeBERT no aporta mejora; el enfoque tabular (Enfoque A) es suficiente.

# 📚 REFERENCIA COMPARTIDA (APA) — incluir en ambos READMEs

Ramachandra, S., Chaudhary, A., Tran, J., Desai, P., Pang, A., & Salloum, M. (2026).
*Detecting AI-Generated Code in Introductory Programming Courses.* SIGCSE TS 2026,
UC Riverside. https://doi.org/10.1145/3770762.3772522

Notas del paper (para contexto, NO son resultados de tus notebooks):
- TF-IDF + Ensemble: F1 = 0.988
- CodeBERT + Ensemble: F1 = 0.995
- Sobre 3,360 muestras de C++.
- Reporta que ~10–15% de estudiantes de CS1 cometen deshonestidad asistida por IA.

