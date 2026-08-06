# Fake Job Postings Detection

Binary classification of fraudulent job adverts from their text and metadata.
~17,880 real postings, ~5% of them scams — a hybrid text + tabular problem where
the headline metric has to be **average precision (PR-AUC)**, because accuracy is
actively misleading at this class balance.

Baseline plus three models are compared on an identical, frozen test split:
LightGBM, a fine-tuned DistilBERT, and frozen sentence-transformer embeddings with
a classifier on top.

---

## 1. Business task & goal

Job boards carry adverts that are not jobs. They exist to harvest personal data
(passport scans, bank details "for payroll"), to charge up-front fees for training
or equipment, or to recruit money mules. Every one that stays up costs the platform
trust; every legitimate advert wrongly pulled costs it a paying customer.

**Goal:** rank incoming postings by fraud risk so a small moderation team can review
the top slice, instead of reading everything or trusting keyword rules.

Framed this way, the product question is not "what's the accuracy?" but:

> If we auto-flag the postings we are most confident about, **what share of the
> scams do we catch, and how many honest employers do we insult along the way?**

That is a precision/recall trade-off at a chosen operating point — which is why
threshold selection is a deliberate step here, not left at the default 0.5.

## 2. Data

| | |
|---|---|
| Source | [Kaggle: `shivamb/real-or-fake-fake-jobposting-prediction`](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (the EMSCAD dataset) |
| Size | 17,880 postings × 18 columns, ~50 MB |
| Target | `fraudulent` — 1 = fake (~866 rows, **4.8%**), 0 = real |
| Licence | Public, published for research by the University of the Aegean |

Two kinds of feature in one table:

* **Text** — `title`, `company_profile`, `description`, `requirements`, `benefits`.
  Contains HTML markup, and emails/URLs/phones are masked as `#EMAIL_...#`,
  `#URL_...#`, `#PHONE_...#`.
* **Tabular** — `telecommuting`, `has_company_logo`, `has_questions`,
  `employment_type`, `required_experience`, `required_education`, `industry`,
  `function`, `location`, `salary_range`, `department`.

Missing values are not noise here. `company_profile` is absent from a large share
of the fraudulent postings and `has_company_logo` is one of the strongest single
predictors in the whole dataset — so absence is encoded explicitly as `*_is_missing`
features rather than imputed away. See `notebooks/01_eda.ipynb`.

## 3. Evaluation approach & chosen metric

**Headline metric: average precision (PR-AUC).**

With 4.8% positives, a model that predicts "real" for every posting scores **~95%
accuracy** and catches **zero** scams. Accuracy is not a weak metric here; it is a
misleading one. ROC-AUC is also flattering under heavy imbalance, because a large
true-negative count keeps the false-positive rate low no matter what.

Average precision summarises the precision/recall curve across every threshold and
only rewards ranking the rare positive class well. A random ranker scores AP ≈ 0.048
(the positive rate) — that is the real floor.

Reported alongside: **ROC-AUC** (comparability with other published work) and
**precision / recall / F1 for the minority class** at the chosen threshold.

**Protocol — the same for every model:**

1. Split once, stratified, `SEED = 42` → **70% train / 15% val / 15% test**, written
   to `data/processed/`. Every model, local or Colab, reads those exact files.
2. Train on train.
3. **Tune the decision threshold on validation** (maximise F1; `tune_threshold(beta=2)`
   is available when recall matters more). Never leave it at 0.5.
4. Report on test **once**, at that frozen threshold. Test never informs a choice.

Every run appends to [`reports/experiments.csv`](reports/experiments.csv).

## 4. Approach & tools

| # | Model | Idea |
|---|---|---|
| — | **Baseline A** — most frequent class | Demonstrates the accuracy trap |
| — | **Baseline B** — TF-IDF → Logistic Regression | The bar the real models must clear |
| 1 | **LightGBM** | TF-IDF → TruncatedSVD (200 dims) + one-hot categoricals + hand-crafted features; `scale_pos_weight` for the imbalance. The only model that sees text *and* metadata together. |
| 2 | **DistilBERT (fine-tuned)** | `distilbert-base-uncased`, 256 tokens, 3 epochs, class-weighted loss. Needs a GPU → separate Colab notebook. |
| 3 | **Sentence-Transformers (frozen)** | `all-MiniLM-L6-v2` embeddings, encoder frozen, LogReg / LightGBM head. Cheap middle ground; runs on CPU. |
| ✳ | **LLM few-shot** *(optional, not graded)* | `src/llm_baseline.py`. Stretch experiment, needs an API key. Not wired into any notebook. |

**Hand-crafted features** (`src/features.py`): missingness flags per field; character
and word counts per field; CAPS ratio, digit ratio, exclamation/question counts;
email / URL / phone / money regex hits; country and location-detail depth parsed out
of `location`; low/high/span parsed out of `salary_range`.

**Stack:** Python 3.9, pandas, scikit-learn, LightGBM, PyTorch, Transformers,
sentence-transformers, SHAP, matplotlib/seaborn.

## 5. Results

Test split. Each model's threshold is tuned on validation with **F2** (recall-leaning,
since a missed scam costs more than a false alarm) and frozen before scoring test, so
precision/recall/F1 are read at that one operating point. Rows are ranked by AP, the
primary metric. Full log: [`reports/experiments.csv`](reports/experiments.csv).

| Model | AP (PR-AUC) ↑ | ROC-AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|
| LightGBM + SVD + metadata | **0.9211** | 0.9918 | 0.7055 | 0.8984 | 0.7904 | 0.0948 |
| Baseline: TF-IDF + LogReg | 0.9095 | 0.9912 | 0.7097 | 0.8594 | 0.7774 | 0.4481 |
| LightGBM + SVD + metadata (Optuna-tuned) | 0.9037 | 0.9883 | 0.7208 | 0.8672 | 0.7872 | 0.1159 |
| DistilBERT fine-tuned | 0.8877 | 0.9847 | 0.8030 | 0.8281 | 0.8154 | 0.0208 |
| MiniLM frozen + LightGBM | 0.8360 | 0.9798 | 0.5909 | 0.8125 | 0.6842 | 0.0048 |
| MiniLM frozen + LogReg | 0.5454 | 0.9466 | 0.4485 | 0.6797 | 0.5404 | 0.7342 |
| Baseline: most frequent *(trivial reference)* | 0.0485 | 0.500 | 0.00 | 0.00 | 0.00 | – |

Figures land in `reports/figures/`: target imbalance, missingness by class, length
distributions, top n-grams and word clouds, fraud rate by category, PR/ROC curves,
confusion matrices, LightGBM feature importance and SHAP summary, and a combined
PR-curve comparison across all models on identical test rows.

## 6. Conclusions

> **Placeholder — write after the runs.** Points worth answering:
>
> * Which model wins on AP, and is the margin over the TF-IDF baseline worth its cost?
> * Does metadata beat wording, or the reverse? (LightGBM feature importance answers this.)
> * Do the transformers justify the compute over TF-IDF on ~18k rows?
> * At 90% precision, what recall do we get — and is that a shippable product?
> * **Error analysis:** which fakes slip through? Early expectation is the polished
>   ones that filled in every field — the missingness signal has nothing to bite on.
> * What would help most next: more data, better features, or an ensemble?

## 7. Target roles & skills

To ground this project in the real job market, I analyzed three real vacancies
that map my transition path — from data & BI analysis toward machine learning
engineering. Comparing them honestly shows what my background already covers,
what this project adds, and where I still have room to grow.

A small meta-note: this is a project about detecting fake **job postings** — and
I built the case for my own profile around real job postings too.

### My starting point

I'm not coming to ML from zero. I have 8+ years in applied analytics (pharma
sales) and 1+ year of hands-on BI in telecom (IP telephony): I write complex SQL,
build dashboards (Apache Superset, Power BI, QlikView), and validate data
end-to-end. That telecom/CRM background maps directly onto role 1 below. What my
CV did *not* yet show was a complete, from-scratch ML pipeline — and that gap is
exactly what this project fills.

### The three roles

**1. Data Analyst, CRM (Kyivstar, telecom) ([job posting](https://djinni.co/jobs/811531-data-analyst/)) — closest to my current profile.**
A CRM-analytics role that explicitly wants someone who "writes predictive models
using Machine Learning, knows the criteria for evaluating their quality, and runs
pilot testing" — alongside strong SQL and reporting. I cover the analytics core
(SQL, reporting, Excel/BI) from work experience, and this project is direct proof
of the ML half: churn/campaign prioritization is the same family of problems as
fraud scoring — imbalanced binary classification with a defensible metric and
threshold tuning. The domain lines up too: the role is telecom/CRM, and I already
have 1+ year of BI in IP telephony. Strongest overall match, because I meet both
the analytics core and the ML expectation.

**2. Strong Junior ML Engineer (Artellence) ([job posting](https://djinni.co/jobs/830060-strong-junior-machine-learning-engineer/)) — where I'm heading.**
The closest match to what this project *is*. It centers on "NLP and Tabular Data
ML tasks" — exactly this project's hybrid setup — plus feature engineering,
clean maintainable Python, end-to-end model development, and Git. All shown
directly. The remaining gap is production tooling (Docker).

**3. Strong Junior Data Scientist (Dataforest) ([job posting](https://djinni.co/jobs/815160-strong-junior-data-scientist/)) — where the market is pulling.**
A GenAI-first role (LLM, LangChain, RAG, prompt engineering). The project covers
the classical ML/NLP foundation these roles still list (text classification,
predictive modeling, Pandas), but not the LLM-agent stack. I include it to be
honest about a real market shift, not to claim skills I don't have.

### Most-requested skills and where I cover them

Across all three postings, the recurring core — and how it's covered by
experience (E) and/or this project (P):

| Skill | Covered by | Evidence |
|---|---|---|
| Python + ML libraries | E + P | pandas/NumPy at work; sklearn, LightGBM, DistilBERT here |
| SQL (incl. complex queries) | E | live SQL in telecom & pharma analytics roles (optimization, MS SQL Server) |
| BI / dashboards | E | Apache Superset, Power BI, QlikView, advanced Excel |
| Feature engineering | P | `src/features.py` — missingness flags, text-length, CAPS ratio, regex |
| Imbalanced data & metric choice | P | ~4.8% positives → PR-AUC primary (not accuracy) + threshold tuning |
| Evaluation (precision/recall/F1, ROC/AUC) | P | `src/evaluation.py` — train and val evaluated separately |
| Text + tabular (hybrid) data | P | TF-IDF + SVD on text, encoded categoricals, engineered features |
| Data validation / cleaning | E + P | end-to-end validation at work; dedup + EDA cleaning here |
| Clean modular code + Git | E + P | GitHub profile; small functions, docstrings, structured repo |

### Honest gaps (my growth roadmap)

- **Docker / deployment / MLOps** — asked by the MLE role; this project ends at
  evaluation, not production serving. My clearest next step.
- **Deep-learning frameworks at depth** — I use PyTorch via DistilBERT here, but
  wouldn't yet claim strong from-scratch DL architecture experience.
- **LLM-agent stack (RAG, LangChain)** — the market direction in the DS role; the
  optional LLM few-shot module (`src/llm_baseline.py`) is a stub, not evidence.

### What this project demonstrates

Set against these roles, the project is the missing piece of an analyst's move
into ML. My CV already shows the data foundation — SQL, BI, validation,
stakeholder communication. What it lacked was proof I can take a messy,
imbalanced, text+tabular dataset and carry it end-to-end: EDA, feature
engineering, several models, a defensible metric, threshold tuning, and error
analysis — as clean, reproducible code. That's what this repository is.

### A market insight worth noting

All three "junior" roles expected roughly 1+ year of experience plus Git and, in
most cases, SQL. Entry-level ML in this market is not truly entry-level — which
is why a mid-career analyst with real SQL/BI experience plus a solid ML portfolio
project is a more credible profile than either half alone.

## 8. Installation & usage

```bash
git clone <your-repo-url>
cd fake-job-postings-detection

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

**Step 1 — download the data.** Needs Kaggle credentials
(`~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` / `KAGGLE_KEY`; see `.env.example`).

```bash
python -m src.data          # -> data/raw/fake_job_postings.csv
```

**Step 2 — build the splits.** De-duplicates, cleans, builds features, splits
stratified with `SEED = 42`. Run this before any notebook.

```bash
python -m src.preprocessing # -> data/processed/{train,val,test}.csv
```

**Step 3 — run the notebooks in order.**

```bash
jupyter lab
```

| Notebook | What it does | Needs |
|---|---|---|
| `01_eda.ipynb` | 6 figures + insights, train split only | CPU, ~1 min |
| `02_baseline.ipynb` | Most-frequent + TF-IDF/LogReg | CPU, ~2 min |
| `03_models_lightgbm.ipynb` | LightGBM, importance, SHAP, error analysis | CPU, ~5 min |
| `04_sentence_transformers.ipynb` | MiniLM embeddings + head; model comparison | CPU ~15 min (GPU ~1 min); embeddings cached |
| `05_distilbert_colab.ipynb` | Fine-tuned DistilBERT | **Colab GPU**, ~15 min |

**Step 4 — DistilBERT in Colab.** Upload `05_distilbert_colab.ipynb`, set
*Runtime → Change runtime type → T4 GPU*, and upload `data/processed/*.csv` when the
notebook asks (or mount Drive). It exports `predictions_distilbert_test.csv` and two
rows for `reports/experiments.csv`. Drop the predictions file into `models/` and
re-run the comparison cell in notebook 04 to put DistilBERT on the shared PR curve.

**Optional — the LLM experiment.** Costs money, needs a key, not part of the grade:

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY
pip install anthropic python-dotenv
python -m src.llm_baseline  # scores 20 validation postings
```

### Reproducibility

`SEED = 42` everywhere — `config.set_seed()` covers `random`, `numpy` and `torch`,
and the split, LightGBM, the SVD and the Colab notebook all take the same seed.
The splits are written to disk precisely so that the comparison is honest: every
model is scored on byte-identical test rows.

## 9. Requirements

Python 3.9+. Versions are pinned in [`requirements.txt`](requirements.txt) as a
compatible set — notably `numpy==1.26.4`, because numpy 2.x breaks the binary ABI of
packages built against 1.x.

**Do not `pip install -r requirements.txt` in Colab.** Colab ships a working
torch/numpy/CUDA combination and pinning on top of it is what triggers
`numpy.dtype size changed`. Notebook 05 installs only what is missing.

`torch` (~2 GB) is only needed for notebooks 04 and 05; notebooks 01–03 run on the
core dependencies alone.

## Project structure

```
fake-job-postings-detection/
├── data/
│   ├── raw/                     # downloaded CSV            (gitignored)
│   └── processed/               # train/val/test splits     (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_models_lightgbm.ipynb
│   ├── 04_sentence_transformers.ipynb
│   └── 05_distilbert_colab.ipynb
├── src/
│   ├── config.py                # SEED, paths, columns, hyper-parameters
│   ├── data.py                  # kagglehub download + load + de-duplicate
│   ├── preprocessing.py         # clean, split, save, fit/transform
│   ├── features.py              # hand-crafted features
│   ├── models.py                # train / predict per model
│   ├── evaluation.py            # metrics, threshold tuning, plots, experiment log
│   └── llm_baseline.py          # optional LLM few-shot
├── models/                      # saved models, embeddings   (gitignored)
└── reports/
    ├── figures/                 # generated plots            (gitignored)
    └── experiments.csv          # the experiment log         (tracked)
```
