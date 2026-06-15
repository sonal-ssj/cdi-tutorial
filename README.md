# Confidence-Driven Inference Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sonal-ssj/cdi-tutorial/blob/main/tutorial_version_1_adaptive.ipynb)

Click the badge to open the tutorial in Google Colab. The notebook's first
setup cell clones this repo and installs dependencies automatically — just run
it, then add your `GROQ_API_KEY` in the Colab **Secrets** panel.

## What this tutorial does
The Jupyter notebook walks you through an **end-to-end, automated human-AI annotation pipeline** based on **Confidence-Driven Inference (CDI)**. We collect human annotations via **Prolific** or **Amazon Mechanical Turk**, and LLM annotations through either **Groq** or **OpenAI** (selectable with a single `PROVIDER` switch). The example is based on the method introduced in:

Can Unconfident LLM Annotations Be Used for Confident Conclusions? Kristina Gligorić*, Tijana Zrnic*, Cinoo Lee*, Emmanuel Candès, and Dan Jurafsky. NAACL, 2025.  
https://aclanthology.org/2025.naacl-long.179/#


The goal is to estimate a target statistic about a text corpus while **minimizing costly human labels** by:

1. **Obtaining cheap-but-noisy labels from an LLM** together with its confidence scores.  
2. **Collecting a small sample of human labels** to calibrate.  
3. **Iteratively sampling the most informative texts** for additional human annotation using the CDI sampling rule.  
4. **Computing the final point estimate & bootstrap confidence interval** once the annotation budget is exhausted.

We focus on annotating texts for **politeness** and showcase the estimation of **two target statistics**:
1. $mean(H)$: prevalence of politeness, i.e., the fraction of texts in the corpus that are polite.
2. $\beta_{hedge}$: the impact of linguistic features of hedging ($X$) on the perceived politeness ($H$), estimated with a logistic regression.

Although the example focuses on detecting *politeness* and estimating these two target statistics, you can adapt the flow to any text classification task and any other target statistic.

---

## Notebook outline

| Section | Purpose |
|---------|---------|
| **Colab setup** | Installs the LLM client libraries (`openai`, `python-dotenv`). Run this first on Google Colab; skip it when running locally from `requirements.txt`. |
| **Import libraries** | Loads scientific stack (`numpy`, `scipy`, `pandas`, `tqdm`, Qualtrics/Prolific/MTURK helpers, and `openai` for LLM calls). |
| **Parameter blocks** | Separate cells let you tune *CDI hyper-parameters*, *LLM sampling settings* (including the `PROVIDER` switch for Groq vs OpenAI and the model name), and *human‑annotation settings* (batch size, budget, etc.). |
| **Step&nbsp;1 – LLM annotation** | Loads a CSV of raw texts (`data/politeness_dataset.csv`), queries the LLM for a label & confidence for each row, and stores results in the working `data` frame. |
| **Step&nbsp;2 – Initial human labels** | Publishes the first batch of texts to Prolific or MTURK, waits for responses, and merges them back into `data`. Initialize the sampling rule to obtain per‑item selection probabilities. |
| **Step&nbsp;3 – Iterative sampling loop** | For each batch: choose texts with highest CDI scores → post new survey → ingest responses → update CDI state. |
| **Step&nbsp;4 – Estimation** | After the last batch, calculate the CDI estimator and a bootstrap 90 % confidence interval. Timing information for the whole pipeline is also logged. |


---

## Repository structure

```
project/
├── tutorial_version_1_adaptive.ipynb
├── tutorial_version_2_non-adaptive.ipynb
├── groq_api_colab.ipynb    # standalone guide for getting/using a Groq API key
├── data/
│   └── politeness_dataset.csv
├── utils/                  # helper modules
│   ├── config.py           # loads LLM API keys (Colab Secrets / .env / credentials.txt)
│   ├── llms.py             # provider-agnostic LLM client (Groq or OpenAI)
│   └── ...                 # survey API wrappers, inference modules
├── requirements.txt
├── credentials.txt         # human-annotation keys (AWS/Qualtrics/Prolific); git-ignored
├── .env.example            # template for LLM API keys; copy to .env and fill in
└── README.md
```

---

## Tutorial Notebooks

This repository includes five versions of a tutorial:

1. **`tutorial_version_1_adaptive.ipynb`**  
   Adaptive label collection using high-level functions and pre-collected human labels (for demonstration purposes).

2. **`tutorial_version_2_non-adaptive.ipynb`**  
   Non-adaptive label collection on a random sample of texts, using pre-collected human labels.

---

## Setup & run

You can run the tutorial **locally** or on **Google Colab**. Both need an API
key — see [API keys](#api-keys-env-and-colab-secrets) below for details.

### Option A — Local (VSCode)

1. **Python ≥ 3.9.**
2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Add your API key:** copy the template and fill it in (see [API keys](#api-keys-env-and-colab-secrets)):

   ```bash
   cp .env.example .env
   # edit .env and set GROQ_API_KEY (or OPENAI_API_KEY)
   ```

4. **Open the notebook in VSCode** and **select the venv as the kernel** — use the
   kernel picker in the top-right and choose the `./venv` interpreter. Then run
   the cells top-to-bottom. You can skip the **Colab setup** cell at the top —
   it's only needed on Colab.

### Option B — Google Colab

1. Click the **Open In Colab** badge at the top of this README (or use **File ▸ Open notebook ▸ GitHub** in Colab).
2. Run the **Setup** cell at the top — it clones this repo (so `utils/` and `data/` are available) and installs `requirements.txt`.
3. Add your API key via the **Secrets** panel (see [API keys](#api-keys-env-and-colab-secrets)).
4. Run the cells top-to-bottom.

---

## API keys (`.env` and Colab Secrets)

The LLM API key is resolved by `utils/config.py` (`load_llm_api_key(PROVIDER)`),
which looks in three places, in order:

1. **Colab Secrets** — when running on Google Colab.
2. **`.env`** file — when running locally (read via `python-dotenv`).
3. **`credentials.txt`** — the repository's original convention (fallback).

This applies to **all** keys — the LLM key (`GROQ_API_KEY` for Groq,
`OPENAI_API_KEY` for OpenAI) **and** the human-annotation keys
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `QUALTRICS_API_KEY`,
`QUALTRICS_API_URL`, `PROLIFIC_API_KEY`). Get a Groq key at
[console.groq.com](https://console.groq.com) (free tier, no card needed); see
`groq_api_colab.ipynb` for a step-by-step walkthrough.

**Local (`.env`):**

```bash
cp .env.example .env
# then edit .env and fill in the key(s) you need, e.g.:
#   GROQ_API_KEY=gsk_...
#   PROLIFIC_API_KEY=...
```

`.env` is git-ignored, so your keys are never committed. The human-annotation
keys are only needed when `COLLECT_HUMAN = True`; leave them blank otherwise.

**Google Colab (Secrets):**

1. Open the **Secrets** panel (key icon in the left sidebar).
2. Add a secret named exactly `GROQ_API_KEY` (or any other key name above) and paste its value.
3. Toggle **Notebook access** on for that secret.

> `credentials.txt` still works as a final fallback for backwards compatibility,
> but `.env` (or Colab Secrets) is now the recommended place for all keys.

---

## Notebook settings (both options)

Once the notebook is open and the cells are running, these settings control its behavior:

- **Choosing the LLM backend:** Set `PROVIDER = "groq"` or `PROVIDER = "openai"`
  in the LLM-annotation parameter cell. The model and API key follow automatically
  (edit `MODEL_BY_PROVIDER` to change models). Groq deprecates models periodically —
  if a call fails with an invalid-model error, run
  `get_client(PROVIDER, LLM_API_KEY).models.list()` to see what's live.
- **Live vs. pre-collected:** Set `COLLECT_LLM = True` to query the LLM live, or
  `False` to load the pre-collected `gpt-4o` labels shipped in the dataset.
- **Dry‑run mode:** Keep `COLLECT_LLM = False` and `COLLECT_HUMAN = False` in the
  parameter cells to skip all external API calls while you familiarize yourself
  with the flow.  

---

## Expected outputs

- Console log showing batch progression and total wall‑clock time.  
- Printed estimate and 90 % CI for the target metric.

---

## Adapting the tutorial

- Swap in your own dataset with **`Text`** and **feature column(s)**.  
- Update the `mapping_categories` dict to match your label set.  
- Tweak `burnin_steps`, `batch_size`, and `budget` to suit annotation cost constraints.  
- Plug in a different LLM prompt, provider (`PROVIDER`), or model name to target alternative tasks.

---

## References

Can Unconfident LLM Annotations Be Used for Confident Conclusions? Kristina Gligorić*, Tijana Zrnic*, Cinoo Lee*, Emmanuel Candès, and Dan Jurafsky. NAACL, 2025.  

This tutorial is presented as part of:
1. The International Conference on Computational Social Science (IC2S2) tutorial on Bridging Human and LLM Annotations for Statistically Valid Computational Social Science. URL: https://sites.google.com/view/ic2s2-bridging-human/home
2. The Summer Institute in Computational Social Science at Stanford, 2025

