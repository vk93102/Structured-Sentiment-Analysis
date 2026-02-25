<div align="center">

# 🧠 Structured Sentiment Analysis

### IIIT Delhi · Natural Language Processing Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A structured approach to opinion mining — extracting **who** expresses **what sentiment** toward **whom**, with a **polarity label**.*

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Methodology](#-methodology)
  - [1. Initial Processing & Tokenization](#1-initial-processing--tokenization)
  - [2. Tag Classification Layer (BIO Tagging)](#2-tag-classification-layer-bio-tagging)
  - [3. Conditional Random Field (CRF)](#3-conditional-random-field-crf)
  - [4. Polarity Classification](#4-polarity-classification)
  - [5. Structured Opinion Generation](#5-structured-opinion-generation)
- [Baseline System](#-baseline-system)
- [Features](#-features)
- [Datasets](#-datasets)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Further Work](#-further-work)

---

## 🎯 Problem Statement

**Structured Sentiment Analysis (SSA)** extends traditional sentiment analysis by extracting complete *sentiment graphs* from raw text. Instead of simply classifying a text as *positive* or *negative*, SSA identifies all the structured opinion tuples present in a sentence.

### Formal Definition

Given a text $T = O_1, O_2, \ldots, O_n$, the goal is to predict all opinion tuples of the form:

$$\mathcal{G} = \{(h, b, t, p)\}$$

Where each element is defined as:

| Symbol | Role | Description |
|--------|------|-------------|
| $h$ | **Holder** | The entity who expresses the opinion |
| $b$ | **Expression** | The polar expression (words conveying sentiment) |
| $t$ | **Target** | The entity toward which the opinion is directed |
| $p$ | **Polarity** | The sentiment label: `positive`, `negative`, or `neutral` |

### Illustrative Example

```
Sentence: "Even though the price is decent for Paris, I would not recommend this hotel."

Opinion Tuple 1:
  Holder     → "I"
  Expression → "would not recommend"
  Target     → "this hotel"
  Polarity   → NEGATIVE

Opinion Tuple 2:
  Holder     → (implicit)
  Expression → "decent"
  Target     → "the price"
  Polarity   → POSITIVE
```

> Unlike standard sentiment analysis, SSA captures **complex, overlapping, and multi-target opinions** within a single sentence — making it far more expressive and practically useful.

---

## 🏗️ Architecture

Our model follows a unified end-to-end architecture built on top of a pretrained BERT-based encoder, enhanced with a CRF decoding layer and a separate polarity head.

```
                    ┌──────────────────────────────────────────┐
                    │              Input Text                   │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │   [CLS]   w₁    w₂   ...   wₙ   [SEP]   │
                    │         Token Embedding Layer             │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │        Norwegian BERT Encoder             │
                    │      (NbAiLab / nb-bert-base)             │
                    │   Contextual Hidden State Representations │
                    └──────────┬────────────────────┬──────────┘
                               │                    │
               ┌───────────────▼──────┐   ┌─────────▼────────────┐
               │   CRF for BIO Tagging│   │  Polarity Head        │
               │                      │   │  ([CLS] token)        │
               │  B-Source  I-Source  │   │                       │
               │  B-Target  I-Target  │   │  ┌──────────────┐    │
               │  B-Polar   I-Polar   │   │  │   Positive   │    │
               │  O                   │   │  │   Negative   │    │
               └──────────┬───────────┘   │  │   Neutral    │    │
                          │               │  └──────────────┘    │
                          │               └─────────┬────────────┘
                          │                         │
               ┌──────────▼─────────────────────────▼──────────┐
               │         Structured Opinion Generation           │
               │                                                 │
               │   Merge CRF spans + polarity predictions       │
               │   Apply offset mapping & span filtering        │
               │                                                 │
               │  Output: (Source, Target, Expression, Polarity)│
               └─────────────────────────────────────────────────┘
```

### Model Components at a Glance

| Component | Description | Technology |
|-----------|-------------|------------|
| **Encoder** | Contextual representation of input tokens | RoBERTa / BERT (multilingual) |
| **BIO Tagger** | Sequence labeling for span identification | Linear Projection |
| **CRF Decoder** | Structured decoding with label constraints | Conditional Random Field |
| **Polarity Head** | Classifies sentiment of identified spans | Multi-layer Classifier on `[CLS]` |
| **Opinion Builder** | Assembles final structured output tuples | Post-processing pipeline |

---

## 🔬 Methodology

### 1. Initial Processing & Tokenization

- SSA extends basic sentiment analysis by extracting four key components: **Holder, Target, Expression, Polarity**
- We use a pre-trained **RoBERTa model** (`PolitAi Lab/nb-bert-base`, [Liu et al., 2019]) as our encoder
- The encoder transforms raw input text into rich **contextual embeddings** for each token
- We benchmarked RoBERTa with `bert-base` and `bert-medium` — **RoBERTa consistently outperforms** both

```python
# Encoder produces hidden states for every token position
hidden_states = roberta_encoder(input_ids, attention_mask)
# Shape: [batch_size, seq_len, hidden_dim]
```

---

### 2. Tag Classification Layer (BIO Tagging)

- A **linear projection layer** maps contextual embeddings to BIO tag logits
- We use the **BIO (Beginning-Inside-Outside)** tagging scheme to represent the boundaries of each opinion component

#### BIO Tag Space

```
O           → Outside any opinion component
B-Source    → Beginning of a Holder span
I-Source    → Inside a Holder span
B-Target    → Beginning of a Target span
I-Target    → Inside a Target span
B-Polar     → Beginning of a Polar Expression span
I-Polar     → Inside a Polar Expression span
```

#### Projection Formula

$$Z = W \cdot z + b$$

where $W \in \mathbb{R}^{t \times H}$ is the weight matrix, $z^{t,i} \in \mathbb{R}^H$ represents the hidden state at position $i$, and $t$ is the number of possible BIO tags.

$$\hat{Y} = \mathbb{R}^{N \times t}$$

---

### 3. Conditional Random Field (CRF)

After obtaining token-level hidden states from BERT/RoBERTa, a **CRF layer** is applied to decode BIO tag sequences in a globally-consistent manner.

- The CRF learns **transition weights** between adjacent tags
- This enforces structural constraints (e.g., `I-Source` cannot follow `B-Target`)
- Enables **span detection** while explicitly modeling tag dependencies

#### CRF Training Objective

$$\mathcal{L}_{CRF} = -\log\frac{\exp(\text{score}(\text{gold tags}))}{\sum_{\hat{y}} \exp(\text{score}(\hat{y}, \text{ valid tags}))}$$

> **Key Insight:** CRF outperforms a plain softmax decoder because it captures *inter-tag dependencies* critical for multi-span structured extraction.

---

### 4. Polarity Classification

- A **separate multi-layer classification head** takes the `[CLS]` token representation as input
- Predicts one of three sentiment labels:

```
Sentiment Labels:
  ✅ Positive   →  Favorable / approving sentiment
  ❌ Negative   →  Critical / disapproving sentiment
  ➖ Neutral    →  Factual / non-opinionated expression
```

- The model **struggles with ambiguous sentences** like:
  > *"The food was great, but the service was terrible."*
  
  where polarity is mixed at the span level — a key challenge that motivates future work on span-level polarity.

---

### 5. Structured Opinion Generation

In the final stage, we combine CRF-predicted spans with polarity predictions to produce the complete **structured opinion quadruple** `(holder, expression, target, polarity)`.

**Post-processing pipeline:**
1. Convert span boundaries to **character offsets** using BERT's offset mapping with adjacent token merging
2. **Filter short spans** (< 3 characters) to remove noise
3. Assemble final output as JSON-formatted opinion tuples

> ⚠️ This filtering step leads to a small loss (~a few percent) of true predictions.

---

## 🔁 Baseline System

As a comparison point, we implement a **Sequence Labeling + Relation Classification** pipeline using BiLSTM models.

### Pipeline Overview

```
Step 1: Span Extraction
        ├── BiLSTM → Extract Holders
        ├── BiLSTM → Extract Targets
        └── BiLSTM → Extract Polar Expressions

Step 2: Relation Prediction
        └── BiLSTM + Max Pooling
              ├── Full text representation
              ├── Holder / Target representation
              └── Expression representation
                    └── Concatenate → Linear → Sigmoid
                          → Binary: has_relation? (threshold = 0.5)

Step 3: Assemble tuples → (holder, target, expression, polarity)
```

> The baseline uses **GloVe / FastText embeddings** and trains three separate BiLSTMs, one per annotation type, followed by a relation prediction model.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌍 **Multilingual Support** | Works across Norwegian, English, Spanish, Catalan, and Basque |
| 🏷️ **BIO Sequence Labeling** | Precise span-level identification using structured tagging |
| 🔗 **CRF Decoding** | Globally-consistent tag sequence prediction with structural constraints |
| 🎭 **Polarity Classification** | 3-class (Positive / Negative / Neutral) sentiment head |
| 🧩 **Quadruple Extraction** | Complete `(holder, target, expression, polarity)` output |
| 📊 **Weighted F1 Evaluation** | Partial-overlap scoring using token-level Jaccard intersection |
| 🔄 **Cross-lingual Transfer** | Train on English, evaluate on low-resource target languages |
| 📦 **Modular Architecture** | Encoder, CRF, and classifier heads are independently configurable |

---

## 📦 Datasets

### Subtask 1 — Monolingual

| Dataset | Language | Domain |
|---------|----------|--------|
| `norec` | Norwegian | Professional reviews (multi-domain) |
| `opener_en` | English | Hotel reviews |
| `opener_es` | Spanish | Hotel reviews |
| `multibooked_ca` | Catalan | Hotel reviews |
| `multibooked_eu` | Basque | Hotel reviews |
| `darmstadt_unis` | English | University reviews (online) |
| `mpqa` | English | News (opinion annotations) |

### Subtask 2 — Cross-Lingual

Trained on high-resource English data, evaluated on:

| Test Dataset | Language |
|-------------|----------|
| `opener_es` | Spanish |
| `multibooked_ca` | Catalan |
| `multibooked_eu` | Basque |

### Data Format

Each dataset is in JSON format with the following schema:

```json
{
  "sent_id": "opener/en/hotel/english00164-6",
  "text": "Even though the price is decent for Paris, I would not recommend this hotel.",
  "opinions": [
    {
      "Source":           [["I"],                    ["44:45"]],
      "Target":           [["this hotel"],           ["66:76"]],
      "Polar_expression": [["would not recommend"],  ["46:65"]],
      "Polarity":         "negative",
      "Intensity":        "average"
    }
  ]
}
```

---

## 📊 Results

### Monolingual Performance

> Average across all 7 datasets

| Metric | Score |
|--------|-------|
| **SF1** (Sentiment F1) | **0.46** |
| **SP** (Sentiment Precision) | **0.62** |
| **SR** (Sentiment Recall) | **0.65** |

**Per-dataset Breakdown:**

| Dataset | SF1 | SP | SR |
|---------|-----|----|----|
| Opener_en | 0.41 | 0.37 | 0.47 |
| Opener_es | 0.35 | 0.33 | 0.38 |
| NoReC | 0.23 | 0.30 | 0.18 |
| Multibooked_ca | 0.57 | 0.53 | 0.63 |
| Multibooked_eu | 0.53 | 0.40 | 0.71 |
| darmstadt_unis | 0.55 | 0.58 | 0.00 |
| MPQA | 0.52 | 0.55 | 0.00 |

---

### Cross-Lingual Performance

> Average across 3 target language datasets

| Metric | Score |
|--------|-------|
| **SF1** (Sentiment F1) | **0.35** |
| **SP** (Sentiment Precision) | **0.85** |
| **SR** (Sentiment Recall) | **0.63** |

**Per-dataset Breakdown:**

| Dataset | SF1 | Precision | Recall |
|---------|-----|-----------|--------|
| Opener_es | 0.000 | 0.000 | 0.000 |
| Multibooked_ca | 0.481 | 0.461 | 0.503 |
| Multibooked_eu | 0.671 | 0.618 | 0.733 |

---

### 🔍 Key Observations

- 🟢 **BERT significantly improves** span extraction results over CRF baseline alone — especially for English and language-similar corpora
- 🟡 **`Multibooked_eu` performs best** in cross-lingual settings — likely due to the smaller dataset size and consistent hotel-review characteristics
- 🔴 **Complex/ambiguous expressions** (different polarity in different contexts) present a challenge across all datasets
- 📌 **Character-level BERT representations** outperform word-level representations as a comparison baseline

---

## 🚀 Installation

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.9
- CUDA (optional, for GPU acceleration)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Structured-Sentiment-Analysis-IIITD-NLP-PROJECT.git
cd Structured-Sentiment-Analysis-IIITD-NLP-PROJECT
```

### 2. Install Core Dependencies

```bash
pip install torch torchvision transformers
pip install nltk scikit-learn tqdm gensim
```

### 3. Install Baseline Dependencies

```bash
pip install -r baseline/sequence_labeling/requirements.txt
```

### 4. Install Data Processing Dependencies

```bash
pip install -r data/requirements.txt
```

### 5. Prepare External Datasets

**MPQA 2.0** — Download from the [MPQA website](http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/) and run:
```bash
cd data/mpqa && bash process_mpqa.sh
```

**Darmstadt Service Review Corpus** — Download from [TU Darmstadt](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2448) and run:
```bash
cd data/darmstadt_unis && bash process_darmstadt.sh
```

---

## 🛠️ Usage

### Evaluation

Run the official evaluation script on model predictions:

```bash
python evaluate.py <input_dir> <output_dir>
```

Where:
- `<input_dir>/res/` contains your `predictions.json` per dataset
- `<input_dir>/ref/data/` contains the gold test files

### Baseline — Training

```bash
# Train all BiLSTM baseline models across datasets
cd baseline/sequence_labeling
bash get_baselines.sh
```

### Baseline — Inference

```bash
# Run inference on a specific dataset and split
python baseline/sequence_labeling/inference.py \
    --DATADIR opener_en \
    --FILE dev.json
```

Output will be saved to:
```
baseline/sequence_labeling/saved_models/relation_prediction/<DATADIR>/prediction.json
```

### Predictions Format

Prediction files must match the gold data format. Each entry should look like:

```json
{
  "sent_id": "unique-sentence-id",
  "text": "Raw input sentence here.",
  "opinions": [
    {
      "Source":           [["holder text"],     ["start:end"]],
      "Target":           [["target text"],     ["start:end"]],
      "Polar_expression": [["expression text"], ["start:end"]],
      "Polarity":         "positive"
    }
  ]
}
```

---

## 📁 Project Structure

```
Structured-Sentiment-Analysis-IIITD-NLP-PROJECT/
│
├── 📄 evaluate.py                  ← Official evaluation script (SF1 / SP / SR)
│
├── 📁 baseline/
│   └── sequence_labeling/
│       ├── extraction_module.py    ← BiLSTM span extractor (Holder / Target / Expr)
│       ├── relation_prediction_module.py  ← BiLSTM relation classifier
│       ├── inference.py            ← End-to-end inference pipeline
│       ├── convert_to_bio.py       ← Convert JSON data to BIO format
│       ├── convert_to_rels.py      ← Convert predictions to relation pairs
│       ├── utils.py                ← Data loading & vocabulary utilities
│       ├── WordVecs.py             ← Pretrained word embedding loader
│       ├── get_baselines.sh        ← Script to train all baseline models
│       └── requirements.txt
│
├── 📁 data/
│   ├── norec/                      ← Norwegian multi-domain reviews
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   ├── opener_en/                  ← English hotel reviews
│   ├── opener_es/                  ← Spanish hotel reviews
│   ├── multibooked_ca/             ← Catalan hotel reviews
│   ├── multibooked_eu/             ← Basque hotel reviews
│   ├── mpqa/                       ← MPQA news corpus
│   ├── darmstadt_unis/             ← English university reviews
│   └── README.md                   ← Data format documentation
│
└── 📁 predictions/
    └── norec/
        └── predictions.json        ← Sample model predictions
```

---

## 🔭 Further Work

We identify several promising directions to build upon this work:

### 🕸️ Dependency Graph Parsing
In the future, we would likely move to a **dependency graph parsing approach** for structured sentiment — augmenting the token-level representation with their heads in a dependency tree ([Kurtz et al., 2020](https://arxiv.org/abs/2005.01450)). This allows richer relational reasoning between opinion components.

### 🌐 Multi-task & Cross-lingual Learning
- Exploring **multi-task learning** across languages to better leverage shared structure in multilingual sentiment graphs
- Joint training on all monolingual datasets with language-specific adapters

### 🔗 Advanced Graph Parsers
- Point Network ([Samuel & Straka, 2020](https://arxiv.org/abs/2009.02040)) — a strong graph parser for SSA
- PERIN — a permutation-invariant structured prediction framework

### 📐 Span-level Polarity
- Currently, polarity is predicted globally per sentence via the `[CLS]` token
- Moving to **span-level polarity** prediction would handle cases like *"The food was great, but the service was terrible"*

### 🤖 Serialized Large Language Models
- Explore whether large pre-trained models (e.g., GPT-4, LLaMA) can directly predict structured opinion tuples via **in-context learning or fine-tuning**, without an explicit CRF layer

---

## 📚 References

```
Barnes, J. et al. (2021). SemEval-2022 Task 10: Structured Sentiment Analysis.
  Proceedings of the 16th Workshop on Semantic Evaluation.

Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
  arXiv:1907.11692.

Kurtz, et al. (2020). Improving Low-Resource NMT through Relevance Based Linguistic Features.
  ACL 2020.

Samuel, D. & Straka, M. (2020). ÚFAL at MRP 2020: Permutation-Invariant Semantic Parsing.
  CoNLL 2020 Shared Task.
```

---

<div align="center">

Made with ❤️ at **IIIT Delhi** | NLP Course Project

*For questions or contributions, please open an issue or pull request.*

</div>
