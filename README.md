# **Hotel Reviews Summarization**

This repository contains the implementation of an aspect-based summarization system designed to generate concise summaries from hotel reviews, developed as part of a Web Mining project. The project explores three summarization paradigms: **Extractive Summarization**, **Abstractive Summarization**, and **Hybrid Extractive-Abstractive Summarization** with polarity analysis.

## **Project Overview**

This project aims to develop a system for automatically summarizing hotel reviews by identifying key aspects (rooms, location, service, cleanliness, building, food) and generating aspect-specific summaries. The system tackles challenges such as aspect extraction, sentiment polarity classification, and generating coherent summaries from multiple reviews. It compares three approaches, evaluating their performance using both automatic metrics (BLEU, ROUGE, METEOR, BERTScore) and LLM-based human-aligned evaluation.

## **Dataset**

The dataset used is the **SPACE** (Semantic-based Pre-training for Aspect-based Classification and Extraction) dataset, a large-scale benchmark constructed from TripAdvisor hotel reviews:

* **Scale:** ~1.1 million user-generated reviews from over 11,000 hotels
* **Evaluation Set:** 1,050 human-written abstractive summaries covering 50 hotels
* **Aspects Covered:** Rooms, Location, Service, Cleanliness, Building, and Food
* **Task Complexity:** Each summary is generated from 100 reviews

Download from: [SPACE Dataset](https://github.com/stangelid/qt)

## **Methods and Models**

### **Pre-processing**

* **Sentence Segmentation:** Reviews are split into individual sentences for aspect-based analysis.
* **Aspect Classification:** Each sentence is classified using zero-shot classification with pretrained BART (`facebook/bart-large-mnli`) or DeBERTa models.
* **Polarity Classification:** (Hybrid method only) Sentences are classified as positive or negative.

### **Models**

**1. Baseline: Random Extractive Summarization**

Uses zero-shot aspect classification, then randomly selects top-K sentences per aspect.
* **Classifiers:** `facebook/bart-large-mnli`, `MoritzLaworski/DeBERTa-v3-base-mnli-fever-anli`
* **Threshold:** 0.5 confidence for aspect assignment
* **Selection:** K=5 sentences per aspect

**2. Fine-tuned Abstractive Summarization (FAS)**

Fine-tunes instruction-tuned LLMs using **QLoRA** (Quantized Low-Rank Adaptation) for efficient training.

* **Models:**
  - `google/gemma-3-1b-it` (270M effective params)
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `meta-llama/Llama-3.2-1B-Instruct`

* **Data Recipes:**
  - **Human-25:** 25 entities with human-annotated summaries (450 examples)
  - **Mixed:** 75 synthetic + 25 human entities (1,800 examples)
  - **Synthetic-100:** 100 entities with LLM-generated summaries (1,800 examples)

* **Training Configuration:** LoRA rank=16, α=32, batch size=8, learning rate=2e-4, 1 epoch

**3. Hybrid Extractive-Abstractive Framework (HEAF)**

A five-stage pipeline combining extractive filtering with abstractive generation:
1. **Aspect Classification:** Zero-shot classification using BART/DeBERTa
2. **Polarity Classification:** Sentiment classification (positive/negative)
3. **Sentence Selection:** Top-K=20 sentences per aspect-polarity pair using `sentence-transformers/all-mpnet-base-v2`
4. **Few-Shot Retrieval (Optional):** Hybrid sparse-dense retrieval with reranking
5. **Abstractive Summarization:** LLM-based generation with sentiment-aware prompts

## **Installation**

```bash
git clone https://github.com/fisherman611/hotel-reviews-summarization.git
cd hotel-reviews-summarization
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```

## **Usage**

### **1. Baseline (Extractive Summarization)**

```bash
python methods/baseline/pipeline.py
```
Output: `outputs/baseline/`

### **2. Fine-tuned Abstractive Summarization**

**Training:**
```bash
python methods/finetune_abstractive/train.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --recipe mixed
```

**Inference:**
```bash
python methods/finetune_abstractive/inference_pipeline.py
```
Output: `outputs/finetune_abstractive/`

### **3. Hybrid Extractive-Abstractive**

```bash
python methods/hybrid_extractive_abstractive/pipeline.py
```
Output: `outputs/hybrid_extractive_abstractive/`

## **Evaluation**

```python
from metrics.metrics import compute_all_metrics_single

scores = compute_all_metrics_single(prediction, references, lang="en")
```

**Automatic Metrics:** ROUGE-1/2/L, BLEU, METEOR, BERTScore

**LLM-based Human-Aligned Metrics:** Naturalness, Coherence, Engagingness, Groundedness (via `metrics/llm_evaluation/`)

## **Project Structure**

```
hotel-reviews-summarization/
├── data/                           # Dataset files (SPACE)
├── methods/
│   ├── baseline/                   # Random extractive baseline
│   ├── finetune_abstractive/       # QLoRA fine-tuning + inference
│   ├── hybrid_extractive_abstractive/  # 5-stage hybrid pipeline
│   └── teacher/                    # Teacher LLM summaries
├── metrics/
│   ├── metrics.py                  # Automatic evaluation metrics
│   └── llm_evaluation/             # Human-aligned LLM evaluation
├── outputs/                        # Generated summaries & models
├── results/                        # Evaluation results
├── notebooks/                      # Jupyter notebooks
└── requirements.txt
```

## **Contributors**
- [Lương Hữu Thành](https://github.com/fisherman611) - 20225458
- [Vũ Trung Thành](https://github.com/thanh309) - 20220066
- [Nguyễn Mậu Trung](https://github.com/Pearlcentt) - 20225534
- [Đoàn Anh Vũ](https://github.com/bluff-king) - 20225465

## **License**

This project is licensed under the [MIT License](LICENSE).
