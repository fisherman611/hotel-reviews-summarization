# **Hotel Reviews Summarization**

This repository contains the implementation of an aspect-based summarization system designed to generate concise summaries from hotel reviews, developed as part of a Web Mining project. The project explores three summarization paradigms: **Extractive Summarization**, **Abstractive Summarization**, and **Hybrid Extractive-Abstractive Summarization** with polarity analysis.

## **Project Overview**

This project aims to develop a system for automatically summarizing hotel reviews by identifying key aspects (rooms, location, service, cleanliness, building, food) and generating aspect-specific summaries. The system tackles challenges such as aspect extraction, sentiment polarity classification, and generating coherent summaries from multiple reviews. It compares three approaches, evaluating their performance using metrics such as BLEU, ROUGE, METEOR, and BERTScore.

## **Dataset**

The dataset used is the **SPACE** (Semantic-based Pre-training for Aspect-based Classification and Extraction) dataset, consisting of hotel reviews with aspect-specific summaries. Key characteristics:

* **Aspects Covered:** Rooms, Location, Service, Cleanliness, Building, and Food.
* **Data Structure:** Each entity (hotel) contains multiple reviews with sentences and corresponding gold-standard aspect-specific summaries.
* **Format:** JSON files with review sentences and ground-truth summaries for evaluation.

Download from this repo: https://github.com/stangelid/qt

Place the dataset files (`space_summ.json`, `space_train.json`, `space_summ_split.txt`) in the `data/` directory.

## **Methods and Models**

### **Pre-processing**

* **Sentence Segmentation:** Reviews are split into individual sentences for aspect-based analysis.
* **Aspect Classification:** Each sentence is classified into one or more of six predefined aspects using zero-shot classification.
* **Polarity Classification:** (Hybrid method only) Sentences are classified as positive, neutral, or negative.

### **Models**

**1. Baseline: Extractive Summarization**

* **Pipeline:**
  1. **Aspect Classification:** Uses `facebook/bart-large-mnli` for zero-shot aspect classification of review sentences.
  2. **Sentence Selection:** Employs `sentence-transformers/all-mpnet-base-v2` to select the top-K most representative sentences per aspect using semantic similarity.
  3. **Extractive Summary:** Concatenates selected sentences to form aspect-specific summaries.

* **Strengths:** Simple, interpretable, preserves original wording.
* **Weaknesses:** May produce redundant or less coherent summaries; limited to existing sentences.

**2. Finetuned Abstractive Summarization**

* **Pipeline:**
  1. **Aspect Classification:** Same as baseline, using `facebook/bart-large-mnli`.
  2. **Abstractive Summarization:** Uses `Qwen/Qwen2.5-0.5B-Instruct`, a finetuned language model, to generate fluent aspect-specific summaries from grouped sentences.

* **Key Components:** 
  - Prompt-based generation with aspect-specific instructions.
  - Maximum 256 new tokens per summary.
  
* **Strengths:** Generates fluent, coherent summaries; can paraphrase and combine information.
* **Weaknesses:** May introduce hallucinations; requires more computational resources.

**3. Hybrid Extractive-Abstractive with Polarity Analysis**

* **Pipeline:**
  1. **Aspect Classification:** Uses `facebook/bart-large-mnli` for aspect detection.
  2. **Polarity Classification:** Uses `facebook/bart-large-mnli` for sentiment classification (positive/neutral/negative).
  3. **Sentence Selection:** Employs `sentence-transformers/all-mpnet-base-v2` to select top-K sentences per aspect-polarity pair.
  4. **Abstractive Summarization:** Uses `Qwen/Qwen2.5-0.5B-Instruct` to generate summaries that capture both aspects and sentiment nuances.

* **Key Innovation:** Combines extractive filtering (to reduce noise) with abstractive generation (for fluency), while explicitly modeling sentiment polarity.

* **Strengths:** Balances coverage and coherence; sentiment-aware summaries; reduces hallucination risk.
* **Weaknesses:** More complex pipeline; increased processing time.

## **Results**

| Model                          | BLEU   | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore-F1 |
|--------------------------------|--------|---------|---------|---------|--------|--------------|
| Baseline (Extractive)          | TBD    | TBD     | TBD     | TBD     | TBD    | TBD          |
| Finetuned Abstractive          | TBD    | TBD     | TBD     | TBD     | TBD    | TBD          |
| Hybrid Extractive-Abstractive  | TBD    | TBD     | TBD     | TBD     | TBD    | TBD          |

*Note: Run the evaluation pipelines on your test set to populate these results.*

**Key Observations:**
- **Hybrid Extractive-Abstractive** approach is expected to achieve the best balance between fluency and factual accuracy by combining the strengths of both paradigms.
- **Finetuned Abstractive** should excel in fluency and coherence but may risk hallucination without proper constraints.
- **Baseline Extractive** provides a strong, reliable baseline with high factual accuracy but may lack coherence.

## **Conclusion**

The project successfully developed and compared three approaches for aspect-based hotel review summarization. The hybrid extractive-abstractive method with polarity awareness represents a promising direction, leveraging sentence selection to maintain relevance while using neural generation for fluency. Future work could focus on:

- Expanding to other domains (restaurants, products, etc.).
- Incorporating cross-aspect coherence modeling for multi-aspect summaries.
- Exploring lightweight models for deployment in resource-constrained environments.
- Fine-tuning larger language models (e.g., Llama, GPT variants) for improved generation quality.
- Adding controllability features (summary length, formality, emphasis).

## **Installation**

Clone the repository:

```bash
git clone https://github.com/fisherman611/hotel-reviews-summarization.git
```

Navigate to the project directory:

```bash
cd hotel-reviews-summarization
```

Create and activate a virtual environment (recommended):

```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Linux/Mac:
source env/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## **Dataset Setup**

Download or prepare the SPACE dataset and place the following files in the `data/` directory:
- `space_summ.json`
- `space_train.json`
- `space_summ_splits.txt`

## **Usage**

### **1. Baseline (Extractive Summarization)**

Run the baseline extractive pipeline:

```bash
python methods/baseline/pipeline.py
```

Configuration can be modified in `methods/baseline/config.json`:
- `aspect_model`: Model for aspect classification
- `threshold`: Confidence threshold for aspect classification
- `selector_model`: Model for sentence similarity
- `k`: Number of sentences to select per aspect

Output: `outputs/baseline_summaries.json`

### **2. Finetuned Abstractive Summarization**

Run the abstractive summarization pipeline:

```bash
python methods/finetune_abstractive/pipeline.py
```

Configuration can be modified in `methods/finetune_abstractive/config.json`:
- `aspect_model`: Model for aspect classification
- `abstractive_model`: Model for summary generation
- `max_new_tokens`: Maximum length of generated summaries

Output: `outputs/finetune_abstractive_summaries.json`

### **3. Hybrid Extractive-Abstractive with Polarity**

Run the hybrid pipeline:

```bash
python methods/hybrid_extractive_abstractive/pipeline.py
```

Configuration can be modified in `methods/hybrid_extractive_abstractive/config.json`:
- `aspect_model`: Model for aspect classification
- `polarity_model`: Model for sentiment classification
- `selector_model`: Model for sentence similarity
- `k`: Number of sentences to select per aspect-polarity pair
- `abstractive_model`: Model for summary generation

Output: `outputs/hybrid_extractive_abstractive_summaries.json`

## **Evaluation**

To evaluate generated summaries against gold-standard references, use the metrics module:

```python
from metrics.metrics import compute_all_metrics_single

prediction = "Generated summary text"
references = ["Gold summary 1", "Gold summary 2"]

scores = compute_all_metrics_single(prediction, references, lang="en")
print(scores)
```

Available metrics:
- **ROUGE-1, ROUGE-2, ROUGE-L:** N-gram overlap metrics
- **BLEU:** Precision-focused n-gram metric
- **METEOR:** Alignment-based metric with synonym matching
- **BERTScore:** Semantic similarity using contextual embeddings

## **Project Structure**

```
hotel-reviews-summarization/
├── data/                           # Dataset files
│   ├── space_summ_splits.txt
│   ├── space_summ.json
│   └── space_train.json
├── methods/                        # Summarization approaches
│   ├── baseline/                   # Extractive baseline
│   │   ├── aspect_classifier.py
│   │   ├── aspect_sentences_selector.py
│   │   ├── aspect_extractive_summarizer.py
│   │   ├── config.json
│   │   └── pipeline.py
│   ├── finetune_abstractive/       # Abstractive approach
│   │   ├── aspect_classifier.py
│   │   ├── aspect_abstractive_summarizer.py
│   │   ├── config.json
│   │   └── pipeline.py
│   └── hybrid_extractive_abstractive/  # Hybrid approach
│       ├── aspect_classifier.py
│       ├── polarity_classifier.py
│       ├── aspect_polarity_sentences_selector.py
│       ├── aspect_abstractive_summarizer.py
│       ├── config.json
│       └── pipeline.py
├── metrics/                        # Evaluation metrics
│   └── metrics.py
├── utils/                          # Helper functions
│   └── helpers.py
├── outputs/                        # Generated summaries
├── notebooks/                      # Jupyter notebooks (optional)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## **References**

[1] S. Amplayo, S. Angelidis, and M. Lapata. "Aspect-Controllable Opinion Summarization." *Proceedings of EMNLP 2021*. URL: https://aclanthology.org/2021.emnlp-main.528/

[2] R. Angelidis and M. Lapata. "Summarizing Opinions: Aspect Extraction Meets Sentiment Prediction and They Are Both Weakly Supervised." *Proceedings of EMNLP 2018*. URL: https://aclanthology.org/D18-1403/

[3] M. Lewis et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *Proceedings of ACL 2020*. URL: https://aclanthology.org/2020.acl-main.703/

[4] N. Reimers and I. Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of EMNLP 2019*. URL: https://aclanthology.org/D19-1410/

[5] A. Yang et al. "Qwen2.5: A Party of Foundation Models." *Technical Report 2024*. URL: https://qwenlm.github.io/

[6] T. Zhang et al. "BERTScore: Evaluating Text Generation with BERT." *Proceedings of ICLR 2020*. URL: https://openreview.net/forum?id=SkeHuCVFDr

[7] C.-Y. Lin. "ROUGE: A Package for Automatic Evaluation of Summaries." *Text Summarization Branches Out, ACL 2004*. URL: https://aclanthology.org/W04-1013/

[8] K. Papineni et al. "BLEU: A Method for Automatic Evaluation of Machine Translation." *Proceedings of ACL 2002*. URL: https://aclanthology.org/P02-1040/

## **Contributors**
- [Lương Hữu Thành](https://github.com/fisherman611) - 20225458
- [Vũ Trung Thành](https://github.com/thanh309) - 20220066
- [Nguyễn Mậu Trung](https://github.com/Pearlcentt) - 20225534
- [Đoàn Anh Vũ](https://github.com/bluff-king) - 20225465

## **License**

This project is licensed under the [MIT License](LICENSE).
