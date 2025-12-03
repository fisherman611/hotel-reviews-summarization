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

Place the dataset files (`space_summ.json`, `space_train.json`, `space_summ_splits.txt`, `test.json`) in the `data/` directory.
The content of `test.json`:
```json
[
  {
    "entity_id": "100597",
    "entity_name": "Doubletree by Hilton Seattle Airport",
    "reviews": [
      {
        "review_id": "UR59977476",
        "sentences": [
          "We stayed here on a lay over home from Cancun.",
          "It was great to have a comfortable bed and room on our final night of holidays.",
          "The kids loved the pool which was warmer than the ones at the resort in Cancun which we could not believe as we were in Seattle!",
          "The staff was friendly and we appreciated the cookies after a long flight when we were waiting to check inn.",
          "Just a nice touch!",
          "Shuttle was convenient and would definitely stay here again."
        ],
        "rating": 5
      },
      {
        "review_id": "UR116963461",
        "sentences": [
          "I reserved this room through an affiliate travel booking platform and got a great room rate.",
          "Aprox.",
          "75.00 (not including taxes) for a full day stay.",
          "Late check in was effortless and was greeted with a warmed cookie.",
          "I think it is fair to mention it costs to use internet in your room but for business travlers I suppose that is an expected cost.",
          "Great friendly staff took the edge off our long day of intercontinental travel.",
          "I would choose this hotel again as our stop over."
        ],
        "rating": 4
      },
      {
        "review_id": "UR115823712",
        "sentences": [
          "We flew into SEA TAC for a few days before our cruise and our travel agent recommended this hotel.",
          "The hotel was clean, beds were fine, hotel is located across the street from the air port, was within walking distance to a Denny's, Jack in the Box, and a steak place.",
          "Room was ready very early in the morning for us.",
          "The kids liked the pool and I was able to do laundry before the cruise.",
          "It had easy access to the light rail located at the airport (we took the DT shuttle over), and was next to an Enterprise car rental.",
          "Lots of conveniences close by.",
          "Downtown Seattle was about a 30 minute ride by light rail service (cost of $16 for the 4 of us) or $40 cab ride to the pier.",
          "Hope this helps."
        ],
        "rating": 4
      }
    ],
    "summaries": {
      "general": [
        "The staff are friendly and exceptional. Every room (lobby included) was very clean. They are spacious, very quiet, and come with a coffee maker. Though, the rooms are outdated in decor. The hotel itself is conveniently close to the airport and restaurants. There's a chocolate-chip cookie at arrival, and for the prices, the experience is a good value.",
        "Service was exceptional and the quality was great! The rooms are always clean, quiet and spacious with nicely appointed bathrooms. The location is across the street from the airport, was within walking distance to a Denny's and other restaurants. The hotel interior itself is a bit outdated, but the room we stayed was modern.",
        "All the staff was exceptionally helpful, courteous, and friendly, keeping the rooms clean and well-prepared. The interior of the hotel needs updating, but the rooms themselves were very spacious, modern, and comfortable to stay in. The hotel itself is conveniently located near the airport, a steak restaurant, fast food, and has a free shuttle service for broader access to Seattle."
      ],
      "rooms": [
        "The rooms are large and quite, you can't hear the planes taking off at the airport next door. The beds are comfortable and large. The bathrooms are mixed, some need cleaner doors and to be renovated, others seem clean and well appointed. The ice and vending machines are close. The coffee machine in the room is appreciated. The lighting was insufficient, and an old basement smell was present sometimes.",
        "While close to the airport, it was quiet because of thick windows. The beds were large and comfortable with lots of extra pillows. The bathrooms could use some refurbishment. Furnishings were complete with an ottoman, an easy chair, and a coffee maker. A balcony gives a great view of the surrounding city.",
        "This hotel features very comfortable and spacious rooms, with balcony, coffeemaker, comfortable beds and were well furnished. Some things that need work is the bad lighting, unkempt bathrooms and smell of mildew. All that being said, the rooms are very quiet even though the hotel is close to the airport."
      ],
      "location": [
        "It's a convenient location close to the airport, with shuttle service to and from the airport that runs every 15 minutes for 24 hours a day. The shuttle service is very good. It's so close you could even walk to the airport if you wanted. It's also in convenient walking distance of many restaurants.",
        "The airport was convenient to reach with the help of a speedy, twenty-four hour shuttle bus. Also located nearby, within walking distance, was a Denny's, a fast food joint, and a steak house.",
        "Within walking distance from the airport, this hotel's location is great. There is even a 24 hour shuttle that runs every 15 min that will take you to the airport or some near by places to eat like Denny's Jack in the Box and a steak place."
      ],
      "service": [
        "The staff is exceptionally friendly and helpful both at the front desk and the restaurant. Expect sweet welcoming gifts at your check-in.",
        "Helpful, courteous, warm staff helps with a wind down after traveling. There is also a chocolate chip cookie at check-in.",
        "Mostly the staff is extremely helpful and friendly, helping to take the stress out of traveling. The cookies given at check in were greatly appreciated."
      ],
      "cleanliness": [
        "The spacious hotel lobby and rooms are very clean, comfortable, and well-appointed.",
        "Although the hotel's architecture feels dated, the rooms and bathrooms are clean.",
        "Even thought there was a minor issue with gaining access to the room because of a faulty magnetic door, the room and bedding were clean and comfortable."
      ],
      "building": [
        "The historical hotel lobby were very attractive. The balcony had a great view of trees . The spa and heated pool is a kid-friendly area and also has wi-fi. There is even a laundry room available to the guests.",
        "Hotel with very nice lobby and relaxing spa/pool area with lounge and free wifi. The pool is big and kid-friendly. There is also a beautiful view of the trees from the balcony.",
        "Warm, beautiful, large pool for the family. Old fashioned interior but pleasant rooms, great balcony, and the view outside to the trees was relaxing."
      ],
      "food": [
        "The hotel restaurant's food was nicely presented, and sometimes good. However, sometimes it was bland and tasteless, and a bit pricey. The restaurant's clam chowder was good. The breakfast buffet isn't a bad deal for what you get. The fresh cookies given at check in were delicious.",
        "Food was well presented and some of it was tasty, if a little pricey, but the clam chowder at the restaurant and the breakfast buffet made the trip all the more worth it. Dave's Diner next door was also enjoyable.",
        "Although some of the food was bland and a little overpriced, the clam chowder was good. The staff even gave out these delicious freshly baked cookies int he reception area and the breakfast buffet is also a great value for what is offered."
      ]
    }
  }
]
```

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
