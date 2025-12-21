# Hotel Review Summary Evaluation

This directory contains scripts for evaluating hotel review summaries using standard NLP metrics.

## Usage

You can run the evaluator from the command line:

### Evaluate a Full Dataset
```bash
python metrics/evaluator.py --input_file path/to/your/summaries.json --output path/to/your/results.json
```

### Key Arguments

-   `--input_file`: (Required) Path to the JSON file containing the generated summaries and golden references.
-   `--output` or `-o`: (Required) Path to save the evaluation results in JSON format.

### Metrics Computed

The script computes the following metrics for each hotel aspect (rooms, location, service, cleanliness, building, food):
-   **ROUGE** (1, 2, L)
-   **BLEU**
-   **METEOR**
-   **BERTScore** (Precision, Recall, F1)
