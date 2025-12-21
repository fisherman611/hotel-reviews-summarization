# LLM Evaluation for Hotel Review Summaries

This directory contains scripts for evaluating hotel review summaries using Large Language Models (LLMs), primarily following the **G-Eval** framework.

## Usage

You can run the evaluator from the command line:

### Evaluate a Full Dataset
```bash
python metrics/llm_evaluation/llm_evaluator.py --input_file path/to/your/summaries.json --output path/to/your/results.json
```

### Evaluate a Single Sample (for testing)
```bash
python metrics/llm_evaluation/llm_evaluator.py --input_file path/to/your/summaries.json --sample 0
```

### Key Arguments

-   `--input_file`: (Required) Path to the JSON file containing the summaries and context.
-   `--output` or `-o`: Path to save the evaluation results in JSON format.
-   `--summary_type`: Choose between `generated` (default) or `golden` summaries to evaluate.
-   `--temperature` or `-t`: LLM temperature (default: 0 for consistency).
-   `--sample`: Integer index to evaluate only one specific sample from the input file.
