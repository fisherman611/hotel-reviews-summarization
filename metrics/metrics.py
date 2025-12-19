from typing import List, Dict, Any
import evaluate

# Load metrics once at import time
_rouge = evaluate.load("rouge")
_bleu = evaluate.load("bleu")
_meteor = evaluate.load("meteor")
_bertscore = evaluate.load("bertscore")


def _prepare_single_pair(
    prediction: str,
    references: List[str],
):
    """
    Given a single prediction and a list of references, create
    parallel lists for evaluate: one (prediction, ref_i) pair per ref.
    """
    if not references:
        raise ValueError("references must be a non-empty list of strings")
    
    if not prediction or not prediction.strip():
        raise ValueError("prediction must be a non-empty string")

    predictions = [prediction] * len(references)
    return predictions, references


def compute_rouge_single(
    prediction: str,
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L for a single prediction string
    against a list of reference summaries. Scores are averaged across
    all (prediction, reference_i) pairs.
    """
    try:
        predictions, refs = _prepare_single_pair(prediction, references)

        scores = _rouge.compute(
            predictions=predictions,
            references=refs,
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

        return {
            "rouge1": float(scores["rouge1"]),
            "rouge2": float(scores["rouge2"]),
            "rougeL": float(scores["rougeL"]),
        }
    except (ZeroDivisionError, ValueError, KeyError, TypeError) as e:
        print(f"Warning: Error computing ROUGE scores: {e}. Returning zeros.")
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }


def compute_bleu_single(
    prediction: str,
    references: List[str],
) -> Dict[str, float]:
    """
    Compute BLEU for a single prediction string against multiple references.

    We evaluate (prediction, ref_i) as separate pairs and let the BLEU
    implementation aggregate at corpus level.
    """
    try:
        predictions, refs = _prepare_single_pair(prediction, references)
        # BLEU expects references as List[List[str]]
        wrapped_refs = [[r] for r in refs]

        scores = _bleu.compute(
            predictions=predictions,
            references=wrapped_refs,
        )

        return {
            "bleu": float(scores["bleu"]),
        }
    except (ZeroDivisionError, ValueError, KeyError, TypeError) as e:
        print(f"Warning: Error computing BLEU scores: {e}. Returning zeros.")
        return {
            "bleu": 0.0,
        }


def compute_meteor_single(
    prediction: str,
    references: List[str],
) -> Dict[str, float]:
    """
    Compute METEOR for a single prediction string against multiple references.
    We treat (prediction, ref_i) as separate pairs and average.
    """
    try:
        predictions, refs = _prepare_single_pair(prediction, references)
        scores = _meteor.compute(
            predictions=predictions,
            references=refs,
        )
        return {
            "meteor": float(scores["meteor"]),
        }
    except (ZeroDivisionError, ValueError, KeyError, TypeError) as e:
        print(f"Warning: Error computing METEOR scores: {e}. Returning zeros.")
        return {
            "meteor": 0.0,
        }


def compute_bertscore_single(
    prediction: str,
    references: List[str],
    lang: str = "en",
) -> Dict[str, float]:
    """
    Compute BERTScore P/R/F1 for a single prediction string against
    multiple reference summaries. We compare the same prediction to each
    reference separately and average P/R/F1 across references.
    """
    def _avg(xs):
        """Safely compute average, handling empty lists and division by zero."""
        if not xs or len(xs) == 0:
            return 0.0
        try:
            return float(sum(xs) / len(xs))
        except (ZeroDivisionError, TypeError, ValueError):
            return 0.0
    
    try:
        predictions, refs = _prepare_single_pair(prediction, references)

        scores = _bertscore.compute(
            predictions=predictions,
            references=refs,
            lang=lang,
        )

        precisions = scores["precision"]
        recalls = scores["recall"]
        f1s = scores["f1"]

        return {
            "bertscore_precision": _avg(precisions),
            "bertscore_recall": _avg(recalls),
            "bertscore_f1": _avg(f1s),
        }
    except (ZeroDivisionError, ValueError, KeyError, TypeError) as e:
        print(f"Warning: Error computing BERTScore: {e}. Returning zeros.")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }


def compute_all_metrics_single(
    prediction: str,
    references: List[str],
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute ROUGE-1/2/L, BLEU, METEOR, BERTScore
    for a single prediction string vs. a list of reference strings.
    
    Returns a dictionary of metrics. If any metric fails to compute,
    it will return 0.0 for that metric and print a warning.
    """
    metrics: Dict[str, Any] = {}
    
    # Each function has its own error handling, so these should be safe
    metrics.update(compute_rouge_single(prediction, references))
    metrics.update(compute_bleu_single(prediction, references))
    metrics.update(compute_meteor_single(prediction, references))
    metrics.update(compute_bertscore_single(prediction, references, lang=lang))
    
    return metrics


if __name__ == "__main__":
    pred = "The room was large and very clean, with comfortable beds."
    gold_refs = [
        "The room was spacious and very clean.",
        "Guests mention that the rooms are large and clean with comfy beds.",
    ]

    scores = compute_all_metrics_single(pred, gold_refs, lang="en")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")