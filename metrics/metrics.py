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


def compute_bleu_single(
    prediction: str,
    references: List[str],
) -> Dict[str, float]:
    """
    Compute BLEU for a single prediction string against multiple references.

    We evaluate (prediction, ref_i) as separate pairs and let the BLEU
    implementation aggregate at corpus level.
    """
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


def compute_meteor_single(
    prediction: str,
    references: List[str],
) -> Dict[str, float]:
    """
    Compute METEOR for a single prediction string against multiple references.
    We treat (prediction, ref_i) as separate pairs and average.
    """
    predictions, refs = _prepare_single_pair(prediction, references)
    scores = _meteor.compute(
        predictions=predictions,
        references=refs,
    )
    return {
        "meteor": float(scores["meteor"]),
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
    predictions, refs = _prepare_single_pair(prediction, references)

    scores = _bertscore.compute(
        predictions=predictions,
        references=refs,
        lang=lang,
    )

    precisions = scores["precision"]
    recalls = scores["recall"]
    f1s = scores["f1"]

    def _avg(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "bertscore_precision": _avg(precisions),
        "bertscore_recall": _avg(recalls),
        "bertscore_f1": _avg(f1s),
    }


def compute_all_metrics_single(
    prediction: str,
    references: List[str],
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute ROUGE-1/2/L, BLEU, METEOR, BERTScore
    for a single prediction string vs. a list of reference strings.
    """
    metrics: Dict[str, Any] = {}
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
