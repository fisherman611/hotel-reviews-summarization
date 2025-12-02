import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

ASPECTS = ["rooms", "location", "service", "cleanliness", "building", "food"]


def group_sentences_by_aspect(entities, aspects=ASPECTS, output_file=None):
    """
    entities: list of entities with reviews, each review having:
        - "sentences": [...]
        - "sentence_aspects": [[aspect1, aspect2, ...], ...]
    """
    result = []
    for entity in entities:
        aspect_buckets = {aspect: set() for aspect in aspects}

        for review in entity.get("reviews", []):
            sentences = review.get("sentences", [])
            sentence_aspects = review.get("sentence_aspects", [])

            for sent, sent_aspects in zip(sentences, sentence_aspects):
                for aspect in sent_aspects:
                    if aspect in aspect_buckets:
                        aspect_buckets[aspect].add(sent)

        # convert sets to lists
        for aspect, _ in aspect_buckets.items():
            aspect_buckets[aspect] = list(aspect_buckets[aspect])

        grouped_entity = {
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "reviews": aspect_buckets,
        }

        # if original entity already has summaries, keep them
        if "summaries" in entity:
            grouped_entity["summaries"] = entity["summaries"]

        result.append(grouped_entity)

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    return result


def aspect_extractive_summarizer(topk_sentences: dict):
    """
    topk_sentences: dict[aspect -> list[str]]
    returns: dict[aspect -> str]
    """
    aspect_summaries = {}

    # BUGFIX: .items() not .item()
    for aspect, sentences in topk_sentences.items():
        if not sentences:
            aspect_summaries[aspect] = ""
            continue

        # simple concatenation; you can later swap for nicer formatting / re-writing
        aspect_summary = " ".join(sentences).strip()
        aspect_summaries[aspect] = aspect_summary

    return aspect_summaries
