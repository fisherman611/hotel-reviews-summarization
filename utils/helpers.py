import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

ASPECTS = ["rooms", "location", "service", "cleanliness", "building", "food"]
POLARITIES = ["positive", "neutral", "negative"]


def group_sentences_by_aspect(entities, aspects=ASPECTS, output_file=None):
    """
    entities: list of entities with reviews, each review having:
        - "sentences": [...]
        - "sentence_aspects": [[aspect1, aspect2, ...], ...]
    
    Output: entities with:
        - "reviews": original reviews without sentence_aspects
        - "aspect_reviews": reviews with sentence_aspects
        - "grouped_reviews": sentences grouped by aspect
    """
    result = []
    for entity in entities:
        aspect_buckets = {aspect: set() for aspect in aspects}
        
        # Copy reviews to aspect_reviews and create clean reviews without sentence_aspects
        original_reviews = []
        aspect_reviews = entity.get("reviews", [])
        
        for review in aspect_reviews:
            # Create original review without sentence_aspects
            original_review = {k: v for k, v in review.items() if k != "sentence_aspects"}
            original_reviews.append(original_review)
            
            # Process sentence_aspects for grouping
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
            "reviews": original_reviews,
            "aspect_reviews": aspect_reviews,
            "grouped_reviews": aspect_buckets,
        }

        # if original entity already has summaries, keep them
        if "summaries" in entity:
            grouped_entity["summaries"] = entity["summaries"]

        result.append(grouped_entity)

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    return result

def group_sentences_by_aspect_polarity(entities, aspects=ASPECTS, output_file=None):
    """
    entities: list of entities with reviews, each review having:
        - "sentences": [...]
        - "sentence_aspects": [[aspect1, aspect2, ...], ...]
        - "sentence_polarity": ["positive"/"negative"/..., ...]
    
    Output format per entity:
    {
        "entity_id": ...,
        "reviews": original reviews without sentence_aspects/sentence_polarity,
        "aspect_reviews": reviews with sentence_aspects and sentence_polarity,
        "grouped_reviews": {
            "rooms": {
                "negative": [...],
                "positive": [...],
            },
            "location": {
                "negative": [...],
                "positive": [...],
            },
            ...
        },
        "summaries": ... (if present in input)
    }
    """
    result = []

    for entity in entities:
        # aspect -> {"negative": set(), "positive": set()}
        aspect_buckets = {
            aspect: {"negative": set(), "positive": set()}
            for aspect in aspects
        }
        
        # Copy reviews to aspect_reviews and create clean reviews
        original_reviews = []
        aspect_reviews = entity.get("reviews", [])

        for review in aspect_reviews:
            # Create original review without sentence_aspects and sentence_polarity
            original_review = {k: v for k, v in review.items() 
                             if k not in ("sentence_aspects", "sentence_polarity")}
            original_reviews.append(original_review)
            
            # Process sentence_aspects and sentence_polarity for grouping
            sentences = review.get("sentences", [])
            sentence_aspects = review.get("sentence_aspects", [])
            sentence_polarity = review.get("sentence_polarity", [])

            # iterate aligned by index
            for sent, sent_asps, polarity in zip(sentences, sentence_aspects, sentence_polarity):
                # only care about positive/negative
                if polarity not in ("positive", "negative"):
                    continue

                for aspect in sent_asps:
                    # if aspect not in predefined list, optionally add it dynamically
                    if aspect not in aspect_buckets:
                        aspect_buckets[aspect] = {"negative": set(), "positive": set()}
                    aspect_buckets[aspect][polarity].add(sent)

        # convert sets to lists
        for aspect, pol_dict in aspect_buckets.items():
            for pol in ("negative", "positive"):
                pol_dict[pol] = list(pol_dict[pol])

        grouped_entity = {
            "entity_id": entity["entity_id"],
            "reviews": original_reviews,
            "aspect_reviews": aspect_reviews,
            "grouped_reviews": aspect_buckets,
        }

        # keep summaries if present
        if "summaries" in entity:
            grouped_entity["summaries"] = entity["summaries"]

        result.append(grouped_entity)

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    return result

def reformat_output_for_llm_evaluation(sample, output_file=None):
    """
    Reformat model output summaries for LLM evaluation.
    
    Args:
        sample: a single entity dictionary containing:
            - entity_id
            - reviews: list of reviews with review_id and sentences
            - generated_summaries: dict of aspect -> summary
            - golden_summaries: dict of aspect -> list of summaries
        output_file: optional path to save the markdown output
    
    Returns:
        markdown formatted string
    """
    markdown_output = []
    
    entity_id = sample.get("entity_id", "")
    reviews = sample.get("reviews", [])
    generated_summaries = sample.get("generated_summaries", {})
    golden_summaries = sample.get("golden_summaries", {})
    
    # Format reviews section
    markdown_output.append(f"## Entity ID: {entity_id}\n")
    markdown_output.append("### Reviews:\n")
    for review in reviews:
        review_id = review.get("review_id", "")
        sentences = review.get("sentences", [])
        review_text = " ".join(sentences)
        markdown_output.append(f"- **{review_id}**: {review_text}\n")
    
    markdown_output.append("\n")
    
    # Get all aspects from generated_summaries
    aspects = [aspect for aspect in generated_summaries.keys() if aspect != "general"]
    
    # Format each aspect
    for aspect in aspects:
        generated_summary = generated_summaries.get(aspect, "")
        golden_summary_list = golden_summaries.get(aspect, [])
        
        # Skip if both generated and golden are empty
        if not generated_summary and not golden_summary_list:
            continue
        
        markdown_output.append(f"### Aspect: {aspect}\n")
        
        # Golden summaries
        markdown_output.append("### Golden Summaries:\n")
        if golden_summary_list:
            for golden in golden_summary_list:
                markdown_output.append(f"- {golden}\n")
        else:
            markdown_output.append("- (No golden summary available)\n")
        
        markdown_output.append("\n")
        
        # Generated summary
        markdown_output.append("### Generated Summary:\n")
        if generated_summary:
            markdown_output.append(f"- {generated_summary}\n")
        else:
            markdown_output.append("- (No generated summary)\n")
        
        markdown_output.append("\n---\n\n")
    
    result = "".join(markdown_output)
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
    
    return result

with open("outputs/baseline_summaries.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    sample = data[0]  # Get first entity as sample
    
prompt = reformat_output_for_llm_evaluation(sample)
print(prompt)