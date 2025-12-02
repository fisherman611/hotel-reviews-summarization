def aspect_extractive_summarizer(topk_sentences: dict):
    """
    topk_sentences: dict[aspect -> list[str]]
    returns: dict[aspect -> str]
    """
    aspect_summaries = {}

    for aspect, sentences in topk_sentences.items():
        if not sentences:
            aspect_summaries[aspect] = ""
            continue

        # simple concatenation; you can later swap for nicer formatting / re-writing
        aspect_summary = " ".join(sentences).strip()
        aspect_summaries[aspect] = aspect_summary

    return aspect_summaries