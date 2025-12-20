import os
import sys
import json
import time
from pathlib import Path
from tqdm.auto import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import logic from generate_summaries
from methods.teacher.generate_summaries import get_summary, call_api_with_rate_limit

def run_teacher_pipeline():
    config_path = PROJECT_ROOT / "methods/teacher/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ASPECTS = config["aspects"]
    MODEL_NAME = config["model_name"]
    DATA_PATH = PROJECT_ROOT / config["data_path"]
    OUTPUT_TEMPLATE = config["summary_output_path_template"]

    if not DATA_PATH.exists():
        print(f"Error: Data path {DATA_PATH} does not exist.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        entities = json.load(f)

    # We want to generate 3 separate files for 3 styles
    for style_idx in range(3):
        print(f"\n--- STAGE: GENERATING TEACHER SUMMARIES (STYLE {style_idx}) ---")
        final_summaries = []
        
        output_path = PROJECT_ROOT / OUTPUT_TEMPLATE.format(style=style_idx)
        os.makedirs(output_path.parent, exist_ok=True)

        for ent in tqdm(entities, desc=f"Style {style_idx}"):
            reviews = ent.get("reviews", [])
            
            review_paragraphs = []
            for r in reviews:
                paragraph = " ".join(r['sentences']) or ""
                review_paragraphs.append(paragraph)
            
            reviews_text = "\n- ".join(review_paragraphs)
            
            # Use get_summary with the specific style index and configured model
            res = get_summary(reviews_text, style_idx, model=MODEL_NAME)
            
            generated_summaries = {}
            if res and 'summaries' in res:
                # Ensure all aspects are present, even if empty
                for aspect in ASPECTS:
                    generated_summaries[aspect] = res['summaries'].get(aspect, "")
            else:
                print(f"  Warning: Failed to generate summaries for entity {ent['entity_id']}")
                generated_summaries = {aspect: "" for aspect in ASPECTS}

            final_summaries.append({
                "entity_id": ent["entity_id"],
                "reviews": ent["reviews"],
                "generated_summaries": generated_summaries,
                "golden_summaries": ent.get("summaries", {})
            })

        print(f"Saving Style {style_idx} results to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_summaries, f, ensure_ascii=False, indent=4)

    print("\nTeacher Pipeline Complete!")

if __name__ == "__main__":
    run_teacher_pipeline()
