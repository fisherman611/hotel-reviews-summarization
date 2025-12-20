import os
import sys
import json
import re
import requests
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

# Add parent directory to path to import helpers
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.helpers import reformat_output_for_llm_evaluation

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("Missing NVIDIA_API_KEY in environment variables.")

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {NVIDIA_API_KEY}",
}

ASPECTS = ["rooms", "location", "service", "cleanliness", "building", "food"]

# Load system prompt from file
SCRIPT_DIR = Path(__file__).parent
SYSTEM_PROMPT_FILE = SCRIPT_DIR / "system_prompt.txt"
with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()


def is_json_incomplete(text: str) -> bool:
    """
    Check if JSON appears to be incomplete/truncated.
    
    Returns:
        bool: True if JSON appears incomplete
    """
    if not text:
        return True
    
    text = text.strip()
    
    # Check if it's in a markdown code block
    if text.startswith("```"):
        # Find the closing ```
        end_marker = text.find("```", 3)
        if end_marker == -1:
            return True  # Incomplete markdown block
        text = text[3:end_marker].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    
    # Check for balanced braces
    if text.startswith("{"):
        brace_count = 0
        in_string = False
        escape_next = False
        
        for char in text:
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
        
        # If braces are not balanced, JSON is incomplete
        if brace_count > 0:
            return True
    
    # Check if text ends abruptly (common truncation patterns)
    # If it ends in the middle of a string, array, or object
    text_clean = text.rstrip()
    if text_clean and not text_clean.endswith(('}', ']', '"')):
        # Check if we're in the middle of a string
        if text_clean.count('"') % 2 != 0:
            return True
        # Check if we're in the middle of an array/object
        if text_clean.endswith((',', ':', '[', '{')):
            return True
    
    return False


def extract_and_parse_json(text: str) -> Tuple[dict, bool]:
    """
    Robust JSON extraction and parsing from LLM output.
    Handles markdown code blocks, extra text, and various formatting issues.
    
    Returns:
        tuple[dict, bool]: (Parsed JSON object, is_incomplete flag)
                          Returns empty dict and True if JSON is incomplete
    """
    if not text:
        print("Warning: Empty text provided to extract_and_parse_json")
        return {}, True
    
    text = text.strip()
    
    # Strategy 1: Try direct JSON parse if it looks like JSON
    if text.startswith("{"):
        try:
            # First try parsing as-is
            result = json.loads(text)
            return result, False  # Success - not incomplete
        except json.JSONDecodeError as e:
            print(f"Strategy 1 failed - JSON parse error: {e}")
            # If it starts with { but fails, try finding the end
            try:
                # Try to find balanced braces
                brace_count = 0
                end_pos = -1
                for i, char in enumerate(text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    json_text = text[:end_pos]
                    result = json.loads(json_text)
                    return result, False
            except (json.JSONDecodeError, Exception) as e2:
                print(f"Strategy 1b failed - Balanced brace attempt: {e2}")
    
    # Strategy 2: Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    markdown_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    markdown_match = re.search(markdown_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if markdown_match:
        json_content = markdown_match.group(1).strip()
        try:
            result = json.loads(json_content)
            return result, False
        except json.JSONDecodeError as e:
            print(f"Strategy 2 failed - Markdown block parse error: {e}")
            # Check if markdown block itself is incomplete
            if not text.rstrip().endswith("```"):
                return {}, True
    
    # Strategy 3: Remove common markdown artifacts and try again
    cleaned = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^json\s*\n", "", cleaned, flags=re.IGNORECASE)
    try:
        result = json.loads(cleaned)
        return result, False
    except json.JSONDecodeError as e:
        print(f"Strategy 3 failed - Cleaned parse error: {e}")
    
    # Strategy 4: Find first {...} block using regex with balanced braces
    try:
        start = text.find('{')
        if start >= 0:
            brace_count = 0
            end_pos = -1
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > 0:
                json_text = text[start:end_pos]
                result = json.loads(json_text)
                return result, False
    except (json.JSONDecodeError, Exception) as e:
        print(f"Strategy 4 failed - Balanced brace search: {e}")
    
    # All strategies failed - check if JSON appears incomplete
    is_incomplete = is_json_incomplete(text)
    print(f"Warning: Could not parse JSON from model output")
    if is_incomplete:
        print("JSON appears to be incomplete/truncated")
    print(f"Output length: {len(text)} characters")
    print(f"First 200 chars: {text[:200]}")
    print(f"Last 200 chars: {text[-200:]}")
    return {}, is_incomplete


def build_messages(formatted_context: str, summary_type: str = "generated"):
    """
    Build messages for LLM evaluation following G-Eval approach.
    
    Args:
        formatted_context: full formatted context from reformat_output_for_llm_evaluation
        summary_type: "generated" or "golden" - controls what to evaluate
    
    Raises:
        ValueError: if summary_type is not "generated" or "golden"
    """
    if summary_type == "generated":
        summary_description = "Generated summaries"
        evaluation_instruction = "Evaluate the GENERATED summary on 4 dimensions (naturalness, coherence, engagingness, groundedness)"
        json_schema = """{{
  "aspects": [
    {{
      "aspect": "rooms",
      "generated_summary_eval": {{
        "scores": {{
          "naturalness": 1-5,
          "coherence": 1-5,
          "engagingness": 1-5,
          "groundedness": 1-5,
          "overall_quality": (average of above 4)
        }}
      }}
    }}
  ]
}}"""
    elif summary_type == "golden":
        summary_description = "Golden reference summaries"
        evaluation_instruction = "Evaluate EACH golden summary on 4 dimensions (naturalness, coherence, engagingness, groundedness)"
        json_schema = """{{
  "aspects": [
    {{
      "aspect": "rooms",
      "golden_summaries_eval": [
        {{
          "golden_index": 0,
          "scores": {{
            "naturalness": 1-5,
            "coherence": 1-5,
            "engagingness": 1-5,
            "groundedness": 1-5,
            "overall_quality": (average of above 4)
          }}
        }}
      ]
    }}
  ]
}}"""
    else:
        raise ValueError(f"summary_type must be 'generated' or 'golden', got '{summary_type}'")
    
    user_prompt = f"""The following is a markdown-formatted hotel review summary evaluation task:

{formatted_context}

---

## Evaluation Task

The above content shows:
- Original hotel reviews (in markdown format)
- {summary_description} for each aspect

Evaluate ALL aspects shown above (rooms, location, service, cleanliness, building, food).

For EACH aspect:
{evaluation_instruction}

## Evaluation Steps
1. Carefully read the reviews and summary for the aspect
2. Assign scores (1-5) based on the rubric in the system prompt
3. Calculate overall_quality as: (naturalness + coherence + engagingness + groundedness) / 4

Return JSON with this exact schema:
{json_schema}

IMPORTANT: 
- Use scores 1-5 (not 0-5)
- overall_quality = (naturalness + coherence + engagingness + groundedness) / 4
""".strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_deepseek(messages, *, temperature=0, top_p=0.95, max_tokens=8192):
    payload = {
        "model": "deepseek-ai/deepseek-v3.2",
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": messages,
    }

    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    
    # Check if response was truncated
    finish_reason = choice.get("finish_reason", "")
    is_truncated = finish_reason == "length" or finish_reason == "max_tokens"
    
    return content, is_truncated


def _evaluate_summary_type(sample: dict, summary_type: str, *, temperature=0, max_tokens=8192, max_retries=2):
    """
    Helper function to evaluate either generated or golden summaries.
    
    Args:
        sample: entity dictionary
        summary_type: "generated" or "golden"
        temperature: temperature for LLM
        max_tokens: maximum tokens for LLM response
        max_retries: maximum retry attempts
    
    Returns:
        dict with "aspects" key containing evaluation results, or None on error
    """
    entity_id = sample.get("entity_id", "")
    
    # Format context for the specific summary type
    formatted_context = reformat_output_for_llm_evaluation(sample, summary_type=summary_type)
    
    # Build messages
    messages = build_messages(formatted_context, summary_type=summary_type)
    
    # Try with retries if response is truncated
    current_max_tokens = max_tokens
    for attempt in range(max_retries + 1):
        raw, is_truncated = call_deepseek(messages, temperature=temperature, max_tokens=current_max_tokens)
        
        # Parse JSON from model output
        parsed, is_incomplete = extract_and_parse_json(raw)
        
        # If response was truncated or JSON is incomplete, retry with more tokens
        if (is_truncated or is_incomplete) and attempt < max_retries:
            current_max_tokens = int(current_max_tokens * 1.5)  # Increase by 50%
            print(f"Warning: Response truncated or incomplete for entity {entity_id} ({summary_type}). Retrying with max_tokens={current_max_tokens}...")
            continue
        
        # Validate that we got aspect results
        if not parsed or not isinstance(parsed, dict) or not parsed.get("aspects"):
            error_msg = f"Failed to extract valid evaluation results from model output ({summary_type})"
            if is_truncated:
                error_msg += " (response was truncated by max_tokens limit)"
            elif is_incomplete:
                error_msg += " (JSON appears incomplete)"
            
            print(f"Warning: {error_msg} for entity {entity_id}")
            print(f"Raw output length: {len(raw)} characters")
            
            # Try to save the raw output to a separate file for debugging
            try:
                error_file = f"error_entity_{entity_id}_{summary_type}_raw.txt"
                with open(error_file, "w", encoding="utf-8") as f:
                    f.write(raw)
                print(f"Raw output saved to: {error_file}")
            except Exception as e:
                print(f"Could not save raw output: {e}")
            
            return None
        
        # Success - return parsed results
        return parsed
    
    return None


def evaluate_one_sample(sample: dict, *, summary_type="generated", temperature=0, max_tokens=8192, max_retries=2):
    """
    Evaluate one sample for either generated or golden summaries.
    
    Args:
        sample: entity dictionary containing:
            - entity_id: str
            - reviews: [{"review_id": str, "sentences": [str]}, ...]
            - generated_summaries: {aspect: str, ...}
            - golden_summaries: {aspect: [str, ...], "general": [...]}
        summary_type: "generated" or "golden" - which type to evaluate (default: "generated")
        temperature: temperature for LLM
        max_tokens: maximum tokens for LLM response
        max_retries: maximum retry attempts
    
    Returns:
        dict with evaluation results for the specified summary type
    """
    if summary_type not in ("generated", "golden"):
        raise ValueError(f"summary_type must be 'generated' or 'golden', got '{summary_type}'")
    
    entity_id = sample.get("entity_id", "")
    
    # Evaluate the specified summary type
    print(f"Evaluating {summary_type} summaries for entity {entity_id}...")
    parsed = _evaluate_summary_type(sample, summary_type, temperature=temperature, 
                                   max_tokens=max_tokens, max_retries=max_retries)
    
    # Check for errors
    if parsed is None:
        return {
            "entity_id": entity_id,
            "error": f"Failed to evaluate {summary_type} summaries"
        }
    
    # Extract per-aspect results
    results = parsed.get("aspects", [])
    
    # Compute overall averages across all aspects
    score_keys = ["naturalness", "coherence", "engagingness", "groundedness", "overall_quality"]
    
    # Calculate averages based on summary_type
    score_vals = {k: [] for k in score_keys}
    
    if summary_type == "generated":
        for r in results:
            if isinstance(r, dict) and "generated_summary_eval" in r:
                gen_eval = r["generated_summary_eval"]
                if "scores" in gen_eval:
                    for k in score_keys:
                        v = gen_eval["scores"].get(k)
                        if isinstance(v, (int, float)):
                            score_vals[k].append(float(v))
    else:  # golden
        for r in results:
            if isinstance(r, dict) and "golden_summaries_eval" in r:
                for gold_eval in r["golden_summaries_eval"]:
                    if "scores" in gold_eval:
                        for k in score_keys:
                            v = gold_eval["scores"].get(k)
                            if isinstance(v, (int, float)):
                                score_vals[k].append(float(v))
    
    # Build overall statistics
    overall = {
        "entity_id": entity_id,
        "num_aspects_evaluated": len(results),
        f"{summary_type}_summary_avg": {}
    }
    
    for k in score_keys:
        overall[f"{summary_type}_summary_avg"][k] = round(sum(score_vals[k]) / len(score_vals[k]), 3) if score_vals[k] else None

    return {
        "entity_id": entity_id,
        "per_aspect": results,
        "overall": overall,
    }


def evaluate_output_file(input_file: str, output_file: str = None, summary_type: str = "generated", 
                         temperature: float = 1, max_tokens: int = 8192):
    """
    Evaluate all samples in an output file.
    
    Args:
        input_file: path to JSON file with summaries
        output_file: optional path to save evaluation results
        summary_type: "generated" or "golden" - which type to evaluate (default: "generated")
        temperature: temperature for LLM
        max_tokens: maximum tokens for LLM response (default: 8192)
    
    Returns:
        list of evaluation results
    """
    if summary_type not in ("generated", "golden"):
        raise ValueError(f"summary_type must be 'generated' or 'golden', got '{summary_type}'")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    for i, sample in enumerate(data):
        print(f"Evaluating sample {i+1}/{len(data)} - Entity {sample.get('entity_id', '?')}...")
        try:
            report = evaluate_one_sample(sample, summary_type=summary_type, temperature=temperature, max_tokens=max_tokens)
            results.append(report)
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            results.append({
                "entity_id": sample.get("entity_id", ""),
                "error": str(e)
            })
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nEvaluation results saved to: {output_file}")
    
    # Compute aggregate statistics
    valid_results = [r for r in results if "overall" in r and "error" not in r]
    if valid_results:
        score_keys = ["naturalness", "coherence", "engagingness", "groundedness", "overall_quality"]
        
        # Overall averages across all entities
        aggregate = {
            "num_entities": len(valid_results),
            f"{summary_type}_summary": {}
        }
        
        summary_avg_key = f"{summary_type}_summary_avg"
        for k in score_keys:
            vals = [r["overall"][summary_avg_key].get(k) for r in valid_results 
                   if r["overall"][summary_avg_key].get(k) is not None]
            aggregate[f"{summary_type}_summary"][k] = round(sum(vals) / len(vals), 3) if vals else None
        
        # Per-aspect statistics across all entities
        aspect_stats = {}
        for aspect in ASPECTS:
            aspect_stats[aspect] = {
                f"{summary_type}_summary": {}
            }
            
            for k in score_keys:
                vals = []
                
                for r in valid_results:
                    for asp_result in r.get("per_aspect", []):
                        if asp_result.get("aspect") == aspect:
                            if summary_type == "generated":
                                if "generated_summary_eval" in asp_result:
                                    gen_scores = asp_result["generated_summary_eval"].get("scores", {})
                                    v = gen_scores.get(k)
                                    if isinstance(v, (int, float)):
                                        vals.append(float(v))
                            else:  # golden
                                if "golden_summaries_eval" in asp_result:
                                    for gold_eval in asp_result["golden_summaries_eval"]:
                                        gold_scores = gold_eval.get("scores", {})
                                        v = gold_scores.get(k)
                                        if isinstance(v, (int, float)):
                                            vals.append(float(v))
                
                aspect_stats[aspect][f"{summary_type}_summary"][k] = round(sum(vals) / len(vals), 3) if vals else None
                aspect_stats[aspect][f"{summary_type}_summary"]["num_samples"] = len(vals)
        
        print("\n" + "="*60)
        print("AGGREGATE STATISTICS - OVERALL")
        print("="*60)
        print(json.dumps(aggregate, indent=2))
        
        print("\n" + "="*60)
        print("AGGREGATE STATISTICS - PER ASPECT")
        print("="*60)
        print(json.dumps(aspect_stats, indent=2))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate hotel review summaries using LLM")
    parser.add_argument("--input_file", help="Path to JSON file with summaries to evaluate")
    parser.add_argument("-o", "--output", help="Path to save evaluation results (optional)")
    parser.add_argument("--summary_type", choices=["generated", "golden"], default="generated",
                       help="Type of summaries to evaluate: 'generated' or 'golden' (default: generated)")
    parser.add_argument("-t", "--temperature", type=float, default=0, 
                       help="Temperature for LLM (default: 0)")
    parser.add_argument("--max_tokens", type=int, default=8192,
                       help="Maximum tokens for LLM response (default: 8192)")
    parser.add_argument("--sample", type=int, help="Evaluate only one sample by index")
    
    args = parser.parse_args()
    
    if args.sample is not None:
        # Evaluate single sample
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if args.sample >= len(data):
            print(f"Error: Sample index {args.sample} out of range (max: {len(data)-1})")
        else:
            sample = data[args.sample]
            print(f"Evaluating sample {args.sample} - Entity {sample.get('entity_id', '?')}...")
            report = evaluate_one_sample(sample, summary_type=args.summary_type, 
                                       temperature=args.temperature, max_tokens=args.max_tokens)
            
            # Pretty print with per-aspect breakdown
            print("\n" + "="*60)
            print(f"ENTITY: {report.get('entity_id')}")
            print("="*60)
            
            if "per_aspect" in report:
                for aspect_result in report["per_aspect"]:
                    print(f"\n--- {aspect_result.get('aspect', 'Unknown').upper()} ---")
                    
                    # Display based on summary_type
                    if args.summary_type == "generated":
                        if "generated_summary_eval" in aspect_result:
                            print("\nGENERATED SUMMARY:")
                            gen_eval = aspect_result["generated_summary_eval"]
                            if "scores" in gen_eval:
                                print("  Scores:")
                                for score_name, score_val in gen_eval["scores"].items():
                                    print(f"    {score_name}: {score_val}")
                    else:  # golden
                        if "golden_summaries_eval" in aspect_result:
                            for i, gold_eval in enumerate(aspect_result["golden_summaries_eval"]):
                                print(f"\nGOLDEN SUMMARY {i+1}:")
                                if "scores" in gold_eval:
                                    print("  Scores:")
                                    for score_name, score_val in gold_eval["scores"].items():
                                        print(f"    {score_name}: {score_val}")
            
            if "overall" in report:
                print("\n" + "="*60)
                print("OVERALL AVERAGES")
                print("="*60)
                print(json.dumps(report["overall"], indent=2))
            
            print("\n" + "="*60)
            print("FULL JSON OUTPUT")
            print("="*60)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"\nSaved to: {args.output}")
    else:
        # Evaluate all samples
        results = evaluate_output_file(args.input_file, args.output, args.summary_type, 
                                      args.temperature, args.max_tokens)
        print(results)