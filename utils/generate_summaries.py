import json
import requests
import dotenv
import os
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# Load environment variables
dotenv.load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
LAST_REQUEST_TIME = 0

def call_api_with_rate_limit(url, json_payload, headers):
    """
    Makes an API call while ensuring at least 2 seconds between request starts.
    """
    global LAST_REQUEST_TIME
    now = time.time()
    elapsed = now - LAST_REQUEST_TIME
    if elapsed < 2.0:
        sleep_time = 2.0 - elapsed
        time.sleep(sleep_time)
    
    # Update time immediately before the request
    LAST_REQUEST_TIME = time.time()
    return requests.post(url, json=json_payload, headers=headers)

def sanitize_text(text):
    """
    Cleans up the LLM output by replacing non-standard dashes and other artifacts.
    """
    if not text:
        return ""
    text = text.replace('\u2011', '-') # Non-breaking hyphen
    text = text.replace('\u2013', '-') # En dash
    text = text.replace('\u2014', '-') # Em dash
    text = text.replace('\u2018', "'") # Left single quote
    text = text.replace('\u2019', "'") # Right single quote/apostrophe
    text = text.replace('\u201c', '"') # Left double quote
    text = text.replace('\u201d', '"') # Right double quote
    text = text.strip('"').strip("'")
    return text

def get_summary(reviews_text, version_index):
    """
    Calls the LLM API to generate aspect-based summaries and extract the hotel name.
    Uses version_index to vary the style and ensures sanitization.
    """
    
    styles = [
        "Write in a balanced and descriptive tone, like a professional travel guide.",
        "Write in a direct and concise tone, focusing heavily on the specific pros and cons mentioned by guests.",
        "Write in a more conversational and subjective tone, as if one guest is giving a detailed recap to another."
    ]
    style_instruction = styles[version_index % len(styles)]

    prompt = f"""You are an expert hotel reviewer. Based on the following reviews for a hotel, please:
1. Identify the Name of the Hotel.
2. Provide a concise, human-natural summary (1-3 sentences, <100 words) for each of these 6 aspects:
   - rooms
   - location
   - service
   - cleanliness
   - building
   - food

{style_instruction} 
Avoid using robotic phrases like "The hotel offers..." or "Guests noted that...". Speak naturally and vary your sentence structures.

Output MUST be a valid JSON object with the following structure:
{{
    "summaries": {{
        "rooms": "Summary...",
        "location": "Summary...",
        "service": "Summary...",
        "cleanliness": "Summary...",
        "building": "Summary...",
        "food": "Summary..."
    }}
}}

Reviews:
{reviews_text}
"""
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 4096,
        "stream": False
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {API_KEY}"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = call_api_with_rate_limit(URL, payload, headers)
            if response.status_code != 200:
                print(f"API Error (Status {response.status_code}): {response.text}")
                continue
                
            data = response.json()
            content = data['choices'][0]['message']['content'].strip()
            
            # Extract JSON from potential markdown markers
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Clean up potential leading/trailing non-json chars
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                content = content[start:end]
                
            res_json = json.loads(content)
            
            # Sanitize each summary string
            if 'summaries' in res_json:
                for k, v in res_json['summaries'].items():
                    res_json['summaries'][k] = sanitize_text(v)
            
            return res_json
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    return None

def process_file(input_path, output_path):
    print(f"Loading {input_path}...")
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    modified_data = []
    
    for entry_idx, entry in enumerate(data):
        entity_id = entry['entity_id']
        reviews = entry['reviews']
        
        review_paragraphs = []
        for r in reviews:
            paragraph = " ".join(r['sentences']) or ""
            review_paragraphs.append(paragraph)
        
        reviews_text = "\n- ".join(review_paragraphs)

        print(f"[{entry_idx + 1}/{len(data)}] Processing Entity ID: {entity_id}")
        
        all_generations = []
        entity_name = "Unknown Hotel"
        
        # Generate 3 versions with different styles
        for i in range(3):
            print(f"  Generating version {i+1}/3...")
            res = get_summary(reviews_text, i)
            if res and 'summaries' in res:
                all_generations.append(res['summaries'])
            else:
                print(f"    Failed to get valid response for version {i+1}")
        
        if not all_generations:
            print(f"  Failed all generations for {entity_id}. Skipping.")
            continue
            
        # Reformat summaries: aspect -> [summary1, summary2, summary3]
        aspects = ["rooms", "location", "service", "cleanliness", "building", "food"]
        formatted_summaries = {aspect: [] for aspect in aspects}
        
        for gen in all_generations:
            for aspect in aspects:
                # Use empty string if aspect missing in a particular generation
                formatted_summaries[aspect].append(gen.get(aspect, ""))
        
        new_entry = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "reviews": reviews,
            "summaries": formatted_summaries
        }
        modified_data.append(new_entry)
        
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(modified_data, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    input_file = 'data/sampled_space_train.json'
    output_file = 'data/space_summ_train.json'
    process_file(input_file, output_file)
