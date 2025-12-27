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
MODEL_DEFAULT = "openai/gpt-oss-120b"
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

def get_summary(reviews_text, version_index, model=None):
    """
    Calls the LLM API to generate aspect-based summaries and extract the hotel name.
    Uses version_index to vary the style and ensures sanitization.
    """
    if model is None:
        model = MODEL_DEFAULT
    
    # styles = [
    #     "Write in a balanced and descriptive tone, like a professional travel guide.",
    #     "Write in a direct and concise tone, focusing heavily on the specific pros and cons mentioned by guests.",
    #     "Write in a more conversational and subjective tone, as if one guest is giving a detailed recap to another."
    # ]

    focus_instructions = [
        "Identify the single most common sentiment (Positive or Negative). Write 1-2 short, punchy sentences stating this main feeling and the primary reason for it. Ignore minor details.",
        "Focus on physical objects and specific nouns mentioned (e.g., 'flat screen TV', 'marble floors', 'hot water', 'wifi'). List these features in a natural sentence describing the room/hotel.",
        "State directly. Do not use softening language (e.g., 'however', 'although'). If something is bad, just say it is bad. If something is good, just say it is good."
    ]
    instruction = focus_instructions[version_index % len(focus_instructions)]

    prompt = f"""You are an expert abstractive summarizer. Your task is to write a summary of opinions for specific aspects of a hotel.

**CRITICAL RULES (Failure to follow these lowers the score):**
1.  **NO META-LANGUAGE:** NEVER say "Guests said," "Reviewers mentioned," "The consensus is," or "Reports indicate."
    * *BAD:* "Guests found the location convenient."
    * *GOOD:* "The location is convenient."
2.  **DIRECT ASSERTIONS:** State opinions as objective facts.
3.  **BREVITY:** Keep it short (15-40 words), and strictly under 75 words. Do not fill the space just because you can.
4.  **FOCUS:** {instruction}

**EXAMPLE SUMMARIES**
```
"rooms": [
    "The rooms had excellent creature comforts, towels pillows and bedding . I felt that it was a very comfortable, nicely decorated, spacious sleeping room. The rooms overlooking the bay were fantastic with floor to ceiling windows with amazing views. The bathroom is also really nice, as well.",
    "The room was very basic with some odor. It seems like it wasn't getting aired. The room was recently nicely decorated and in a style similar to most US motel rooms with two large queen sized beds, window-mounted air conditioner mounted and came with safe, refrigerator and microwave. The A/C is noisy, but works well. Otherwise, the rooms are quite. The beds are comfortable. The bedroom and bathroom were clean and comfortable.",
    "The rooms were clean, and comfortable; which includes them being quiet. They also had an amazing view overlooking the harbor and float planes.",
],
"location": [
    "This hotel has a very good location near the Biscayne Bay with beautiful views! It's very close to the Bayside area, and just a $15 ride to Lincoln Rd/South Beach. There were always taxi's at the hotel. There was a lot of construction around the hotel, but overall it was relaxing for downtown location.",
    "It was a good location, plenty of restaurants nearby and easy access to major routes. It's very close to Super Target, Olive Garden, and Publix. It's in a good place to fuel up for the long days at the parks.",
    "The hotel is in downtown Vancouver, with a wonderful view of the harbor from the rooms. It is also near a SkyTrain station and within walking distance of the convention center.",
],
"service": [
    "The staff was always friendly, accommodating, and helpful. The valet employees are very quick and nice.",
    "The staff are fast, friendly & provide a good service. The guest services stand was helpful. The reception staff was mostly very pleasant and helpful, except for one older woman who was abrupt and unhelpful. Otherwise, the housekeeping and ticket agents were the best!The directions given by the hotel staff were also very clear.",
    "Very friendly and helpful staff offering local suggestions. Excellent housekeeping service that was very efficient.",
],
"cleanliness": [
    "The rooms at Hilton Miami Downtown were very clean.",
    "Everyday the rooms were cleaned and had fresh linen and towels. The grounds were also tidy.",
    "The rooms and the hotel itself were all very clean and comfortable",
],
"building": [
    "The lobby, pool, and gym of the hotel were all very beautiful.",
    "There was a good pool. And on the ground floor nearby is a self-service laundry machine."
    "The very obviously updated lobby, restaurant, and bar area is beautiful. Loved the decor in eclectic reds, golds, and blacks. The free WiFi in the lobby area is good. The pool, sauna, and training room were excellent, along with security. There were very nice harbor views.",
],
"food": [
    "The breakfast was generally very good , although it was the same every day, and sometimes reported as cold. Complimentary coffee, fruit, and bottled water were offered once one exists the elevator in the morning into the lobby. The restaurant on the main level was excellent, had a variety of food and drinks. The breakfast buffet was expensive.",
    "The hotels simple continental breakfast was good. However, it was sometimes hard to get a seat.",
    "The hotel has a good quality menu. There's a breakfast buffet with lots of choices including tons of fruit. The menu changed during the day to include a wide variety of options.",
]
```

Output MUST be a valid JSON object:
{{
    "summaries": {{
        "rooms": "...",
        "location": "...",
        "service": "...",
        "cleanliness": "...",
        "building": "...",
        "food": "..."
    }}
}}

Reviews:
{reviews_text}
"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 8192,
        "stream": False,
        "reasoning_effort": "low"
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
