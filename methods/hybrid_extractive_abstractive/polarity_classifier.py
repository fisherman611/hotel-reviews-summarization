import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

with open("methods/hybrid_extractive_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

POLARITY_MODEL = config["polarity_model"]
POLARITIES = config["polarities"]

class PolarityClassifier:
    def __init__(self,
                 model_name: str=POLARITY_MODEL,
                 polarities: list[str]=POLARITIES,
                 device: str=None) -> None:
        
        self.polarities = polarities
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def classify_polarity(self, text: str, polarity: str):
        hypothesis = f"This text is about {polarity}"
        inputs = self.tokenizer(text,
                                hypothesis,
                                return_tensors="pt",
                                truncation=True).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        entail_contra = logits[:, [0, 2]]
        probs = F.softmax(entail_contra, dim=1)
        entail_score = probs[:, 1].item()
        
        return entail_score
            
    def predict(self, text: str):
        scores = {}
        
        for polarity in self.polarities:
            score = self.classify_polarity(text, polarity)
            scores[polarity] = score
        
        return {
            "scores": scores,
            "predicted_polarity": max(scores, key=scores.get)
        }