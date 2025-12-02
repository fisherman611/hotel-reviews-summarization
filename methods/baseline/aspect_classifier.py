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

with open("methods/baseline/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ASPECTS = config["aspects"]
ASPECT_MODEL = config["aspect_model"]


class AspectClassifier:
    def __init__(
        self,
        model_name: str = ASPECT_MODEL,
        aspects: list[str] = ASPECTS,
        threshold: float = 0.5,
        device: str = None,
    ) -> None:

        self.aspects = aspects
        self.threshold = threshold
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def add_aspects(self, aspects: list[str]):
        self.aspects.extend(aspects)

    def classify_aspect(self, text: str, aspect: str):
        hypothesis = f"This text is about {aspect}"
        inputs = self.tokenizer(
            text, hypothesis, return_tensors="pt", truncation=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # MNLI: [contradiction=0, neutral=1, entailment=2]
        entail_contra = logits[:, [0, 2]]
        probs = F.softmax(entail_contra, dim=1)
        entail_score = probs[:, 1].item()

        return entail_score

    def predict(self, text: str):
        scores = {}
        predicted_aspects = []

        for aspect in self.aspects:
            score = self.classify_aspect(text, aspect)
            scores[aspect] = score
            if score >= self.threshold:
                predicted_aspects.append(aspect)

        return {"scores": scores, "predicted_aspects": predicted_aspects}
