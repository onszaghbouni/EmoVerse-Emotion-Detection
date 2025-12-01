from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

class TextEmotionDetector:
    def __init__(self):
        # Modèle pré-entraîné HuggingFace
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        # Labels
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

    def predict(self, text):
        # Tokenization
        inputs = self.tokenizer(text, return_tensors="pt")
        # Inférence
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Probabilités
        probs = F.softmax(outputs.logits, dim=-1)[0].numpy()
        # Emotion dominante
        idx_max = np.argmax(probs)
        dominant_emotion = self.labels[idx_max]
        confidence = float(probs[idx_max])
        # Toutes les émotions
        all_emotions = {self.labels[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        return {
            "dominant_emotion": dominant_emotion,
            "confidence": round(confidence,4),
            "all_emotions": all_emotions
        }
