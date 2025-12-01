from text_emotion_detector import TextEmotionDetector

detector = TextEmotionDetector()

while True:
    text = input("Enter your text (or 'quit' to exit): ")
    if text.lower() == "quit":
        break
    result = detector.predict(text)
    print("\nDominant emotion :", result["dominant_emotion"])
    print("Confidence :", result["confidence"])
    print("All emotions :", result["all_emotions"])
    print("-"*50)
