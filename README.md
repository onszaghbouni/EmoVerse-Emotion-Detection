# EmoVerse-Emotion-Detection
EmoVerse is a multimodal AI system capable of detecting emotions from text and speech. It integrates a Transformer-based NLP classifier, an MFCC-powered audio pipeline with an MLP model, and a full Streamlit interface. Designed for academic evaluation, reproducibility, and practical human–computer interaction experiments.
Description

EmoVerse est une application multimodale permettant l’analyse des émotions à partir de :

texte (modèle Transformer – NLP)

audio (MFCC + modèle MLP)

Développée avec Streamlit, elle offre une interface moderne et intuitive pour tester la reconnaissance des émotions.

🧠 Technologies utilisées

Python 3.9+

Streamlit

Transformers (Hugging Face)

Librosa

Scikit-learn

SoundFile

NumPy

PyTorch

CSS personnalisé

🔧 Installation & Exécution
1. Cloner le projet
git clone https://github.com/<TON-USERNAME>/EmoVerse-Emotion-Detection.git
cd EmoVerse-Emotion-Detection

2. Créer un environnement virtuel
python -m venv env
env\Scripts\activate

3. Installer les dépendances
pip install -r requirements.txt

4. Lancer l'application
streamlit run app.py

🎤 Test du modèle audio
python main.py --audio fichier.wav

✍️ Test du modèle texte
python main.py --text "I am happy today"
