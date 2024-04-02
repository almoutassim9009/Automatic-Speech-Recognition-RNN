# Reconnaissance Automatique de la Parole avec RNN (Automatic Speech Recognition with RNN)

Ce projet présente une implémentation de la reconnaissance automatique de la parole (ASR) en utilisant un réseau de neurones récurrent (RNN). Contrairement au modèle CNN qui utilise des spectrogrammes audio en entrée, le modèle RNN est conçu pour traiter les coefficients Mel-frequency cepstral (MFCC) comme entrée et prédire la classe correspondante, représentant le mot ou le phonème prononcé.

## Fonctionnalités

- **Modèle RNN pour l'ASR**: Le modèle RNN est conçu pour apprendre les structures temporelles des données audio à partir des MFCC et effectuer une classification précise des données vocales.

- **Prétraitement des Données**: Les données audio sont prétraitées pour extraire les coefficients MFCC, qui sont ensuite utilisés comme entrée pour le modèle RNN.

- **Entraînement et Évaluation**: Le modèle est entraîné sur un ensemble de données d'entraînement et évalué sur un ensemble de données de test pour évaluer sa performance en termes de précision de classification.

## Contenu du Projet

- `speech_rnn.py`: Fichier principal contenant l'implémentation du modèle RNN pour l'ASR, ainsi que le code d'entraînement et d'évaluation du modèle.

- `data_preprocessing.py`: Script de prétraitement des données audio, convertissant les fichiers audio en coefficients MFCC pour l'entraînement et l'évaluation du modèle.

## Utilisation

1. **Prétraitement des Données**: Utilisez `data_preprocessing.py` pour prétraiter les données audio et générer les coefficients MFCC nécessaires pour l'entraînement du modèle.

2. **Entraînement du Modèle**: Exécutez `speech_rnn.py` pour entraîner le modèle RNN sur les données prétraitées.

3. **Évaluation du Modèle**: Une fois l'entraînement terminé, évaluez les performances du modèle en termes de précision de classification.
