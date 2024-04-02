# Importation des librairies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import recall_score, f1_score


import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
import torchtext
import torchsummary

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Chemin vers l'ensemble de données 
Data_chemin = "C:/Users/BMD TECH/MSI-2/Projet_Speech/augmented_dataset"
Classes = os.listdir(Data_chemin)
print("-"*100)
print(f"Nombre total de classes : {len(Classes)} Classe")
print("-"*100)
print("Liste des classes :", Classes)
print("-"*100)

def count(path):
    size=[]
    for file in os.listdir(path):
        size.append(len(os.listdir(os.path.join(path,file))))
    return pd.DataFrame(size,columns=['Number of Classes', ],index=os.listdir(path))  

tr = count(Data_chemin)
tr

root_dir = Data_chemin
folders = os.listdir(Data_chemin)

# Liste pour stocker les chemins des fichiers audio et leurs étiquettes
audio_paths = []
labels = []

# Parcourir chaque dossier
for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    # Parcourir chaque fichier dans le dossier
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        # Vérifier si le fichier est un fichier audio
        if file_path.endswith('.wav') or file_path.endswith('.mp3'):
            # Ajouter le chemin du fichier audio et son étiquette (nom du dossier)
            audio_paths.append(file_path)
            labels.append(folder)

            def charger_et_extraire_mfcc(audio_paths):
    mfccs = []
    for audio_path in audio_paths:
        waveform, sample_rate = torchaudio.load(audio_path)
        # Calculer les coefficients MFCC
        mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=64,  # Nombre de coefficients MFCC à extraire
            melkwargs={'n_fft': 2048, 'hop_length': 256, 'n_mels': 64}  # Paramètres pour le calcul du spectre Mel
        )(waveform)
        mfccs.append(mfcc)

    return torch.stack(mfccs)

mfccs = charger_et_extraire_mfcc(audio_paths)
print("MFCCs shape:", mfccs.shape)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

mlb.fit(pd.Series(labels).fillna("missing").str.split(', '))
y_mlb = mlb.transform(pd.Series(labels).fillna("missing").str.split(', '))
print(mlb.classes_)

y_mlb = torch.tensor(y_mlb)
#y_mlb_labels = torch.max(y_mlb, 1)[1]
y_mlb_labels = torch.argmax(y_mlb, dim=1)
print(y_mlb_labels.shape)

# Données d'exemple
input_data = mfccs # Exemple de données d'entrée
input_data_reshaped = input_data.view(41849, 63, 64)  # Redimensionner pour correspondre à la forme attendue
labels = y_mlb_labels  

# Diviser les données en ensembles d'entraînement et de test
train_features, test_features, train_labels, test_labels = train_test_split(
    input_data_reshaped, labels, test_size=0.3, random_state=62)

print("Train data :")
print(train_features.shape)
print(train_labels.shape)
print("*"*20)
print("Test data :")
print(test_features.shape)
print(test_labels.shape)