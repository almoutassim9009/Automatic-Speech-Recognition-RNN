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

import torch.nn.functional as F

class SpeechRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechRNN, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)  # Nouvelle couche LSTM
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)  # Nouvelle couche LSTM
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)  # Passer à travers la troisième couche LSTM
        out, _ = self.lstm4(out)  # Passer à travers la quatrième couche LSTM
        out = self.fc(out[:, -1, :])  # Prendre la dernière sortie temporelle
        return out


# Paramètres du modèle
input_dim = 64  # Dimension des caractéristiques MFCC, par exemple
hidden_dim = 128  # Nombre d'unités dans les couches LSTM
output_dim = 30  # Nombre de classes de sortie
# Création du modèle avec deux couches LSTM supplémentaires
model_rnn = SpeechRNN(input_dim, hidden_dim, output_dim)
model_rnn

input_rnn = torch.randn(1, 63, 64)
# Passe en avant
output_rnn = model_rnn(input_rnn)

print("*"*30)
target_rnn= torch.randn(1, 30)
print("shape target :",target_rnn.shape)
print("shape input  :", input_rnn.shape)
print("*"*30)
print("shape output :", output_rnn.shape)
print("*"*30)  


from torchviz import make_dot
# Visualiser le graphique computationnel
dot = make_dot(output_rnn, params=dict(model_rnn.named_parameters()))
dot.render("model_graph_rnn", format="png")
#dot.render("model_graph")  # Enregistre le graphique au format PDF
dot


# Déplacez le modèle sur le GPU s'il est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_rnn.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_rnn.parameters(), lr=0.001)


# Training loop
num_epochs = 30
train_losses = []
for epoch in range(num_epochs):
    train_total_loss = 0.0  # Réinitialisation de la perte totale pour chaque époque
    for inputs, targets in dataloader_train:    
        # Déplacez les données d'entrée et les cibles sur le même périphérique que le modèle
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_rnn(inputs)
        # Calculer la perte
        loss = criterion(outputs, targets)
        train_total_loss += loss.item()
        
        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()
        # Réinitialiser les gradients
        optimizer.zero_grad()

    # Afficher la perte moyenne de l'époque
    average_loss = train_total_loss / len(dataloader_train)
    train_losses.append(average_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss average = [{average_loss}]")
    
    
model_rnn.eval()
predictions1 = []
targets_list1 = []
n_tot_test = 0
nb_correct_test = 0

with torch.no_grad():
    for data, targets in dataloader_test:
        pred = model_rnn(data)
        pred_class = torch.argmax(pred, dim=1)
        # Ajouter les prédictions et les cibles aux listes
        predictions1.extend(pred_class.tolist())
        targets_list1.extend(targets.tolist())
        
        nb_correct_test += torch.sum(pred_class == targets)
        n_tot_test += data.shape[0]
        
    acc_test = nb_correct_test / n_tot_test    
    print(f"Sur les {n_tot_test} audios de test, {nb_correct_test} ont été reconus par le modèle")
    print(f"Accuracy test  : {acc_test.item()}")




from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import accuracy_score

# Convertir les listes de prédictions et de cibles en tenseurs
predictions_tensor1 = torch.tensor(predictions1)
targets_tensor1 = torch.tensor(targets_list1)

# Calculer la précision
precision1 = accuracy_score(targets_tensor1, predictions_tensor1)
# Calculer le rappel (recall)
recall1 = recall_score(targets_tensor1, predictions_tensor1, average='weighted')
# Calculer le score F1 (F1 score)
f11 = f1_score(targets_tensor1, predictions_tensor1, average='weighted')

print(f"Précision : {precision1}")
print(f"Rappel (recall) : {recall1}")
print(f"Score F1 (F1 score) : {f11}")


# Tracer la courbe d'apprentissage
plt.plot(range(num_epochs), train_losses, label='Train Loss',
         color='red', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Courbe d'apprentissage")
plt.legend()
plt.grid(True)

# Ajouter un fond de couleur
plt.savefig('Ec111.png')
plt.show()


# Calculer la matrice de confusion
conf_matrix11 = confusion_matrix(df['Targets'], df['Predictions'])

# Afficher la matrice de confusion sous forme de graphique
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix11, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matrice de Confusion")
plt.xlabel("Classe Prédite")
plt.ylabel("Classe Réelle")
plt.savefig('Matrice_rnn.png')
plt.show()


word_list = Classes
predictions_tensor = torch.tensor(predictions1)
targets_tensor = torch.tensor(targets_list)
# Création d'un dictionnaire associant chaque mot à son indice
word_to_index = {word: index for index, word in enumerate(word_list)}
# Inversion de l'encodage
targets = [word_list[index] for index in targets_tensor]
predictions = [word_list[index] for index in predictions_tensor]
# Création du DataFrame
df = pd.DataFrame({'Targets': targets, 'Predictions': predictions})
# Affichage du DataFrame
df