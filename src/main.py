from .dataset import WineQualityDataset
from .mlp import SimpleMLP
from .model import Model
from .views import show_views, plot_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier

# 1- Carregar os dados do dataset Wine

feature_translation = {
    'fixed acidity': 'Acidez fixa',
    'volatile acidity': 'Acidez volátil',
    'citric acid': 'Ácido cítrico',
    'residual sugar': 'Açúcar residual',
    'chlorides': 'Cloretos',
    'free sulfur dioxide': 'Dióxido de enxofre livre',
    'total sulfur dioxide': 'Dióxido de enxofre total',
    'density': 'Densidade',
    'pH': 'pH',
    'sulphates': 'Sulfatos',
    'alcohol': 'Álcool',
    'quality': 'Qualidade'
}

dataset = "winequality-" + input(f"Escolha um dataset ('red' ou 'white'): ") + ".csv"
df = pd.read_csv(f"data/{dataset}", sep=';')

df.rename(columns=feature_translation, inplace=True) # Renomeia as colunas para português

X = df.drop(['Qualidade'], axis=1).values
Y = df['Qualidade'].values

# 2 - Apresentar os dados do dataset wine utilizando a biblioteca pandas para criar um dataframe

show_df = df.sample(frac = 1) # Embaralha as linhas do DataFrame

print("Amostra de 5 dados do dataset qualidade vinho:\n")
print(df.head()) # Exibe os 5 primeiros dados do dataframe df
print("\n Legenda de qualidade do vinho:\n 3 e 4 = Ruim\n 5 e 6 = Médio\n 7 e 8 = Bom\n 9 e 10 = Excelente\n")

# 3 - Visualizar dados do dataset wine quality com biblioteca matplotlib (Eu farei)
show_views(df)

# 4 - Dividir os dados em treino e teste utilizando train_test_split

Y = Y - 3

# Y[(Y == 0) | (Y == 1)] = 0
# Y[(Y == 2) | (Y == 3)] = 1
# Y[(Y == 4) | (Y == 5)] = 2
# Y[Y == 6] = 3

X = X.astype(np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)

# 5 - Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 6 - Preparar Dataset e Loader (Arquivo: dataset.py)
train_dataset = WineQualityDataset(X_train, Y_train)
test_dataset = WineQualityDataset(X_test, Y_test)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 6.5 criando modelo KNN
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train)

# 7 - Criar um modelo de MLP (Arquivo: mlp.py)
input_dim = X_train.shape[1]
hidden_dim = 512
output_dim = len(np.unique(Y_train))

model = SimpleMLP(input_dim, hidden_dim, output_dim)

# 8 - Definir a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# 9 - Treinar o modelo (Fazer uma função de treino)
epochs = 150

Model.train_model(model, optimizer, criterion, train_loader, epochs)

# 9.5 - Testar o modelo - knn

X_new = np.array([[6.3,0.45,0.1,1.2,0.03335,15.5,21.0,0.9946,3.39,0.47,10.0]]) # Array com caracteristicas de um vinho, para ser testado retornando sua qualidade
Model.test_model(X_new, knn)

# 10 - Avaliar o modelo (Fazer uma função de avaliação)

y_true, y_pred = Model.eval_model(model, test_loader)
# class_names = ['Ruim', 'Médio', 'Bom', 'Excelente']
class_names = ['3','4', '5', '6', '7', '8', '9']
plot_confusion_matrix(y_true, y_pred, class_names)