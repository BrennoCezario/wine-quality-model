import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .mlp import SimpleMLP
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier

def parameter_tuning(X_train, X_test, Y_train, Y_test):

    net = NeuralNetClassifier(
        module=SimpleMLP,
        module__input_dim=X_train.shape[1],
        module__output_dim=2,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        batch_size=4,
        iterator_train__shuffle=True,
        verbose=0
    )
    params = {

    }

    gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy', verbose=2)
    gs.fit(X_train, Y_train)
    
    print("\nMelhores parâmetros encontrados:")
    print(gs.best_params_)

    print(f"\nMelhor acurácia: {gs.best_score_ * 100:.4f}%")

    print("\nAvaliando no conjunto de teste...")
    y_pred = gs.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    print(f'Acurácia no teste: {acc * 100:.2f}%')

def train_model(): # Definir a função que treina o modelo
    return

def test_model(): # Definir a função que testa o modelo 
    return

# 1- Carregar os dados do dataset Wine
df = pd.DataFrame()

wine = load_wine() # Função da biblioteca sklearn que carrega o dataset Wine

X = wine.data # Entrada X
Y = wine.target # Saída Esperada Y

# 2 - Apresentar os dados do dataset wine utilizando a biblioteca pandas para criar um dataframe

feature_translation = {
    'alcohol': 'Álcool',
    'malic_acid': 'Ácido málico',
    'ash': 'Cinza',
    'alcalinity_of_ash': 'Alcalinidade das cinzas',
    'magnesium': 'Magnésio',
    'total_phenols': 'Fenóis totais',
    'flavanoids': 'Flavonoides',
    'nonflavanoid_phenols': 'Fenóis não flavonoides',
    'proanthocyanins': 'Proantocianidinas',
    'color_intensity': 'Intensidade da cor',
    'hue': 'Matiz',
    'od280/od315_of_diluted_wines': 'OD280/OD315 de vinhos diluídos',
    'proline': 'Prolina'
}

df = pd.DataFrame(X, columns=wine.feature_names)  # Cria um dataframe com os dados de entrada X e os rotula de acorodo com as features do dataset iris

df.rename(columns=feature_translation, inplace=True) # Renomeia as colunas para português
df['Qualidade'] = Y # Adiciona a coluna 'species' ao dataframe df com os dados de saída Y

show_df = df.sample(frac = 1) # Embaralha as linhas do DataFrame

print("Amostra de 5 dados do dataset iris:\n")
print(show_df.head()) # Exibe os 5 primeiros dados do dataframe df

# 3 - Visualizar dados do dataset iris com biblioteca matplotlib (Eu farei)

# 4 - Dividir os dados em treino e teste utilizando train_test_split

# 5 - Normalizar os dados

# 6 - Preparar Dataset e Loader (Arquivo: dataset.py)

# 7 - Criar um modelo de MLP (Arquivo: mlp.py)

# 8 - Definir a função de perda e o otimizador

# 9 - Treinar o modelo (Fazer uma função de treino)

# 10 - Avaliar o modelo (Fazer uma função de avaliação)