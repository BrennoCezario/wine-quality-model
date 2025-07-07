from dataset import QualidadeVinhoDataset
from mlp import SimpleMLP
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# from skorch import NeuralNetClassifier

# configura dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo em uso:", device)


# def parameter_tuning(X_train, X_test, Y_train, Y_test):

#     net = NeuralNetClassifier(
#         module=SimpleMLP,
#         module__input_dim=X_train.shape[1],
#         module__output_dim=2,
#         criterion=nn.CrossEntropyLoss,
#         optimizer=optim.Adam,
#         batch_size=4,
#         iterator_train__shuffle=True,
#         verbose=0
#     )
#     params = {

#     }

#     gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy', verbose=2)
#     gs.fit(X_train, Y_train)
    
#     print("\nMelhores parâmetros encontrados:")
#     print(gs.best_params_)

#     print(f"\nMelhor acurácia: {gs.best_score_ * 100:.4f}%")

#     print("\nAvaliando no conjunto de teste...")
#     y_pred = gs.predict(X_test)
#     acc = accuracy_score(Y_test, y_pred)
#     print(f'Acurácia no teste: {acc * 100:.2f}%')

def train_model(n_epochs): # Definir a função que treina o modelo
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Época [{epoch+1}/{n_epochs}] - Perda média: {epoch_loss: .4f}")

    print("Treinamento realizado")

    return

def test_model(array_teste): # Definir a função que testa o modelo 
    prediction = knn.predict(array_teste)
    
    # previsao = wine['target_names'][prediction]
    print("O vinho de teste possui qualidade:", prediction)

    return

# 1- Carregar os dados do dataset Wine
# df = pd.DataFrame()
df = pd.read_csv('WineQT.csv')

# wine = load_wine() # Função da biblioteca sklearn que carrega o dataset Wine

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
    'proline': 'Prolina',
    'pH': 'pH',
    'fixed acidity': 'acidez fixa',
    'volatile acidity': 'acidez volátil',
    'citric acid': 'ácido cítrico',
    'residual sugar': 'açúcar residual',
    'chlorides': 'cloretos',
    'free sulfur dioxide': 'dióxido de enxofre livre',
    'total sulfur dioxide': 'dióxido de enxofre total',
    'density': 'densidade',
    'sulphates': 'sulfatos',
    'quality': 'qualidade',
    'Id': 'id'
}

df.rename(columns=feature_translation, inplace=True) # Renomeia as colunas para português

# X = wine.data # Entrada X
# Y = wine.target # Saída Esperada Y

X = df.drop(['qualidade','id'], axis=1).values
Y = df['qualidade'].values

print("Valores X:", X)
# Y vai ter os valores de qualidade
print("Valores Y:", Y)

# 2 - Apresentar os dados do dataset wine utilizando a biblioteca pandas para criar um dataframe



# df = pd.DataFrame(X, columns=wine.feature_names)  # Cria um dataframe com os dados de entrada X e os rotula de acorodo com as features do dataset wine

# df['Qualidade'] = Y # Adiciona a coluna 'species' ao dataframe df com os dados de saída Y

show_df = df.sample(frac = 1) # Embaralha as linhas do DataFrame

print("Amostra de 5 dados do dataset qualidade vinho:\n")
print(show_df.head()) # Exibe os 5 primeiros dados do dataframe df
print("\n Legenda de qualidade do vinho:\n 0 - 3 = Ruim\n 4 - 7 = Normal\n 8 - 10 = Excelente\n")
print(df.columns)

# 3 - Visualizar dados do dataset wine quality com biblioteca matplotlib (Eu farei)

# grafico de dispersão de fixed acidity e pH, relacionando qualidade
plt.scatter(df['acidez fixa'], df['pH'], c=df['qualidade'], cmap='viridis')
plt.xlabel('Fixed Acidity')
plt.ylabel('pH')
plt.title('Fixed Acidity vs pH colored by Quality')
plt.colorbar(label='Quality')
plt.show()

# histograma de Álcool
df['Álcool'].hist()
plt.xlabel('Álcool')
plt.title('Histograma Álcool')
plt.show()

# histograma de pH
df['pH'].hist()
plt.xlabel('pH')
plt.title('Histograma pH')
plt.show()

# box plot Álcool e quality
df.boxplot(column='Álcool', by='qualidade')
plt.xlabel('Qualidade')
plt.ylabel('Álcool')
plt.title('Boxplot Álcool por qualidade')
plt.show()


# 4 - Dividir os dados em treino e teste utilizando train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# 5 - Normalizar os dados

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 6 - Preparar Dataset e Loader (Arquivo: dataset.py)

train_dataset = QualidadeVinhoDataset(X_train, Y_train)
test_dataset = QualidadeVinhoDataset(X_test, Y_test)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 6.5 criando modelo KNN
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train)

# 7 - Criar um modelo de MLP (Arquivo: mlp.py)

input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 12

model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)

# 8 - Definir a função de perda e o otimizador

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# 9 - Treinar o modelo (Fazer uma função de treino)

#configura o número de épocas
num_epochs = 100

train_model(num_epochs)

# 9.5 - Testar o modelo - knn

#array com um valor das caracteristicas de um vinho, para ser testado retornando sua qualidade
X_new = np.array([[6.3,0.45,0.1,1.2,0.03335,15.5,21.0,0.9946,3.39,0.47,10.0]])
test_model(X_new)

# 10 - Avaliar o modelo (Fazer uma função de avaliação)

def eval_model():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nCorretos: {correct}")
    print(f"Total: {total}\n")

    accuracy = 100.0 * correct / total
    print(f"Acurácia: {accuracy: .2f}%")
    if (accuracy < 70):
        print("A acurácia para este dataset está bem baixo se comparado às das flores, que vinha acurácia de aproximadamente 95%\n" \
        "Isso é por conta das qualidades que variam de 0 à 10. Se agrupássemos em 3 grupos ao invés de 11, obteríamos resultados melhores.")
    return

eval_model()