from .mlp import SimpleMLP
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier

class Model():
    def parameter_tuning(X_train, X_test, Y_train, Y_test):

        net = NeuralNetClassifier(
            module=SimpleMLP,
            module__input_dim=X_train.shape[1],
            module__output_dim=len(np.unique(Y_train)),
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            batch_size=32,
            iterator_train__shuffle=True,
            verbose=0
        )
        params = {
            'module__hidden_dim': [256, 512],
            'lr': [0.0005, 0.0001],
            'max_epochs': [10, 100, 150]
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

    def train_model(model, optimizer, criterion, train_loader, epochs):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for _, (features, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                outputs = model(features)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Época [{epoch+1}/{epochs}] - Perda média: {epoch_loss: .4f}")

        print("Treinamento realizado")

    def test_model(array_test, knn): 
        prediction = knn.predict(array_test)
        
        print("O vinho de teste possui qualidade:", prediction+1)

    # Dentro do seu arquivo model.py, na classe Model

    def eval_model(model, test_loader):
        model.eval()  # Coloca o modelo em modo de avaliação
        all_labels = []
        all_predicted = []

        with torch.no_grad(): # Desativa o cálculo de gradientes
            for features, labels in test_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

        # Cálculo da acurácia (como você já tinha)
        correct = (np.array(all_predicted) == np.array(all_labels)).sum()
        total = len(all_labels)
        accuracy = 100.0 * correct / total
        print(f"\nAcurácia do modelo MLP no teste: {accuracy:.2f}%")

        return all_labels, all_predicted