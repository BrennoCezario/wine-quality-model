import torch
import torch.utils.data as data

class WineQualityDataset(data.Dataset): # Cria uma classe IrisDataset que herda de data.Dataset
    def __init__(self, X, Y): # Método construtor da classe IrisDataset
        self.X = torch.tensor(X, dtype=torch.float32) # Converte os dados de entrada X para tensor
        self.Y = torch.tensor(Y, dtype=torch.long) # Converte os dados de saída Y para tensor

    def __len__(self): # Método que retorna o tamanho do dataset
        return len(self.X)

    def __getitem__(self, idx): # Método que retorna um item do dataset
        return self.X[idx], self.Y[idx]