import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def show_views(df):
    
    # grafico de dispersão de álcool e pH, relacionando qualidade
    plt.scatter(df['Álcool'], df['pH'], c=df['Qualidade'], cmap='viridis')
    plt.xlabel('Álcool')
    plt.ylabel('pH')
    plt.title('Gráfico de dispersão: Álcool x pH')
    plt.colorbar(label='Quality')
    plt.savefig('data/img/dispersao.png', dpi=600)
    plt.show()

    # Contagem de cada valor de qualidade
    quality_counts = df['Qualidade'].value_counts().sort_index()

    # Criando o gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(quality_counts.index, quality_counts.values, color=sns.color_palette("viridis", len(quality_counts)))

    # Adicionando título e rótulos
    ax.set_title('Distribuição das Notas de Qualidade do Vinho', fontsize=16, fontweight='bold')
    ax.set_xlabel('Qualidade', fontsize=12)
    ax.set_ylabel('Número de Amostras', fontsize=12)
    ax.set_xticks(quality_counts.index) # Garante que todos os valores de qualidade apareçam no eixo x

    # Adicionando o número em cima de cada barra
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center') # va: vertical alignment
    
    plt.savefig('data/img/distribuicao.png', dpi=600)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # Criando o box plot
    sns.boxplot(x='Qualidade', y='Álcool', data=df, ax=ax, palette="coolwarm")

    # Adicionando título e rótulos
    ax.set_title('Relação entre Qualidade e Teor Alcoólico', fontsize=16, fontweight='bold')
    ax.set_xlabel('Qualidade', fontsize=12)
    ax.set_ylabel('Teor Alcoólico (%)', fontsize=12)

    plt.savefig('data/img/box_plot.png', dpi=600)
    plt.show()
    
    # Calcula a matriz de correlação
    correlation_matrix = df.corr()

    # Criando o heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, annot_kws={"size": 10})
    ax.set_title('Mapa de Calor da Correlação entre Variáveis', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig('data/img/heatmap.png', dpi=600)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    # Calcula a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # Cria a figura e o eixo com matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plota o heatmap com seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)

    # Adiciona títulos e rótulos
    ax.set_title('Matriz de Confusão', fontsize=16, fontweight='bold')
    ax.set_xlabel('Valores Previstos', fontsize=12)
    ax.set_ylabel('Valores Reais', fontsize=12)
    
    plt.savefig('data/img/confusao.png', dpi=600)
    plt.show()