````markdown

# SOLUÇÃO DA QUESTÃO 1 DO TRABALHO FINAL

# --- Questão 1: Classificador de Sudoku 4x4 ou 9x9 (Google Colab) ---

import numpy as np
import pandas as pd

def carregar_sudoku_csv(caminho):
    df = pd.read_csv(caminho, header=None)
    sudoku = df.to_numpy()
    return sudoku

def valido_linha(linha):
    n = len(linha)
    return set(linha) == set(range(1, n + 1))

def valido_coluna(tabuleiro, col):
    coluna = tabuleiro[:, col]
    return valido_linha(coluna)

def valido_subgrade(tabuleiro, start_row, start_col, bloco):
    sub = tabuleiro[start_row:start_row+bloco, start_col:start_col+bloco].flatten()
    return valido_linha(sub)

def classificar_tabuleiro(tabuleiro):
    n = tabuleiro.shape[0]
    bloco = int(n**0.5)

    # Verificar se está todo preenchido
    if np.any(tabuleiro == 0):
        return 0  # inválido

    # Verificar linhas
    for i in range(n):
        if not valido_linha(tabuleiro[i]):
            return 0

    # Verificar colunas
    for j in range(n):
        if not valido_coluna(tabuleiro, j):
            return 0

    # Verificar subgrades
    for i in range(0, n, bloco):
        for j in range(0, n, bloco):
            if not valido_subgrade(tabuleiro, i, j, bloco):
                return 0

    return 1  # válido

...

if __name__ == "__main__":
    testar_sudoku()

````
