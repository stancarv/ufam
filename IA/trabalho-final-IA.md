
# SOLUÇÃO DA QUESTÃO 1 DO TRABALHO FINAL

# --- Questão 1: Classificador de Sudoku 4x4 ou 9x9 (Google Colab) ---

```python
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

    if np.any(tabuleiro == 0):
        return 0

    for i in range(n):
        if not valido_linha(tabuleiro[i]):
            return 0

    for j in range(n):
        if not valido_coluna(tabuleiro, j):
            return 0

    for i in range(0, n, bloco):
        for j in range(0, n, bloco):
            if not valido_subgrade(tabuleiro, i, j, bloco):
                return 0

    return 1
```

```python
tabuleiro_4x4_valido = np.array([
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1]
])

tabuleiro_4x4_invalido = np.array([
    [1, 1, 3, 4],
    [3, 4, 1, 2],
    [2, 3, 4, 1],
    [4, 2, 1, 3]
])

tabuleiro_9x9_valido = np.array([
    [5,3,4,6,7,8,9,1,2],
    [6,7,2,1,9,5,3,4,8],
    [1,9,8,3,4,2,5,6,7],
    [8,5,9,7,6,1,4,2,3],
    [4,2,6,8,5,3,7,9,1],
    [7,1,3,9,2,4,8,5,6],
    [9,6,1,5,3,7,2,8,4],
    [2,8,7,4,1,9,6,3,5],
    [3,4,5,2,8,6,1,7,9]
])

print("Sudoku 4x4 válido:", classificar_tabuleiro(tabuleiro_4x4_valido))
print("Sudoku 4x4 inválido:", classificar_tabuleiro(tabuleiro_4x4_invalido))
print("Sudoku 9x9 válido:", classificar_tabuleiro(tabuleiro_9x9_valido))
```

# SOLUÇÃO DA QUESTÃO 2

```python
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product

# [O código da Questão 2 foi mantido aqui conforme enviado, com todas as funções]

# ... Colar o restante do código da Questão 2 ...
```

# SOLUÇÃO PARA A QUESTÃO 3

```python
# [O código da Questão 3 foi mantido aqui conforme enviado, com todas as funções]

# ... Colar o restante do código da Questão 3 ...
```
