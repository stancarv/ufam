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


# ---------- Casos de Teste ----------

# Exemplo válido 4x4
tabuleiro_4x4_valido = np.array([
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1]
])

# Exemplo inválido 4x4 (repetição na 1ª linha)
tabuleiro_4x4_invalido = np.array([
    [1, 1, 3, 4],
    [3, 4, 1, 2],
    [2, 3, 4, 1],
    [4, 2, 1, 3]
])

# Exemplo válido 9x9
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

print("Sudoku 4x4 válido:", classificar_tabuleiro(tabuleiro_4x4_valido))      # 1
print("Sudoku 4x4 inválido:", classificar_tabuleiro(tabuleiro_4x4_invalido))  # 0
print("Sudoku 9x9 válido:", classificar_tabuleiro(tabuleiro_9x9_valido))      # 1
```

... (conteúdo completo omitido aqui por brevidade)


```python
# PARA TESTAR:

from google.colab import files
uploaded = files.upload()

# Nome do arquivo CSV que você carregou:
nome_arquivo = list(uploaded.keys())[0]
tabuleiro = carregar_sudoku_csv(nome_arquivo)

# Ver resultado
classificar_tabuleiro_aberto(tabuleiro)

# EXEMPLO SUDOKU 4X4 VALIDO
#    1,2,3,4
#    3,4,1,2
#    2,1,4,3
#    4,3,2,0

# EXEMPLO SUDOKU 4X4 INVALIDO
#    1, 0, 3, 4
#    3, 4, 0, 1
#    0, 1, 4, 3
#    4, 0, 0, 1
```

# SOLUÇÃO PARA A QUESTÃO 3

```python
import numpy as np
import pandas as pd
from itertools import product, combinations
from pysat.formula import CNF
from pysat.solvers import Solver
from google.colab import files


class SudokuSolver:
    def __init__(self, tamanho=9):
        self.n = tamanho
        self.bloco = int(tamanho ** 0.5)
        self.cnf = CNF()
        self.var_map = {}

    def _var(self, r, c, v):
        if (r, c, v) not in self.var_map:
            self.var_map[(r, c, v)] = len(self.var_map) + 1
        return self.var_map[(r, c, v)]

    def carregar_sudoku_csv(self, caminho):
        df = pd.read_csv(caminho, header=None)
        tabuleiro = df.to_numpy()

        if tabuleiro.shape[0] != tabuleiro.shape[1]:
            raise ValueError("Tabuleiro deve ser quadrado")

        self.n = tabuleiro.shape[0]
        self.bloco = int(self.n ** 0.5)
        return tabuleiro.astype(int)

    def validar_tabuleiro(self, tabuleiro):
        if np.any(tabuleiro < 0) or np.any(tabuleiro > self.n):
            return False

        for i in range(self.n):
            linha = tabuleiro[i, :]
            numeros = linha[linha != 0]
            if len(numeros) != len(set(numeros)):
                return False

        for j in range(self.n):
            coluna = tabuleiro[:, j]
            numeros = coluna[coluna != 0]
            if len(numeros) != len(set(numeros)):
                return False

        for bi, bj in product(range(self.bloco), repeat=2):
            bloco = tabuleiro[
                bi * self.bloco:(bi + 1) * self.bloco,
                bj * self.bloco:(bj + 1) * self.bloco
            ]
            numeros = bloco[bloco != 0]
            if len(numeros) != len(set(numeros)):
                return False

        return True

    def construir_cnf(self, tabuleiro):
        self.cnf = CNF()
        self.var_map = {}

        for r, c in product(range(self.n), repeat=2):
            if tabuleiro[r][c] == 0:
                self.cnf.append([self._var(r, c, v) for v in range(self.n)])
            else:
                v = tabuleiro[r][c] - 1
                self.cnf.append([self._var(r, c, v)])

        for r, c in product(range(self.n), repeat=2):
            for v1, v2 in combinations(range(self.n), 2):
                self.cnf.append([-self._var(r, c, v1), -self._var(r, c, v2)])

        for r, v in product(range(self.n), range(self.n)):
            self.cnf.append([self._var(r, c, v) for c in range(self.n)])
            for c1, c2 in combinations(range(self.n), 2):
                self.cnf.append([-self._var(r, c1, v), -self._var(r, c2, v)])

        for c, v in product(range(self.n), range(self.n)):
            self.cnf.append([self._var(r, c, v) for r in range(self.n)])
            for r1, r2 in combinations(range(self.n), 2):
                self.cnf.append([-self._var(r1, c, v), -self._var(r2, c, v)])

        for bi, bj, v in product(range(self.bloco), range(self.bloco), range(self.n)):
            celulas = [
                self._var(bi * self.bloco + i, bj * self.bloco + j, v)
                for i, j in product(range(self.bloco), repeat=2)
            ]
            self.cnf.append(celulas)
            for i, j in combinations(range(len(celulas)), 2):
                self.cnf.append([-celulas[i], -celulas[j]])

    def resolver_sat(self, tabuleiro):
        if not self.validar_tabuleiro(tabuleiro):
            print("Tabuleiro inválido.")
            return None

        self.construir_cnf(tabuleiro)
        solver = Solver(bootstrap_with=self.cnf)

        if solver.solve():
            modelo = solver.get_model()
            solucao = np.zeros((self.n, self.n), dtype=int)
            for (r, c, v), var in self.var_map.items():
                if var in modelo:
                    solucao[r][c] = v + 1
            return solucao
        return None

    def analisar_heuristicas(self, tabuleiro):
        heuristicas = {
            "Naked Single": False,
            "Hidden Single": False,
            "Locked Candidates": False,
            "Naked Pair": False,
            "X-Wing": False
        }

        for r in range(self.n):
            for c in range(self.n):
                if tabuleiro[r, c] == 0:
                    candidatos = [v for v in range(1, self.n + 1)
                                  if self.pode_colocar(tabuleiro, r, c, v)]
                    if len(candidatos) == 1:
                        heuristicas["Naked Single"] = True

        for v in range(1, self.n + 1):
            for r in range(self.n):
                cols = [c for c in range(self.n)
                        if tabuleiro[r, c] == 0 and self.pode_colocar(tabuleiro, r, c, v)]
                if len(cols) == 1:
                    heuristicas["Hidden Single"] = True

            for c in range(self.n):
                rows = [r for r in range(self.n)
                        if tabuleiro[r, c] == 0 and self.pode_colocar(tabuleiro, r, c, v)]
                if len(rows) == 1:
                    heuristicas["Hidden Single"] = True

        print("
Heurísticas recomendadas:")
        for h, ativa in heuristicas.items():
            if ativa:
                print(f"- {h}")
        return heuristicas

    def pode_colocar(self, tabuleiro, r, c, v):
        if v in tabuleiro[r, :] or v in tabuleiro[:, c]:
            return False
        br, bc = r // self.bloco, c // self.bloco
        bloco = tabuleiro[br * self.bloco:(br + 1) * self.bloco,
                          bc * self.bloco:(bc + 1) * self.bloco]
        return v not in bloco

    def explicacao_ltn(self):
        print("\n=== Sobre a solução com Logic Tensor Networks (LTN) ===")
        print("Teoricamente, sim: LTNs poderiam ser aplicadas para resolver Sudoku,")
        print("pois são redes neurais que combinam lógica simbólica com aprendizado profundo.")
        print("\nNo entanto, na prática:")
        print("- LTNs são mais complexas para implementar que métodos tradicionais")
        print("- O desempenho geralmente é inferior a algoritmos dedicados como SAT")
        print("- Requerem grande quantidade de dados e tempo de treinamento")
        print("- Resultados são aproximados, não exatos como métodos lógicos")
        print("\nEste solver optou por manter apenas a implementação SAT, que:")
        print("- Garante soluções exatas")
        print("- É computacionalmente eficiente")
        print("- Tem implementação robusta e testada")
        print("\nReferências sobre LTN:")
        print("- Badreddine et al. (2020): Logic Tensor Networks")
        print("- Serafini & Garcez (2016): Logic and Neural Networks")


def testar_sudoku():
    uploaded = files.upload()
    if not uploaded:
        print("Nenhum arquivo carregado!")
        return

    nome_arquivo = list(uploaded.keys())[0]
    solver = SudokuSolver()

    try:
        tabuleiro = solver.carregar_sudoku_csv(nome_arquivo)
        print("\n=== Tabuleiro Carregado ===")
        print(tabuleiro)

        if not solver.validar_tabuleiro(tabuleiro):
            print("Tabuleiro inválido!")
            return

        print("\n=== Análise de Heurísticas ===")
        solver.analisar_heuristicas(tabuleiro)

        print("\n=== Solução SAT ===")
        sol_sat = solver.resolver_sat(tabuleiro)
        print(sol_sat)

        solver.explicacao_ltn()

    except Exception as e:
        print(f"Erro: {str(e)}")


if __name__ == "__main__":
    testar_sudoku()
```
