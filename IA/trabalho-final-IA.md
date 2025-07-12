

#### \# SOLUÇÃO DA QUESTÃO 1 DO TRABALHO FINAL



\# --- Questão 1: Classificador de Sudoku 4x4 ou 9x9 (Google Colab) ---



import numpy as np

import pandas as pd



def carregar\_sudoku\_csv(caminho):

&nbsp;   df = pd.read\_csv(caminho, header=None)

&nbsp;   sudoku = df.to\_numpy()

&nbsp;   return sudoku



def valido\_linha(linha):

&nbsp;   n = len(linha)

&nbsp;   return set(linha) == set(range(1, n + 1))



def valido\_coluna(tabuleiro, col):

&nbsp;   coluna = tabuleiro\[:, col]

&nbsp;   return valido\_linha(coluna)



def valido\_subgrade(tabuleiro, start\_row, start\_col, bloco):

&nbsp;   sub = tabuleiro\[start\_row:start\_row+bloco, start\_col:start\_col+bloco].flatten()

&nbsp;   return valido\_linha(sub)



def classificar\_tabuleiro(tabuleiro):

&nbsp;   n = tabuleiro.shape\[0]

&nbsp;   bloco = int(n\*\*0.5)



&nbsp;   # Verificar se está todo preenchido

&nbsp;   if np.any(tabuleiro == 0):

&nbsp;       return 0  # inválido



&nbsp;   # Verificar linhas

&nbsp;   for i in range(n):

&nbsp;       if not valido\_linha(tabuleiro\[i]):

&nbsp;           return 0



&nbsp;   # Verificar colunas

&nbsp;   for j in range(n):

&nbsp;       if not valido\_coluna(tabuleiro, j):

&nbsp;           return 0



&nbsp;   # Verificar subgrades

&nbsp;   for i in range(0, n, bloco):

&nbsp;       for j in range(0, n, bloco):

&nbsp;           if not valido\_subgrade(tabuleiro, i, j, bloco):

&nbsp;               return 0



&nbsp;   return 1  # válido





\# ---------- Casos de Teste ----------



\# Exemplo válido 4x4

tabuleiro\_4x4\_valido = np.array(\[

&nbsp;   \[1, 2, 3, 4],

&nbsp;   \[3, 4, 1, 2],

&nbsp;   \[2, 1, 4, 3],

&nbsp;   \[4, 3, 2, 1]

])



\# Exemplo inválido 4x4 (repetição na 1ª linha)

tabuleiro\_4x4\_invalido = np.array(\[

&nbsp;   \[1, 1, 3, 4],

&nbsp;   \[3, 4, 1, 2],

&nbsp;   \[2, 3, 4, 1],

&nbsp;   \[4, 2, 1, 3]

])



\# Exemplo válido 9x9

tabuleiro\_9x9\_valido = np.array(\[

&nbsp;   \[5,3,4,6,7,8,9,1,2],

&nbsp;   \[6,7,2,1,9,5,3,4,8],

&nbsp;   \[1,9,8,3,4,2,5,6,7],

&nbsp;   \[8,5,9,7,6,1,4,2,3],

&nbsp;   \[4,2,6,8,5,3,7,9,1],

&nbsp;   \[7,1,3,9,2,4,8,5,6],

&nbsp;   \[9,6,1,5,3,7,2,8,4],

&nbsp;   \[2,8,7,4,1,9,6,3,5],

&nbsp;   \[3,4,5,2,8,6,1,7,9]

])



print("Sudoku 4x4 válido:", classificar\_tabuleiro(tabuleiro\_4x4\_valido))      # 1

print("Sudoku 4x4 inválido:", classificar\_tabuleiro(tabuleiro\_4x4\_invalido))  # 0

print("Sudoku 9x9 válido:", classificar\_tabuleiro(tabuleiro\_9x9\_valido))      # 1





#### \# SOLUÇÃO DA QUESTÃO 2



import numpy as np

import pandas as pd

from copy import deepcopy

from itertools import product



def carregar\_sudoku\_csv(caminho):

&nbsp;   df = pd.read\_csv(caminho, header=None)

&nbsp;   return df.to\_numpy()



def validar\_tabuleiro(tabuleiro):

&nbsp;   n = tabuleiro.shape\[0]

&nbsp;   bloco = int(n \*\* 0.5)

&nbsp;   

&nbsp;   # Verificar linhas

&nbsp;   for i in range(n):

&nbsp;       linha = tabuleiro\[i, :]

&nbsp;       numeros = linha\[linha != 0]

&nbsp;       if len(numeros) != len(set(numeros)):

&nbsp;           return False, f"Valores repetidos na linha {i+1}"

&nbsp;   

&nbsp;   # Verificar colunas

&nbsp;   for j in range(n):

&nbsp;       coluna = tabuleiro\[:, j]

&nbsp;       numeros = coluna\[coluna != 0]

&nbsp;       if len(numeros) != len(set(numeros)):

&nbsp;           return False, f"Valores repetidos na coluna {j+1}"

&nbsp;   

&nbsp;   # Verificar blocos

&nbsp;   for bi in range(bloco):

&nbsp;       for bj in range(bloco):

&nbsp;           ini\_i, ini\_j = bi \* bloco, bj \* bloco

&nbsp;           bloco\_atual = tabuleiro\[ini\_i:ini\_i+bloco, ini\_j:ini\_j+bloco]

&nbsp;           numeros = bloco\_atual\[bloco\_atual != 0]

&nbsp;           if len(numeros) != len(set(numeros)):

&nbsp;               return False, f"Valores repetidos no bloco ({bi+1},{bj+1})"

&nbsp;   

&nbsp;   return True, "Tabuleiro válido"



def pode\_colocar(tabuleiro, linha, col, valor):

&nbsp;   n = tabuleiro.shape\[0]

&nbsp;   bloco = int(n \*\* 0.5)



&nbsp;   if valor in tabuleiro\[linha]: return False

&nbsp;   if valor in tabuleiro\[:, col]: return False



&nbsp;   ini\_linha = (linha // bloco) \* bloco

&nbsp;   ini\_col = (col // bloco) \* bloco

&nbsp;   subgrade = tabuleiro\[ini\_linha:ini\_linha+bloco, ini\_col:ini\_col+bloco]



&nbsp;   if valor in subgrade: return False



&nbsp;   return True



def verificar\_sem\_solucao(tabuleiro):

&nbsp;   n = tabuleiro.shape\[0]

&nbsp;   impossiveis = \[]



&nbsp;   for numero in range(1, n + 1):

&nbsp;       total\_no\_tabuleiro = np.count\_nonzero(tabuleiro == numero)



&nbsp;       if total\_no\_tabuleiro > n:  # Verifica se já excedeu o número permitido

&nbsp;           impossiveis.append((numero, f"já aparece {total\_no\_tabuleiro} vezes (máximo é {n})"))

&nbsp;           continue



&nbsp;       if total\_no\_tabuleiro == n:

&nbsp;           continue



&nbsp;       pode\_inserir = False

&nbsp;       posicoes\_invalidas = \[]

&nbsp;       for i, j in product(range(n), repeat=2):

&nbsp;           if tabuleiro\[i]\[j] == 0:

&nbsp;               if pode\_colocar(tabuleiro, i, j, numero):

&nbsp;                   pode\_inserir = True

&nbsp;               else:

&nbsp;                   posicoes\_invalidas.append((i, j))



&nbsp;       if not pode\_inserir:

&nbsp;           impossiveis.append((numero, posicoes\_invalidas))



&nbsp;   if impossiveis:

&nbsp;       return True, impossiveis

&nbsp;   else:

&nbsp;       return False, None



def numeros\_faltando(tabuleiro):

&nbsp;   n = tabuleiro.shape\[0]

&nbsp;   return \[num for num in range(1, n+1) if np.count\_nonzero(tabuleiro == num) < n]



def aplicar\_movimentos(tabuleiro):

&nbsp;   n = tabuleiro.shape\[0]

&nbsp;   analise = {}

&nbsp;   numeros\_faltantes = numeros\_faltando(tabuleiro)



&nbsp;   for numero in numeros\_faltantes:

&nbsp;       movimentos\_validos = \[]

&nbsp;       leva\_a\_sem\_solucao = False



&nbsp;       for i, j in product(range(n), repeat=2):

&nbsp;           if tabuleiro\[i]\[j] == 0 and pode\_colocar(tabuleiro, i, j, numero):

&nbsp;               novo\_tabuleiro = deepcopy(tabuleiro)

&nbsp;               novo\_tabuleiro\[i]\[j] = numero



&nbsp;               bloqueado, \_ = verificar\_sem\_solucao(novo\_tabuleiro)

&nbsp;               if bloqueado:

&nbsp;                   leva\_a\_sem\_solucao = True



&nbsp;               movimentos\_validos.append((i, j))



&nbsp;       analise\[numero] = {

&nbsp;           "movimentos\_validos": movimentos\_validos,

&nbsp;           "leva\_a\_sem\_solucao": leva\_a\_sem\_solucao

&nbsp;       }



&nbsp;   return analise



def classificar\_tabuleiro\_aberto(tabuleiro):

&nbsp;   n = tabuleiro.shape\[0]

&nbsp;   print(f"Tamanho do tabuleiro: {n}x{n}")

&nbsp;   

&nbsp;   # Validação inicial rigorosa

&nbsp;   valido, mensagem = validar\_tabuleiro(tabuleiro)

&nbsp;   if not valido:

&nbsp;       print(f"\\nSITUAÇÃO: TABULEIRO INVÁLIDO - {mensagem}")

&nbsp;       print("\\nCLASSIFICAÇÃO: 1 (Tabuleiro sem solução possível)")

&nbsp;       return 1

&nbsp;   

&nbsp;   nums\_faltantes = numeros\_faltando(tabuleiro)

&nbsp;   print("\\nNúmeros que ainda podem ser colocados:", nums\_faltantes)

&nbsp;   

&nbsp;   sem\_solucao, info = verificar\_sem\_solucao(tabuleiro)



&nbsp;   if sem\_solucao:

&nbsp;       print("\\nSITUAÇÃO: SEM SOLUÇÃO")

&nbsp;       for item in info:

&nbsp;           numero, detalhes = item

&nbsp;           if isinstance(detalhes, str):  # Caso de número repetido

&nbsp;               print(f"\\nNúmero {numero}: {detalhes}")

&nbsp;           else:

&nbsp;               print(f"\\nNúmero {numero}:")

&nbsp;               print("  Não pode ser colocado em nenhuma posição válida")

&nbsp;               print(f"  Posições testadas: {detalhes}")

&nbsp;       print("\\nCLASSIFICAÇÃO: 1 (Tabuleiro sem solução possível)")

&nbsp;       return 1

&nbsp;   else:

&nbsp;       print("\\nSITUAÇÃO: SOLUÇÃO POSSÍVEL")

&nbsp;       analise = aplicar\_movimentos(tabuleiro)

&nbsp;       

&nbsp;       for num in nums\_faltantes:

&nbsp;           info = analise\[num]

&nbsp;           print(f"\\nNúmero {num}:")

&nbsp;           

&nbsp;           if not info\["movimentos\_validos"]:

&nbsp;               print("  Nenhuma posição válida encontrada")

&nbsp;           else:

&nbsp;               print(f"  Posições válidas: {info\['movimentos\_validos']}")

&nbsp;               if info\["leva\_a\_sem\_solucao"]:

&nbsp;                   print("  Atenção: Algumas destas posições podem levar a um estado sem solução")



&nbsp;       print("\\nCLASSIFICAÇÃO: 2 (Tabuleiro com solução possível)")

&nbsp;       return 2



\# PARA TESTAR:



from google.colab import files

uploaded = files.upload()



\# Nome do arquivo CSV que você carregou:

nome\_arquivo = list(uploaded.keys())\[0]

tabuleiro = carregar\_sudoku\_csv(nome\_arquivo)



\# Ver resultado

classificar\_tabuleiro\_aberto(tabuleiro)



\# EXEMPLO SUDOKU 4X4 VALIDO

\#    1,2,3,4

\#    3,4,1,2

\#    2,1,4,3

\#    4,3,2,0



\# EXEMPLO SUDOKU 4X4 INVALIDO

\#    1, 0, 3, 4

\#    3, 4, 0, 1

\#    0, 1, 4, 3

\#    4, 0, 0, 1





#### \# SOLUÇÃO PARA A QUESTÃO 3



import numpy as np

import pandas as pd

from itertools import product, combinations

from pysat.formula import CNF

from pysat.solvers import Solver

from google.colab import files





class SudokuSolver:

&nbsp;   def \_\_init\_\_(self, tamanho=9):

&nbsp;       self.n = tamanho

&nbsp;       self.bloco = int(tamanho \*\* 0.5)

&nbsp;       self.cnf = CNF()

&nbsp;       self.var\_map = {}



&nbsp;   def \_var(self, r, c, v):

&nbsp;       if (r, c, v) not in self.var\_map:

&nbsp;           self.var\_map\[(r, c, v)] = len(self.var\_map) + 1

&nbsp;       return self.var\_map\[(r, c, v)]



&nbsp;   def carregar\_sudoku\_csv(self, caminho):

&nbsp;       df = pd.read\_csv(caminho, header=None)

&nbsp;       tabuleiro = df.to\_numpy()



&nbsp;       if tabuleiro.shape\[0] != tabuleiro.shape\[1]:

&nbsp;           raise ValueError("Tabuleiro deve ser quadrado")



&nbsp;       self.n = tabuleiro.shape\[0]

&nbsp;       self.bloco = int(self.n \*\* 0.5)

&nbsp;       return tabuleiro.astype(int)



&nbsp;   def validar\_tabuleiro(self, tabuleiro):

&nbsp;       if np.any(tabuleiro < 0) or np.any(tabuleiro > self.n):

&nbsp;           return False



&nbsp;       for i in range(self.n):

&nbsp;           linha = tabuleiro\[i, :]

&nbsp;           numeros = linha\[linha != 0]

&nbsp;           if len(numeros) != len(set(numeros)):

&nbsp;               return False



&nbsp;       for j in range(self.n):

&nbsp;           coluna = tabuleiro\[:, j]

&nbsp;           numeros = coluna\[coluna != 0]

&nbsp;           if len(numeros) != len(set(numeros)):

&nbsp;               return False



&nbsp;       for bi, bj in product(range(self.bloco), repeat=2):

&nbsp;           bloco = tabuleiro\[

&nbsp;               bi \* self.bloco:(bi + 1) \* self.bloco,

&nbsp;               bj \* self.bloco:(bj + 1) \* self.bloco

&nbsp;           ]

&nbsp;           numeros = bloco\[bloco != 0]

&nbsp;           if len(numeros) != len(set(numeros)):

&nbsp;               return False



&nbsp;       return True



&nbsp;   def construir\_cnf(self, tabuleiro):

&nbsp;       self.cnf = CNF()

&nbsp;       self.var\_map = {}



&nbsp;       for r, c in product(range(self.n), repeat=2):

&nbsp;           if tabuleiro\[r]\[c] == 0:

&nbsp;               self.cnf.append(\[self.\_var(r, c, v) for v in range(self.n)])

&nbsp;           else:

&nbsp;               v = tabuleiro\[r]\[c] - 1

&nbsp;               self.cnf.append(\[self.\_var(r, c, v)])



&nbsp;       for r, c in product(range(self.n), repeat=2):

&nbsp;           for v1, v2 in combinations(range(self.n), 2):

&nbsp;               self.cnf.append(\[-self.\_var(r, c, v1), -self.\_var(r, c, v2)])



&nbsp;       for r, v in product(range(self.n), range(self.n)):

&nbsp;           self.cnf.append(\[self.\_var(r, c, v) for c in range(self.n)])

&nbsp;           for c1, c2 in combinations(range(self.n), 2):

&nbsp;               self.cnf.append(\[-self.\_var(r, c1, v), -self.\_var(r, c2, v)])



&nbsp;       for c, v in product(range(self.n), range(self.n)):

&nbsp;           self.cnf.append(\[self.\_var(r, c, v) for r in range(self.n)])

&nbsp;           for r1, r2 in combinations(range(self.n), 2):

&nbsp;               self.cnf.append(\[-self.\_var(r1, c, v), -self.\_var(r2, c, v)])



&nbsp;       for bi, bj, v in product(range(self.bloco), range(self.bloco), range(self.n)):

&nbsp;           celulas = \[

&nbsp;               self.\_var(bi \* self.bloco + i, bj \* self.bloco + j, v)

&nbsp;               for i, j in product(range(self.bloco), repeat=2)

&nbsp;           ]

&nbsp;           self.cnf.append(celulas)

&nbsp;           for i, j in combinations(range(len(celulas)), 2):

&nbsp;               self.cnf.append(\[-celulas\[i], -celulas\[j]])



&nbsp;   def resolver\_sat(self, tabuleiro):

&nbsp;       if not self.validar\_tabuleiro(tabuleiro):

&nbsp;           print("Tabuleiro inválido.")

&nbsp;           return None



&nbsp;       self.construir\_cnf(tabuleiro)

&nbsp;       solver = Solver(bootstrap\_with=self.cnf)



&nbsp;       if solver.solve():

&nbsp;           modelo = solver.get\_model()

&nbsp;           solucao = np.zeros((self.n, self.n), dtype=int)

&nbsp;           for (r, c, v), var in self.var\_map.items():

&nbsp;               if var in modelo:

&nbsp;                   solucao\[r]\[c] = v + 1

&nbsp;           return solucao

&nbsp;       return None



&nbsp;   def analisar\_heuristicas(self, tabuleiro):

&nbsp;       heuristicas = {

&nbsp;           "Naked Single": False,

&nbsp;           "Hidden Single": False,

&nbsp;           "Locked Candidates": False,

&nbsp;           "Naked Pair": False,

&nbsp;           "X-Wing": False

&nbsp;       }



&nbsp;       for r in range(self.n):

&nbsp;           for c in range(self.n):

&nbsp;               if tabuleiro\[r, c] == 0:

&nbsp;                   candidatos = \[v for v in range(1, self.n + 1)

&nbsp;                                 if self.pode\_colocar(tabuleiro, r, c, v)]

&nbsp;                   if len(candidatos) == 1:

&nbsp;                       heuristicas\["Naked Single"] = True



&nbsp;       for v in range(1, self.n + 1):

&nbsp;           for r in range(self.n):

&nbsp;               cols = \[c for c in range(self.n)

&nbsp;                       if tabuleiro\[r, c] == 0 and self.pode\_colocar(tabuleiro, r, c, v)]

&nbsp;               if len(cols) == 1:

&nbsp;                   heuristicas\["Hidden Single"] = True



&nbsp;           for c in range(self.n):

&nbsp;               rows = \[r for r in range(self.n)

&nbsp;                       if tabuleiro\[r, c] == 0 and self.pode\_colocar(tabuleiro, r, c, v)]

&nbsp;               if len(rows) == 1:

&nbsp;                   heuristicas\["Hidden Single"] = True



&nbsp;       print("\\nHeurísticas recomendadas:")

&nbsp;       for h, ativa in heuristicas.items():

&nbsp;           if ativa:

&nbsp;               print(f"- {h}")

&nbsp;       return heuristicas



&nbsp;   def pode\_colocar(self, tabuleiro, r, c, v):

&nbsp;       if v in tabuleiro\[r, :] or v in tabuleiro\[:, c]:

&nbsp;           return False

&nbsp;       br, bc = r // self.bloco, c // self.bloco

&nbsp;       bloco = tabuleiro\[br \* self.bloco:(br + 1) \* self.bloco,

&nbsp;                         bc \* self.bloco:(bc + 1) \* self.bloco]

&nbsp;       return v not in bloco



&nbsp;   def explicacao\_ltn(self):

&nbsp;       """Explicação sobre a possibilidade teórica de resolver Sudoku com LTN"""

&nbsp;       print("\\n=== Sobre a solução com Logic Tensor Networks (LTN) ===")

&nbsp;       print("Teoricamente, sim: LTNs poderiam ser aplicadas para resolver Sudoku,")

&nbsp;       print("pois são redes neurais que combinam lógica simbólica com aprendizado profundo.")

&nbsp;       print("\\nNo entanto, na prática:")

&nbsp;       print("- LTNs são mais complexas para implementar que métodos tradicionais")

&nbsp;       print("- O desempenho geralmente é inferior a algoritmos dedicados como SAT")

&nbsp;       print("- Requerem grande quantidade de dados e tempo de treinamento")

&nbsp;       print("- Resultados são aproximados, não exatos como métodos lógicos")

&nbsp;       print("\\nEste solver optou por manter apenas a implementação SAT, que:")

&nbsp;       print("- Garante soluções exatas")

&nbsp;       print("- É computacionalmente eficiente")

&nbsp;       print("- Tem implementação robusta e testada")

&nbsp;       print("\\nReferências sobre LTN:")

&nbsp;       print("- Badreddine et al. (2020): Logic Tensor Networks")

&nbsp;       print("- Serafini \& Garcez (2016): Logic and Neural Networks")





def testar\_sudoku():

&nbsp;   uploaded = files.upload()

&nbsp;   if not uploaded:

&nbsp;       print("Nenhum arquivo carregado!")

&nbsp;       return



&nbsp;   nome\_arquivo = list(uploaded.keys())\[0]

&nbsp;   solver = SudokuSolver()



&nbsp;   try:

&nbsp;       tabuleiro = solver.carregar\_sudoku\_csv(nome\_arquivo)

&nbsp;       print("\\n=== Tabuleiro Carregado ===")

&nbsp;       print(tabuleiro)



&nbsp;       if not solver.validar\_tabuleiro(tabuleiro):

&nbsp;           print("Tabuleiro inválido!")

&nbsp;           return



&nbsp;       print("\\n=== Análise de Heurísticas ===")

&nbsp;       solver.analisar\_heuristicas(tabuleiro)



&nbsp;       print("\\n=== Solução SAT ===")

&nbsp;       sol\_sat = solver.resolver\_sat(tabuleiro)

&nbsp;       print(sol\_sat)



&nbsp;       # Adicionando a explicação sobre LTN

&nbsp;       solver.explicacao\_ltn()



&nbsp;   except Exception as e:

&nbsp;       print(f"Erro: {str(e)}")





if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   testar\_sudoku()











