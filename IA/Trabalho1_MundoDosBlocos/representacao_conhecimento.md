
# PARTE REPRESENTAÇÃO DO CONHECIMENTO

#### Integrantes:

Jhonatas Costa Oliveira

Stanley de Carvalho Monteiro

Ícaro Costa Moreira

### 1) Descrição do Problema em Linguagem Natural

O problema trata de um mundo com blocos de diferentes tamanhos, mas de mesma altura. Esses blocos podem ser empilhados uns sobre os outros, desde que respeitem a condição de estabilidade. Ou seja, um bloco só pode ser colocado sobre outro se este último não tiver nenhum outro bloco em cima dele, garantindo a estabilidade.

As trajetórias de movimento dos blocos são simples:
- Os blocos são movidos verticalmente até ficarem sobre outros blocos ou em uma posição livre.
- Após isso, os blocos podem ser movidos horizontalmente para uma nova posição.

Neste mundo, os blocos são sempre estáveis e não são rotacionados, sendo movidos apenas para cima, para baixo ou lateralmente.

### 2) Definição dos Conceitos

- **Clear(X):** Um bloco `X` é considerado **livre** (ou "clear") se não houver nenhum outro bloco em cima dele. Em Prolog, isso pode ser representado da seguinte forma:
  ```prolog
  clear(X) :- \+ (on(_, X)).
  ```

- **On(X, Y):** A relação `on(X, Y)` indica que o bloco `X` está colocado sobre o bloco `Y`. Em Prolog, isso pode ser representado como:
  ```prolog
  on(a, b).  % O bloco 'a' está em cima de 'b'
  ```

- **Estado do Mundo:** O estado do mundo pode ser representado por uma lista de relações de sobreposição. Cada relação `on(X, Y)` define que o bloco `X` está em cima do bloco `Y`, e cada `clear(X)` indica que o bloco `X` está livre. Por exemplo:
  ```prolog
  on(a, b), on(b, c), clear(d).
  ```
  Neste exemplo, o bloco `a` está sobre `b`, `b` está sobre `c`, e o bloco `d` está livre.

### 3) Representação Lógica em Prolog

A representação lógica em Prolog para os elementos descritos no documento pode ser feita utilizando os predicados `on/2` e `clear/1`, sem o uso de `assign` ou `retract`.

#### Representação de Relacionamentos:

```prolog
% Definindo as relações de sobreposição entre os blocos
on(a, b).  % O bloco 'a' está sobre o bloco 'b'
on(b, c).  % O bloco 'b' está sobre o bloco 'c'
clear(d).  % O bloco 'd' está livre

% Definindo que um bloco está livre se não houver nada em cima dele
clear(X) :- \+ (on(_, X)).
```

#### Representação de Estado do Mundo:

O estado do mundo pode ser representado por uma lista de fatos que descrevem a posição dos blocos e as condições de "livre" (clear).

```prolog
% Estado do Mundo
on(a, b).
on(b, c).
clear(d).
```

Este conjunto de fatos define que:
- O bloco `a` está sobre `b`.
- O bloco `b` está sobre `c`.
- O bloco `d` está livre (não há nenhum bloco sobre ele).

### Observação

- **Sem o uso de `assign` e `retract`:** A representação dos estados e das relações entre blocos é feita diretamente com predicados lógicos em Prolog. O uso de `assign` e `retract` não é necessário, pois o estado é simplesmente representado por relações lógicas fixas.

#### Exemplo de Saída. s_inicial=i1 ate o estado s_final=i2 do T1_MundoDosBlocos.pdf

![image](https://github.com/user-attachments/assets/5bb8b3f9-a223-40a4-b356-595c6d0e9f64)

