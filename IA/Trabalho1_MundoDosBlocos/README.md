
# Planejador de Blocos com Comprimentos Diferentes (Capítulo 17 - Bratko)

## 📚 Descrição Geral

Este projeto implementa um planejador para o **mundo dos blocos com comprimentos diferentes**, conforme descrito na página 403 do capítulo 17 do livro *Prolog Programming for Artificial Intelligence* de Ivan Bratko.

A implementação foi feita em **Prolog**, utilizando as técnicas de **goal regression** e **means-ends analysis**, respeitando as restrições de estabilidade dos blocos.

## 📌 Entregáveis

✔️ Um texto explicando a resposta de cada pergunta do arquivo `T1_MundoDosBlocos.pdf` (veja a seção abaixo)  
✔️ O código-fonte em Prolog (`.pl`)  
✔️ Instruções claras de execução no ambiente Prolog

---

## 🧠 Representação do Conhecimento

### 1) Descrição do Problema

O mundo é composto por blocos com **comprimentos diferentes**, porém **mesma altura**, dispostos sobre uma **mesa de posições inteiras**. Os blocos podem ser empilhados, desde que estejam **estáveis** (sem outros blocos em cima).

### 2) Definição dos Conceitos

- `on(X, Y)`: o bloco `X` está sobre o bloco ou posição `Y`
- `clear(X)`: não há nada em cima de `X`
- O **estado do mundo** é representado como uma **lista de fatos**, sem o uso de `assert/retract`

### 3) Representação Lógica

Veja [representacao_conhecimento.md](./representacao_conhecimento.md) para todos os detalhes.

---

## 🤖 Raciocínio e Planejamento Automático

- Utilizamos **regressão de objetivos** para decompor o problema e **planejamento orientado a metas**
- A ação principal é `move(Bloco, De, Para)`
- Pré-condições e efeitos de cada ação são modelados com regras declarativas

---

## 📁 Estrutura do Projeto

```
projeto/
├── planejador.pl               # Código-fonte principal em Prolog
├── representacao_conhecimento.md  # Parte textual da entrega 1 a 3
├── README.md                   # Instruções e explicação geral (você está aqui)
```

---

## ▶️ Como Executar

1. **Abra o SWI-Prolog** ou outro interpretador compatível.
2. Carregue o arquivo:

```prolog
?- [planejador].
```

3. Modifique os fatos em:
   - `initial_state/1` para mudar a configuração inicial
   - `goal_list/1` para mudar os objetivos

4. Execute o planejador:

```prolog
?- execute_plan.
```

5. O sistema irá:
   - Mostrar o estado inicial
   - Listar os objetivos
   - Exibir o plano gerado (sequência de ações)
   - Mostrar os estados intermediários até atingir o objetivo

---

## 🧪 Testes e Situações

- A situação 1 foi testada com sucesso (I1 → I2)
- Situações adicionais podem ser testadas alterando `initial_state/1` e `goal_list/1` com base no arquivo `T1_MundoDosBlocos.pdf`

---

## ✍️ Observações

- Nenhum `assert/retract` é utilizado.
- A implementação é puramente declarativa.
- O código segue o modelo fornecido no livro de Bratko.

---
