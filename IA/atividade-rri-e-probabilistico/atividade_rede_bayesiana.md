# Atividade - Rede Bayesiana

O código adaptando o alarm_sensor.pl e bayes_net_interpreter.pl postado no classroom para um código calls_alarm.pl baseado na figura 13.2 do material *StuartRussell_PeterNorvig_2020_AIMA_Chap13.pdf*.

A causalidade **burglary** e **lightning** dispara **sensor** foi modificada para causalidade **burglary** e **earthquake** dispara **alarm**.

---

### Antes (exemplo simplificado do `parent` e `p/2` e `p/3`):

```prolog
parent(burglary, sensor).
parent(lightning, sensor).
parent(sensor, alarm).
parent(sensor, call).

p(burglary, 0.001).
p(lightning, 0.02).
p(sensor, [burglary, lightning], 0.9).
p(sensor, [burglary, not(lightning)], 0.9).
p(sensor, [not(burglary), lightning], 0.1).
p(sensor, [not(burglary), not(lightning)], 0.001).
p(alarm, [sensor], 0.95).
p(alarm, [not(sensor)], 0.001).
p(call, [sensor], 0.9).
p(call, [not(sensor)], 0.0).
```

---

### Depois (adaptado para a rede da figura 13.2):

```prolog
parent(burglary, alarm).
parent(earthquake, alarm).
parent(alarm, johncalls).
parent(alarm, marycalls).

p(burglary, 0.001).
p(earthquake, 0.002).

p(alarm, [burglary, earthquake], 0.7).
p(alarm, [burglary, not(earthquake)], 0.01).
p(alarm, [not(burglary), earthquake], 0.7).
p(alarm, [not(burglary), not(earthquake)], 0.01).

p(johncalls, [alarm], 0.9).
p(johncalls, [not(alarm)], 0.05).

p(marycalls, [alarm], 0.7).
p(marycalls, [not(alarm)], 0.01).
```

---

**Resumo das mudanças principais:**

- `sensor` foi removido, substituído por `alarm`.
- Os pais de `alarm` são `burglary` e `earthquake`.
- Os filhos de `alarm` são `johncalls` e `marycalls`.
- As probabilidades condicionais foram atualizadas conforme a tabela da figura 13.2.
- Nomes das variáveis e fatos atualizados para refletir a nova topologia e CPTs.

---
###Integrantes:###

STANLEY DE CARVALHO MONTEIRO
JHONATAS COSTA OLIVEIRA
FERNANDA DE OLIVEIRA DA COSTA
ANA LETÍCIA DOS SANTOS SOUZA
ÍCARO COSTA MOREIRA
ROSINEIDE SANTANA SANTOS
