% Aluno: Stanley de Carvalho Monteiro - 21950882

% Definindo as probabilidades a priori
0.7::str(dry).
0.2::str(wet).
0.1::str(snow_covered).

0.1::fiw.
0.9::not_fiw :- \+fiw.

% Regras para R (Dínamo)
0.8::r :- (fiw; str(snow_covered)).
0.1::r :- \+fiw, \+str(snow_covered).

% Regras para V (Tensão)
0.9::v :- r.
0.2::v :- \+r.

% Componentes independentes
0.95::b.
0.9::k.

% Luz ligada
0.99::li :- v, b, k.
0.1::li :- v, (\+b; \+k).
0.0::li :- \+v.

% Consulta condicionada P(V=t|Str=snow_covered)
evidence(str(snow_covered), true).
query(v).
