
% -------------------------------------------------------------
% Planejador de blocos com comprimentos diferentes - Bratko Cap.17
% -------------------------------------------------------------
% Este arquivo implementa um sistema de planejamento baseado em goal regression
% para organizar blocos de tamanhos diferentes sobre uma mesa.
% -------------------------------------------------------------

% Integrantes:

% Jhonatas Costa Oliveira

% Stanley de Carvalho Monteiro

% Ícaro Costa Moreira

% -------------------------------------------------------------
% Predicado principal: executa o plano completo
% -------------------------------------------------------------
execute_plan :-
    initial_state(State),
    goal_list(Goals),
    format('Initial State: ~w~n~n', [State]),
    format('Goal: ~w~n~n', [Goals]),
    (plan(State, Goals, [], Plan) ->
        format('~nPlan: ~w~n', [Plan]),
        show_plan(Plan, State, 1)
    ;
        format('~nPlan Not Found~n', [])
    ).

% Mostra o plano passo a passo
show_plan([], _, _).
show_plan([Action|Actions], State, N) :-
    apply_action(Action, State, NewState),
    format('~n~w. ~w~nNew State: ~w~n', [N, Action, NewState]),
    N1 is N + 1,
    show_plan(Actions, NewState, N1).

% Verifica se todos os objetivos estão satisfeitos
all_goals_satisfied(State, Goals) :-
    forall(member(Goal, Goals), holds(Goal, State)).

% Caso base: todos os objetivos já satisfeitos
plan(State, Goals, _, []) :-
    all_goals_satisfied(State, Goals).

% Caso recursivo: seleciona objetivo, encontra ação, aplica e planeja recursivamente
plan(State, Goals, Visited, Plan) :-
    select_goal(State, Goals, Goal),
    format('~n> Select Goal: ~w~n', [Goal]),

    find_valid_action(Goal, State, Action),
    format('> Action: ~w~n', [Action]),

    action_preconditions(Action, Preconditions),
    check_conditions(Preconditions, State),
    format('> Precondition: ~w~n', [Preconditions]),

    preserves_action(Action, Goals),

    regress_goals(State, Goals, Action, RegressedGoals),
    format('> Regressed Goals: ~w~n', [RegressedGoals]),

    apply_action(Action, State, NewState),
    \+ member(NewState, Visited),

    plan(NewState, RegressedGoals, [State|Visited], PrePlan),
    append(PrePlan, [Action], Plan).

% -------------------------------------------------------------
% SUPORTE E RESTRIÇÕES
% -------------------------------------------------------------

% Verifica se uma condição é satisfeita no estado
holds(Cond, State) :- member(Cond, State).
holds(different(X,Y), _) :- X \= Y.

% Verifica todas as condições
check_conditions([], _).
check_conditions([Cond|Conds], State) :-
    holds(Cond, State),
    check_conditions(Conds, State).

% Seleciona um objetivo ainda não satisfeito
select_goal(State, Goals, Goal) :-
    findall(G, (member(G, Goals), \+ holds(G, State)), Unsatisfied),
    (member(clear(X), Unsatisfied), member(on(_,X), State) -> Goal = clear(X) ;
     member(on(c,a), Unsatisfied) -> Goal = on(c,a) ;
     member(on(b,d), Unsatisfied) -> Goal = on(b,d) ;
     member(on(a,d), Unsatisfied) -> Goal = on(a,d) ;
     member(on(d,4), Unsatisfied) -> Goal = on(d,4) ;
     member(Goal, Unsatisfied)).

% Define ações válidas baseadas no objetivo
find_valid_action(on(Block,Pos), State, move(Block,From,Pos)) :-
    block(Block),
    (place(Pos); block(Pos)),
    member(on(Block,From), State),
    From \= Pos,
    \+ member(on(_,Pos), State).

find_valid_action(clear(Pos), State, move(Block,Pos,To)) :-
    (place(Pos); block(Pos)),
    block(Block),
    member(on(Block,Pos), State),
    (place(To); block(To)),
    To \= Pos,
    Block \= To,
    \+ member(on(_,To), State).

% Pré-condições para uma ação de movimento
action_preconditions(move(Block,From,To),
    [clear(Block), clear(To), on(Block,From),
     different(Block,To), different(From,To)]).

% Efeitos da ação: adiciona e remove condições
action_adds(move(Block,From,To), [on(Block,To), clear(From)]).
action_deletes(move(Block,From,To), [on(Block,From), clear(To)]).

% Garante que a ação não destrua metas já satisfeitas
preserves_action(Action, Goals) :-
    action_deletes(Action, Dels),
    \+ (member(Goal, Dels), member(Goal, Goals)).

% Regride metas com base nos efeitos da ação
regress_goals(State, Goals, Action, RegressedGoals) :-
    action_adds(Action, AddList),
    delete_all(Goals, AddList, TempGoals),
    action_preconditions(Action, Preconds),
    exclude(holds_in(State), Preconds, NewPreconds),
    add_new_goals(NewPreconds, TempGoals, RegressedGoals).

holds_in(State, Cond) :- holds(Cond, State).

% Aplica os efeitos da ação no estado atual
apply_action(move(Block,From,To), State, NewState) :-
    delete(State, on(Block,From), Temp1),
    (member(clear(To), State) -> delete(Temp1, clear(To), Temp2) ; Temp2 = Temp1),
    append(Temp2, [on(Block,To), clear(From)], NewState).

% -------------------------------------------------------------
% DEFINIÇÕES DO MUNDO
% -------------------------------------------------------------

add_new_goals([], L, L).
add_new_goals([Goal|Rest], Goals, NewGoals) :-
    (impossible(Goal, Goals), \+ holds(Goal, Goals) -> !, fail
    ; add_new_goals(Rest, Goals, G1),
      (member(Goal, Goals) -> NewGoals = G1 ; NewGoals = [Goal|G1])).

delete_all([], _, []).
delete_all([X|Xs], L, Result) :- member(X, L), !, delete_all(Xs, L, Result).
delete_all([X|Xs], L, [X|Result]) :- delete_all(Xs, L, Result).

% Regras de impossibilidade
impossible(on(X,X), _) :- !.
impossible(on(Block,Pos), Goals) :-
    member(on(Block,OtherPos), Goals), OtherPos \= Pos, !.
impossible(on(Block,Pos), Goals) :-
    member(on(OtherBlock,Pos), Goals), OtherBlock \= Block, !.
impossible(clear(Pos), Goals) :-
    member(on(_,Pos), Goals), !.

% Blocos e posições válidas
block(a). block(b). block(c). block(d).
place(0). place(1). place(3). place(4). place(5). place(6).

% Estado inicial do mundo
initial_state([
    clear(0), clear(3), clear(c), clear(d),
    on(c,1),
    on(a,4),
    on(b,6),
    on(d,a)
]).

% Lista de objetivos a serem atingidos
goal_list([
    clear(0),
    on(a,c),
    on(c,1),
    on(d,3),
    on(b,6)
]).

% Deleta um elemento da lista
delete([X|Xs], X, Xs) :- !.
delete([Y|Xs], X, [Y|Ys]) :- delete(Xs, X, Ys).
