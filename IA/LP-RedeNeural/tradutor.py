# Integrantes:
# STANLEY DE CARVALHO MONTEIRO
# JHONATAS COSTA OLIVEIRA
# FERNANDA DE OLIVEIRA DA COSTA
# ANA LETÍCIA DOS SANTOS SOUZA
# ÍCARO COSTA MOREIRA
# ROSINEIDE SANTANA SANTOS

import numpy as np
import csv

class CILPNetwork:
    def __init__(self):
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []
        self.weights = {}
        self.thresholds = {}
        self.A_min = 0.7
        self.beta = 1.0
        
    def bipolar_sigmoid(self, x):
        return (2 / (1 + np.exp(-self.beta * x))) - 1
    
    def linear(self, x):
        return x
    
    def translate_program(self, program_file):
        with open(program_file, 'r') as f:
            reader = csv.reader(f)
            clauses = [row[0] for row in reader]
        
        self.process_clauses(clauses)
        self.calculate_parameters()
    
    def process_clauses(self, clauses):
        variables = set()
        clause_info = []
        
        for clause in clauses:
            head, body = clause.split('<-')
            head = head.strip()
            body_literals = [lit.strip() for lit in body.split(',')] if body.strip() else []
            
            pos_literals = [lit for lit in body_literals if not lit.startswith('not ')]
            neg_literals = [lit[4:] for lit in body_literals if lit.startswith('not ')]
            
            variables.add(head)
            variables.update(pos_literals)
            variables.update(neg_literals)
            
            clause_info.append({
                'head': head,
                'pos_literals': pos_literals,
                'neg_literals': neg_literals,
                'k_l': len(body_literals),
                'p_l': len(pos_literals),
                'n_l': len(neg_literals)
            })
        
        head_counts = {}
        for info in clause_info:
            head = info['head']
            head_counts[head] = head_counts.get(head, 0) + 1
        
        for info in clause_info:
            info['mu_l'] = head_counts[info['head']]
        
        self.variables = list(variables)
        self.clause_info = clause_info
    
    def calculate_parameters(self):
        max_k = max(info['k_l'] for info in self.clause_info)
        max_mu = max(info['mu_l'] for info in self.clause_info)
        max_P = max(max_k, max_mu)
        
        numerator = 2 * (np.log(1 + self.A_min) - np.log(1 - self.A_min))
        denominator = max_P * (self.A_min - 1) + self.A_min + 1
        self.W = numerator / (self.beta * denominator)
        
        self.input_neurons = self.variables
        self.output_neurons = list(set(info['head'] for info in self.clause_info))
        self.hidden_neurons = [f'N_{i}' for i in range(len(self.clause_info))]
        
        self.weights = {}
        
        for i, info in enumerate(self.clause_info):
            N_l = f'N_{i}'
            
            for lit in info['pos_literals']:
                self.weights[(lit, N_l)] = self.W
            for lit in info['neg_literals']:
                self.weights[(lit, N_l)] = -self.W
            
            self.weights[(N_l, info['head'])] = self.W
            self.thresholds[N_l] = ((1 + self.A_min) * (info['k_l'] - 1) / 2) * self.W
            
            if info['head'] not in self.thresholds:
                self.thresholds[info['head']] = ((1 + self.A_min) * (1 - info['mu_l']) / 2) * self.W
        
        for inp in self.input_neurons:
            for hid in self.hidden_neurons:
                if (inp, hid) not in self.weights:
                    self.weights[(inp, hid)] = 0
        
        for hid in self.hidden_neurons:
            for out in self.output_neurons:
                if (hid, out) not in self.weights:
                    self.weights[(hid, out)] = 0
    
    def evaluate(self, input_values):
        for var in self.input_neurons:
            if var not in input_values:
                raise ValueError(f"Valor de entrada faltando para variável: {var}")
        
        input_activations = {var: self.linear(input_values[var]) for var in self.input_neurons}
        
        hidden_activations = {}
        for N_l in self.hidden_neurons:
            input_potential = -self.thresholds[N_l]
            for var in self.input_neurons:
                input_potential += input_activations[var] * self.weights.get((var, N_l), 0)
            hidden_activations[N_l] = self.bipolar_sigmoid(input_potential)
        
        output_activations = {}
        for out in self.output_neurons:
            input_potential = -self.thresholds[out]
            for N_l in self.hidden_neurons:
                input_potential += hidden_activations[N_l] * self.weights.get((N_l, out), 0)
            output_activations[out] = self.bipolar_sigmoid(input_potential)
        
        return output_activations
    
    def generate_truth_table(self):
        input_neurons = [v for v in self.input_neurons if v != 'T']
        n = len(input_neurons)

        truth_table = []
        
        for i in range(2**n):
            bits = format(i, f'0{n}b')
            
            input_values = {
                var: 1 if bit == '1' else -1
                for var, bit in zip(input_neurons, bits)
            }
            input_values['T'] = 1  # fixa T como true

            outputs = self.evaluate(input_values)
            rounded_outputs = {k: 1 if v >= self.A_min else (-1 if v <= -self.A_min else 0)
                             for k, v in outputs.items()}
            
            truth_table.append({**input_values, **rounded_outputs})
        
        return truth_table
    
    def print_network(self):
        print("=== Estrutura da Rede Neural ===")
        print(f"Neurônios de Entrada: {self.input_neurons}")
        print(f"Neurônios Ocultos: {self.hidden_neurons}")
        print(f"Neurônios de Saída: {self.output_neurons}")
        print("\nPesos das Conexões:")
        for (src, tgt), weight in self.weights.items():
            if weight != 0:
                print(f"{src} -> {tgt}: {weight:.2f}")
        print("\nThresholds:")
        for neuron, threshold in self.thresholds.items():
            print(f"{neuron}: {threshold:.2f}")
        print(f"\nParâmetros: A_min={self.A_min}, W={self.W:.2f}, beta={self.beta}")
    
    def print_truth_table(self, truth_table):
        print("\n=== Tabela Verdade ===")
        # Imprime cabeçalho
        header = self.input_neurons + self.output_neurons
        print(" | ".join(header))
        print("-" * (4 * len(header) + 1))
        
        # Imprime cada linha
        for row in truth_table:
            print(" | ".join(f"{row[var]:2}" for var in header))

if __name__ == "__main__":
    translator = CILPNetwork()
    
    caminho = input("Digite o nome ou caminho do arquivo .csv: ").strip()
    
    try:
        translator.translate_program(caminho)
        translator.print_network()
        
        truth_table = translator.generate_truth_table()
        translator.print_truth_table(truth_table)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        
# Para utilizar: python tradutor.py
# Se der erro, colocar versao ao lado de python(Ex: python3 tradutor.py)
