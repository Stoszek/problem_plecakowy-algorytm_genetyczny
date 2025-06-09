import random
import matplotlib.pyplot as plt
from collections import defaultdict

class Knapsack:
    def __init__(self, capacity, numberOfItems, minWeight, minValue, maxWeight, maxValue, seed):
        self.capacity = capacity
        self.numberOfItems = numberOfItems
        self.listOfItems = []
        self.seed = seed
        self.rng = random.Random(self.seed)

        for i in range(self.numberOfItems):
            value = self.rng.randint(minValue, maxValue)
            weight = self.rng.randint(minWeight, maxWeight)
            self.listOfItems.append([value, weight])

    def __str__(self):
        result = f"{self.numberOfItems} items in knapsack with {self.capacity} capacity:\n"
        result += f"{'Idx':<4}{'Value':>8}{'Weight':>8}\n"
        result += "-" * 20 + "\n"
        for idx, itemList in enumerate(self.listOfItems, start=1):
            result += f"{idx:<4}{itemList[0]:>6}{itemList[1]:>8}\n"
        return result


class Population:
    def __init__(self, populationSize, knapsack: Knapsack, numberOfGeneretions, tournament_size, mutation_rate, selection_method,crossover_mode):
        """Inicjalizacja"""
        self.tournament_size = tournament_size
        self.populationSize = populationSize
        self.mutation_rate = mutation_rate
        self.knapsack = knapsack
        self.numberOfGeneretions = numberOfGeneretions
        self.selection_method = selection_method
        self.crossover_mode = crossover_mode

        """Listy pomocnicze"""
        self.population = []
        self.selected_indices = []
        self.listOfSums = []

        self.best_fitness_history = []
        self.avg_fitness_history = []

        for _ in range(self.populationSize):
            genotype = [self.knapsack.rng.randint(0, 1) for _ in range(self.knapsack.numberOfItems)]
            self.population.append(genotype)


    def fitnessFunction(self):
        self.listOfSums = [] #Reset

        for genotype in self.population:
            value_sum = 0
            weight_sum = 0
            for idx, i in enumerate(genotype):
                if i == 1:
                    value_sum += self.knapsack.listOfItems[idx][0]
                    weight_sum += self.knapsack.listOfItems[idx][1]

            if weight_sum > self.knapsack.capacity:
                value_sum = 0
                weight_sum = 0

            self.listOfSums.append([value_sum, weight_sum])
        return self.listOfSums

    def selection(self, method):
        if method == "rullete":
            fitness_values = [i[0] for i in self.listOfSums]
            total_fitness = sum(fitness_values)

            if total_fitness == 0:
                return self.population[self.knapsack.rng.randrange(self.populationSize)]
            probabilities = [fitness / total_fitness for fitness in fitness_values]
            r = self.knapsack.rng.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    self.selected_indices.append(i)
                    return self.population[i]

        elif method == "random":
            indx = self.knapsack.rng.randrange(self.populationSize)
            self.selected_indices.append(indx)
            return self.population[indx]

        elif method == "tournament":
            candidates = self.knapsack.rng.sample(range(self.populationSize), self.tournament_size)
            best_index = max(candidates, key=lambda idx: self.listOfSums[idx][0])
            self.selected_indices.append(best_index)
            return self.population[best_index]

        else:
            raise ValueError(f"Unknown selection method: {method}")

    def crossover(self, parent1, parent2):
        if self.crossover_mode == 'half':
            child1 = parent1[:len(parent1) // 2] + parent2[len(parent1) // 2:]
            child2 = parent2[:len(parent2) // 2] + parent1[len(parent2) // 2:]
        elif self.crossover_mode == 'mid':
            third = len(parent1) // 3
            child1 = parent1[:third] + parent2[third:2 * third] + parent1[2 * third:]
            child2 = parent2[:third] + parent1[third:2 * third] + parent2[2 * third:]
        elif self.crossover_mode == 'single_point':
            point = self.knapsack.rng.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]

        return child1, child2

    def mutation(self, child):
        for i in range(len(child)):
            if self.knapsack.rng.random() < self.mutation_rate:
                child[i] = 1 - child[i]
        return child

    def newPopulation(self):
        self.selected_indices = [] #Reset

        self.fitnessFunction()

        best_index = max(range(self.populationSize), key=lambda i: self.listOfSums[i][0])
        best_genotype = self.population[best_index]

        new_population = [best_genotype]

        while len(new_population) < self.populationSize:
            parent1 = self.selection(self.selection_method)
            parent2 = self.selection(self.selection_method)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.append(child1)
            if len(new_population) < self.populationSize:
                new_population.append(child2)

        self.population = new_population[:self.populationSize]
        self.listOfSums = []
        self.fitnessFunction()

        best = max(self.listOfSums, key=lambda x: x[0])
        self.best_fitness_history.append(best[0])

        average = sum([x[0] for x in self.listOfSums]) / len(self.listOfSums)
        self.avg_fitness_history.append(average)

        #print(f"Best fitness in this generation: value = {best[0]}, weight = {best[1]}")

    def run_generations(self):
        for generation in range(self.numberOfGeneretions):
            #print(f"\n=== Generation {generation + 1} ===")
            self.newPopulation()
            #print(self)

    def plot_best_fitness_per_generation(self):
        plt.plot(self.best_fitness_history)
        plt.title("Best fitness per generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.show()

    def table_average_fitness_per_generation(self):
        print(f"Tabela średniej wartości fitnessu, dla ziarna {self.knapsack.seed}, przy {self.numberOfGeneretions} generacjach, metoda selekcji to {self.selection_method}, metoda krzyżowania to {self.crossover_mode}:")
        for i, j in enumerate(self.avg_fitness_history):
            print(i, j)

    def table_best_fitness_per_generation(self):
        print(f"Tabela najlepszej wartości fitnessu, dla ziarna {self.knapsack.seed}, przy {self.numberOfGeneretions} generacjach, metoda selekcji to {self.selection_method}, metoda krzyżowania to {self.crossover_mode}::")
        for i, j in enumerate(self.best_fitness_history):
            print(i, j)


    def plot_average_fitness_per_generation(self):
        plt.plot(self.best_fitness_history, label='Max Fitness')
        plt.plot(self.avg_fitness_history, label='Avg Fitness')
        plt.title("Fitness Evolution")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()

    def __str__(self):
        result = "-" * 20 + "\n"
        for genotype, sum in zip(self.population, self.listOfSums):
            result += f"{genotype} sums: {sum}\n"
        return result

def full_experiment(selection_methods, crossover_modes, mutation_rate, knapsack, generations=15):
    results = defaultdict(list)
    best_overall = []

    crossover_names = {
        "half": "Jednolite (Half)",
        "mid": "Trójdzielne (Mid)",
        "single_point": "Jednopunktowe"
    }

    selection_names = {
        "tournament": "Turniejowa",
        "random": "Losowa",
        "rullete": "Ruletkowa"
    }

    for selection_method in selection_methods:
        tournament_sizes = [2, 3, 4] if selection_method == "tournament" else [None]

        for tournament_size in tournament_sizes:
            for crossover_mode in crossover_modes:
                label = crossover_names[crossover_mode]
                key = (selection_method, tournament_size)

                print(f"Running: {selection_method}, crossover: {crossover_mode}, mutation: {mutation_rate}, t_size: {tournament_size if tournament_size else '-'}")

                pop = Population(
                    populationSize=300,
                    knapsack=knapsack,
                    numberOfGeneretions=generations,
                    tournament_size=tournament_size if tournament_size else 3,
                    mutation_rate=mutation_rate,
                    selection_method=selection_method,
                    crossover_mode=crossover_mode
                )
                pop.run_generations()
                results[key].append((label, pop.best_fitness_history))

                best_overall.append({
                    'key': key,
                    'crossover': crossover_mode,
                    'label': label,
                    'fitness': pop.best_fitness_history,
                    'final_score': pop.best_fitness_history[-1]
                })

    for key, series in results.items():
        selection_method, t_size = key
        title = f"Metoda selekcji: {selection_names[selection_method]}"
        if selection_method == "tournament":
            title += f" (rozmiar turnieju: {t_size})"

        plt.figure(figsize=(10, 6))
        for label, history in series:
            plt.plot(history, label=label)
        plt.title(title)
        plt.xlabel("Generacja")
        plt.ylabel("Najlepszy fitness")
        plt.legend(fontsize='medium')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    best_per_selection = {}
    for record in best_overall:
        key = record['key']
        if key not in best_per_selection or record['final_score'] > best_per_selection[key]['final_score']:
            best_per_selection[key] = record

    plt.figure(figsize=(12, 7))
    for key, record in best_per_selection.items():
        selection_method, t_size = key
        label = f"{selection_names[selection_method]}"
        if selection_method == "tournament":
            label += f" (t={t_size})"
        label += f" + {crossover_names[record['crossover']]}"
        plt.plot(record['fitness'], label=label)
    plt.title("Najlepsze połączenia selekcji i krzyżowania")
    plt.xlabel("Generacja")
    plt.ylabel("Najlepszy fitness")
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    pop.plot_average_fitness_per_generation()
    pop.table_average_fitness_per_generation()
    pop.plot_best_fitness_per_generation()
    pop.table_best_fitness_per_generation()



mutation_rate = 0.01
selection_methods = ["tournament", "random", "rullete"]
crossover_modes = ["half", "mid", "single_point"]

knapsack = Knapsack(capacity=1500, numberOfItems=150, minWeight=0, minValue=0, maxWeight=10, maxValue=10, seed=42)

full_experiment(selection_methods, crossover_modes, mutation_rate, knapsack, generations=200)