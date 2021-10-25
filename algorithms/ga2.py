import numpy as np


class GeneticAlgorithm2:

    # TODO: implementar elitismo.
    # TODO: bolar uma técnica que aumente a taxa de mutação assim que as gerações perdem disversidade.
    def __init__(self, bits, dimensions, n_population, search_range, k, cp, mp, obj_function, max_iter, selection_type,
                 crossover_type, mutation_type, pc_variation, cp_final):

        self.bits = bits
        self.dimensions = dimensions
        self.n_population = n_population
        self.search_range = [search_range for _ in range(self.dimensions)]
        self.k = k
        self.cp = cp
        self.mp = mp
        self.obj_function = obj_function
        self.max_iter = max_iter
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.pc_variation = pc_variation
        self.cp_final = cp_final

        self.selected = np.array([])
        self.population_decimal = np.array([])
        self.population_eval = np.array([])
        self.best_ind = []
        self.best_eval = np.inf

        self.bitstring_size = self.bits * self.dimensions
        self.population = np.array([np.random.randint(0, 2, self.bits * self.dimensions).tolist() for _ in
                                    range(self.n_population)])

    def binary_to_decimal(self):
        population_decimal = []
        for ind in self.population:
            bit_string = ''.join([str(i) for i in ind])
            d = []
            for i in range(self.dimensions):
                di = int(bit_string[i * self.bits:self.bits * (i + 1)], 2)
                di = self.search_range[i][0] + (di / 2 ** self.bits) * \
                    (self.search_range[i][1] - self.search_range[i][0])
                d.append(di)
            population_decimal.append(d)
        self.population_decimal = np.array(population_decimal)

    def evaluate(self):
        self.population_eval = np.array([self.obj_function(d) for d in self.population_decimal])

    def tournament(self):
        selected = []
        for _ in range(self.n_population):
            fighters = np.random.randint(0, self.n_population, self.k)
            fighters_fitness = self.population_eval[fighters]
            winner = np.argmin(fighters_fitness)
            selected.append(self.population[winner])
        self.selected = np.array(selected)

    def roulette_wheel(self):
        invert = 1 / (self.population_eval + 1)
        prob = invert / np.sum(invert)
        ids = np.random.choice(a=range(self.n_population), size=self.n_population, p=prob)
        self.selected = np.array(self.population)[ids].tolist()

    def selection(self):
        if self.selection_type is 'tournament':
            self.tournament()
        elif self.selection_type is 'roulette_wheel':
            self.roulette_wheel()

    def one_point(self, p1, p2):
        cutoff = np.random.randint(1, self.bitstring_size - 2)
        c1 = np.concatenate((p1[:cutoff], p2[cutoff:]))
        c2 = np.concatenate((p2[:cutoff], p1[cutoff:]))
        return c1, c2

    def two_points(self, p1, p2):
        c1, c2 = p1, p2
        cutoff = np.sort(np.random.randint(0, self.bitstring_size, 2))
        while cutoff[0] == cutoff[1]:
            cutoff = np.sort(np.random.randint(0, self.bitstring_size, 2))
        c1[cutoff[0]:cutoff[1]] = p2[cutoff[0]:cutoff[1]]
        c2[:cutoff[0]], p2[cutoff[1]:] = p1[:cutoff[0]], p1[cutoff[1]:]
        return c1, c2

    def uniform(self, p1, p2):
        uniform_c1 = np.random.randint(0, 2, self.bitstring_size)
        uniform_c2 = 1 - uniform_c1
        parents = [p1, p2]
        c1 = np.array([parents[uniform_c1[j]][j] for j in range(self.bitstring_size)])
        c2 = np.array([parents[uniform_c2[j]][j] for j in range(self.bitstring_size)])
        return c1, c2

    def crossover_prob_variation(self):
        if self.pc_variation is 'constant':
            self.cp = np.array([self.cp for _ in range(self.max_iter)])
        elif self.pc_variation is 'linear':
            self.cp = np.arange(0, self.max_iter)*(self.cp_final - self.cp)/(self.max_iter - 1) + self.cp
        self.mp = 1. - self.cp

    def crossover(self, gen):
        childrens = []
        c1 = []
        c2 = []
        for i in range(0, self.n_population, 2):
            parent1 = self.selected[i]
            parent2 = self.selected[i+1]
            if np.random.rand() < self.cp[gen]:
                if self.crossover_type is 'one_point':
                    c1, c2 = self.one_point(parent1, parent2)
                elif self.crossover_type is 'two_points':
                    c1, c2 = self.two_points(parent1, parent2)
                elif self.crossover_type is 'uniform':
                    c1, c2 = self.uniform(parent1, parent2)
                childrens += [c1] + [c2]
            else:
                childrens += [parent1] + [parent2]
        self.population = np.array(childrens)

    def bit_by_bit(self, gen):
        for i in range(self.n_population):
            for j in range(self.bitstring_size):
                if np.random.rand() < self.mp[gen]:
                    self.population[i, j] = 1 - self.population[i, j]

    def random_choice(self, gen):
        for i in range(self.n_population):
            if np.random.rand() < self.mp[gen]:
                bit = np.random.randint(0, self.bitstring_size, 1)[0]
                self.population[i, bit] = 1 - self.population[i, bit]

    def mutation(self, gen):
        if self.mutation_type is 'bit_by_bit':
            self.bit_by_bit(gen)
        elif self.mutation_type is 'random_choice':
            self.random_choice(gen)

    def fit(self):
        self.crossover_prob_variation()
        for gen in range(self.max_iter):
            self.binary_to_decimal()
            self.evaluate()
            best_id = np.argmin(self.population_eval)
            if self.population_eval[best_id] < self.best_eval:
                self.best_eval = self.population_eval[best_id]
                self.best_ind = self.population_decimal[best_id]
                print(f'[{gen+1}] --> Novo melhor indivíduo: f({self.best_ind}) = {self.best_eval}.')
            if self.best_eval == 0.0:
                break
            self.selection()
            self.crossover(gen)
            self.mutation(gen)
