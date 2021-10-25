import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


class GeneticAlgorithm:
    """
    Class that contains all the functions to perform the genetic algorithm to minimize an objective function
    of interest.
    """

    def __init__(self, bits: int, n_population: int, max_iter: int, dimensions: int, search_range: list, **kwargs):
        """
        Description: Initializes the parameters for running the algorithm.
        
        bits        : Number of bits for the representation of the individual.
        n_population: Number of individuals in the population.
        max_iter    : Maximum number of iterations, in this case maximum number of generations.
        dimensions  : Number of variables to be optimized.
        search_range: Lower and upper limit for the variables that are to be optimized.
        **kwargs    :

            selection_type: Type of selection of individuals. It can be by 'tournament' or by 'roulette_wheel' roulette.
                            Default: 'tournament'.
            function_name : Name of the function to be minimized. It can be 'bent_cigar' or 'rastrigin_function'.
                            Default: 'bent_cigar'.
            elitism       : You can choose whether or not elitism will occur during the selection. Default: False.
            mutation_type : Type of mutation, if 'bit_bt_bit' or 'random_choice'. Default: 'random_choice'.
            crossover_prob: Probability of crossover occurrence. Default: 0.95.
            mutation_prob : Probability of occurrence of mutation. Default: 1 - crossover_prob.
        """

        # storing the parameters
        self.bits = bits
        self.n_population = n_population
        self.max_iter = max_iter
        self.dimensions = dimensions
        self.search_range = [search_range for _ in range(self.dimensions)]

        # storing optional parameter
        self.selection_type = 'tournament'
        self.function_name = 'bent_cigar'
        self.elitism = False
        self.mutation_type = 'random_choice'
        self.crossover_prob = .95
        self.mutation_prob = None

        # overriding optional parameters if they have been declared
        for (k, v) in kwargs.items():
            setattr(self, k, v)

        # calculating mutation probability if not passed
        if not self.mutation_prob:
            self.mutation_prob = 1. - self.crossover_prob

        # initializing the population
        self.population = np.array([np.random.randint(0, 2, self.bits * self.dimensions).tolist() for _ in
                                    range(self.n_population)])

        # list to store ratings by generation
        self.best_chrom_gen = []
        self.mean_chrom_gen = []

        # others
        self.gen = 0
        self.selected = np.array([])
        self.population_decimal = np.array([])
        self.population_evaluate = np.array([])

    def function(self, x: list):
        """
        Description: 
        
        Function_name: Name of the objective function to be minimized can be 'bent_cigar' or 'rastrigin_function'.
        x            : Vector containing the variables.
        """

        # transforming the list into an array to facilitate operations
        x = np.array(x)

        if self.function_name == "bent_cigar":
            return (x[0] ** 2.0 + (10 ** 6) * np.sum(x[1:] ** 2)).tolist()
        elif self.function_name == 'rastrigin_function':
            return np.sum((x ** 2) - 10 * np.cos(2 * np.pi * x) + 10).tolist()

    def binary_to_decimal(self):
        """
        Description: Converts bitstring to decimal values.
        """

        # initializing
        population_decimal = []

        # converting
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
        """
        Description: Calculates the score for each individual (fitness).
        """

        # list to store individuals and their ratings.
        self.population_evaluate = np.array([self.function(x=c) for c in self.population_decimal])

        # storing ratings
        self.best_chrom_gen.append(np.min(self.population_evaluate))
        self.mean_chrom_gen.append(np.mean(self.population_evaluate))

    def selection(self):
        """
        Description: It selects individuals, either by 'tournament' or 'roulette_wheel', with or without elitism.
        """

        # picking up the best individual if elitism == True
        if self.elitism is True:
            k = 1
            best_chrom = np.argmax(self.population_evaluate)
            selected = self.population[best_chrom]

        else:
            k = 0
            selected = []

        # tournament selection
        if self.selection_type == 'tournament':

            # iterating through population length
            for _ in range(len(self.population) - k):
                idx_figthers = np.random.randint(0, self.n_population, 2)
                best_value = np.argmin([self.population_evaluate[idx] for idx in idx_figthers])
                idx_winner = idx_figthers[best_value]
                selected.append(self.population[idx_winner])

            self.selected = np.array(selected)

        # roulette selection
        elif self.selection_type == 'roulette_wheel':

            invert = 1 / self.population_evaluate
            prob = invert / np.sum(invert)
            ids = np.random.choice(a=range(self.n_population - k), size=(self.n_population - k), p=prob)
            self.selected = np.array(self.population)[ids].tolist()

    def crossover(self):
        """
        Description: Recombines between selected individuals (parents).
        """

        # list to record crossover results
        childrens = []

        # iterating between the selected (parents) to make the combinations (crossover)
        for i in range(0, self.n_population, 2):

            # capturing parents for iteration i
            parent_01 = self.selected[i]
            parent_02 = self.selected[i + 1]

            # checking if the crossing will happen
            if np.random.rand() < self.crossover_prob:

                # cutoff
                cutoff = np.random.randint(1, self.bits * self.dimensions - 1)

                # crossing
                c1 = np.concatenate((parent_01[:cutoff], parent_02[cutoff:]))
                c2 = np.concatenate((parent_02[:cutoff], parent_01[cutoff:]))

                # saving if crossover occurs
                childrens += [c1] + [c2]

            else:

                # saving if the crossing does not occur
                childrens += [parent_01] + [parent_02]

        self.population = np.array(childrens)

    def mutation(self):
        """
        Description: Causes mutation or not of bits (genes). It can be 'random_choice' or 'bit_by_bit'.
        """

        # random bit-choice mutation
        if self.mutation_type == 'random_choice':

            # iterating through individuals
            for i in range(self.n_population):

                # checking if the mutation will happen
                if np.random.uniform(0, 1, 1)[0] <= self.mutation_prob:
                    gene = np.random.randint(0, self.bits, 1)[0]
                    self.population[i, gene] = 1 - self.population[i, gene]

        # bit by bit
        elif self.mutation_type == 'bit_by_bit':

            # iterating through individuals
            for i in range(self.n_population):

                # iterating bit by bit
                for j in range(self.bits):

                    # checking if the mutation will happen
                    if np.random.uniform(0, 1, 1)[0] <= self.mutation_prob:
                        self.population[i][j] = 1 - self.population[i, j]

    def plot_eval_gen(self, **kwargs):
        """
        Description: Creates the graph of the evolution of the score (fitness) of the best individual and the average
        of the population.
        """

        # plot parameters if not passed to the function
        figsize = (10, 8)
        best_color = 'blue'
        avg_color = 'black'
        x_label = 'Generation'
        y_label = 'Score'
        title = 'Score per Generation'

        # plot parameters if passed to the function
        for (k, v) in kwargs.items():
            setattr(self, k, v)

        plt.figure(figsize=figsize)
        plt.plot(list(range(1, self.gen + 1)), self.best_chrom_gen, '--', c=best_color, label='Best')
        plt.plot(list(range(1, self.gen + 1)), self.mean_chrom_gen, c=avg_color, label='Average')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        sns.despine(bottom=False, left=False)
        plt.show()

    def fit(self):
        """
        Description: Runs the genetic algorithm.
        """

        best_eval_now = np.inf

        # iterating
        for gen in range(self.max_iter):

            self.gen = gen + 1

            # converting binary data to decimal
            self.binary_to_decimal()

            # evaluating individuals
            self.evaluate()

            # looking for the best individual of the generation
            best_eval = np.min(self.population_evaluate)
            idx_best_eval = np.argmin(self.population_evaluate)
            best_chro_d = self.population_decimal[idx_best_eval]

            # early stop criterion
            if np.min(self.population_evaluate) == 0:

                print(f"--- Geração {self.gen} ---")
                print(f'\nObjetivo atingido, parada antecipada.')
                print(f'\nMelhor indivíduo da geração {self.gen}: f({best_chro_d}) = {best_eval}.\n')
                break

            else:

                if best_eval < best_eval_now:
                    best_eval_now = best_eval
                    print(f"--- Geração {self.gen} ---")
                    print(f'\nMelhor indivíduo da geração {self.gen}: f({best_chro_d}) = {best_eval}.\n')

                # making the selection
                self.selection()

                # making the combinations
                self.crossover()

                # making the mutations
                self.mutation()
