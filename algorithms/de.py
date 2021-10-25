import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# setting the graphic style
sns.set_style('whitegrid')


class DifferentialEvolution:
    """
    Class that contains all the functions to perform the Differential Evolution (DE) to minimize an objective
    function of interest.
    """

    def __init__(self, search_range, n_population, dimensions, obj_func, F, CR, max_iter, opposition=False, JR=1.0):
        """
        Description: Initializes the parameters for running the algorithm.

        Args:
            search_range: Lower and upper limit for the variables that are to be optimized.
            n_population: Number of candidates to solve the problem.
            dimensions  : Number of variables to be optimized.
            obj_func    : Function to be minimized.
            F           : Step size towards the difference vector.
            CR          : Probability of crossover occurrence.
            max_iter    : Maximum number of iterations.
            opposition  : Whether the algorithm will run with opposition. Default: False.
            JR          : Jump rate. Probability of occurrence of opposition.
        """

        # storing the parameters
        self.search_range = search_range
        self.n_population = n_population
        self.dimensions = dimensions
        self.obj_func = obj_func
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
        self.opposition = opposition
        self.JR = JR
        self.best_vetor = None
        self.best_eval = None
        self.results = None
        self.avg_tracking = []
        self.best_tracking = []
        self.candidates = np.arange(self.n_population)

        # initializing the population
        self.population = np.random.uniform(self.search_range[0],
                                            self.search_range[1],
                                            (self.n_population, self.dimensions))

        # initializing the population with opposition
        if self.opposition:
            lower = np.min(self.population)
            upper = np.max(self.population)
            diff = lower + upper
            for i in range(round(self.n_population / 2)):
                for j in range(self.population[i].shape[0]):
                    self.population[i][j] = diff - self.population[i][j]

    def evaluate(self):
        """
        Description: Calculates the score for each vector (fitness).
        """

        return [self.obj_func(ind) for ind in self.population]

    def opposition_operator(self, vector, lower, upper):
        """
        Description: Operates opposition in vector.

        Args:
            vector: Solution candidate.
            lower : Lower value within the generation.
            upper : Upper value within the generation.
        """

        diff = lower + upper
        for i in range(vector.shape[0]):
            vector[i] = diff - vector[i]

        return vector

    def mutation(self, vectors):
        """
        Description: Operates mutation on vectors.

        Args:
            vectors: Solution candidates.
        """

        return vectors[0] + self.F * (vectors[1] - vectors[2])

    def crossover(self, mutated_vector, target_vector):
        """
        Description: Operates crossover on vectors.

        Args:
            mutated_vector: Candidate for the mutated solution.
            target_vector : Solution candidate.
        """

        # rng for check if crossover will happen.
        rng = np.random.rand(self.dimensions)

        # doing the crossover.
        trial = np.array([mutated_vector[k] if rng[k] < self.CR else target_vector[k] for k in range(self.dimensions)])

        return trial

    def plot(self, log=False, avg=True):
        """
        Description:

        Args:
            log: Y axis with log scale.
            avg: Whether the plot should consider the average evolution.
        """

        plt.figure(figsize=(10, 7))
        if avg:
            plt.plot(np.arange(len(self.avg_tracking)), self.avg_tracking, c='k', linestyle='--', label='average')
        plt.plot(np.arange(len(self.best_tracking)), self.best_tracking, c='k', label='best')
        plt.xlim((0, len(self.avg_tracking)))
        if log:
            plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Evaluation')
        plt.title('Evaluation by Iteration')
        plt.legend()
        sns.despine(bottom=False, left=False)
        plt.show()

    def fit(self):
        """
        Description: Runs the DE algorithm.
        """

        # evaluating the population.
        eval_pop = self.evaluate()

        # storing the best results
        self.best_vetor = self.population[np.argmin(eval_pop)]
        self.best_eval = np.min(eval_pop)

        # list to store the results
        self.results = [self.best_eval]
        self.avg_tracking.append(np.mean(eval_pop))
        self.best_tracking.append(self.best_eval)

        for i in range(self.max_iter):

            # teste
            lower = np.min(self.population)
            upper = np.max(self.population)

            for j in range(self.n_population):

                if self.opposition:
                    rng = np.random.rand()
                else:
                    rng = 1.0

                if rng < self.JR:
                    trial = self.opposition_operator(self.population[j], lower, upper)
                    trial_eval = self.obj_func(trial)
                else:

                    # selecting candidates for mutation
                    vectors_idx = np.delete(self.candidates, j)
                    vectors = self.population[np.random.choice(vectors_idx, 3)]

                    # mutation
                    mutated = self.mutation(vectors)
                    mutated = np.clip(mutated, self.search_range[0], self.search_range[1])

                    # crossover
                    trial = self.crossover(mutated, self.population[j])

                    # evaluating the trial vector
                    trial_eval = self.obj_func(trial)

                # check if trial evaluation is better
                if trial_eval < eval_pop[j]:
                    self.population[j] = trial
                    eval_pop[j] = trial_eval

            # updating the best evaluation
            self.best_eval = np.min(eval_pop)

            # tracking the results
            self.avg_tracking.append(np.mean(eval_pop))
            self.best_tracking.append(self.best_eval)

            # storing the new best candidate if it exists
            if self.best_eval < self.results[-1]:
                self.best_vetor = self.population[np.argmin(eval_pop)]
                self.results.append(self.best_eval)
                print(f'Melhor solução da iteração {i} ---> Vetor:{self.best_vetor}, avaliação: {self.best_eval}.')
            if self.results[-1] == 0:
                break
