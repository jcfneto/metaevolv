import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plot style
sns.set_style('whitegrid')


class Clonalg:
    """
    Class that contains all the functions to perform the Clonal Selection Algorithm to minimize an
    objective function of interest.
    """

    def __init__(self, search_range, n_population, dimensions, obj_function, sr, cr, gama, max_iter):
        """
        Description: Initializes the parameters for running the algorithm.

        Args:
            search_range: Lower and upper limit for the variables that are to be optimized.
            n_population: Number of candidates to solve the problem.
            dimensions  : Number of variables to be optimized.
            obj_function: Function to be minimized.
            sr          : Selectio rate.
            cr          : Cloning rate.
            gama        : Mutation radius.
            max_iter    : Maximum number of iterations.
        """

        # storing the parameters
        self.max_iter = max_iter
        self.search_range = search_range
        self.n_population = n_population
        self.dimensions = dimensions
        self.obj_function = obj_function
        self.sr = sr
        self.cr = cr
        self.gama = gama

        # lsit to store the results
        self.best_ind = []
        self.avg_top_10 = []

        # initializing the population
        self.population_rank = np.array([])
        self.population = np.random.uniform(self.search_range[0],
                                            self.search_range[1],
                                            (self.n_population, self.dimensions))

        # doing the first ranking
        self.ranking()

        # storing the first best
        self.best_ind.append(self.population_rank[0][1])
        self.avg_top_10.append(np.mean([j for i, j in self.population_rank[:10]]))

    def affinity(self, x):
        """
        Description: Calculates the score for each individual.

        Args:
            x: Array containing the values to be evaluated.

        Returns: Individual score (evaluation).
        """

        return self.obj_function(x)

    def ranking(self):
        """
        Description: Avalia cada indivíduo e classifica-o.
        """

        self.population_rank = np.array([(p, self.affinity(p)) for p in self.population])
        self.population_rank = sorted(self.population_rank, key=lambda x: x[1])

    def mutation(self, clone, alfa):
        """
        Description: Causes the mutation according to the probability of occurrence.

        Args:
            clone: Clone of individual.
            alfa : Mutation rate.

        Returns: Individual after mutation.
        """

        for k in range(clone.shape[0]):
            if np.random.rand() < alfa:
                clone[k] = np.random.uniform(self.search_range[0], self.search_range[1], 1)[0]
        return clone

    def plot(self, avg=True, log=False):
        """
        Description: Plot the results.

        Args:
            avg: Shows the average of the top 10 individuals per generation. Default: True.
            log: Y axis in log scale.

        Returns: Plot.
        """

        x = range(len(self.best_ind))
        plt.figure(figsize=(8, 5))
        plt.plot(x, self.best_ind, c='k', label='best')
        if avg:
            plt.plot(x, self.avg_top_10, c='r', linestyle='--', label='average top 10')
        if log:
            plt.yscale('log')
        plt.xlim(0, np.max(x))
        plt.ylim(0, np.max(self.avg_top_10) if avg else np.max(self.best_ind))
        plt.xlabel('Generations', fontsize=12)
        plt.ylabel('Evaluation', fontsize=12)
        plt.title('Evaluation by Generation', fontsize=14)
        plt.legend()
        sns.despine(bottom=False, left=False)
        plt.show()

    def fit(self):
        """
        Description: Runs the Clonalg algorithm.
        """

        # generations
        for gen in range(self.max_iter):

            # it clones the best individuals
            for i in range(1, int(self.sr * self.n_population) + 1):
                nc = round((self.cr * self.n_population)/i)
                fit = 1 - (i - 1)/(self.n_population - 1)
                clones = np.tile(self.population_rank[i - 1][0], (nc, 1))
                alfa = self.gama * np.exp(-fit)

                # mutation of clones
                for j in range(nc):
                    clone = self.mutation(clones[j], alfa)
                    clone_fit = self.affinity(clone)

                    # if clone fitness is better than the original, the clone replaces then
                    if clone_fit < self.population_rank[i - 1][1]:
                        self.population_rank[i - 1] = (clone, clone_fit)

            # new population
            individuals = np.array([i for i, _ in self.population_rank])
            new_individuals = np.random.uniform(self.search_range[0],
                                                self.search_range[1],
                                                (int(self.n_population - self.sr * self.n_population),
                                                 self.dimensions))
            self.population = np.concatenate((individuals, new_individuals))

            # ranking the individuals
            self.ranking()

            # storing the best results
            self.best_ind.append(self.population_rank[0][1])
            self.avg_top_10.append(np.mean([j for i, j in self.population_rank[:10]]))

            # early stop. Checking if the optimal individual was found
            if np.min(self.best_ind) == 0:
                print(f"[{gen}] The two best solutions:"
                      f"\nf{np.round(self.population_rank[0][0], 4)} = {np.round(self.population_rank[0][1], 4)}"
                      f"\nf{np.round(self.population_rank[1][0], 4)} = {np.round(self.population_rank[1][1], 4)}")
                break
