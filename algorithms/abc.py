import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')


class ABC:

    def __init__(self, search_range, n_population, dimensions, obj_function, max_iter, limit):
        """
        Description: Initializes the parameters for running the algorithm.

        Args:
            search_range: (list) Lower and upper limit for the variables that are to be optimized.
            n_population: (int) Number of candidates to solve the problem.
            dimensions  : (int) Number of variables to be optimized.
            obj_function: (object) Function to be minimized.
            max_iter    : (int) Maximum number of iterations.
            limit       : (int) Number of attempts to improve a food source.
        """

        # storing the parameters
        self.search_range = search_range
        self.n_population = n_population
        self.dimensions = dimensions
        self.obj_function = obj_function
        self.max_iter = max_iter
        self.limit = limit

        # initializing the population
        self.trials = np.zeros(self.n_population)
        self.fs = np.random.uniform(self.search_range[0],
                                    self.search_range[1],
                                    (self.n_population, self.dimensions))

        # Quality of the food source.
        self.fsq = np.array([self.obj_function(i) for i in self.fs])

        # Variables to stores the results.
        self.best_iter = []
        self.avg_iter = []
        self.best_fsq = np.inf
        self.best_fs = None

    def employed_bee(self, fs, fsq, trials, k):
        """
        Description: Routine of the worker bee.

        Args:
            fs    : (array) Vector representation of the food source.
            fsq   : (array) Food source quality.
            trials: (array) Attempts to improve the solution.
            k     : (int) Iteration.

        Returns: (array) Food sources and (array) trials.
        """

        # Produces a new solution to be tested.
        new_fs = []
        partner = np.random.randint(self.n_population)
        while partner == k:
            partner = np.random.randint(self.n_population)
        for i in range(fs.shape[1]):
            new_fs.append(fs[k, i] + np.random.uniform(-1, 1) * (fs[k, i] - fs[partner, i]))
        new_fs = np.array(new_fs)

        # Checking if the new solution is better than the old one.
        if self.obj_function(new_fs) < fsq[k]:
            fsq[k] = self.obj_function(new_fs)
            fs[k] = new_fs
            trials[k] = 0
        else:
            trials[k] += 1

        return fs, fsq, trials

    def employed_bee_phase(self, fs, fsq, trials):
        """
        Description: Execute all routine of the worker bee.

        Args:
            fs    : (array) Vector representation of the food source.
            fsq   : (array) Food source quality.
            trials: (array) Attempts to improve the solution.

        Returns: (array) Food source, (array) food source quality and (array) trials.
        """

        for i in range(fs.shape[0]):
            fs, fsq, trials = self.employed_bee(fs, fsq, trials, i)

        return fs, fsq, trials

    def fitness(self, fsq):
        """
        Description: Calculates the fitness of the food source.

        Args:
            fsq: (array) Food source quality.

        Returns: (array) fitness of the food source.
        """

        return np.array([(1 / (1 + i)) if i >= 0 else (1 / (1 - i)) for i in fsq])

    def onlooker_bee_phase(self, fs, fsq, trials):
        """
        Description: Execute all routine of the onlooker bee.

        Args:
            fs    : (array) Vector representation of the food source.
            fsq   : (array) Food source quality.
            trials: (array) Attempts to improve the solution.

        Returns: (array) Food source, (array) food source quality and (array) trials.
        """

        # Calculate the probabilities.
        fsf = self.fitness(fsq)
        pi = fsf/np.sum(fsf)

        # Visits the best food sources.
        for i in range(fs.shape[0]):
            if np.random.rand() < pi[i]:
                fs, fsq, trials = self.employed_bee(fs, fsq, trials, i)

        return fs, fsq, trials

    def scout_bee_phase(self, fs, fsq, trials):
        """
        Description: Routine of the scout bee.

        Args:
            fs    : (array) Vector representation of the food source.
            fsq   : (array) Food source quality.
            trials: (array) Attempts to improve the solution.

        Returns: (array) Food source, (array) food source quality and (array) trials.
        """

        # Replaces depleted food sources.
        for i in range(fs.shape[0]):
            if trials[i] == self.limit:
                fs[i] = np.random.uniform(self.search_range[0], self.search_range[1], (1, self.dimensions))[0]
                fsq[i] = self.obj_function(fs[i])
                trials[i] = 0

        return fs, fsq, trials

    def fit(self):
        """
        Description:  Runs the ABC algorithm.
        """

        for i in range(self.max_iter):

            # bees routine.
            self.fs, self.fsq, self.trials = self.employed_bee_phase(self.fs, self.fsq, self.trials)
            self.fs, self.fsq, self.trials = self.onlooker_bee_phase(self.fs, self.fsq, self.trials)
            self.fs, self.fsq, self.trials = self.scout_bee_phase(self.fs, self.fsq, self.trials)

            # results.
            self.best_iter.append(np.min(self.fsq))
            self.avg_iter.append(np.mean(self.fsq))
            if np.min(self.fsq) < self.best_fsq:
                self.best_fsq = np.min(self.fsq)
                self.best_fs = self.fs[np.argmin(self.fsq)]

        self.best_fs = self.fs[np.argmin(self.fsq)]
        print(f'f{self.best_fs} = {np.min(self.best_fsq)}')

    def plot(self, avg=False, log=False):
        """
        Description: Plot the results.

        Args:
            avg: Shows the average of the top 10 individuals per generation. Default: True.
            log: Y axis in log scale.

        Returns: Plot.
        """

        x = range(self.max_iter)
        plt.figure(figsize=(8, 5))
        plt.plot(x, self.best_iter, c='k', label='best')
        if avg:
            plt.plot(x, self.avg_iter, c='r', linestyle='--', label='average')
        if log:
            plt.yscale('log')
        plt.xlabel('Generations', fontsize=12)
        plt.ylabel('Iterations', fontsize=12)
        plt.title('Evaluation by Iterations', fontsize=14)
        plt.legend()
        sns.despine(bottom=False, left=False)
        plt.show()
