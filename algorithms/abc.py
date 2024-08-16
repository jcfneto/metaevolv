"""Artificial Bee Colony (ABC) algorithm.

This algorithm is inspired by the foraging behavior of honey bees. 
The algorithm consists of three types of bees: employed bees, onlooker bees, 
and scout bees. The employed bees search for food sources, the onlooker bees 
choose the best food sources found by the employed bees, and the scout bees are 
responsible for finding new food sources.

The ABC algorithm is a population-based optimization algorithm that is used to
solve optimization problems. It is based on the following steps:

1. Initialization: The algorithm starts by generating a population of food sources.
2. Employed bee phase: The employed bees search for food sources.
3. Onlooker bee phase: The onlooker bees choose the best food sources found by the employed bees.
4. Scout bee phase: The scout bees find new food sources.
5. Stopping criterion: The algorithm stops when a stopping criterion is met.
6. Evaluation: The food sources are evaluated.
7. Early stop: The algorithm stops if the optimal solution is found.

The algorithm is controlled by the following parameters:

- Search range: Lower and upper limit for the variables that are to be optimized.
- Population size: Number of food sources.
- Dimensions: Number of variables to be optimized.
- Maximum number of iterations: Maximum number of iterations.
- Limit: Number of attempts to improve the solution.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
import seaborn as sns
from numpy import ndarray


sns.set_style('whitegrid')


class ABCConfig(BaseModel):
    """Configuration for the ABC algorithm."""
    search_range: tuple[float, float]
    n_population: int
    dimensions: int
    max_iter: int
    limit: int


class ABC:
    """Artificial Bee Colony (ABC) algorithm."""

    def __init__(
        self,
        config: ABCConfig,
        obj_function: Callable[[ndarray], float]
    ) -> None:
        """Initializes the parameters for running the algorithm.
        
        Args:
            config (ABCConfig): Configuration object containing search range, population size, dimensions, etc.
            obj_function (Callable[[np.ndarray], float]): Function to be minimized.
        """
        self.config = config
        self.obj_function = obj_function

        self.trials = np.zeros(self.config.n_population, dtype=int)
        self.fs = np.random.uniform(
            self.config.search_range[0], self.config.search_range[1], 
            (self.config.n_population, self.config.dimensions)
        )
        self.fsq = np.array([self.obj_function(ind) for ind in self.fs])

        self.best_iter: list[float] = []
        self.avg_iter: list[float] = []
        self.best_fsq = np.inf
        self.best_fs = None


    def _employed_bee(
        self,
        fs: ndarray,
        fsq: ndarray,
        trials: ndarray,
        k: int
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Routine of the worker bee.

        Args:
            fs (ndarray): Vector representation of the food source.
            fsq (ndarray): Food source quality.
            trials (ndarray): Attempts to improve the solution.
            k (int): Iteration.

        Returns:
            tuple[ndarray, ndarray, ndarray]: Updated food sources and trials.
        """
        partner = np.random.randint(self.config.n_population)
        while partner == k:
            partner = np.random.randint(self.config.n_population)

        new_fs = (
            fs[k] +
            np.random.uniform(-1, 1, self.config.dimensions) *
            (fs[k] - fs[partner])
        )

        new_fsq = self.obj_function(new_fs)
        if new_fsq < fsq[k]:
            fsq[k] = new_fsq
            fs[k] = new_fs
            trials[k] = 0
        else:
            trials[k] += 1

        return fs, fsq, trials


    def _employed_bee_phase(
        self,
        fs: ndarray,
        fsq: ndarray,
        trials: ndarray
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Execute all routine of the worker bee.
        
        Args:
            fs (ndarray): Vector representation of the food source.
            fsq (ndarray): Food source quality.
            trials (ndarray): Attempts to improve the solution.
        
        Returns:
            tuple[ndarray, ndarray, ndarray]: Updated food sources, food source quality, and trials.
        """
        for i in range(self.config.n_population):
            fs, fsq, trials = self._employed_bee(fs, fsq, trials, i)

        return fs, fsq, trials


    def _fitness(self, fsq: ndarray) -> ndarray:
        """Calculates the fitness of the food source.
        
        Args:
            fsq (ndarray): Food source quality.
        
        Returns:
            ndarray: Fitness of the food source.
        """
        return np.where(fsq >= 0, 1 / (1 + fsq), 1 / (1 - fsq))


    def _onlooker_bee_phase(
        self,
        fs: ndarray,
        fsq: ndarray,
        trials: ndarray
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Execute all routine of the onlooker bee.

        Args:
            fs (ndarray): Vector representation of the food source.
            fsq (ndarray): Food source quality.
            trials (ndarray): Attempts to improve the solution.

        Returns:
            tuple[ndarray, ndarray, ndarray]: Updated food sources, food source quality, and trials.
        """
        pi = self._fitness(fsq) / np.sum(self._fitness(fsq))

        for i in range(self.config.n_population):
            if np.random.rand() < pi[i]:
                fs, fsq, trials = self._employed_bee(fs, fsq, trials, i)

        return fs, fsq, trials


    def _scout_bee_phase(
        self,
        fs: ndarray,
        fsq: ndarray,
        trials: ndarray
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Routine of the scout bee.

        Args:
            fs (ndarray): Vector representation of the food source.
            fsq (ndarray): Food source quality.
            trials (ndarray): Attempts to improve the solution.

        Returns:
            tuple[ndarray, ndarray, ndarray]: Updated food sources, food source quality, and trials.
        """
        for i in range(self.config.n_population):
            if trials[i] >= self.config.limit:
                fs[i] = np.random.uniform(
                    self.config.search_range[0],
                    self.config.search_range[1],
                    self.config.dimensions
                )
                fsq[i] = self.obj_function(fs[i])
                trials[i] = 0

        return fs, fsq, trials


    def fit(self) -> None:
        """Runs the ABC algorithm."""
        for _ in range(self.config.max_iter):
            self.fs, self.fsq, self.trials = self._employed_bee_phase(
                self.fs, self.fsq, self.trials
            )
            self.fs, self.fsq, self.trials = self._onlooker_bee_phase(
                self.fs, self.fsq, self.trials
            )
            self.fs, self.fsq, self.trials = self._scout_bee_phase(
                self.fs, self.fsq, self.trials
            )

            best_fsq = np.min(self.fsq)
            self.best_iter.append(best_fsq)
            self.avg_iter.append(np.mean(self.fsq))

            if best_fsq < self.best_fsq:
                self.best_fsq = best_fsq
                self.best_fs = self.fs[np.argmin(self.fsq)]

        print(f'Best solution: {self.best_fs} with objective function value: {self.best_fsq}')


    def plot(self, avg: bool = False, log: bool = False) -> None:
        """Plot the optimization progress.

        Args:
            avg (bool): Shows the average of the top 10 individuals per generation.
            log (bool): Y axis in log scale.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.best_iter, label='Best', color='k')
        if avg:
            plt.plot(self.avg_iter, label='Average', linestyle='--', color='r')
        if log:
            plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function Value')
        plt.title('Optimization Progress')
        plt.legend()
        sns.despine()
        plt.show()
