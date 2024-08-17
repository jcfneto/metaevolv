"""Clonal Selection Algorithm.

The Clonal Selection Algorithm is a population-based optimization algorithm inspired by the immune system.
It is based on the clonal selection principle, which is the process by which B cells are selected to
produce antibodies in response to an antigen. The algorithm is based on the following steps:

1. Initialization: The algorithm starts by generating a population of candidate solutions.
2. Cloning: The best individuals are cloned according to their fitness.
3. Mutation: The clones are mutated to explore the search space.
4. Selection: The best individuals are selected to form the new population.
5. Evaluation: The new population is evaluated.
6. Stopping criterion: The algorithm stops when a stopping criterion is met.
7. Early stop: The algorithm stops if the optimal individual is found.

The algorithm is controlled by the following parameters:

- Search range: Lower and upper limit for the variables that are to be optimized.
- Population size: Number of candidates to solve the problem.
- Dimensions: Number of variables to be optimized.
- Objective function: Function to be minimized.
- Selection rate: Rate of selection.
- Cloning rate: Rate of cloning.
- Mutation radius: Mutation radius.
- Maximum number of iterations: Maximum number of iterations.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from pydantic import BaseModel, Field, field_validator

sns.set_style('whitegrid')


class ClonalgConfig(BaseModel):
    """Configuration for the Clonal Selection Algorithm."""

    search_range: tuple[float, float]
    n_population: int = Field(
        ..., ge=1, description='Number of candidates must be >= 1'
    )
    dimensions: int = Field(
        ..., ge=1, description='Number of variables must be >= 1'
    )
    sr: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Selection rate must be between 0 and 1',
    )
    cr: float = Field(
        ..., ge=0.0, le=1.0, description='Cloning rate must be between 0 and 1'
    )
    gama: float = Field(..., description='Mutation radius')
    max_iter: int = Field(
        ..., ge=1, description='Maximum number of iterations must be >= 1'
    )

    @field_validator('search_range')
    def check_search_range(cls, v): # noqa
        """Check if the search range is valid."""
        if v[0] >= v[1]:
            raise ValueError(
                'search_range must be a tuple where the first value is less than the second value'
            )
        return v


class Clonalg:
    """
    Class that contains all the functions to perform the Clonal Selection Algorithm to minimize an
    objective function of interest.
    """

    def __init__(
        self,
        config: ClonalgConfig,
        obj_function: Callable[[ndarray], float],
    ) -> None:
        """
        Initializes the parameters for running the algorithm.

        Args:
            config (ClonalgConfig): Configuration object containing search range, population size,
                dimensions, etc.
            obj_function (Callable[[ndarray], float]): Function to be minimized.
        """
        self.config = config
        self.obj_function = obj_function

        self.best_ind: list[float] = []
        self.avg_top_10: list[float] = []

        self.population = np.random.uniform(
            self.config.search_range[0],
            self.config.search_range[1],
            (self.config.n_population, self.config.dimensions),
        )
        self.population_rank = self._ranking(self.population)

        self._store_best_results()


    def affinity(self, x: ndarray) -> float:
        """Calculates the score for each individual.

        Args:
            x (ndarray): Array containing the values to be evaluated.

        Returns:
            float: Individual score (evaluation).
        """
        return self.obj_function(x)


    def _ranking(self, population: ndarray) -> ndarray:
        """Evaluates and ranks each individual.

        Args:
            population (ndarray): Population to be ranked.

        Returns:
            ndarray: Ranked population.
        """
        return sorted(
            [(p, self.affinity(p)) for p in population], key=lambda x: x[1]
        )


    def mutation(self, clone: ndarray, alfa: float) -> ndarray:
        """Causes the mutation according to the probability of occurrence.

        Args:
            clone (ndarray): Clone of individual.
            alfa (float): Mutation rate.

        Returns:
            ndarray: Individual after mutation.
        """
        mutation_mask = np.random.rand(clone.shape[0]) < alfa
        clone[mutation_mask] = np.random.uniform(
            self.config.search_range[0],
            self.config.search_range[1],
            np.sum(mutation_mask),
        )
        return clone


    def plot(self, avg: bool = True, log: bool = False) -> None:
        """Plot the results.

        Args:
            avg (bool): Shows the average of the top 10 individuals per generation.
            log (bool): Y axis in log scale.
        """
        x = range(len(self.best_ind))
        plt.figure(figsize=(8, 5))
        plt.plot(x, self.best_ind, c='k', label='best')
        if avg:
            plt.plot(
                x,
                self.avg_top_10,
                c='r',
                linestyle='--',
                label='average top 10',
            )
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


    def fit(self) -> None:
        """Runs the Clonalg algorithm."""
        for gen in range(self.config.max_iter):
            for i in range(
                1, int(self.config.sr * self.config.n_population) + 1
            ):
                nc = round((self.config.cr * self.config.n_population) / i)
                fit = 1 - (i - 1) / (self.config.n_population - 1)
                clones = np.tile(self.population_rank[i - 1][0], (nc, 1))
                alfa = self.config.gama * np.exp(-fit)

                for j in range(nc):
                    clone = self.mutation(clones[j], alfa)
                    clone_fit = self.affinity(clone)

                    if clone_fit < self.population_rank[i - 1][1]:
                        self.population_rank[i - 1] = (clone, clone_fit)

            num_new_individuals = self.config.n_population - int(
                self.config.sr * self.config.n_population
            )
            new_individuals = np.random.uniform(
                self.config.search_range[0],
                self.config.search_range[1],
                (num_new_individuals, self.config.dimensions),
            )
            individuals = np.array([i for i, _ in self.population_rank])
            self.population = np.concatenate((individuals, new_individuals))

            self.population_rank = self._ranking(self.population)

            self._store_best_results()

            if np.min(self.best_ind) == 0:
                print(
                    f'[{gen}] The two best solutions:'
                    f'\nf{np.round(self.population_rank[0][0], 4)} = {np.round(self.population_rank[0][1], 4)}'
                    f'\nf{np.round(self.population_rank[1][0], 4)} = {np.round(self.population_rank[1][1], 4)}'
                )
                break


    def _store_best_results(self) -> None:
        """Stores the best individual and the average of the top 10 individuals."""
        self.best_ind.append(self.population_rank[0][1])
        self.avg_top_10.append(
            np.mean([j for _, j in self.population_rank[:10]])
        )
