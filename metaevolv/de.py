"""Differential Evolution Algorithm.

This module contains the implementation of the Differential Evolution (DE) algorithm. The DE is a
population-based optimization algorithm that uses the difference of vectors to generate new
candidates. The algorithm is based on the work of Storn and Price (1997). The algorithm is
based on the following steps:

1. Initialization: The algorithm starts by generating a population of candidate solutions.
2. Mutation: The algorithm generates a new candidate by adding the difference of two vectors to a third vector.
3. Crossover: The algorithm generates a trial vector by combining the new candidate with the target vector.
4. Selection: The algorithm selects the best candidate to form the new population.
5. Evaluation: The new population is evaluated.
6. Stopping criterion: The algorithm stops when a stopping criterion is met.
7. Early stop: The algorithm stops if the optimal individual is found.

The algorithm is controlled by the following parameters:

- Search range: Lower and upper limit for the variables that are to be optimized.
- Population size: Number of candidates to solve the problem.
- Dimensions: Number of variables to be optimized.
- Objective function: Function to be minimized.
- Step size: Step size towards the difference vector.
- Probability of crossover occurrence: Probability of crossover occurrence.
- Maximum number of iterations: Maximum number of iterations.
- Opposition: Whether the algorithm will run with opposition.
- Jump rate: Probability of occurrence of opposition.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from pydantic import BaseModel, Field, field_validator

sns.set_style('whitegrid')


class DifferentialEvolutionConfig(BaseModel):
    """Configuration for the Differential Evolution Algorithm."""

    search_range: tuple[float, float]
    n_population: int = Field(
        ..., ge=1, description='Number of candidates to solve the problem'
    )
    dimensions: int = Field(
        ..., ge=1, description='Number of variables to be optimized'
    )
    f: float = Field(
        ..., ge=0, description='Step size towards the difference vector'
    )
    cr: float = Field(
        ..., ge=0, le=1, description='Probability of crossover occurrence'
    )
    max_iter: int = Field(
        ..., ge=1, description='Maximum number of iterations'
    )
    opposition: bool = Field(
        False, description='Whether the algorithm will run with opposition'
    )
    jr: float = Field(
        1.0,
        ge=0,
        le=1,
        description='Jump rate. Probability of occurrence of opposition',
    )

    @field_validator('search_range')
    def check_search_range(
        cls, v: tuple[float, float]
    ) -> tuple[float, float]:
        """Check if the search range is valid."""
        if v[0] >= v[1]:
            raise ValueError(
                'search_range must be a tuple where the first value is less than the second value'
            )
        return v


class DifferentialEvolution:
    """
    Class that contains all the functions to perform the Differential Evolution (DE) to minimize an objective
    function of interest.
    """

    def __init__(
        self,
        config: DifferentialEvolutionConfig,
        obj_function: Callable[[ndarray], float],
    ) -> None:
        """Initializes the parameters for running the algorithm.

        Args:
            config (DifferentialEvolutionConfig): Configuration object containing all the parameters.
            obj_function (Callable[[ndarray], float]): Objective function to be minimized.
        """
        self.config = config
        self.obj_function = obj_function

        self.best_vetor = None
        self.best_eval = None
        self.results = None
        self.avg_tracking = []
        self.best_tracking = []

        self.candidates = np.arange(self.config.n_population)
        self.population = np.random.uniform(
            self.config.search_range[0],
            self.config.search_range[1],
            (self.config.n_population, self.config.dimensions),
        )

        if self.config.opposition:
            lower = np.min(self.population)
            upper = np.max(self.population)
            diff = lower + upper
            for i in range(round(self.config.n_population / 2)):
                for j in range(self.population[i].shape[0]):
                    self.population[i][j] = diff - self.population[i][j]


    def _evaluate(self) -> list[float]:
        """Calculates the score for each vector (fitness)."""
        return [self.obj_function(ind) for ind in self.population]


    def _opposition_operator(
        self, vector: ndarray, lower: float, upper: float
    ) -> ndarray:
        """Operates opposition in vector.

        Args:
            vector (ndarray): Vector to operate opposition.
            lower (float): Lower limit for the variables.
            upper (float): Upper limit for the variables.

        Returns:
            ndarray: Vector with opposition.
        """
        diff = lower + upper
        for i in range(vector.shape[0]):
            vector[i] = diff - vector[i]
        return vector


    def _mutation(self, vectors: ndarray) -> ndarray:
        """Operates mutation on vectors.

        Args:
            vectors (ndarray): Vectors to operate mutation.

        Returns:
            ndarray: Mutated vector.
        """
        return vectors[0] + self.config.f * (vectors[1] - vectors[2])


    def _crossover(
        self, mutated_vector: ndarray, target_vector: ndarray
    ) -> ndarray:
        """Operates crossover on vectors.

        Args:
            mutated_vector (ndarray): Mutated vector.
            target_vector (ndarray): Target vector.

        Returns:
            ndarray: Trial vector.
        """
        rng = np.random.rand(self.config.dimensions)
        trial = np.array(
            [
                mutated_vector[k]
                if rng[k] < self.config.cr
                else target_vector[k]
                for k in range(self.config.dimensions)
            ]
        )
        return trial


    def plot(self, log: bool = False, avg: bool = True) -> None:
        """Plot the results.

        Args:
            log (bool, optional): Whether to plot the evaluation in log scale. Defaults to False.
            avg (bool, optional): Whether to plot the average evaluation. Defaults to True.
        """
        plt.figure(figsize=(10, 7))
        if avg:
            plt.plot(
                np.arange(len(self.avg_tracking)),
                self.avg_tracking,
                c='k',
                linestyle='--',
                label='average',
            )
        plt.plot(
            np.arange(len(self.best_tracking)),
            self.best_tracking,
            c='k',
            label='best',
        )
        plt.xlim((0, len(self.avg_tracking)))
        if log:
            plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Evaluation')
        plt.title('Evaluation by Iteration')
        plt.legend()
        sns.despine(bottom=False, left=False)
        plt.show()


    def fit(self) -> None:
        """Runs the DE algorithm."""
        eval_pop = self._evaluate()
        self.best_vetor = self.population[np.argmin(eval_pop)]
        self.best_eval = np.min(eval_pop)
        self.results = [self.best_eval]
        self.avg_tracking.append(np.mean(eval_pop))
        self.best_tracking.append(self.best_eval)

        for i in range(self.config.max_iter):
            lower = np.min(self.population)
            upper = np.max(self.population)

            for j in range(self.config.n_population):
                rng = 1.0
                if self.config.opposition:
                    rng = np.random.rand()

                if rng < self.config.jr:
                    trial = self._opposition_operator(
                        self.population[j], lower, upper
                    )
                    trial_eval = self.obj_function(trial)
                else:
                    vectors_idx = np.delete(self.candidates, j)
                    vectors = self.population[np.random.choice(vectors_idx, 3)]
                    mutated = self._mutation(vectors)
                    mutated = np.clip(
                        mutated,
                        self.config.search_range[0],
                        self.config.search_range[1],
                    )
                    trial = self._crossover(mutated, self.population[j])
                    trial_eval = self.obj_function(trial)

                if trial_eval < eval_pop[j]:
                    self.population[j] = trial
                    eval_pop[j] = trial_eval

            self.best_eval = np.min(eval_pop)
            self.avg_tracking.append(np.mean(eval_pop))
            self.best_tracking.append(self.best_eval)

            if self.best_eval < self.results[-1]:
                self.best_vetor = self.population[np.argmin(eval_pop)]
                self.results.append(self.best_eval)
                print(
                    f'Best solution at iteration {i} -> Vector:{self.best_vetor}, evaluation: {self.best_eval}.'
                )
            if self.results[-1] == 0:
                break
