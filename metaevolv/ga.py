"""Genetic Algorithm (GA) algorithm.

This module contains the Genetic Algorithm (GA) algorithm. The GA is a 
population-based optimization algorithm that uses the principles of natural
selection to evolve a population of candidate solutions. The algorithm is
based on the following steps:

1. Initialization: The algorithm starts by generating a population of individuals.
2. Evaluation: The individuals are evaluated using an objective function.
3. Selection: The individuals are selected based on their fitness.
4. Crossover: The individuals are combined to generate new individuals.
5. Mutation: The individuals are mutated to introduce diversity.
6. Stopping criterion: The algorithm stops when a stopping criterion is met.
7. Early stop: The algorithm stops if the optimal individual is found.

The algorithm is controlled by the following parameters:

- Number of bits: Number of bits for each variable.
- Number of dimensions: Number of variables to be optimized.
- Population size: Number of individuals in the population.
- Search range: Lower and upper limit for the variables that are to be optimized.
- Number of individuals in the tournament selection: Number of individuals in the tournament selection.
- Crossover probability: Probability of crossover.
- Mutation probability: Probability of mutation.
- Maximum number of generations: Maximum number of generations.
- Selection type: Type of selection method (tournament or roulette wheel).
- Crossover type: Type of crossover method (one-point, two-points, or uniform).
- Mutation type: Type of mutation method (bit-by-bit or random choice).
- Probability variation: Variation of the crossover probability.
- Final crossover probability: Final crossover probability.
"""

from typing import Callable, Literal

import numpy as np
from numpy import ndarray
from pydantic import BaseModel, Field, field_validator


# TODO: implement elitism.
# TODO: devise a technique that increases the mutation rate as generations lose diversity.
class GAConfig(BaseModel):
    """Configuration class for Genetic Algorithm (GA) algorithm."""

    bits: int = Field(
        ..., ge=1, description='Number of bits for each variable'
    )
    dimensions: int = Field(
        ...,
        ge=1,
        description='Number of variables in the optimization problem',
    )
    n_population: int = Field(
        ..., ge=1, description='Number of individuals in the population'
    )
    search_range: tuple[float, float]
    k: int = Field(
        ...,
        ge=1,
        description='Number of individuals in the tournament selection',
    )
    cp: float = Field(..., ge=0, le=1, description='Crossover probability')
    mp: float = Field(..., ge=0, le=1, description='Mutation probability')
    max_iter: int = Field(..., ge=1, description='Number of generations')
    selection_type: Literal['tournament', 'roulette_wheel']
    crossover_type: Literal['one_point', 'two_points', 'uniform']
    mutation_type: Literal['bit_by_bit', 'random_choice']
    pc_variation: Literal['constant', 'linear']
    cp_final: float = Field(
        ..., ge=0, le=1, description='Final crossover probability'
    )

    @field_validator('search_range')
    def check_search_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Check if the search range is valid."""
        if v[0] >= v[1]:
            raise ValueError(
                'search_range must be a tuple where the first value is less than the second value'
            )
        return v


class GeneticAlgorithm:
    """Genetic Algorithm (GA) algorithm."""

    def __init__(
        self, config: GAConfig, obj_function: Callable[[ndarray], float]
    ) -> None:
        """
        Initializes the parameters for running the Genetic Algorithm.

        Args:
            config (GAConfig): Configuration object containing all the parameters.
        """
        self.config = config
        self.config.search_range = [
            self.config.search_range for _ in range(config.dimensions)
        ]
        self.obj_function = obj_function

        self.selected = np.array([])
        self.population_decimal = np.array([])
        self.population_eval = np.array([])
        self.best_ind = []
        self.best_eval = np.inf
        self.mp = np.array([])

        self.bitstring_size = self.config.bits * self.config.dimensions
        self.population = np.array(
            [
                np.random.randint(0, 2, self.bitstring_size).tolist()
                for _ in range(self.config.n_population)
            ]
        )


    def _binary_to_decimal(self) -> None:
        """Converts the binary population to decimal."""
        population_decimal = []
        for ind in self.population:
            bit_string = ''.join([str(i) for i in ind])
            d = []
            for i in range(self.config.dimensions):
                di = int(
                    bit_string[
                        i * self.config.bits : self.config.bits * (i + 1)
                    ],
                    2,
                )
                di = self.config.search_range[i][0] + (
                    di / 2**self.config.bits
                ) * (
                    self.config.search_range[i][1]
                    - self.config.search_range[i][0]
                )
                d.append(di)
            population_decimal.append(d)
        self.population_decimal = np.array(population_decimal)


    def _evaluate(self) -> None:
        """Evaluates the population."""
        self.population_eval = np.array(
            [self.obj_function(d) for d in self.population_decimal]
        )


    def _tournament(self) -> None:
        """Selects the best individual from a tournament."""
        selected = []
        for _ in range(self.config.n_population):
            fighters = np.random.randint(
                0, self.config.n_population, self.config.k
            )
            fighters_fitness = self.population_eval[fighters]
            winner = np.argmin(fighters_fitness)
            selected.append(self.population[winner])
        self.selected = np.array(selected)


    def _roulette_wheel(self) -> None:
        """Selects individuals using the roulette wheel method."""
        invert = 1 / (self.population_eval + 1)
        prob = invert / np.sum(invert)
        ids = np.random.choice(
            a=range(self.config.n_population),
            size=self.config.n_population,
            p=prob,
        )
        self.selected = np.array(self.population)[ids].tolist()


    def _selection(self) -> None:
        """Selects the individuals for the next generation."""
        if self.config.selection_type == 'tournament':
            self._tournament()
        elif self.config.selection_type == 'roulette_wheel':
            self._roulette_wheel()


    def _one_point(self, p1: ndarray, p2: ndarray) -> tuple[ndarray, ndarray]:
        """Performs the one-point crossover.

        Args:
            p1 (ndarray): First parent.
            p2 (ndarray): Second parent.

        Returns:
            tuple[ndarray, ndarray]: Children generated by the crossover.
        """
        cutoff = np.random.randint(1, self.bitstring_size - 2)
        c1 = np.concatenate((p1[:cutoff], p2[cutoff:]))
        c2 = np.concatenate((p2[:cutoff], p1[cutoff:]))
        return c1, c2


    def _two_points(self, p1: ndarray, p2: ndarray) -> tuple[ndarray, ndarray]:
        """Performs the two-point crossover.

        Args:
            p1 (ndarray): First parent.
            p2 (ndarray): Second parent.

        Returns:
            tuple[ndarray, ndarray]: Children generated by the crossover."""
        c1, c2 = p1.copy(), p2.copy()
        cutoff = np.sort(np.random.randint(0, self.bitstring_size, 2))
        while cutoff[0] == cutoff[1]:
            cutoff = np.sort(np.random.randint(0, self.bitstring_size, 2))
        c1[cutoff[0] : cutoff[1]] = p2[cutoff[0] : cutoff[1]]
        c2[cutoff[0] : cutoff[1]] = p1[cutoff[0] : cutoff[1]]
        return c1, c2


    def _uniform(self, p1: ndarray, p2: ndarray) -> tuple[ndarray, ndarray]:
        """Performs the uniform crossover.

        Args:
            p1 (ndarray): First parent.
            p2 (ndarray): Second parent.

        Returns:
            tuple[ndarray, ndarray]: Children generated by the crossover.
        """
        uniform_c1 = np.random.randint(0, 2, self.bitstring_size)
        uniform_c2 = 1 - uniform_c1
        parents = [p1, p2]
        c1 = np.array(
            [parents[uniform_c1[j]][j] for j in range(self.bitstring_size)]
        )
        c2 = np.array(
            [parents[uniform_c2[j]][j] for j in range(self.bitstring_size)]
        )
        return c1, c2


    def _crossover_prob_variation(self) -> None:
        """Variates the crossover probability."""
        if self.config.pc_variation == 'constant':
            self.config.cp = np.array(
                [self.config.cp for _ in range(self.config.max_iter)]
            )
        elif self.config.pc_variation == 'linear':
            self.config.cp = (
                np.arange(0, self.config.max_iter)
                * (self.config.cp_final - self.config.cp)
                / (self.config.max_iter - 1)
                + self.config.cp
            )
        self.mp = 1.0 - self.config.cp


    def _crossover(self, gen: int) -> None:
        """Performs the crossover operation.

        Args:
            gen (int): Current generation.
        """
        childrens = []
        for i in range(0, self.config.n_population, 2):
            parent1 = self.selected[i]
            parent2 = self.selected[i + 1]
            if np.random.rand() < self.config.cp[gen]:
                if self.config.crossover_type == 'one_point':
                    c1, c2 = self._one_point(parent1, parent2)
                elif self.config.crossover_type == 'two_points':
                    c1, c2 = self._two_points(parent1, parent2)
                elif self.config.crossover_type == 'uniform':
                    c1, c2 = self._uniform(parent1, parent2)
                childrens += [c1] + [c2]
            else:
                childrens += [parent1] + [parent2]
        self.population = np.array(childrens)


    def _bit_by_bit(self, gen: int) -> None:
        """Performs the bit-by-bit mutation.

        Args:
            gen (int): Current generation.
        """
        for i in range(self.config.n_population):
            for j in range(self.bitstring_size):
                if np.random.rand() < self.mp[gen]:
                    self.population[i, j] = 1 - self.population[i, j]


    def _random_choice(self, gen: int) -> None:
        """Performs the random choice mutation.

        Args:
            gen (int): Current generation.
        """
        for i in range(self.config.n_population):
            if np.random.rand() < self.mp[gen]:
                bit = np.random.randint(0, self.bitstring_size, 1)[0]
                self.population[i, bit] = 1 - self.population[i, bit]


    def _mutation(self, gen: int) -> None:
        """Performs the mutation operation.

        Args:
            gen (int): Current generation.
        """
        if self.config.mutation_type == 'bit_by_bit':
            self._bit_by_bit(gen)
        elif self.config.mutation_type == 'random_choice':
            self._random_choice(gen)


    def fit(self) -> None:
        """Runs the Genetic Algorithm."""
        self._crossover_prob_variation()
        for gen in range(self.config.max_iter):
            self._binary_to_decimal()
            self._evaluate()
            best_id = np.argmin(self.population_eval)
            if self.population_eval[best_id] < self.best_eval:
                self.best_eval = self.population_eval[best_id]
                self.best_ind = self.population_decimal[best_id]
                print(
                    f'[{gen+1}] --> Novo melhor indivíduo: f({self.best_ind}) = {self.best_eval}.'
                )
            if self.best_eval == 0.0:
                break
            self._selection()
            self._crossover(gen)
            self._mutation(gen)
