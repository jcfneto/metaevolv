"""Particle Swarm Optimization (PSO) algorithm.

This module contains the implementation of the Particle Swarm Optimization (PSO)
algorithm. The PSO is a population-based optimization algorithm that uses the
movement of particles to find the optimal solution. The algorithm is based on the
work of Kennedy and Eberhart (1995). The algorithm is based on the following steps:

1. Initialization: The algorithm starts by generating a population of particles.
2. Movement: The algorithm updates the position of particles and their speeds.
3. Update bests: The algorithm updates the best position of particles.
4. Evaluation: The algorithm evaluates the new position of particles.
5. Stopping criterion: The algorithm stops when a stopping criterion is met.
6. Early stop: The algorithm stops if the optimal individual is found.

The algorithm is controlled by the following parameters:

- Number of particles: Number of particles in the swarm.
- Maximum number of iterations: Maximum number of iterations.
- Number of dimensions: Number of variables to be optimized.
- Search range: Range of values for the variables.
- Speed range: Range of speed values for the particles.
- Inertial weighting: Inertial weighting parameter.
- Cognitive and social parameters: Cognitive and social parameters.
- Constrain factor: Activate the constrain factor.
- Constrain factor parameter: Constrain factor parameter.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from pydantic import BaseModel, Field, field_validator

sns.set_style('whitegrid')


class PSOConfig(BaseModel):
    """Configuration class for Particle Swarm Optimization (PSO) algorithm."""

    n_particles: int = Field(..., ge=1, description='Number of particles')
    max_iter: int = Field(
        ..., ge=1, description='Maximum number of iterations'
    )
    dimensions: int = Field(
        ..., ge=1, description='Number of variables to be optimized'
    )
    search_range: tuple[float, float]
    speed_range: tuple[float, float]
    w: tuple[float, float] = Field(
        (0.9, 0.4), description='Inertial weighting'
    )
    c: tuple[float, float] = Field(
        (2.05, 2.05), description='Cognitive (c1) and social (c2) parameters'
    )
    const_factor: bool = Field(
        False, description='Activate the constrain factor'
    )
    k: float = Field(1.0, description='Constrain factor parameter')

    @field_validator('search_range', 'speed_range')
    def check_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Check if the range is valid."""
        if v[0] >= v[1]:
            raise ValueError(
                'Range must be a tuple where the first value is less than the second value'
            )
        return v


class PSO:
    """
    Class that contains all the functions to perform the Particle Swarm Optimization (PSO) to minimize an objective
    function of interest.
    """

    def __init__(
        self, config: PSOConfig, obj_function: Callable[[ndarray], float]
    ) -> None:
        """Initializes the parameters for running the algorithm.

        Args:
            config (PSOConfig): Configuration object containing all the parameters.
            obj_function (Callable[[ndarray], float]): Objective function to be minimized.
        """
        self.config = config
        self.obj_function = obj_function

        self.phi = np.sum(self.config.c)

        if not self.config.const_factor:
            self.config.w = self.config.w[1] - (
                (self.config.w[1] - self.config.w[0]) / self.config.max_iter
            ) * np.linspace(0, self.config.max_iter, self.config.max_iter)
        else:
            self.const_factor_value = (2 * self.config.k) / np.abs(
                (2 - self.phi - np.sqrt((self.phi**2) - (4 * self.phi)))
            )

        self.r1 = np.random.random(self.config.max_iter)
        self.r2 = np.random.random(self.config.max_iter)

        self.particles = np.random.uniform(
            self.config.search_range[0],
            self.config.search_range[1],
            (self.config.n_particles, self.config.dimensions),
        )

        self.velocities = np.random.uniform(
            self.config.speed_range[0],
            self.config.speed_range[1],
            (self.config.n_particles, self.config.dimensions),
        )

        self.p_best = self.particles
        self.p_best_eval = np.array(
            [self.obj_function(p) for p in self.p_best]
        )
        self.g_best_eval = [np.min(self.p_best_eval)]
        self.g_best = self.particles[np.argmin(self.p_best_eval)]
        self.evol_best_eval = []
        self.movements = 1
        self.hist = []


    def _update_velocities_particles(self, k: int) -> None:
        """Updates the position of particles and their speeds.

        Args:
            k (int): Current iteration.
        """
        commom = self.config.c[0] * self.r1[k] * (
            self.p_best - self.particles
        ) + self.config.c[1] * self.r2[k] * (self.g_best - self.particles)

        if self.config.const_factor:
            self.velocities = self.const_factor_value * commom
        else:
            self.velocities = (self.config.w[k] * self.velocities) + commom

        self.particles += self.velocities
        self.particles = np.clip(
            self.particles,
            self.config.search_range[0],
            self.config.search_range[1],
        )

        self.velocities = np.clip(
            self.velocities,
            self.config.speed_range[0],
            self.config.speed_range[1],
        )
        self.velocities = np.where(
            (self.particles == self.config.search_range[0])
            | (self.particles == self.config.search_range[1]),
            0,
            self.velocities,
        )


    def _update_bests(self, i: int) -> None:
        """Updates the p_best and g_best.

        Args:
            i (int): Particle index.
        """
        xi = self.particles[i]
        xi_eval = self.obj_function(xi)

        if xi_eval < self.p_best_eval[i]:
            self.p_best_eval[i] = xi_eval
            self.p_best[i] = xi

            if xi_eval < self.g_best_eval[-1]:
                self.g_best_eval.append(xi_eval)
                self.g_best = xi


    def plot(self, log: bool = False) -> None:
        """Plot the score evolution graph.

        Args:
            log (bool, optional): If True, the y-axis will be in logarithmic scale. Defaults to False.
        """
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(0, self.movements), self.evol_best_eval, c='black')
        if log:
            plt.yscale('log')
        plt.xlim((0, self.movements))
        plt.xlabel('Iteration')
        plt.ylabel('Evaluation')
        plt.title('Evaluation by Iteration')
        sns.despine(bottom=False, left=False)
        plt.show()


    def fit(self) -> None:
        """Runs the PSO algorithm."""
        for k in range(self.config.max_iter):
            self.hist.append(self.particles)
            self.movements = k + 1
            self._update_velocities_particles(k=k)
            self.evol_best_eval.append(np.min(self.p_best_eval))

            if self.g_best_eval[-1] == 0.0:
                break

            for i in range(self.config.n_particles):
                self._update_bests(i=i)
