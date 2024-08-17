# MetaEvolv

**MetaEvolv** is a Python library that implements various evolutionary metaheuristic optimization algorithms, including ABC (Artificial Bee Colony), Clonalg (Clonal Selection Algorithm), DE (Differential Evolution), GA (Genetic Algorithm), and PSO (Particle Swarm Optimization).


## Installation

To install the library:

```bash
pip install metaevolv
```

## Implemented Algorithms

### Artificial Bee Colony (ABC)

ABC is an optimization algorithm based on the foraging behavior of honeybees. It is used to find minima of an objective function.

### Clonal Selection Algorithm (Clonalg)
Clonalg is an algorithm inspired by the clonal selection process of the immune system. It is used for function minimization.

### Differential Evolution (DE)
DE is an evolutionary algorithm that optimizes continuous problems. It uses mutation and crossover operations to evolve a population of solutions.

### Genetic Algorithm (GA)
GA is an algorithm inspired by the theory of natural evolution. It uses genetic operators like selection, crossover, and mutation to find optimal solutions.

## Particle Swarm Optimization (PSO)
PSO is inspired by the behavior of bird flocks and fish schools. It optimizes an objective function by moving particles through the search space.

## How to Use

To use any of the algorithms, you need to import the specific module and configure the algorithm with the desired parameters.

### Example: Genetic Algorithm (GA)

Here is an example of how to use the Genetic Algorithm (GA) with MetaEvolv:

```python
import numpy as np
from metaevolv.ga import GeneticAlgorithm, GAConfig

# Define your objective function
def objective_function(x: np.ndarray) -> float:
    return np.sum(x**2)

# Configure the genetic algorithm
config = GAConfig(
    bits=10,
    dimensions=2,
    n_population=50,
    search_range=(-10.0, 10.0),
    k=3,
    cp=0.8,
    mp=0.01,
    max_iter=100,
    selection_type='tournament',
    crossover_type='one_point',
    mutation_type='bit_by_bit',
    pc_variation='constant',
    cp_final=0.9
)

# Instantiate the genetic algorithm with the configuration
ga = GeneticAlgorithm(config, objective_function)

# Run the algorithm
ga.fit()

# Display the results
print(f"Best solution found: {ga.best_ind[-1]}")
print(f"Objective function value at the best solution: {ga.best_eval}")
```

## License

This project is licensed under the terms of the MIT license.