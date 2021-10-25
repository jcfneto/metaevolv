import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# graphic style
sns.set_style("whitegrid")


class PSO:
    """
    Class that contains all the functions to perform the Particle Swarm Optimization (PSO) to minimize an objective
    function of interest.
    """

    def __init__(self, n_particles: int, max_iter: int, dimensions: int, search_range: tuple,
                 speed_range: tuple, obj_func, **kwargs):
        """
        Description: Initializes the parameters for running the algorithm.

        Args:
            n_particles : Number of particles.
            max_iter    : Maximum number of iterations.
            dimensions  : Number of variables to be optimized.
            search_range: Lower and upper limit for the variables that are to be optimized.
            speed_range : Lower and upper limit for speed parameter.
            obj_func    : Function to be minimized
            **kwargs    :

                w           : Inertial weighting. Default: (0.9, 0.4).
                c           : Cognitive (c1) and social c2) parameter (learning rate). Default: (2.05, 2.05).
                const_factor: Bool value to activate the constrain factor. Default: False.
                k           : ***
        """

        # hyperparameters
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.dimensions = dimensions
        self.search_range = search_range
        self.speed_range = speed_range
        self.obj_func = obj_func
        self.w = (0.9, 0.4)
        self.c = (2.05, 2.05)
        self.const_factor = False
        self.k = 1

        # getting options parameters
        for (k, v) in kwargs.items():
            setattr(self, k, v)

        # calculating variable parameters
        self.phi = np.sum(self.c)

        # if not constrain factor, calculetes w
        if not self.const_factor:
            self.w = self.w[1] - ((self.w[1] - self.w[0]) / max_iter) * np.linspace(0, self.max_iter, self.max_iter)
        else:
            print('Constrain Factor')
            self.const_factor = (2 * self.k) / np.abs((2 - self.phi - np.sqrt((self.phi ** 2) - (4 * self.phi))))

        # rng
        self.r1 = np.random.random(max_iter)
        self.r2 = np.random.random(max_iter)

        # initializing the particles
        self.particles = np.random.uniform(self.search_range[0],
                                           self.search_range[1],
                                           (self.n_particles, self.dimensions))

        # initializing the velocities
        self.velocities = np.random.uniform(self.speed_range[0],
                                            self.speed_range[1],
                                            (self.n_particles, self.dimensions))

        # variables to store the results
        self.p_best = self.particles
        self.p_best_eval = np.array([self.obj_func(p) for p in self.p_best])
        self.g_best_eval = [np.min(self.p_best_eval)]
        self.g_best = self.particles[np.argmin(self.p_best_eval)]
        self.evol_best_eval = []
        self.movements = 1

        # remover depois
        self.hist = []

    def update_velocities_particles(self, k: int):
        """
        Description: Updates the position of particles and their speeds.

        Args:
            k: Current iteration (iteration n).
        """

        # the commom part of equation
        commom = self.c[0] * self.r1[k] * (self.p_best - self.particles) + \
                 self.c[1] * self.r2[k] * (self.g_best - self.particles)

        # updating velocities (if constrain factor was true or false)
        if self.const_factor:
            self.velocities = self.const_factor * commom
        else:
            self.velocities = (self.w[k] * self.velocities) + commom

        # updating particles
        self.particles += self.velocities

        # limiting if values exceed limits
        self.particles = np.where(self.particles < self.search_range[0], self.search_range[0], self.particles)
        self.particles = np.where(self.particles > self.search_range[1], self.search_range[1], self.particles)
        self.velocities = np.where(self.velocities < self.speed_range[0], self.speed_range[0], self.velocities)
        self.velocities = np.where(self.velocities > self.speed_range[1], self.speed_range[1], self.velocities)
        res_min1, res_min2 = np.where(self.particles == self.search_range[0])
        res_max1, res_max2 = np.where(self.particles == self.search_range[1])

        # Set speed to 0 when needed
        for idx_min in list(zip(res_min1, res_min2)):
            self.velocities[idx_min] = 0

        for idx_max in list(zip(res_max1, res_max2)):
            self.velocities[idx_max] = 0

    def update_bests(self, i: int):
        """
        Description: Updates the p_best and g_best.

        Args:
            i: Current particle (particle i).
        """

        xi = self.particles[i]
        xi_eval = self.obj_func(xi)

        if xi_eval < self.p_best_eval[i]:
            self.p_best_eval[i] = xi_eval
            self.p_best[i] = xi

            if xi_eval < self.g_best_eval[-1]:
                self.g_best_eval.append(xi_eval)
                self.g_best = xi

    def plot(self, log: bool = False):
        """
        Description: Plot the score evolution graph.

        Args:
             log: Bool value to change scale of y axis to log. Default: False.
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

    def fit(self):
        """
        Description: Runs the PSO algorithm.
        """

        # moving
        for k in range(self.max_iter):
            self.hist.append(self.particles)
            self.movements = k + 1
            self.update_velocities_particles(k=k)
            self.evol_best_eval.append(np.min(self.p_best_eval))

            # check if optimal solution was found
            if self.g_best_eval[-1] == 0.0:
                break

            # updating p_best and g_best
            for i in range(self.n_particles):
                self.update_bests(i=i)
