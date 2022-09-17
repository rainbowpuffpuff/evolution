#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import dataclasses
import sys, os

from matplotlib import pyplot as plt

sys.path.insert(0, 'evoman')
from environment import Environment
from controller_task1 import MaskedController

# imports other libs
import numpy as np

LOGS_DIRECTORY = 'solutions_task1'
EXPERIMENTS_DIRECTORY = 'experiments_task1'

if not os.path.exists(LOGS_DIRECTORY):
    os.makedirs(LOGS_DIRECTORY)

if not os.path.exists(EXPERIMENTS_DIRECTORY):
    os.makedirs(EXPERIMENTS_DIRECTORY)

@dataclasses.dataclass
class ExperimentConfig:
    N = 100
    D = 4
    W = 10
    generations = 100

class Experiment(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

        n_hidden_neurons = 0
        # initializes environment for single objective mode (specialist)  with static enemy and ai player
        self.env = Environment(experiment_name=LOGS_DIRECTORY,
                               playermode="ai",
                               player_controller=MaskedController(n_hidden_neurons),
                               speed="normal",
                               enemymode="static",
                               level=2)

    def initialize(self):
        # D layers of W neurons each
        weights = np.random.uniform(-1, 1, (self.cfg.N, self.cfg.D, self.cfg.W, self.cfg.W))
        biases = np.random.uniform(-1, 1, (self.cfg.N, self.cfg.D, self.cfg.W))
        masks = np.random.choice(a=[0., 1.], size=(self.cfg.N, self.cfg.D, self.cfg.W, self.cfg.W), p=[0.5, 0.5])
        # Final layer that maps to 5 outputs
        final_weight = np.random.uniform(-1, 1, (self.cfg.N, self.cfg.W, 5))
        final_bias = np.random.uniform(-1, 1, (self.cfg.N, 5))
        final_mask = np.random.choice(a=[0., 1.], size=(self.cfg.N, self.cfg.W, 5), p=[0.5, 0.5])

        # Stack the parameters into a list of tuples
        population = list(zip(weights, biases, masks, final_weight, final_bias, final_mask))

        return population

    def select(self, population, fitness):
        # input is the population array and the fitness for each agent
        # return the indices of agent that you want to keep as a list
        subset = ...
        return subset

    def reproduce(self, parents, fitness):
        # input is the 'parents' selected by self.select(...) and their fitnesses
        # return the new population for the next iteration
        new_population = ...
        return new_population

    def best(self, population, fitness):
        # Return the best agent in the population
        return population[np.argmax(fitness)], np.max(fitness)

    def finish(self, best, best_fitness):
        # Stop condition, return True if we want to stop running the program
        return self.generation >= self.cfg.generations

    def criterion(self, fitness, player_life, enemy_life, time):
        # We can experiment with different fitness functions here
        return fitness

    def evaluate(self, population):
        fitness = []
        for sol in population:
            # sol is a tuple with weights, biases, masks, final_weight, final_bias, final_mask
            # fitness, player_life, enemy_life, time
            f, p, e, t = self.env.play(sol)
            fitness = self.criterion(f, p, e, t)
            fitness.append(fitness)
        return np.array(fitness)

    def run_iteration(self, population):
        fitness = self.evaluate(population)
        self.logs['average_fitness'].append(np.mean(fitness))
        best, best_fitness = self.best(population, fitness)
        self.logs['best_fitness'].append(best_fitness)
        parent_indices = self.select(population, fitness)

        population = self.reproduce(population[parent_indices], fitness[parent_indices])
        self.generation += 1
        return population, best, best_fitness

    def run(self):
        assert self.generation == 0, 'Experiment already ran, call reset() to clear the logs before running it again'
        population, best, best_fitness = self.run_iteration(self.initialize())
        while self.finish(best, best_fitness):
            population = self.run_iteration(population)

        return best, best_fitness

    def plot(self, show=False):
        fig = plt.figure()
        # Plot the logs
        plt.plot(self.logs['best_fitness'], label='best')
        plt.plot(self.logs['average_fitness'], label='average')
        plt.legend()
        if show:
            plt.show()
        else:
            fig.savefig(f'{LOGS_DIRECTORY}/fitness.png')

    def reset(self):
        """ Clear logs and reset generation counter, needed to run the experiment again """
        self.generation = 0
        self.logs = {
            'best_fitness': [],
            'average_fitness': [],
        }


# For the second method we can just superclass stuff
class ExperimentVariation(Experiment):
    def select(self, population, fitness):
        pass


if __name__ == '__main__':
    cfg = ExperimentConfig()
    experiment = Experiment(cfg)

    experiment.run()
    experiment.plot(show=True)

    # TODO: Save the best solution (pickle) to EXPERIMENTS_DIRECTORY
    # TODO: Save the logs and plots as well to EXPERIMENTS_DIRECTORY
    # TODO: Best if all experiments are saved maybe in subfolders like experiment1, experiment2, etc.

