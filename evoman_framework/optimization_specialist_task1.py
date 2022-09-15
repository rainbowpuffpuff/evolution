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

sys.path.insert(0, 'evoman')
from environment import Environment
from controller_task1 import player_controller

# imports other libs
import numpy as np

experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 0

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="normal",
                  enemymode="static",
                  level=2)

@dataclasses
class ExperimentConfig():
    N = 100
    D = 4
    W = 10
    generations = 100

model =  [intialize, select, select2, reproduce, best]

class Experiment(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.generation = 0

    def initialize(self):
        # For now this just assumes that all layers are the same size
        population = np.random.choice(a=[False, True], size=(self.cfg.N, self.cfg.D, self.cfg.W, self.cfg.W), p=[0.5, 0.5])
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

    def run_iteration(self, population):
        fitness = self.evaluate(population)
        best, best_fitness = self.best(population, fitness)
        parent_indices = self.select(population, fitness)

        population = self.reproduce(population[parent_indices], fitness[parent_indices])
        self.generation += 1
        return population, best, best_fitness

    def run(self):
        population, best, best_fitness = self.run_iteration(self.initialize())
        while self.finish(population, best, best_fitness):
            population = self.run_iteration(population)

        return best, best_fitness

# For the second method we can just superclass stuff
class ExperimentVariation(Experiment):
    def select(self, population, fitness):
        pass

cfg = ExperimentConfig()
experiment = Experiment(cfg)


# tests saved demo solutions for each enemy
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])

    # Load specialist controller
    sol = np.loadtxt('solutions_demo/demo_' + str(en) + '.txt')
    print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
    # fitness, player_life, enemy_life, time
    fitnesses = []
    for sol in population:
        f, p, e, t = env.play(sol)
        fitnesses.append(f)

    top10 = list(sorted(fitnesses, reverse=True))[:10]

    # rcombine top10, mutate

    population = ...
