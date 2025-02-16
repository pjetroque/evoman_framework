################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np
import neat
import pandas as pd
import pickle
sys.path.insert(0, 'evoman')
from environment import Environment
from specialist_controller import NEAT_Controls



# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# runs simulation



def eval_genomes(genomes, config):
    fitness_array = []
    fitness_array_smop = []
    global generation
    generation += 1
    for genome_id, genome in genomes:
        genome.fitness = 0

        f, p, e, t = env.play(pcont=genome)

        fitness_new = 0.9 * (100 - e) + 0.1 * p - np.log(t)
        fitness_smop = (100 / (100 - (0.9 * (100 - e) + 0.1 * p - np.log10(t))))

        children_index.append([generation, f, p, e, t])
        #children_data.append(genome)

        fitness_array.append(fitness_new)
        fitness_array_smop.append(fitness_smop)

        genome.fitness = fitness_new

    total_fitness_data.append([np.max(fitness_array),
                               np.mean(fitness_array),
                               np.std(fitness_array)])


def evaluate_best_sol(environment, run, experiment_name, enemy):
    global env

    env = environment
    best_sol_data = []

    infile = open(f'{experiment_name}/enemy_{enemy}/best_sol_{run}.obj', 'rb')
    best_sol = pickle.load(infile)
    infile.close()

    for j in range(5):
        best_sol.fitness = 0

        f, p, e, t = env.play(pcont=best_sol)

        best_sol_data.append([f, p, e, t])

    if not os.path.exists(experiment_name+"/enemy_"+str(enemy)+"/boxplot"):
        os.makedirs(experiment_name+"/enemy_"+str(enemy)+"/boxplot")


    best_sol_df = pd.DataFrame(best_sol_data, columns=['fitness', 'p_health',
                                                              'e_health', 'time'])
    best_sol_df.to_csv(f'{experiment_name}/enemy_{enemy}/boxplot/data_{run}.csv', index=False)


def run(environment, generations, config_file, run, experiment_name, enemy):
    """
    runs the NEAT algorithm to train a neural network to play mega man.
    It uses the config file named config-feedforward.txt. After running it stores it results in CSV files.
    """
    global env
    global total_fitness_data
    global children_index
    global children_data
    global generation

    env = environment
    total_fitness_data = []
    children_index = []
    children_data = []
    generation = 0


    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to generations generations.
    winner = p.run(eval_genomes, generations)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    if not os.path.exists(experiment_name+"/enemy_"+str(enemy)):
        os.makedirs(experiment_name+"/enemy_"+str(enemy))

    # stats to csv

    total_fitness_data_df = pd.DataFrame(total_fitness_data, columns = ["max", "mean", "std_dev"])
    total_fitness_data_df.to_csv(f'{experiment_name}/enemy_{enemy}/fitness_data_{run}.csv', index = False)

    children_index_df = pd.DataFrame(children_index, columns = ['generation', 'fitness', 'p_health',
                         'e_health', 'time'])
    children_index_df.to_csv(f'{experiment_name}/enemy_{enemy}/full_data_index_{run}.csv', index = False)

    file = open(f'{experiment_name}/enemy_{enemy}/best_sol_{run}.obj', 'wb')
    pickle.dump(winner, file)
    file.close()

    # children_data_df = pd.DataFrame(children_data)
    # children_data_df.to_csv(f'{experiment_name}/full_data_{run}.csv', index = False)

