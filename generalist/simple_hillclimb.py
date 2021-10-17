"""
Regular hillclimber
"""
# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import copy
import numpy as np
from math import fabs,sqrt
import glob, os
from bio_functions import crossover, mutation, get_children, fitfunc
import csv
import matplotlib.pyplot as plt


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'data_normal/hillclimbtest'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

def play_game(player, env):
    'Play a game, return gain'
    enemy = [1,2,3,4,5,6,7,8]   #which enemy

    gain = 0
    for en in enemy:
        f, p, e, t = env.run_single(en, pcont=player, econt="None")
        gain += p
        gain -= e

    return gain


def initial(folder):
    n_hidden_neurons = 10       #number of hidden neurons
    n_vars = (20+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    weights_data = []

    #open data
    with open(f'{folder}/best_sol_0.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            weights_data.append(row)
        player = np.array(weights_data[0])[:n_vars]

    #initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      randomini = "no")

    #calculate gain of player
    best_gain = play_game(player, env)
    print('old best gain:', best_gain)
    climb(player, env, best_gain)


def climb(player, env, best_gain):
    value = 0.1

    for i in range(len(player)):
        mutate_add = copy.deepcopy(player)
        mutate_substract = copy.deepcopy(player)

        # add value
        mutate_add[i] += value
        new_gain = play_game(mutate_add, env)
        if new_gain > best_gain:
            print('gen (increased)', i)
            print('new gain', new_gain)

            return climb(mutate_add, env, new_gain)

        # substract value
        mutate_substract[i] -= value
        new_gain = play_game(mutate_substract, env)
        if new_gain > best_gain:
            print('gen (decreased)', i)
            print('new gain', new_gain)

            return climb(mutate_substract, env, new_gain)

    print('no better solution found')

    #save best option
    with open(f'{experiment_name}/bestsol.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(player)

    return player

folder_best_sol = 'data_normal/winnertest' # folder with best solution
initial(folder_best_sol)
