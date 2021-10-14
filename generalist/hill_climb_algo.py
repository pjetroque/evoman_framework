# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:02:42 2021

@author: Sicco
Events file
"""


import numpy as np
import scipy
import pandas as pd
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from scipy import optimize
from demo_controller import player_controller
import glob, os
from scipy.optimize import differential_evolution

best_sol_file = "best_sol_0.csv"#"data_normal/enemy_[1, 2, 3, 4, 6, 7]_errfoscilation/best_sol_1.csv"
experiment_name = "hill_climb_algo"

#make save folder
if not os.path.exists(f'{experiment_name}'):
    os.makedirs(f'{experiment_name}')


env = Environment(experiment_name=f'{experiment_name}',
                          playermode="ai",
                          player_controller=player_controller(10),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          randomini = "no")

# choose this for not using visuals and thus making experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

#make sure to not print every startup of the pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

k = 0


def optimization_func(player, enemies):
        player = np.tanh(player)

        gains = []
        health_player = []
        health_enemies = []
        times = []
        kills = []


        # battle each enemy in this list
        for enemy in enemies:
            gain_avg = 0
            time_avg = 0
            health = 0
            health_enemy = 0
            kill = 0

            # repeat each player to counter the randomness

            f, p, e, t = env.run_single(enemy, pcont=player[:265], econt="None")


            health +=  p
            health_enemy +=  e
            time_avg +=  t
            gain_avg +=  p -  e
            if e == 0:
                kill += 1

            gains.append(gain_avg)
            health_player.append(health)
            health_enemies.append(health_enemy)
            times.append(time_avg)
            kills.append(kill)

        global k
        k+=1
        print(k)

        print(np.sum(gains), kills)

        return [player, np.sum(gains), gains,  times, health_player, health_enemies,
                kills]

def minimizer_optim(utility, theta_ini):
    # print(theta_ini)
    options = {'eps': 1e-09,  # argument convergence criteria
               'disp': True,  # display iterations
               'maxiter': 10}  # maximum number of iterations

    results = scipy.optimize.minimize(utility, theta_ini,
                                      options=options,
                                      method='Nelder-Mead')

    print(results.x)
    print(np.tanh(results.x))

def define_bounds(array, c):
    array = array.T[0]
    low = array-c
    high = array+c
    low = low.tolist()
    high = high.tolist()
    bounds = list(zip(low, high))
    return bounds

def diff_evolution(utility, theta_ini):

    bounds = define_bounds(theta_ini, 0.001)

    print(bounds)

    result = differential_evolution(utility, bounds)

    print(result.X)


def main():
    df = pd.read_csv(best_sol_file, header = None)

    theta_ini = df.values.T
    theta_ini[np.where(theta_ini == 1)] = 0.9999
    theta_ini[np.where(theta_ini == -1)] = -0.9999

    theta_ini = np.arctanh(theta_ini)

    enemies = [1,2,3,4,5,6,7,8]

    utility = lambda y: (-(optimization_func(y, enemies)[1]))

    #minimizer_optim(utility, theta_ini)

    diff_evolution(utility, theta_ini)


if __name__ == "__main__":
    main()