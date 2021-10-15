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
               'maxiter': 40}  # maximum number of iterations

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

    bounds = define_bounds(theta_ini, 0.1)

    print(bounds)

    result = differential_evolution(utility, bounds)

    print(result.X)

    utility(result.X)


def main():
    #df = pd.read_csv(best_sol_file, header = None)
    #theta_ini = df.values.T
    #print(theta_ini)
    theta_ini = [-0.385303911,0.445471417,0.28652132,0.5635923,0.021597644,0.108016858,-0.664716997,-0.015371625,-0.733936594,-0.426020786,0.578805611,-0.197852799,0.596904692,-0.873708003,0.30375544,0.434910483,-0.19333661,0.75754787,-0.431303542,-0.432567367,-0.616580994,0.69937126,-0.749194027,0.498421222,-0.776467056,-0.576748388,-0.436926976,-0.735865276,-0.581057634,-0.383402476,-0.454060078,-0.404066764,0.454916264,0.542537929,-0.144840705,0.549972056,-0.735161377,0.351538909,0.084061525,-0.412626433,-0.798309076,-0.220599025,0.645220865,0.108839794,0.658300906,0.737118929,-0.763631697,-0.498044128,-0.39620437,-0.605733365,-0.367443642,-0.589702536,0.455484659,-0.390708522,0.540705856,0.134821106,-0.538463373,-0.765312918,0.725636735,-0.545819988,0.381558991,-0.67994626,-0.341324106,0.745963828,-0.262247466,-0.48607213,0.68559782,-0.596428212,-0.647708606,0.49438855,-0.446596411,-0.460813469,-0.302459686,0.235422481,-0.269071191,0.013905246,-0.817570898,0.528787308,-0.530463048,-0.858418521,-0.484191819,-0.247267189,0.127959465,-0.420521634,0.706211432,0.334019056,0.266944525,0.637602247,-0.763905445,-0.588023419,0.164105183,-0.837985673,-0.813899423,-0.516291799,0.361905125,-0.373259572,-0.309151758,-0.44288066,0.666065288,-0.482689584,0.67063394,-0.487249214,0.722587506,0.16992198,-0.580509543,-0.076041889,-0.309084524,-0.460481036,-0.149568088,-0.358248774,-0.505939504,0.299487828,0.519289411,-0.515637647,0.170535265,0.251583078,-0.50482725,-0.469434743,-0.059965467,-0.690037772,0.765864743,-0.418247641,-0.728397575,0.663954164,-0.848288074,0.63036719,-0.742376384,0.325512929,-1,0.65900152,-0.19959352,-0.496247294,0.665985965,0.482322429,-0.629610367,-0.640739085,0.213080607,-0.306136141,-0.392157107,-0.361258506,-0.949617348,-0.617096986,-0.721427162,0.443955133,0.416606882,-0.500720022,-1,0.066408841,0.450934443,-0.73752401,-0.159732189,-0.234744394,-0.758625901,-0.490409969,-0.548103867,-0.257449195,0.815236383,-0.206118942,0.362699883,-0.040517128,-0.443006192,-0.43376949,0.838289491,-0.394860209,-0.575197954,-0.422908196,-0.287141945,0.425477854,-0.291572968,0.729585914,-0.311978652,-0.735926339,-0.375931859,-0.681765155,-0.049339081,0.347402962,0.612258696,-0.118298788,-0.327812421,0.384280841,0.149699226,0.697667455,0.218272191,-0.383266419,0.367942031,-0.362670327,-0.244671744,0.38925683,-0.554517712,0.761759164,-0.657818899,0.631236714,-0.589820172,0.487213875,-0.036569525,-0.22485888,0.419801337,0.379954855,-0.712919843,-0.688064701,-0.01722747,0.340424588,0.427744353,0.545753565,-0.464505369,-0.595362007,-0.982018524,-0.635102315,0.346310577,0.595192812,-0.679430401,-0.753191756,0.817632844,0.870999462,0.636783416,0.679928723,-0.541302227,-0.616889932,0.129551615,0.489011717,0.438366673,0.777299321,-0.032210643,0.494785886,-0.696592038,-0.562271325,-0.195887569,0.465844005,0.407486005,-0.720668606,-0.267494643,0.892366774,0.654742424,-0.29318585,0.579298909,-0.666194485,0.155561105,-0.381725687,0.691369701,-0.041071395,-0.636825139,-0.705428395,-0.112774147,-0.062527719,-0.157479006,0.538181001,0.698251791,-0.339485155,0.681193027,-0.761682587,-0.843136437,-0.486097499,0.113424539,0.74064333,-0.609547383,0.257974362,-0.366344354,0.328558558,0.686980175,0.277756965,0.710376699,0.758187798,-0.68017135,-0.232763581,-0.617930006,0.185457374,0.150530362,0.152488588,0.087206872]#
    theta_ini = np.array([np.array([item]) for item in theta_ini])
    print(theta_ini)
    theta_ini[np.where(theta_ini == 1)] = 0.9999
    theta_ini[np.where(theta_ini == -1)] = -0.9999

    theta_ini = np.arctanh(theta_ini)

    enemies = [1,2,3,4,5,6,7,8]

    utility = lambda y: (-(optimization_func(y, enemies)[1]))

    #minimizer_optim(utility, theta_ini)

    diff_evolution(utility, theta_ini)


if __name__ == "__main__":
    main()