"""
make data for boxplots and make boxplots
"""
# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
# from bio_functions import crossover, mutation, get_children, fitfunc
import csv
import matplotlib.pyplot as plt


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'data_normal/winnertest'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def run_winners(folder, runs, enemy):
    n_hidden_neurons = 10       #number of hidden neurons
    n_vars = (20+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    population_size = 100       #pop size
    test_number = 5             #times to run the best individual

    all_gains = []
    for run in range(runs):
        weights_data = []
        total_fitness_data = []

        #initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          randomini = "no")

        #open data
        with open(f'{folder}/best_sol_{run}.csv', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                weights_data.append(row)
            player = np.array(weights_data[0])


        #test the specific individual multiple times
        gain_list = []
        for r in range(test_number):
            p_list, e_list = [], []
            for en in enemy:
                f, p, e, t = env.run_single(en, pcont=player[:n_vars], econt="None")
                p_list.append(p)
                e_list.append(e)
            gain = np.sum(np.array([p_list])) - np.sum(np.array([e_list]))
            gain_list.append(gain)
        all_gains.append(np.mean(gain_list))

    return all_gains


def boxplots(folders, runs):
    ticklabels = ['             Enemies 2, 3, 5, 7','', '             Enemies 1, 4, 6, 8', '']
    fig, ax = plt.subplots()
    save = 'final_plot_data/boxplot_final_per_group.png'
    plt.grid(axis='y', zorder=1)
    plt.axhline(y=0, color='Grey', linestyle='--', zorder=1)
    enemy = [[2, 3, 5, 7], [1, 4, 6, 8]]   #which enemy
    data = [[[113.36000000000033, 104.2000000000003, 197.6400000000006, 152.64000000000053, 224.16000000000045, 130.08000000000033, 195.12000000000063, 135.3600000000003, 134.56000000000031, 126.44000000000032],
[154.5600000000005, 203.60000000000062, 129.08000000000033, 205.84000000000052, 188.68000000000035, 193.40000000000035, 145.48000000000067, 156.7600000000003, 125.08000000000068, 147.12000000000032]], [[-42.59999999999949, 71.80000000000055, 36.40000000000056, -75.59999999999947, 14.600000000000804, 71.80000000000086, -1.1999999999992639, -43.999999999999275, -70.39999999999951, 56.20000000000084],
[73.2000000000005, 62.80000000000077, 34.000000000000824, 55.60000000000082, 30.2000000000007, 18.80000000000055, 43.60000000000073, 149.6000000000006, 73.60000000000082, 4.600000000000776]]]
    for idx, f in enumerate(folders):
        print(enemy[idx])
        gain_list_alg1 = data[idx][0] #run_winners(f[0], runs, enemy[idx])
        gain_list_alg2 = data[idx][1] #run_winners(f[1], runs, enemy[idx])

        print(gain_list_alg1)
        print(gain_list_alg2)

        #create boxplot
        bp1= ax.boxplot(gain_list_alg1, positions = [((idx+1)*4-2)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor=[(1), (102/255), (102/255)], alpha = 1), medianprops = dict(color = 'k'), widths = 0.4, zorder=10000)
        bp2 = ax.boxplot(gain_list_alg2, positions=[((idx + 1) * 4 - 1)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor=[(102/255), (102/255), (1)], alpha = 1), medianprops = dict(color = 'k'), widths = 0.4, zorder=10000)

    ax.set_xticklabels(ticklabels, fontsize = 14)
    plt.ylim(-100, 300)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Islands", "Regulatory"], loc='lower left')
    plt.ylabel('Gain', fontsize=14)
    plt.savefig(save, dpi=300)
    plt.show()

# , per trainingroep


folder_alg1_group1 = 'data_normal/enemy_[2, 3, 5, 7]_standard_eucl' # folder with data for algorithm 1 for enemy group 1
folder_alg2_group1 = 'data_normal/enemy_[2, 3, 5, 7]_standard_reg'# folder with data for algorithm 2 for enemy group 1
folder_alg1_group2 = 'data_normal/enemy_[1, 4, 6, 8]_standard_eucl' # folder with data for algorithm 1 for enemy group 2
folder_alg2_group2 = 'data_normal/enemy_[1, 4, 6, 8]_standard_reg' # folder with data for algorithm 2 for enemy group 2
runs = 10 #should be 10 for final runs
folders = [(folder_alg1_group1, folder_alg2_group1), (folder_alg1_group2, folder_alg2_group2)]
boxplots(folders, runs)
