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
from bio_functions import crossover, mutation, get_children, fitfunc
import csv
import matplotlib.pyplot as plt


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'data_normal/winnertest'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def run_winners(folder, runs):
    n_hidden_neurons = 10       #number of hidden neurons
    n_vars = (20+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    enemy = [1,2,3,4,5,6,7,8]   #which enemy
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
                          randomini = "yes")

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
    ticklabels = ['             Training group 1','', '             Training group 2', '']
    fig, ax = plt.subplots()
    save = 'final_plot_data/boxplot_test.png'

    for idx, f in enumerate(folders):
        gain_list_alg1 = run_winners(f[0], runs)
        gain_list_alg2 = run_winners(f[1], runs)

        #create boxplot
        bp1= ax.boxplot(gain_list_alg1, positions = [((idx+1)*4-2)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor='r', alpha = 0.6), medianprops = dict(color = 'k'), widths = 0.4)
        bp2 = ax.boxplot(gain_list_alg2, positions=[((idx + 1) * 4 - 1)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor='b', alpha = 0.6), medianprops = dict(color = 'k'), widths = 0.4 )

    ax.set_xticklabels(ticklabels, fontsize = 14)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Alg 1", "Alg 2"], loc='lower right')
    plt.ylabel('Gain', fontsize=14)
    plt.savefig(save, dpi=300)
    plt.show()


folder_alg1_group1 = 'data_normal/enemy_[2, 6]_standard_testforplots' # folder with data for algorithm 1 for enemy group 1
folder_alg2_group1 = 'data_normal/enemy_[2, 6]_standard_testforplots' # folder with data for algorithm 2 for enemy group 1
folder_alg1_group2 = 'data_normal/enemy_[2, 6]_standard_testforplots' # folder with data for algorithm 1 for enemy group 2
folder_alg2_group2 = 'data_normal/enemy_[2, 6]_standard_testforplots' # folder with data for algorithm 2 for enemy group 2
runs = #should be 10 for final runs
folders = [(folder_alg1_group1, folder_alg2_group1), (folder_alg1_group2, folder_alg2_group2)]
boxplots(folders, runs)
