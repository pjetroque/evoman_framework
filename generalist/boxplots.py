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
    ticklabels = ['             Trained on 2, 3, 5, 7','', '             Trained on 1, 4, 6, 8', '']
    fig, ax = plt.subplots()
    save = 'final_plot_data/boxplot_final_norandomini.png'
    plt.grid(axis='y')
    plt.axhline(y=0, color='Grey', linestyle='--')
    data = [[[-196.99999999999966, -237.43999999999974, 15.00000000000091, -80.11999999999945, -83.19999999999953, -200.6399999999997, 23.200000000000898, -112.71999999999966, -193.51999999999967, -185.71999999999971],[-144.3599999999995, -55.91999999999937, -168.03999999999968, -104.63999999999949, -119.99999999999969, -107.91999999999966, -134.51999999999936, -154.31999999999968, -136.83999999999932, -162.15999999999968]],[[-179.47999999999922, -35.079999999999195, -100.71999999999916, -274.1599999999992, -289.3999999999992, -50.199999999999136, -153.83999999999898, -253.95999999999898, -281.19999999999925, -261.79999999999916], [-70.43999999999929, -122.31999999999888, -121.71999999999893, -280.3999999999992, -219.19999999999908, -117.7199999999992, -248.39999999999927, -105.03999999999932, -220.39999999999918, -287.31999999999914]]]
    for idx, f in enumerate(folders):
        gain_list_alg1 = data[idx][0] #run_winners(f[0], runs)
        gain_list_alg2 = data[idx][1] #run_winners(f[1], runs)

        print(gain_list_alg1)
        print(gain_list_alg2)

        #create boxplot
        bp1= ax.boxplot(gain_list_alg1, positions = [((idx+1)*4-2)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor=[(1), (102/255), (102/255)], alpha = 1), medianprops = dict(color = 'k'), widths = 0.4)
        bp2 = ax.boxplot(gain_list_alg2, positions=[((idx + 1) * 4 - 1)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor=[(102/255), (102/255), (1)], alpha = 1), medianprops = dict(color = 'k'), widths = 0.4 )

    ax.set_xticklabels(ticklabels, fontsize = 14)
    plt.ylim(-300, 50)
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
