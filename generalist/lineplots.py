# -*- coding: utf-8 -*-
"""
make lineplots for a training group of enemies
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def retrieve_data(data_folder, runs):
    max_values = np.array([])
    mean_values = np.array([])

    for run in range(runs):
        total_data = []

        #open data
        with open(f'{data_folder}/data_{run}.csv', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                total_data.append(row)

        #extract basic info and fitness data
        enemies, generations = total_data[0]
        total_fitness_data = np.array(total_data[1:])
        generations = int(generations)
        population_size = len(total_fitness_data)

        #extract means and max
        max_values = np.append(max_values, total_fitness_data[:,0])
        mean_values = np.append(mean_values, total_fitness_data[:,1])

    #extract mean of max and mean of mean and their standard deviations
    max_values = max_values.reshape(runs, generations)
    mean_values = mean_values.reshape(runs, generations)
    mean_of_max = np.mean(max_values, axis=0)
    mean_of_mean = np.mean(mean_values, axis=0)
    std_of_max = np.std(max_values, axis=0) / np.sqrt(runs)
    std_of_mean = np.std(mean_values, axis=0) / np.sqrt(runs)

    return mean_of_max, mean_of_mean, std_of_max, std_of_mean, generations


def line_plots(folders, runs):
    '''Creates a line plot for one training group of enemies showing the performance
    of different algorithm results located in folers.'''

    labels = [['Max Islands', 'Mean Islands'], ['Max Regulatory', 'Mean Regulatory']]
    colors = ['Red', 'Blue']
    save = 'final_plot_data/lineplot_[1, 4, 6, 8].png'
    fig, ax = plt.subplots()

    for i, f in enumerate(folders):
        mean_of_max, mean_of_mean, std_of_max, std_of_mean, generations = retrieve_data(f, runs)

        #plotting
        x = range(0, (len(mean_of_max)))
        ax.plot(x, mean_of_max, linestyle = "dashed", color = colors[i], label=labels[i][0])
        ax.plot(x, mean_of_mean, color = colors[i], label=labels[i][1])
        ax.fill_between(x,
                        mean_of_max - std_of_max,
                        mean_of_max + std_of_max ,color = colors[i],  alpha=0.2)
        ax.fill_between(x,
                        mean_of_mean - std_of_mean,
                        mean_of_mean + std_of_mean, color = colors[i], alpha=0.2)

    ax.set_xlim(0, generations)
    ax.set_xticks(np.arange(0, generations+1, int(generations/10)))
    # plt.ylim(-110,120)
    plt.text(12, -400, 'Enemies 1, 4, 6, 8', fontsize=14)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Gain', fontsize=14)
    # plt.legend(loc = 'lower right')
    plt.grid()
    plt.savefig(save, dpi=300)
    plt.show()


data_folder_alg1 = 'data_normal/enemy_[1, 4, 6, 8]_standard_eucl' #folder with data for algorithm 1
data_folder_alg2 = 'data_normal/enemy_[1, 4, 6, 8]_standard_reg' #folder with data for algorithm 2
runs = 10  #should be 10 for final runs
folders = [data_folder_alg1, data_folder_alg2]
line_plots(folders, runs)
