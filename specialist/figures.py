# -*- coding: utf-8 -*-
"""
make figures
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

generations = 50
population = 100

def prepare_data(enemies):
    data_name = 'data_memory' #NEAT
    folder_base = 'data_memory/final_dataset/enemy_'

    ### LINE PLOT DATA ###
    df_MAX = pd.DataFrame([], index = range(1,(generations+1)))
    df_MEAN = pd.DataFrame([],index = range(1,(generations+1)))

    for idx, enemy in enumerate(enemies):
        folder = folder_base+str(enemy)+"_standard"

        max_gain = []
        mean_gain = []
        max_stdev_gain = []
        mean_stdev_gain =[]

        for generation in range(0,(generations+0)):
            gain_list = np.array([])
            max_list = []
            mean_list = []

            for run in range(runs):

                df = pd.read_csv(f'{folder}/full_data_index_{run}.csv')
                player_health = df["p_health"]
                enemy_health = df["e_health"]
                gain = player_health - enemy_health

                gain_list = np.append(gain_list, np.array(gain[df["generation"]==generation]))
                max_list.append(np.max(gain_list))
                mean_list.append(np.mean(gain_list))



            gain_list = np.array(gain_list)
            max_gain.append(np.mean(max_list))
            mean_gain.append(np.mean(mean_list))
            max_stdev_gain.append(np.std(max_list))



            mean_stdev_gain.append(np.std(mean_list))


        df_MAX["max_gain_enemy_"+str(enemy)] = max_gain
        df_MAX["std_max_gain_enemy_"+str(enemy)] = max_stdev_gain
        df_MEAN["mean_gain_enemy_"+str(enemy)] = mean_gain
        df_MEAN["std_mean_gain_enemy_"+str(enemy)] = mean_stdev_gain


    df_MAX.to_csv(f'final_plot_data/lineplot_MAX_{data_name}_gains.csv')
    df_MEAN.to_csv(f'final_plot_data/lineplot_MEAN_{data_name}_gains.csv')

    # ### BOX PLOT DATA###
    # df_gain = pd.DataFrame([])
    # for idx, enemy in enumerate(enemies):
    #     #folder = 'NEAT/final_results/enemy_'+str(enemy)+'/boxplot'  # folder with the data for boxplots
    #     folder = f'{folder_base}{enemy}_standard/boxplots'  # folder with the data for boxplots
    #
    #     gain_list = []
    #     for run in range(runs):
    #         df = pd.read_csv(f'{folder}/fitness_data_{run}.csv')
    #         gain = (df["p_health"] - df["e_health"]).mean()
    #         gain_list.append(gain)
    #     df_gain["enemy_"+str(enemy)] = gain_list
    #
    # if not os.path.exists('final_plot_data'):
    #     os.makedirs('final_plot_data')
    # df_gain.to_csv(f'final_plot_data/boxplot_{data_name}_gains.csv')



def line_plots(runs, enemies):
    folder_MAX_NEAT = 'final_plot_data/lineplot_MAX_NEAT_gains.csv'   # folder with the data for line plots
    folder_MEAN_NEAT = 'final_plot_data/lineplot_MEAN_NEAT_gains.csv'

    folder_MAX_NORMAL = 'final_plot_data/lineplot_MAX_data_normal_gains.csv'   # folder with the data for line plots
    folder_MEAN_NORMAL = 'final_plot_data/lineplot_MEAN_data_normal_gains.csv'

    folder_MAX_MEMORY = 'final_plot_data/lineplot_MAX_data_memory_gains.csv'   # folder with the data for line plots
    folder_MEAN_MEMORY = 'final_plot_data/lineplot_MEAN_data_memory_gains.csv'


    df_MAX_NEAT = pd.read_csv(folder_MAX_NEAT)
    df_MEAN_NEAT = pd.read_csv(folder_MEAN_NEAT)

    df_MAX_NORMAL = pd.read_csv(folder_MAX_NORMAL)
    df_MEAN_NORMAL = pd.read_csv(folder_MEAN_NORMAL)

    df_MAX_MEMORY = pd.read_csv(folder_MAX_MEMORY)
    df_MEAN_MEMORY = pd.read_csv(folder_MEAN_MEMORY)



    for idx, enemy in enumerate(enemies):

        mean_of_max_NEAT = df_MAX_NEAT["max_gain_enemy_"+str(enemy)]
        std_of_max_NEAT = df_MAX_NEAT["std_max_gain_enemy_"+str(enemy)]

        mean_of_max_NORMAL = df_MAX_NORMAL["max_gain_enemy_"+str(enemy)]
        std_of_max_NORMAL = df_MAX_NORMAL["std_max_gain_enemy_"+str(enemy)]

        mean_of_max_MEMORY = df_MAX_MEMORY["max_gain_enemy_"+str(enemy)]
        std_of_max_MEMORY = df_MAX_MEMORY["std_max_gain_enemy_"+str(enemy)]


        mean_of_mean_NEAT = df_MEAN_NEAT["mean_gain_enemy_"+str(enemy)]
        std_of_mean_NEAT = df_MEAN_NEAT["std_mean_gain_enemy_"+str(enemy)]

        mean_of_mean_NORMAL = df_MEAN_NORMAL["mean_gain_enemy_"+str(enemy)]
        std_of_mean_NORMAL = df_MEAN_NORMAL["std_mean_gain_enemy_"+str(enemy)]

        mean_of_mean_MEMORY = df_MEAN_MEMORY["mean_gain_enemy_"+str(enemy)]
        std_of_mean_MEMORY = df_MEAN_MEMORY["std_mean_gain_enemy_"+str(enemy)]



        #plotting
        x = range(1, (len(mean_of_max_NEAT)+1))

        fig, ax = plt.subplots()
        ax.plot(x, mean_of_max_NEAT, linestyle = "dashed", color = "C0", label='Max NEAT')
        ax.plot(x, mean_of_mean_NEAT, color = "C0", label='Mean NEAT')
        ax.fill_between(x,
                        mean_of_max_NEAT - std_of_max_NEAT,
                        mean_of_max_NEAT + std_of_max_NEAT,color = "C0",  alpha=0.2)
        ax.fill_between(x,
                        mean_of_mean_NEAT - std_of_mean_NEAT,
                        mean_of_mean_NEAT + std_of_mean_NEAT, color = "C0", alpha=0.2)

        x = range(0, (len(mean_of_max_NORMAL) ))
        ax.plot(x, mean_of_max_NORMAL, linestyle = "dashed", color = "C1", label='Max BBEA')
        ax.plot(x, mean_of_mean_NORMAL, color = "C1", label='Mean BBEA')
        ax.fill_between(x,
                        mean_of_max_NORMAL - std_of_max_NORMAL,
                        mean_of_max_NORMAL + std_of_max_NORMAL, color = "C1",alpha=0.2)
        ax.fill_between(x,
                        mean_of_mean_NORMAL - std_of_mean_NORMAL,
                        mean_of_mean_NORMAL + std_of_mean_NORMAL, color = "C1", alpha=0.2)

        ax.plot(x, mean_of_max_MEMORY, linestyle = "dashed", color = "C2",label='Max MBBEA')
        ax.plot(x, mean_of_mean_MEMORY, color = "C2", label='Mean MBBEA')
        ax.fill_between(x,
                        mean_of_max_MEMORY - std_of_max_MEMORY,
                        mean_of_max_MEMORY + std_of_max_MEMORY, color = "C2", alpha=0.2)
        ax.fill_between(x,
                        mean_of_mean_MEMORY - std_of_mean_MEMORY,
                        mean_of_mean_MEMORY + std_of_mean_MEMORY, color = "C2", alpha=0.2)

        ax.set_xlim(1, generations)
        ax.set_xticks(np.arange(1, generations+1, int(generations/10)))
        plt.ylim(-110,120)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Gain', fontsize=14)
        plt.title(f'Fitness over the generations against enemy {int(enemy)}', fontsize=14)
        plt.legend(loc = 'lower right')
        plt.grid()
        plt.savefig(f'final_plot_data/lineplot_{enemy}.png')
        plt.show()


def box_plots(runs, enemies):
    NEAT_file = 'final_plot_data/boxplot_NEAT_gains.csv'  # file with the data for boxplots
    MEMORY_file = 'final_plot_data/boxplot_data_memory_gains.csv' #'final_plot_data/boxplot_data_memory_gains.csv'
    NORMAL_file = 'final_plot_data/boxplot_data_normal_gains.csv'

    fig, ax = plt.subplots()

    df_NEAT = pd.read_csv(NEAT_file)
    df_MEMORY = pd.read_csv(MEMORY_file)
    df_NORMAL = pd.read_csv(NORMAL_file)

    for idx, enemy in enumerate(enemies):

        gain_list_NEAT = df_NEAT["enemy_"+str(enemy)]
        gain_list_MEMORY = df_MEMORY["enemy_"+str(enemy)]
        gain_list_NORMAL = df_NORMAL["enemy_"+str(enemy)]
        #create boxplot
        bp1= ax.boxplot(gain_list_NEAT, positions = [((idx+1)*4-2)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor='r', alpha = 0.6), medianprops = dict(color = 'k'), widths = 0.4)
        bp2 = ax.boxplot(gain_list_NORMAL, positions=[((idx + 1) * 4 - 1)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor='b', alpha = 0.6), medianprops = dict(color = 'k'), widths = 0.4 )
        bp3 = ax.boxplot(gain_list_MEMORY, positions = [((idx + 1) * 4 - 0)],  patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor='g', alpha = 0.6), medianprops = dict(color = 'k'), widths = 0.4)
    ax.set_xticklabels(['','enemy 1','', '', 'enemy 5', '', '','enemy 8', ''], fontsize = 14)
    plt.title("Boxplots NEAT, BBEA and MBBEA algorithm", fontsize =18, fontweight = 18)
    #plt.xlabel('Algorithm 1', fontsize=14)
    #plt.xticks([])
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ["NEAT", "BBEA", "MBBEA"], loc='lower right')

    plt.ylabel('Gain', fontsize=14)

    plt.savefig('final_plot_data/boxplots.png')
    plt.show()




os.chdir(r'C:\Users\Sicco\PycharmProjects\evoman_framework')
runs = 10 #amount of runs
enemies = [1, 5, 8]
# folder = 'final_results/enemy_1' #folder with info per generation and best weights
#prepare_data(enemies)
line_plots(runs, enemies)
#box_plots(runs, enemies)
