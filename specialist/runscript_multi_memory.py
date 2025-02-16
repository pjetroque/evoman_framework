################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller_memory import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from bio_dingen import crossover, mutation, get_children, fitfunc
import csv
import time
import multiprocessing as mp

# choose this for not using visuals and thus making experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

#make sure to not print every startup of the pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

class evo_algorithm:
    def __init__(self, n_hidden_neurons, enemy, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, run):
        self.n_hidden_neurons = n_hidden_neurons       #number of hidden neurons
        self.enemy = int(enemy)          #which enemy
        self.run_nr = run_nr                  #number of runs
        self.generations = generations           #number of generations per run
        self.population_size = population_size       #pop size
        self.mutation_baseline = mutation_baseline    #minimal chance for a mutation event
        self.mutation_multiplier = mutation_multiplier  #fitness dependent multiplier of mutation chance
        self.repeats = repeats
        self.fitter = fitter
        self.experiment_name = f'enemy_{enemy}_{fitter}'
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = (20*2+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
        self.total_fitness_data = []
        self.best = []
        self.max_health = 0
        self.run = run
        self.min_health_enemy = 100
        self.max_gain = -100
        self.children_index = []
        
        #make save folder
        if not os.path.exists(f'data_memory/{self.experiment_name}'):
            os.makedirs(f'data_memory/{self.experiment_name}')  
    
    def play_game(self, player, g, avg_fitness):
        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=f'data_memory/{self.experiment_name}',
                          enemies=[self.enemy],
                          playermode="ai",
                          player_controller=player_controller(self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          randomini = "yes")
        fitness_new = 0
        fitness_smop = 0
        health = 0
        health_enemy = 0
        repeats = self.repeats
        time_avg = 0
        
        #repeat each player to counter the randomness
        for i in range(repeats):
            f, p, e, t = env.play(pcont=player)
            fitness_new += (0.9*(100 - e) + 0.1*p - np.log(t))*(1/repeats)
            fitness_smop += fitfunc(self.fitter, self.generations, g, t, e, p)*(1/repeats)
            health += (1/repeats)*p
            health_enemy += (1/repeats)*e
            time_avg += (1/repeats)*t
            surviving_player = False
            
            #if nog good enough 'die'
            if not fitness_smop > avg_fitness/(repeats-i):
                break
            
            #if the player is consistently good, survive
            if i == (repeats-1):
                surviving_player = True
                
        return  [fitness_smop, health, surviving_player, player, health_enemy, time_avg]
    
    
    def simulate(self, pop = []):
        #initiate 100 parents
        if not len(pop) == self.population_size:
            pop = np.random.uniform(-1, 1, (self.population_size ,self.n_vars))
        max_health = 0
        avg_fitness = 0
        
        for g in range(self.generations):
            gen_start = time.time()
            fitness_array = []
            fitness_array_smop = []
            health_array = []
            surviving_players = []
            pool = mp.Pool(mp.cpu_count())
            
            results = [pool.apply_async(self.play_game, args=(player, g, avg_fitness)) for player in pop]
            
            pop = []
            
            #extract the results 
            for ind, result in enumerate(results):
                r = result.get()
                fitness_array_smop.append(r[0])
                pop.append(r[3])
                health = r[1]
                health_array.append(r[1])
                survive = r[2]
                health_enemy = r[4]
                gain = health - health_enemy
                self.children_index.append([g, r[0], health, health_enemy, r[5]])
                
                if gain > self.max_gain:
                    self.max_health = health
                    self.min_health_enemy = health_enemy
                    self.max_gain = gain
                    self.best = r[3]
                    
                if survive:
                    surviving_players.append(ind)
            pool.close()
            
            if np.mean(fitness_array_smop) > avg_fitness:
                avg_fitness = np.mean(fitness_array_smop)
            else:
                avg_fitness *= 0.9
            
            self.total_fitness_data.append([np.max(fitness_array_smop),
                                            np.mean(fitness_array_smop),
                                            np.std(fitness_array_smop),
                                            np.max(health_array),
                                            np.mean(health_array),
                                            np.std(health_array)])
            
            pop = get_children(pop, surviving_players, np.array(fitness_array_smop),
                               mutation_baseline, mutation_multiplier)
            
            print(f'Run: {self.run}, Fitter: {self.fitter}, Generation {g}, fit_mean = {round(np.mean(fitness_array_smop),2)} pm {round(np.std(fitness_array_smop),2)}, fitness_best = {round(np.max(fitness_array_smop),2)}, best_avg_health = {np.round(self.max_health,2)}, best_gain = {self.max_gain}, time={round(time.time()-gen_start)}')
        return
    
    def save_results(self, extended = False, full = False):
    
        with open(f'data_memory/{self.experiment_name}/fitness_data_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.enemy, self.generations, self.max_health])
            writer.writerows(self.total_fitness_data)
        
        with open(f'data_memory/{self.experiment_name}/best_sol_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.best)
        
        if extended:
            with open(f'data_memory/{self.experiment_name}/full_data_index_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['generation', 'fitness', 'p_health', 
                                 'e_health', 'time'])
                writer.writerows(self.children_index)
        if full:
            with open(f'data_memory/{self.experiment_name}/full_data_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.children_data)
            
        return


if __name__ == '__main__':
    for enemy in [1]:
        for fitter in ["standard"]:
            n_hidden_neurons = 10       #number of hidden neurons
            enemy = int(enemy)          #which enemy
            run_nr = 10                 #number of runs
            generations = 100            #number of generations per run
            population_size = 100       #pop size
            mutation_baseline = 0.02    #minimal chance for a mutation event
            mutation_multiplier = 0.20  #fitness dependent multiplier of mutation chance
            repeats = 20
            start = time.time()
            
            for run in range(run_nr):
                evo = evo_algorithm(n_hidden_neurons, enemy, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, run)
                evo.simulate()
                evo.save_results(extended = True)