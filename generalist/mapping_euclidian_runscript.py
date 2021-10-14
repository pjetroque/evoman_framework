################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

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
from helper_mapping_euclidian import crossover, mutation, get_children, fitfunc
import csv
import time
import multiprocessing as mp

# choose this for not using visuals and thus making experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

#make sure to not print every startup of the pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"    

def play_game(play_map, g, avg_gains, enemies, total_generations, nvars, exp_name, fittert):
    '''
    A function that starts a game against a given set of enemies for a number of repeats in order to evaluate a specific solution vector.
    
    Inputs:
        player      = [array] the DNA (weights + biases neural network) and the sigmas
        g           = [int]   the current generation
        avg_fitness = [float] the avg_fitness of the previous generation
        enemies     = [list]  the enemies to train on
        
    Outputs:
        player      = [array] the DNA
        surviving_player = [boolean] if the player survived or not
        mean_gains  = [float] average gains
        gains       = [list]  the gains of the individual enemies
        fitness_smop= [float] average fitness values
        times       = [list]  the runtimes
        healths     = [list]  healths_player
        healths_e   = [list]  healths_enemies
        kills       = [list]  list with kill fraction per enemy
    '''
    
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=f'data_normal/{exp_name}',
                      playermode="ai",
                      player_controller=player_controller(10),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      randomini = "no")
    fitness_smop = 0
    repeats = 1
    gains = []
    health_player = []
    health_enemies = []
    times = []
    kills = []
    surviving_player = False
    
    player, mapping = play_map
    
    contr = player[:nvars]*mapping
    
    #battle each enemy in this list
    for enemy in enemies:
        gain_avg = 0
        time_avg = 0
        health = 0
        health_enemy = 0
        kill = 0
        
        #repeat each player to counter the randomness
        for i in range(repeats):
            #env.randomini = i
            f, p, e, t = env.run_single(enemy, pcont=contr, econt="None")
            
            fitness_smop += fitfunc(fittert, total_generations, g, t, e, p)*(1/repeats)
            health += (1/repeats)*p
            health_enemy += (1/repeats)*e
            time_avg += (1/repeats)*t
            gain_avg += (1/repeats)*p - (1/repeats)*e
            if e == 0:
                kill += (1/repeats)
            
        gains.append(gain_avg)
        health_player.append(health)
        health_enemies.append(health_enemy)
        times.append(time_avg)
        kills.append(kill)
        
    #if nog good enough 'die'
    if fitness_smop > avg_gains:
        surviving_player = True
    return [player, surviving_player, np.sum(gains), gains, fitness_smop, times, health_player, health_enemies, kills, mapping]
    
class evo_algorithm:
    '''
    The main Class containing the algorithm for a set of inputs.
    
    
    enemy               = [list] list of enemies (can be a single one) to train on
    run_nr              = [int] total number of runs, a run is a complete iteration of X generations
    generations         = [int] number of generations per run
    population_size     = [int] size of the population
    mutation_baseline   = [float] minimal chance for a mutation
    mutation_multiplier = [float] fitness dependent multiplier of mutation chance
    repeats             = [int] number of repeated fight of an agent within a generation
    fitter              = [string] name of the fitter to use
    experiment_name     = [string] the used name to save the data to
    n_hidden_neurons    = [int] number of hidden neurons in the hidden layer
    n_vars              = [int] total number of variables needed for the neural network
    total_data          = [list] place to save data to for later printing to csv\
    best                = [array] the best solution
    n_sigmas            = [int] number of sigmas to use (only use 1 or 4)
    '''
    def __init__(self, n_hidden_neurons, enemy, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, run, cores='max', current_generation = 0):
        self.enemy = enemy          #which enemy
        self.run_nr = run_nr                  #number of runs
        self.generations = generations           #number of generations per run
        self.population_size = population_size       #pop size
        self.mutation_baseline = mutation_baseline    #minimal chance for a mutation event
        self.mutation_multiplier = mutation_multiplier  #fitness dependent multiplier of mutation chance
        self.repeats = repeats
        self.fitter = fitter
        self.experiment_name = f'enemy_{enemy}_{fitter}'
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = (20+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
        self.total_data = []
        self.total_sigma_data = []
        self.best = []
        self.run = run
        self.max_gain = -100*len(enemy)
        self.n_sigmas = 4
        self.cores = cores
        self.survival_number = 4
        self.current_generation = current_generation
        
        #make save folder
        if not os.path.exists(f'data_normal/{self.experiment_name}'):
            os.makedirs(f'data_normal/{self.experiment_name}')  
    

    
    def simulate(self, pop = []):
        '''
        Core function of the evo_algorithm dividing the games of a generation over the cores and storing/printing 
        the most important information of a generation is saved.
        '''
        #initiate 100 parents, the size of an agent is n_vars + sigmas
        if not len(pop) == self.population_size:
            DNA = np.random.uniform(-1, 1, (self.population_size ,self.n_vars))
            #set bias of shoot to 1
            for k in range(len(DNA)):
                DNA[k,213] = -5
                DNA[k,212] = 0
                DNA[k,211] = 0
                DNA[k,210] = 0
                DNA[k,214] = 0
            sigmas = np.random.uniform(0.2, 0.5, (self.population_size ,self.n_sigmas))
            sigmas[0] = 0.1
            sigmas[3] = 0.1
            pop = np.hstack((DNA, sigmas))
            mapping = np.ones((self.population_size ,self.n_vars))
        
        avg_gains = self.max_gain
        max_kills = 0
        
        for g in range(self.generations):
            gen_start = time.time()
            gains_array = []
            fitness_array = []
            surviving_players = []
            times_array = []
            health_array = []
            health_e_array = []
            kills_array = [] 
            sigma_array = []
            enemy = self.enemy

            
            #multiple cores implementation
            if self.cores == 'max':
                pool = mp.Pool(mp.cpu_count())
            else:
                pool = mp.Pool(cores)

            results = [pool.apply_async(play_game, args=(play_map, self.current_generation, avg_gains, enemy, self.generations, self.n_vars, self.experiment_name, self.fitter)) for play_map in zip(pop, mapping)]

            pop = []
            mapping = []
            
            #retrieve all the data
            for ind, result in enumerate(results):
                r = result.get()
                pop.append(r[0])
                survive = r[1]
                avg_gain = r[2]
                gains = r[3]
                fitness = r[4]
                times = r[5]
                health = r[6]
                health_e = r[7]
                kills = r[8]
                mapping.append(r[9])
                
                #save all the data to arrays
                gains_array.append(avg_gain)
                fitness_array.append(fitness)
                times_array.append(times)
                health_array.append(health)
                health_e_array.append(health_e)
                kills_array.append(kills)
                sigma_array.append(r[0][265:])
                
                #sigma data + some others
                #self.total_sigma_data.append([self.current_generation]+list(np.concatenate([gains, kills, r[0][265:]]).flat))
                
                if fitness > self.max_gain:
                    self.max_gain = fitness
                    self.best = r[0]*np.hstack([r[9], [1, 1, 1, 1]])
                    print(f'connections: {np.sum(r[9])}, kills = {kills}')
                    #self.best[213] = 1
                    with open(f'data_normal/{self.experiment_name}/best_sol_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.best)
                if np.sum(kills) > max_kills:
                    max_kills = np.sum(kills)
                    with open(f'data_normal/{self.experiment_name}/kills_sol_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(r[0]*np.hstack([r[9], [1, 1, 1, 1]]))
                    
                if survive:
                    surviving_players.append(ind)
            pool.close()
            avg_gains = 0.25*np.mean(fitness_array) + 0.75*np.max(fitness_array)

            self.total_data.append([np.max(gains_array), np.mean(gains_array), np.std(gains_array), np.max(fitness_array), np.mean(fitness_array), np.std(fitness_array)])
            
            ##survival of X best players
            best_players = np.sort(fitness_array, axis=None)[len(fitness_array) - self.survival_number]
            indexes = np.where(fitness_array >= best_players)[0]
            
            for index in indexes:
                if not index in surviving_players:
                    surviving_players.append(index)
            
#            if len(surviving_players) > 10:
#                surviving_players = surviving_players[-10:]
            
            self.current_generation += 1
            #backup population each X gen
            if self.current_generation%10==9:
                self.backup_pop(pop, self.current_generation)
            
            
            mix_pop=False
            if g%20==19:
                mix_pop = True
                print('MINGLETIME')
            pop, mapping = get_children(pop, mapping, surviving_players, np.array(fitness_array),
                               self.mutation_baseline, self.mutation_multiplier, mix_pop)
            
            
            mean_sigmas = np.around(np.mean(np.array(pop)[:,265:], axis=0), decimals=2)
            max_sigmas = np.around(np.max(np.array(pop)[:,265:], axis=0), decimals=2)
            min_sigmas = np.around(np.min(np.array(pop)[:,265:], axis=0), decimals=2)
            killmax = np.where(np.sum(kills_array, axis=1) == np.max(np.sum(kills_array, axis=1)))[0][0]
            
            
            print(f'Run: {self.run}, G: {self.current_generation}, F_mean = {round(np.mean(fitness_array),1)} pm {round(np.std(fitness_array),1)}, F_best = {round(np.max(fitness_array),1)}, G_mean = {np.round(np.mean(gains_array),1)}, G_best = {np.round(np.max(gains_array))}, S_mean={mean_sigmas} max:{max_sigmas} min:{min_sigmas}, kills={kills_array[killmax]}, surviving={len(surviving_players)}, thr={np.round(avg_gains, 1)} time={round(time.time()-gen_start)}')
        return
    
    def save_results(self, full = False, append = False):
        writing_style = 'w'
        if append:
            writing_style = 'a'
        with open(f'data_normal/{self.experiment_name}/data_{self.run}.csv', writing_style, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.enemy, self.generations])
            writer.writerows(self.total_data)
        
        with open(f'data_normal/{self.experiment_name}/best_sol_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.best)
            
        if full:
            title = ['generation']
            for enemy in self.enemy:
                title.append(f'gain_enemy_{enemy}')
            for enemy in self.enemy:
                title.append(f'kill_enemy_{enemy}')
            for sig in range(self.n_sigmas):
                title.append(f'sigma_{sig}')
            with open(f'data_normal/{self.experiment_name}/full_data_{self.run}.csv', writing_style, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not append:
                    writer.writerow(title)
                writer.writerows(self.total_sigma_data)
        return
    
    def backup_pop(self, population, generation):
        with open(f'data_normal/{self.experiment_name}/pop_backup_{generation}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(population)

if __name__ == '__main__':
    n_hidden_neurons = 10       #number of hidden neurons
    enemies = [1, 3, 4, 7]          #which enemies
    run_nr = 1                  #number of runs
    generations = 150           #number of generations per run
    population_size = 200        #pop size
    mutation_baseline = 0       #minimal chance for a mutation event
    mutation_multiplier = 0.8  #fitness dependent multiplier of mutation chance
    repeats = 1
    fitter = 'errfoscilation'
    start = time.time()
    cores = 'max'
    new = True
    
    for run in range(run_nr):
        evo = evo_algorithm(n_hidden_neurons, enemies, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, run, cores)
        
        if new:
            #start a new run
            evo.simulate()
            evo.save_results(full=False)
        
        else:
            #continue an old run
            population = []
            load_from_generation = 0
            backup_name = f'data_normal/enemy_[1, 4, 6]_{fitter}/pop_backup_{load_from_generation}.csv'
            with open(backup_name, newline='', encoding='utf-8') as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                for row in reader:
                    population.append(row)
            evo.current_generation = load_from_generation
            evo.simulate(np.array(population))
            
            evo.save_results(full=True, append=False)