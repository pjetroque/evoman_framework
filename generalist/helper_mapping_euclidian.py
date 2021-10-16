# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:02:42 2021

@author: pjotr
Events file
"""

import numpy as np
import copy
import random
from scipy.spatial import distance_matrix
import time
from sklearn.cluster import KMeans

def fitfunc(fitfunction, generations, g, t, e, p):

    if fitfunction == "standard":
        fitness_smop = 0.9*(100 - e) + 0.1*p
        if e < 0.1:
            fitness_smop = 200

    if fitfunction == "oscilation":
        period = .5*generations
        fitness_smop = (1 + np.cos((2*np.pi/period) * g)) * (0.1*t) + (1 + np.cos((2*np.pi/period) * g + np.pi)) * (100-e+p - np.log(t))

    if fitfunction == "exponential":
        fitness_smop =  100/(100-(0.9*(100 - e) + 0.1*p - np.log(t)))

    if fitfunction == "errfoscilation":
        if g < generations+10:
            fitness_smop =  (0.005*t)**2
            if t > 800-g:
                fitness_smop += .5*( 100 - e + p)
            if e < 0.1:
                fitness_smop += 100
        else:
            fitness_smop = (150 - e + p)
            if e < 0.1:
                fitness_smop = 250

    return fitness_smop


#uniform crossover (no position bias)
def crossover(p1, p2):
    length = len(p1)
    crossing_index = np.random.randint(2, size=length)
    c1 = p1*crossing_index + p2*(1-crossing_index)
    c2 = p1*(1-crossing_index) + p2*crossing_index

    return c1, c2


#weighted crossover based on fitness
def weighted_crossover(p1, p2, f1, f2):
    if f1+f2 == 0:
        w = 0.5
    else:
        w = f1/(f1+f2)
    c1 = w*p1 + (1-w)*p2

    return c1

#mutate a chromosome based on the mutation rate: the chance that a gene mutates
#and sigma: the average size of the mutation (taken from normal distribution)
def mutation(DNA, mutation_rate, sigma, m_b, m_m):

    #sigma function to map sigmas to the weights
    def sigma_func(length, sigma):

        #simple sigma function
        if len(sigma) == 1:
           sizes = np.random.normal(0, 1, length) * sigma[0]

        #multi sigma function
        #bias1 = [0:10]
        #weights1 = [10:210]
        #bias2 = [210:215]
        #weights2 = [215:]
        elif len(sigma) == 4:
            sizes           = np.zeros(length)
            sizes[:10]      = np.random.normal(0, 1, 10)* sigma[0]
            sizes[10:210]   = np.random.normal(0, 1, 200)* sigma[1]
            sizes[210:215]  = np.random.normal(0, 1, 5)* sigma[2]
            sizes[215:]     = np.random.normal(0, 1, 50)* sigma[3]

        return sizes

    length = len(DNA)

    #first mutate sigma(s)
    tau_ = 1/np.sqrt(2*length)
    tau = 1/np.sqrt(2*np.sqrt(length))
    if len(sigma) == 1:
        sigma = sigma * np.exp(np.sqrt(2)*tau_*np.random.normal(0, 1))
    elif len(sigma) == 4:
        sigma = sigma * np.exp(tau_*np.random.normal(0, 1) + tau*np.random.normal(0, 1, 4))

    #standard point mutations using new sigma(s)
    mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
    mutation_size = sigma_func(length, sigma)
    c1 = DNA + mutation_index*mutation_size

    #deletions (rare)
    if np.random.uniform(0, 1) < m_m*mutation_rate:
        mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
        c1 = c1 * (mutation_index==False) + mutation_index * np.random.uniform(-1, 1, length)

    #insertions (rare)
    if np.random.uniform(0, 1) < m_m * mutation_rate:
        mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
        c1 = c1 * (mutation_index==False) + random.randint(2, 5) * c1 * mutation_index

    return np.hstack((c1, sigma))

def mapmutation(child_map, mutation_rate):
    if np.random.uniform(0, 1) < mutation_rate:
        for c in range(len(child_map)):
            if np.random.uniform(0, 1) < 0.005 and not (c < 10 or 210 < c < 215):
                if child_map[c] == 0:
                    child_map[c] = 1
                else:
                    child_map[c] = 0
    
    return child_map

def get_children(parents, mapping, surviving_players, fitness, mutation_base, mutation_multiplier, mix_populations):
   
    children = np.array(copy.deepcopy(parents))
    mapping = np.array(copy.deepcopy(mapping))
    children_mapping = []
    next_gen = []
    
    cluster_amount = 6
    print(f'diff_pop={np.max(distance_matrix(children[:,:265]*mapping, children[:,:265]*mapping))}')
    kmeans = KMeans(n_clusters=cluster_amount, random_state=0).fit(children[:,:265]*mapping).labels_
    
    #change all fitness <0 to 0
    fitness = np.array(fitness)
    fitness = fitness*(fitness > 0) + 10
    
    if mix_populations:
        #migrate individuals between populations subset --> pops
        pops = np.arange(cluster_amount, dtype=int)
        random.shuffle(pops)
        
    for subset in range(cluster_amount):
        surviving_sub = []
        fit_sub = fitness[kmeans==subset]
        children_sub = children[kmeans==subset]
        #print(mapping)
        mapping_sub = mapping[kmeans==subset]
        ind = 0
        n_pop = len(fit_sub)
        
        for boe in range(len(children)):
            if kmeans[boe]==subset:
                if boe in surviving_players:
                    surviving_sub.append(ind)
                ind += 1
        
        if len(surviving_sub) > 2:
            surviving_sub = random.choices(surviving_sub, k=2)
        
        if mix_populations:
            #random choose 2 members from the other pop to add
            pick = 4
            
            mix = pops[subset]
            extra_pop = random.choices(np.where(kmeans == mix)[0], k=pick)
            fit_sub = np.hstack([fit_sub, fitness[extra_pop]])
            children_sub = np.vstack([children_sub, children[extra_pop]])
            mapping_sub = np.vstack([mapping_sub, mapping[extra_pop]])
        
        if not len(surviving_sub) == n_pop:
            #pick parents based on fitness (fitness = weigth)
            parents_index = np.arange(0, len(children_sub), dtype=int)
            p1 = random.choices(parents_index, weights=fit_sub, k=n_pop-len(surviving_sub))
            p2 = random.choices(parents_index, weights=fit_sub, k=n_pop-len(surviving_sub))

            if len(surviving_sub) > 0:
                p1 = np.hstack((surviving_sub, p1))
                p2 = np.hstack((surviving_sub, p2))

        else:
            p1 = surviving_sub
            p2 = surviving_sub

        #iterate to make children
        dist = distance_matrix(children_sub[:,:265]*mapping_sub, children_sub[:,:265]*mapping_sub)
        diff = 1
        if n_pop > 1:
            diff = np.sum(dist)/((n_pop**2-n_pop)*530)
        
        max_diff_fit = fit_sub.max() - fit_sub.min()
        print(diff, max_diff_fit)
        for i in range(n_pop):
            #crossover the genes of parents and random choose a child
            #child = random.choice(crossover(parents[p1[i]], parents[p2[i]]))
            child = weighted_crossover(children_sub[p1[i]], children_sub[p2[i]], fit_sub[p1[i]], fit_sub[p2[i]])
            
            child_map = []
            
            for k in range(len(mapping_sub[0])):
                if np.random.uniform(0, 1) < fit_sub[p1[i]]/(fit_sub[p1[i]]+fit_sub[p2[i]]):
                    child_map.append(mapping_sub[p1[i]][k])
                else:
                    child_map.append(mapping_sub[p2[i]][k])

            #child_map = np.array(child_map)
            DNA   = child[:265]
            sigma = child[265:]
            
            #mutate based on parents fitness
            mutation_rate = 1-0.5*(fit_sub[p1[i]] + fit_sub[p2[i]])/(np.max(fit_sub)+1)
            
            #if converged change heavy
            converged = False
            if diff < 0.01 and n_pop > 1 and max_diff_fit < 10:
                converged = random.choices([True, False], weights = [90, 10], k=1)
            
            if converged:
                child = mutation(DNA, 0.75, [0.5, 0.5, 0.5, 0.5], 0.5, 0.5)
            else:
                child = mutation(DNA, mutation_rate, sigma, mutation_base, mutation_multiplier)
            child_map = mapmutation(child_map, mutation_rate)
            
            #normalize between min-max
            minimum = -1
            maximum = 1
            min_sigma = -0.5
            max_sigma = 0.5
            thresh = 0.001
            
            
            for j in range(len(child)):
                if j < 265:
                    #child[j] = (maximum-minimum)*(child[j]-child.min())/(child.max()-child.min())+minimum
                    if child[j]< minimum:
                        child[j] = minimum
                    elif child[j] > maximum:
                        child[j] = maximum
                else:
                    if child[j] < 0.1:
                        child[j] = 0.1
                    elif child[j] > 0.5:
                        child[j] = 0.5
#                    if child[j]< min_sigma:
#                        child[j] = min_sigma
#                    elif child[j] > max_sigma:
#                        child[j] = max_sigma
                    
            #child = (maximum-minimum)*(child-child.min())/(child.max()-child.min())+minimum
            next_gen.append(child)
            children_mapping.append(child_map)

    return next_gen, children_mapping