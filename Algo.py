from random import random,uniform,randrange
from csv import reader

import time
import numpy as np
import matplotlib.pyplot as plt


global SAMPLE
SAMPLE = []

FILE = 'position_sample.csv'
#FILE = 'iman_sample.csv'

with open(FILE,'r',newline = '') as csvfile:
    read = reader(csvfile, delimiter = ';')

    next(read)

    for data in read:
        SAMPLE.append([float(value) for value in data])

def individu() -> list[float]:
    return [uniform(-100.,100.) for _ in range(6)]

def fitness(ind : list[float]) -> float:
    error = 0.
    for value in SAMPLE: #value de forme [T,X,Y]
        x = ind[0] * np.sin(ind[1] * value[0] + ind[2])
        y = ind[3] * np.sin(ind[4] * value[0] + ind[5])

        error += (value[1] - x)**2 + (value[2] - y)**2

    return error

def select(ind : list[float], hcount : int, lcount : int) -> list[float]:
    #Selectionne les premiers et quelques derniers pour avoir une variété
    return ind[0:hcount] + ind[-lcount:]

def evaluate(pop : list[float]) -> list[list[float]]:
    return sorted(pop, key = lambda x : fitness(x))

def crossover(a : list[float],b : list[float]) -> tuple[list[float], list[float]]:
    if len(a) != len(b):
        raise ValueError("Les individus doivent être de la même taille")
    r = randrange(1,len(a))
    return a[0:r] + b[r:], b[0:r] + a[r:]

def correction(fitness : float) -> float:
    return 30 * (1 - 1/np.exp(fitness/140))


def mutate(ind : list[float]) -> list[float]:
    r = randrange(0,len(ind))
    new_ind = ind
    fit = fitness(new_ind)

    
    #On commence à corriger à partir d'une fitness à 1000
    if fit < 1000:
        correct = correction(fit) #On obtient notre valeur de correction à partir de la fitness

        u_value = uniform(-correct,correct)

        #on est pas obligé mais on reste dans l'intervalle [-100, 100] pour un paramètre
        while new_ind[r] + u_value >= 100 and new_ind[r] + u_value <= -100:
            u_value = uniform(-correct,correct)

        new_ind[r] += u_value
    else:
        new_ind[r] = uniform(-100.,100.)
    
    return new_ind

def create_pop(n : int) -> list[list[float]]:
    return [individu() for _ in range(n)]

def algoG(npop : int = 100, generation : int = 50000, fitlim : int = 1.1) -> tuple[list[float], int]:
    pop = create_pop(npop)

    #On itère {generation} fois
    for gen in range(generation):

        pop = evaluate(pop) #Trie notre population

        print('{0:7} | {1:120} | {2:.20f}'.format(
            gen,
            ';'.join([str(v) for v in pop[0]]),
            fitness(pop[0])
            )
        )

        if fitness(pop[0]) <= fitlim:
            break

        #Processus de sélection    
        h_percentage = int((npop) * 0.15)
        l_percentage = int((npop) * 0.10)
        new_pop = select(pop, h_percentage, l_percentage)
        #Procéssus de croisement, mutation en même temps
        for j in range(0,npop//2,2):
            a,b = crossover(pop[j],pop[j+1])
            a = mutate(a)
            b = mutate(b)
            new_pop += [a,b]

        new_pop += create_pop(npop - len(new_pop))
        pop = new_pop
    print(pop[0])
    return pop[0],gen

def plot(result : list[float] = None, n : int = 1) -> None:

    if result != None:
        
        p = result
        #Trace notre courbe de x(t) et y(t) avec les paramètres trouvées
        t = np.linspace(0,2*np.pi,100000)
        x = p[0] * np.sin(p[1] * t + p[2])
        y = p[3] * np.sin(p[4] * t + p[5])
        plt.plot(x,y)

        #Place nos points avec les paramètres trouvées
        for sample in SAMPLE:
            plt.plot(p[0] * np.sin(p[1] * sample[0] + p[2]),
                    p[3] * np.sin(p[4] * sample[0] + p[5]),
                    marker="o",
                    color="red"
                    )
    
    #Place les points des données sample
    for sample in SAMPLE:
        plt.plot(sample[1],
                sample[2],
                marker="o",
                color="blue"
                )
    plt.title(f'run n{n}')
    plt.show()

def write_result(n :int, time:float):
    """
    Fonction permettant d'écrire le temps d'execution et le nombre d'itération qui a été faite dans un fichier
    """

    with open('result.txt','a') as f:
        f.write(f'\n{n};{time}')

def overrideIfBetter(result):
    """
    Non fonctionnelle, elle écrase tout le temps la valeur du fichier texte
    """
    with open('XIE_Tom_groupeI.txt','w') as f:
        try :
            before = f.readlines()
        except:
            before = None
            print("No lines in file")

        if before == None:
            f.writelines(';'.join([str(val) for val in result]))
        else :
            if fitness(result) < fitness(before):
                f.writelines(';'.join([str(val) for val in result]))

def plotMean(run : list[float], solutions : list[float] = [], generations : list[int] = []) -> None:
    """
    
    """
    r = np.array(run)

    if solutions != []:

        s = np.array(solutions)
        for i in range(len(r)-1):
            plt.plot(r[i],s[i],marker = "o",color="blue")

        plt.title("run en fonction de solution")
        plt.show()

    if generations != []:
        g = np.array(generations)
        for i in range(len(r)-1):
            plt.plot(r[i],g[i],marker = "o",color="red")
        plt.title("run en fonction du nombre de generation")
        plt.show()



if __name__ == '__main__':

    RUNS = []
    SOLUTIONS = []
    GENERATIONS = []

    for run in range(1,10):
        
        start = time.time()

        solution,n = algoG(npop = 100, generation = 10000, fitlim = 0)

        runtime = time.time() - start

        RUNS.append(run)
        SOLUTIONS.append(fitness(solution))
        #GENERATIONS.append(n)
        print(f'run nº{run} -------- %s seconds -------- {n} iterations' % round(runtime,2))

        write_result(n,runtime)
        #overrideIfBetter(solution)

        #plot(solution, run)
    
    print(sum(SOLUTIONS)/len(RUNS))

    plotMean(RUNS,SOLUTIONS,GENERATIONS)
    None
