# Projet IA : Star Wars

###### Tom XIE

### 1. Quelle est la taille de l'espace de recherche ?

On veut déterminer les paramètres $p_i$ où $i \in [0,6]$ et $p_i \in [-100, 100]$, la taille de l'espace de recherche est donc $[-100,100]^6$

### 2. Quelle est votre fonction de fitness ?

Au départ j'étais parti pour une fitness calculant juste l'erreur la plus petite entre toutes les valeurs connues, mais ce n'était pas optimal, du coup il était mieux de faire la somme des erreurs

```py
def fitness(ind : list[float]) -> float:
    error = 0.
    #value de forme [T,X,Y] et SAMPLE les données de position_sample.csv
    for value in SAMPLE:  

       #On calcul nos positions approximatives
       x = ind[0] * np.sin(ind[1] * value[0] + ind[2])
       y = ind[3] * np.sin(ind[4] * value[0] + ind[5])

       #On calcul et on somme l'erreur
       error += (value[1] - x)**2 + (value[2] - y)**2

    return error
```

### 3. Décrivez les opérateurs mis en oeuvre (mutation, croisement)

**Croisement**

Le processus de croisement se repose sur l'aléatoire, on croise deux individus avec un indice aléatoire qui sera la taille de la séparation.



**Mutation**

Pour la mutation$^{(\bold{1})}$, on inclut la fonction de fitness dans notre fonction puis on procède à une correction si la fitness atteint 300.



La formule$^{\bold{(2)}}$ de cette correction est la suivante :

$$
a \times \left(1-e^{-\dfrac{x}{b}}\right)
$$

où $a,b$ des coefficients à determiner manuellement pour avoir la meilleur formule de correction.

$a$ et $b$ permettent de régler la raideur de notre courbe (*) $x$ notre fitness.



Dans mon cas $a = 30$, $b = 140$

Par exemple la courbe(*) de ma configuration de correction:

![](C:\Users\Tom\AppData\Roaming\marktext\images\2022-03-31-22-03-43-image.png)

Si on a une fitness en dessous de 1000 (on prend une grande valeur car notre erreur est au carré), la plus grande correction va être $\pm\  30$ (c'est à dire $a$), plus la fitness baisse et la correction se rapproche de 0



Le code est fournis ci-dessous :

```python
def crossover(a : list[float],b : list[float]) -> tuple[list[float], list[float]]:
    #Vérifie la taille des individus
    if len(a) != len(b):
        raise ValueError("Les individus doivent être de la même taille")
    #Si on a la même taille, on avec un indice r aléatoire entre 1 et len(a)
    r = randrange(1,len(a))
    return a[0:r] + b[r:], b[0:r] + a[r:]

def correction(fitness : float) -> float:
    return 30 * (1 - 1/np.exp(fitness/140))

def mutate(ind : list[float]) -> list[float]:
    r = randrange(0,len(ind))
    new_ind = ind
    fit = fitness(new_ind)

    #On commence à corriger à partir d'une fitness à 100
    if fit < 1000:
        correct = correction(fit) #On obtient notre valeur de correction à partir de la fitness

        u_value = uniform(-correct,correct)

        #On veut rester entre [-100, 100] pour un paramètre
        while new_ind[r] + u_value >= 100 and new_ind[r] + u_value <= -100:
            u_value = uniform(-correct,correct)

        new_ind[r] += u_value
    else:
        new_ind[r] = uniform(-100.,100.)
    
    return new_ind
```

### 4. Décrivez votre processus de sélection

En triant de manière croissante, on obtient dans les premiers individus, les "meilleurs" dans la population, on prend les 2 premiers et on crois ses 2 individus, puis on remplis le reste de notre propulation en la remplissant avec de nouveaux individus.

```py
def select(ind : list[float], hcount : int, lcount : int) -> list[float]:
    #Selectionne les premiers et quelques derniers pour avoir une variété
    return ind[0:hcount] + ind[-lcount:]

"""
Dans la boucle du main :
- pop est notre population de base
- new_pop est notre nouvelle population auquelle on va faire notre selection
- npop est la taille de notre population (pop) de base, par défaut 100
"""
pop = evaluate(pop)

#On prend 25% de notre population, 15 % des meilleurs, 10 % des pires pour avoir de la variété
h_percentage = int((npop) * 0.15)
l_percentage = int((npop) * 0.10)
new_pop = select(pop, h_percentage, l_percentage)
for j in range(0,npop//2,2):
    a,b = crossover(pop[j],pop[j+1])
    a = mutate(a)
    b = mutate(b)
    new_pop += [a,b]
```

### 5. Quelle est la taille de votre population, combien de générations sont nécessaires avant de converger vers une solution stable ?

Avec une population de taille 100, il faut environ 10 000 générations avant d'arriver à une solution avec une erreur d'environ 1,1.



Sur 10 executions, 2 executions n'atteignent pas une fitness en dessous de 2, cela provient de l'aléatoire.

### 6. Combien de temps votre programme prend en moyenne (sur plusieurs runs) ?

Pour environ 10 000 générations, le temps d'execution est d'environ 130 secondes soit environ 2 minutes.

### 7. Si vous avez testé différentes solutions qui ont moins bien fonctionnées, décrivez-les et discutez-les

#### a. L'utilisation de la POO n'est pas une bonne idée

Au début, en utilisant de la POO, ma fitness n'arrivait pas à descendre en dessous de 250, d'ailleurs je ne comprends toujours pas pourquoi ça ne voulait pas descendre du coup j'ai fait un algorithme sans POO.



#### b. Une mauvaise façon d'obtenir ma fitness

J'avais une fitness au début qui retournait juste la plus petite valeur d'écart mais cette évaluation d'individu n'est pas bonne, la formule :

$$
error = \min(error, |x - x_{approx}| + |y-y_{approx}|)
$$

J'ai donc opté pour la somme des écarts. En effet juste avoir la plus petite écart n'était pas significatif, on connaissait l'écart que sur une des données sample et pas entre toutes les données sample. Alors qu'avec la somme des écarts c'est beaucoup plus parlant vu qu'on a un résultat pour chaque données sample et qu'on les sommes. La formule devient donc :

$$
error = error + |x - x_{approx}| + |y - y_{approx}|
$$

J'ai finalement appliqué le carré sur mes écarts pour obtenir de plus grosse fitness et ainsi mieux comparer les individus. La valeur absolue n'est donc plus obligatoire

$$
error = error + (x - x_{approx})^2 + (y-y_{approx})^2
$$



#### c. Un changement de méthode de mutation

$\bold{^{(1)}}$Ensuite sans changer la façon dont je mutais mes individus (en prenant que les meilleurs, de l'élitisme), ça prenait beaucoup de temps pour converger vers une solution stable, on avait que des individus qui se ressemblait, c'est pourquoi j'échelonne la mutation car au bout d'un moment, la fitness a du mal à changer, cela veut dire qu'on se rapproche de la solution exacte.
Du coup à partir d'une fitness de 300, on commence notre processus de correction.



#### d. La sélection élitiste n'est pas la meilleure

Avant changement de la façon de sélection, je prenais que les 2 premiers car c'était les meilleurs de leur population mais si on ne prend que les meilleurs, toute la population va se ressembler et il n'y aura plus de possibilité ou moins de chance d'avoir du changement.

C'est pourquoi après reflexion, il était mieux de prendre comme dans le TD de python, une partie meilleure et l'autre la pire



### e. Généraliser la correction dans la mutation

Dans une ancienne version, je corrige la valeur d'un paramètre lorsque la fitness va en dessous de 100, or cela était trop arbitraire, il était donc préférable de trouver une formule$^{\bold{(2)}}$ pour corriger nos paramètres.

Le code suivant était la version avec correction par échelle :

```py
def mutate(ind : list[float]) -> list[float]:
    r = randrange(0,len(ind))
    new_ind = ind
    fit = fitness(new_ind)

    if fit > 100 :
        new_ind[r] = random()

    #Si on commence à avoir une bonne fitness, on fait des corrections
    elif fit <= 5:
        new_ind[r] += uniform (-0.2,0.2)
    elif fit <= 10:
        new_ind[r] += uniform(-1,1)
    elif fit <= 20:
        new_ind[r] += uniform(-5,5)
    elif fit <= 50:
        new_ind[r] += uniform(-10,10)
    elif fit <= 100:
        new_ind[r] += uniform(-20,20)

    return new_ind
```
