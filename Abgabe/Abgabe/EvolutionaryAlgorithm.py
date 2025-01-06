from concurrent.futures import ThreadPoolExecutor
import random
from NN import trainAndTestMLP

############## Determine Fitness ##############


def fitness(genome: [], data, label: str, numOfIterations: int = 5, maxIter = 500) -> float:
    """Determines the accuracy of the genome for the given data

    Args:
        genome (_type_): The genome to use
        data (_type_): The data to use for the training and testing
        label (str): The label to optimize
        numOfIterations (int, optional): The number of iterations to create the average score. Defaults to 5.

    Returns:
        float: The average accuracy of the genome rounded to 6 decimal places
    """
    score = 0
    for i in range(numOfIterations):
        score += trainAndTestMLP(data, label, hiddenLayerSizes=genome, randomState=42 + i, maxIter=maxIter)
    return round(score / numOfIterations, 6)

############## Evolve ##############

def evolve(data, label: str, maxIterations = 50, popSize = 20, fitnessIter = 2, maxIter = 500, fitness=fitness):
    population = {}
    population["genomes"] = createRandomPopulation(popSize,
                                                   minLayers=2,
                                                   maxLayers=8,
                                                   minNeurons=4,
                                                   maxNeurons=128,
                                                   maxTotalNeurons=512)
    # print(f"Population: {population}")

    iter = 0
    while maxIterations > iter:
        print(f"Iteration {iter}")
        # population["fitness"] = [fitness(genome, data, label, fitnessIter) for genome in population["genomes"]]

        with ThreadPoolExecutor() as executor:
            population["fitness"] = list(executor.map(
            lambda genome: fitness(genome,
                                   data,
                                   label,
                                   fitnessIter,
                                   maxIter=maxIter),
            population["genomes"]
        ))

        sorted_population = [x for _, x in sorted(
                                                zip(population["fitness"],
                                                    population["genomes"]),
                                                reverse=True)]

        # if iter % 100 == 0:
            # print(f"Population: {population['genomes'][0:10]}")
        print(f"Population: {sorted_population}")
        print(f"Best accuracy: {max(population['fitness'])}")
        # print(f"Best accuracy for: {sorted_population[0]}")

        nextPopulation = sorted_population[:popSize // 2]
        # tournamentWinner = tournamentSelection(population["genomes"], population["fitness"], 5)

        children = createChildren(nextPopulation, popSize - len(nextPopulation))
        for index, child in enumerate(children):
            children[index] = mutate(child)
        population["genomes"] = nextPopulation + children
        iter += 1
    print(f"Population: {population}")
    

############## Create Population ##############

def createRandomPopulation(sizeOfPupulation: int, minLayers: int, maxLayers: int, minNeurons: int, maxNeurons: int, maxTotalNeurons: int = 0):
    population = [createRandomGenome(minLayers, maxLayers, minNeurons, maxNeurons, maxTotalNeurons) for _ in range(sizeOfPupulation)]
    return population

def createRandomGenome(minLayers: int, maxLayers: int, minNeurons: int, maxNeurons: int, maxTotalNumberOfNeurons: int = 0):
    """Creates a random Genome with a random amount of layers and a random number for neurons in each layer

    Args:
        minLayers (int): The minimal number of layers in the neuron. min is included
        maxLayers (int): The maximal number of layers in the neuron. max is included
        minNeurons (int): The minimal number of neuron in the layers. min is included
        maxNeurons (int): The maximal number of neuron in the layers. max is included
        maxTotalNeurons (int): The maximal number of neurons in all layers. If 0 is ignored

    Returns:
        list: List with a random number of elements. The elements have values in between the min and max for neurons
    """
    genome = [random.randint(minNeurons, maxNeurons) for _ in range(random.randint(minLayers, maxLayers))]
    if maxTotalNumberOfNeurons == 0:
        return genome
    else:
        while _countNeurons(genome) > maxTotalNumberOfNeurons:
            genome = [random.randint(minNeurons, maxNeurons) for _ in range(random.randint(minLayers, maxLayers))]
        return genome

def _countNeurons(genome):
    count = 0
    for layer in genome:
        count += layer
    return count


############## Survival Selection ##############

def rouletteWheelSelection(population, fitness_scores, num_selected):
    # Fitness-Wahrscheinlichkeiten berechnen
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    
    # Genomen basierend auf ihren Wahrscheinlichkeiten auswählen
    selected = []
    for _ in range(num_selected):
        r = random.random()
        cumulative = 0
        for i, probability in enumerate(probabilities):
            cumulative += probability
            if r <= cumulative:
                selected.append(population[i])
                break
    return selected

def tournamentSelection(population, fitness_scores, num_selected, tournament_size=3):
    selected = []
    for _ in range(num_selected):
        # Wähle zufällige Teilnehmer
        participants = random.sample(range(len(population)), tournament_size)
        # Das beste Genom basierend auf der Fitness auswählen
        best = max(participants, key=lambda idx: fitness_scores[idx])
        selected.append(population[best])
    return selected


############## Create Mutant Children ##############

def createChildren(parents, numChildren):
    children = []
    while len(children) <= numChildren:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        if parent1 != parent2:
            children.append(crossover(parent1, parent2))
    return children

def crossover(parent1, parent2):
    if (len(parent1) == 2 and len(parent2)) == 2 or len(parent1) < 2 or len(parent2) < 2: # Would create clones otherwise
        cut = 1
    else:
        cut = random.randint(1, min(len(parent1), len(parent2) - 1))
    child = parent1[:cut] + parent2[cut:]
    if child == parent1 or child == parent2:
        child[0] += 1 # Change the child in a small way to stop clones entering the population
        child[-1] += 1
    return child


def mutate(genome, maxNeurons=128, mutationChance=0.6, maxLayers=6):
    if random.random() < mutationChance:
        if random.random() < 0.25 and len(genome) > 2: # Remove Layer
            genome.pop(random.randint(0, len(genome) - 1))
        if random.random() < 0.25 and len(genome) <= maxLayers: # Add Layer
            genome.insert(random.randint(0, len(genome)), random.randint(4, maxNeurons))
        if random.random() < 0.25: # Mutate a layer
            r = random.random()
            i = random.randint(0, len(genome) - 1)
            if r < 0.33:
                genome[i] = random.randint(4, maxNeurons)
            elif r < 0.66:
                genome[i] = min(maxNeurons, genome[i] + random.randint(1, max(1, maxNeurons - genome[i])))
            elif genome[i] > 4:
                genome[i] -= random.randint(1, min(1, genome[i] - 4))
    return genome


