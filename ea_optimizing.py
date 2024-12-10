import random
from deap import base, creator, tools, algorithms
from classification import classify

def ea_optimizing_of_nn():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # neurons def
    toolbox.register("attr_int", random.randint, 4, 32)  

    # layer def
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=3)

    # population of nn combinations
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(layer):
        accuracy = classify(tuple(layer))
        return (accuracy,) 

    toolbox.register("evaluate", evaluate)

    # genetic operators
    toolbox.register("mate", tools.cxTwoPoint)  
    toolbox.register("mutate", tools.mutUniformInt, low=4, up=32, indpb=0.2)  
    toolbox.register("select", tools.selTournament, tournsize=3) 

    def main():
        random.seed(42)

        population = toolbox.population(n=20)

        ngen = 10  
        cxpb = 0.5  
        mutpb = 0.2  

        for gen in range(ngen):
            print(f"Generation {gen}")
            
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            population[:] = offspring

        best_ind = tools.selBest(population, 1)[0]
        print("Beste Architektur:", best_ind)
        print("Fitness:", best_ind.fitness.values[0])

if __name__ == "__main__":
    print("Start ea")
    ea_optimizing_of_nn()