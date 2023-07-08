import random
import matplotlib.pyplot as plt
from deap import base, algorithms, creator, tools
import numpy as np

# константы задачи
ONE_MAX_LENGTH = 100  # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма
POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def one_max_fitness(individual):
    return sum(individual),


creator.create('FitnessMax', base.Fitness, weights=(1, ))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('zero_or_one', random.randint, 0, 1)
toolbox.register('individual_creator', tools.initRepeat, creator.Individual, toolbox.zero_or_one, ONE_MAX_LENGTH)
toolbox.register('population_creator', tools.initRepeat, list, toolbox.individual_creator)

toolbox.register('evaluate', one_max_fitness)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1/ONE_MAX_LENGTH)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('max', np.max)
stats.register('mean', np.mean)

population = toolbox.population_creator(n=POPULATION_SIZE)

population, logbook = algorithms.eaSimple(population, toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select('max', 'mean')
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
