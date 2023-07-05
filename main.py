import copy
import random
import matplotlib.pyplot as plt

# константы задачи
ONE_MAX_LENGTH = 100  # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма
POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class FitnessMax:
    def __init__(self):
        self.values = [0]


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


def one_max_fitness(individual):
    return sum(individual),


def individual_creator():
    return Individual([random.randint(0, 1) for _ in range(ONE_MAX_LENGTH)])


def population_creator(n=0):
    return [individual_creator() for _ in range(n)]


def sel_tournament(population, p_len):
    best_list = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

        best = max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0])
        best_list.append(copy.deepcopy(best))

    return best_list


def cx_one_point(child1, child2):
    s = random.randint(2, len(child1) - 3)
    child1[s:], child2[s:] = child2[s:], child1[s:]


def mut_flip_bit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 1 - mutant[indx]


population = population_creator(n=POPULATION_SIZE)
for individual in population:
    individual.fitness.values = one_max_fitness(individual)

generationCounter = 0
maxFitnessValues = []
meanFitnessValues = []
fitnessValues = [individual.fitness.values[0] for individual in population]

while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    population = sel_tournament(population, len(population))

    for child1, child2 in zip(population[::2], population[1::2]):
        if random.random() < P_CROSSOVER:
            cx_one_point(child1, child2)

    for mutant in population:
        if random.random() < P_MUTATION:
            mut_flip_bit(mutant, indpb=1.0 / ONE_MAX_LENGTH)

    for individual in population:
        individual.fitness.values = one_max_fitness(individual)

    fitnessValues = [ind.fitness.values[0] for ind in population]

    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index], "\n")

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
