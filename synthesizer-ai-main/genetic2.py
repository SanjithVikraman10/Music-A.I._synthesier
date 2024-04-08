from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple

Genome = List[int]
Population = List[Genome]               
populate = Callable[[], Population]     # when called, no arguments passed, generates an initial Population of Genomes.
fitness = Callable[[Genome], int]       # calculates the fitness of a single Genome.
selection = Callable[[Population, fitness], Tuple[Genome, Genome]]
                                        # selects a pair of Genomes (parents) from the Population for crossover
crossover = Callable[[Genome, Genome], Tuple[Genome, Genome]]
                                        # performs the crossover operation on two Genomes (parents) to produce new Genomes (offspring).
mutationfunc = Callable[[Genome], Genome]   # mutates a single Genome

printer = Callable[[Population, int, fitness], None]


# to generate a Genome
def genome_generation(length) -> Genome:
    return choices([0,1], k=length)


# to generate Population
def population_generation(size, genome_length) -> Population:
    return [genome_generation(genome_length) for k in range(size)]


# to crossover at one point
def single_point_crossover(one: Genome, two: Genome) -> Tuple:
    if len(one) != len(two):
        raise ValueError("Genomes one and two must be of same length")

    length = len(one)
    if length <2:
        return one, two
    
    pt = randint(1, length-1)
    return one[:pt] + two[pt:], two[:pt] + one[pt:]


# mutation
def mutationfunc(gen: Genome, num: int=1, prob: float=0.5) -> Genome:
    for i in range(num):
        ind = randrange(len(gen))
        gen[ind] = gen[ind] if random() > prob else abs(gen[ind]-1)
    return Genome


# calc fitness of Population
def population_fitness(population: Population, fitness: fitness) -> int:
    return sum([fitness(gen) for gen in population])

# select parents
def pair_selection(pop: Population, fitness: fitness) -> Population:
    return choices(population=pop, weights=[fitness(gene) for gene in pop],k=2)


# sorting Population
def sort_population(pop: Population, fitness: fitness) -> Population:
    return sorted(pop, key=fitness, reverse=True)


# convert Genome to string
def genome_to_string(gen: Genome) -> str:
    return "".join(map(str, gen))



def print_stat(pop: Population, generation_id: int, fitness: fitness):
    print(f"GENERATION {generation_id}")
    print("-*"*10)
    print("Population: [%s]" % ", ".join([Genome_to_string(gene) for gene in pop]))
    print(f"Avg. Fitness: {population_fitness(pop, fitness / len(pop))}")
    sorted_Population = sort_population(pop, fitness)
    print(f"Best: {Genome_to_string(sorted_population[0])} ({fitness_func(sorted_Population[0])})")
    print(f"Worst: {Genome_to_string(sorted_population[-1])} ({fitness_func(sorted_Population[-1])})")
    print("")

    return sorted_population[0]


def run_evolution(
        populate_func: populate,
        fitness_func: fitness,
        fitness_limit: int,
        selection_func: selection = pair_selection,
        crossover_func: crossover = single_point_crossover,
        mutation_func: mutationfunc = mutationfunc,
        generation_limit: int = 100,
        printer: Optional[printer] = None) \
        -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population, key=lambda Genome: fitness_func(Genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i