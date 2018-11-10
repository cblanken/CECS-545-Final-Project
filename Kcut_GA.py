"""Genetic Algorithm functionality for solving K-cut problem
Includes:  population creation and management
            fitness function(s)
            crossover operator(s)
            etc.

In the case of the TSP, 
# The *chromosome* consists of a complete tour (path) of a given 
  set of nodes (cities) given as a list of integers corresponding
  the the respending node ID.
# The *genes* are represented as each individual city in the path
# The *fitness* is  based on the total distance of the full path.
# The *survivors* are selected by ...
#
"""
import os
import sys
import copy
import math
import time
import random
import Kcut
import GraphParse as gp

LOG_BIT = True
LOG_GEN_BIT = False
TEST_LOG_BIT = False

## Crossover functions
def crossover_davis_order(path1, path2, k = None):
    """Davis's Order Crossover (AKA Order 1 Crossover or OX1)
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
    Returns path with randomized Davis's Order crossover applied.
    """    
    ## test logging
    if TEST_LOG_BIT:    
        print(f"PATH1: {path1}")
        print(f"PATH2: {path2}")

    ## cut off start and end nodes (0) for paths so crossover can be performed
    ## without disrupting start and end of full path
    pathStart = path1[0]
    pathEnd = path1[-1]
    path1 = path1[1:-1]
    path2 = path2[1:-1]

    offspring1 = [None for i in path1]
    offspring2 = [None for i in path2]

    ## start 0 to (1 less than last index)
    start = random.randint(0, len(path1)-2)
    ## end start to (last index)
    end = random.randint(start+1, len(path1)-1)
   
    ## test logging
    if TEST_LOG_BIT: 
        print(f"START: {start}")
        print(f"END: {end}")
        print(f"SEGMENT1: {path1[start:end]}")
        print(f"SEGMENT2: {path2[start:end]}")
    
    ### create offspring 1
    ## segment tranferred from parent to new child
    offspring1[start:end+1] = path1[start:end+1]
    ## list of nodes of other parent beginning from *end* index
    remaining1 = path2[end+1:] + path2[:end+1]
    ## current iterator over list of remaining nodes to be transferred to offspring
    remaining1_iter = 0
    ## current iterator over offspring
    wraparound1_iter = end + 1
    while remaining1_iter < len(remaining1) and wraparound1_iter != start:
        if remaining1[remaining1_iter] not in offspring1[start:end+1]:
            if wraparound1_iter >= len(offspring1):
                wraparound1_iter = 0
            offspring1[wraparound1_iter] = remaining1[remaining1_iter]
            wraparound1_iter += 1
        remaining1_iter += 1

    ### create offspring 2
    ## segment tranferred from parent to new child
    offspring2[start:end+1] = path2[start:end+1]
    ## list of nodes of other parent beginning from *end* index
    remaining2 = path1[end+1:] + path1[:end+1]
    ## current iterator over list of remaining nodes to be transferred to offspring
    remaining2_iter = 0
    ## current iterator over offspring
    wraparound2_iter = end + 1
    while remaining2_iter < len(remaining2) and wraparound2_iter != start:
        if remaining2[remaining2_iter] not in offspring2[start:end+1]:
            if wraparound2_iter >= len(offspring2):
                wraparound2_iter = 0
            offspring2[wraparound2_iter] = remaining2[remaining2_iter]
            wraparound2_iter += 1
        remaining2_iter += 1    


    ## add start and end of path back to offspring paths
    offspring1 = [pathStart] + offspring1 + [pathEnd] 
    offspring2 = [pathStart] + offspring2 + [pathEnd] 
    ## test logging
    if TEST_LOG_BIT:
        print(f"OFFSPRING1: {offspring1}")
        print(f"OFFSPRING2: {offspring2}")
    return (offspring1, offspring2)


## Mutation functions
def mutation_inversion(path, range = None):
    """Mutates path by selecting range of consecutive cities/nodes in 
    *path* and inverting the order in which the cities are visited
    :param path: path of new chromosome to be mutated
    :returns: path with scramble mutation applied.
    """
    ## restrict path to internal nodes (not first and last which are pre-determined)
    pathStart = path[0]
    pathEnd = path[-1]
    path = path[1:-1]
    ## default range
    if range == None:
        range = random.randint( 1, math.floor(len(path)/2) + 1)
    elif range >= len(path):
        print("ERR: mutation_scramble range set too large. Defaulted to 3")
        range = 3
    # ## init start and end of range
    # start = random.randint(0, len(path) - range - 1)
    # end = start + range
    # ## invert range in place
    # invert_range = path[start:end]
    # invert_range.reverse()
    # path[start:end] = invert_range
    # path = [pathStart] + path + [pathEnd]
    # return path

    ## init start and end of range
    start = random.randint(0, len(path) - 1)
    end = (start + range) % len(path)
    ## invert range across end of path
    if start > end:
        ## offset the path so start comes before end
        offset_path = [node for node in path[start:] + path[:end] + path[end:start]]
        ## get inverse of nodes from start -> end
        invert_range = offset_path[(len(path) - 1 - start + end)::-1]
        ## set offset path with new inverted range (start -> end)
        offset_path = invert_range + offset_path[(len(path) - start + end):]
        ## undo offset of offset_path
        path = offset_path[len(path) - start : len(path) - start + end] + \
            offset_path[(len(path) - start + end):] + \
            offset_path[0:len(path)-start]
    ## invert range in place    
    else:
        invert_range = path[start:end]
        invert_range.reverse()
        path[start:end] = invert_range

    ## prepend and append start node for full path    
    path = [pathStart] + path + [pathEnd]
    return path


## Parent Selection functions
def select_parent_tournament(generation, k = None):
    """Tournament Selection
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
    :param generation: Generation object with population to choose from
    :param k: # of individuals randomly chosen from population
    :returns: Chosen parent (*chromosome*)
    """
    ## default k 
    if k == None:
        k = random.randint(1, int(len(generation.population)/2))
    k_selection = [random.randint(0, len(generation.population)-1) for i in range(k)]
    ## get index and appropriate fitness of chromosome at population[index]
    parent_selection = [(index, generation.population[index].fitness) for index in k_selection]

    ## returns fitness portions of parent_selection for minimum fitness (distance)
    def fitness_func(x):
        return x[1]
    ## get index of chromosome with minimum fitness (distance) fitness from *parent_selection*
    best_parent_index = min(parent_selection, key=fitness_func)[0]
    return generation.population[best_parent_index]


def select_parent_roulette(generation, k = None):
    """Roulette Wheel Selection
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
    :param generation:  Generation object with population to choose from
    :param k: does nothing, is present to maintain consistent function call structure
    :returns: Chosen parent (*chromosome*)
    """
    fitnessList = [x.fitness for x in generation.population]
    ## s = sum of fitnesses in population
    s = sum(fitnessList)
    ## r = random num between 0 and s
    r = random.randint(0, int(s))

    if TEST_LOG_BIT:
        print(f"r (rand): {r}")

    ## p = partial sum 
    p = r
    for chromosome in generation.population:
        p += chromosome.fitness
        if p >= s:
            ## test logging
            if TEST_LOG_BIT:
                print("===============================================================")
                print(f"CHROMOSOME: {chromosome.path}")
            return chromosome

    if TEST_LOG_BIT:
        print(f"p (partial): {p}")
        print(f"s (sum): {s}")

class Chromosome:
    """
    """
    def __init__(self, kcut):
        self.kcut = kcut
        self.fitness = Kcut.getKcutFitness(kcut)

    def __str__(self):
        return "Fitness: {}, K-cut: {}".format(self.fitness, self.kcut)

    __repr__ = __str__

class Generation:
    """
    """
    def __init__(self, graph_string):
        self.iteration = 0
        self.graph = Kcut.parseInputGraphString(gp.parseString(graph_string))
        self.population = []
        self.size = len(self.population)
        self.logFile = None

    def calcAverageFitness(self):
        sum = 0
        for path in self.population:
            sum += path.fitness
        return sum / len(self.population)
    
    def calcStandardDeviation(self):
        avgFit = self.calcAverageFitness()
        # print( [x.fitness for x in sel])
        sd = math.sqrt( sum( [abs(x.fitness - avgFit)**2 for x in self.population] ) / len(self.population) )
        return sd

    def initPopulation(self, size, k):
        """Initializes population of paths(chomosomes)
        The population size will be equal to size.
        :param size: Size of population
        """
        self.population = [Chromosome(Kcut.kcut(self.graph, k)) for _ in range(size)]


    def iterateGen(self, selection_op, crossover_op, mutation_op, 
        selection_k = None, 
        crossover_k = None, 
        mutation_k = None, mutation_chance = None,
        elitism = True, elitism_k = 5): 
        """Creates new generation based on current state by modifying current state of population.
        1.) Parents are selected *selection_op* 
        2.) Crossover function *crossover_op* is applied to population
        3.) Mutation function *mutation_op* is applied to population
        Note: all *k* parameters have defaults specified in receiving functions
        :param selection_op: string of desired new population selection function
        :param crossover_op: string of desired crossover function
        :param mutation_op: string of desired mutation function
        :param selection_k: # of randomly selected chromosomes etc. depending on selection_op
        :param crossover_k: # of crossover points or other depending on crossover_op
        :param mutation_k: # of swaps / size of range depending on mutation_op
        :param mutation_chance: probabilty (%) a mutation occurs
        :param elitism: keeps top (elitism_k) number of members from current population
        :param elitism_k: # of "elite" chromosomes to keep
        """
        if mutation_chance == None:
            mutation_chance = 0.1
        elif mutation_chance > 1:
            mutation_chance = 1

        newPopulation = []
        
        ## keep *elitism_k* members of current population
        if elitism:
            def min_func(x):
                return x.fitness
            sample_pop = copy.deepcopy(self.population)
            for dummy in range(elitism_k):
                test = sample_pop.pop(sample_pop.index(min(sample_pop, key = min_func)))
                newPopulation.append(test)

        ## create (self.size) new chromosomes by selecting parents and applying crossover and 
        ## mutation operators 
        for dummy in range(len(self.population) - elitism_k):
            newChromosome = None
            ## select (2) parents
            parent1 = globals()[selection_op](self, selection_k)
            parent2 = globals()[selection_op](self, selection_k)
            
            ## test logging
            if TEST_LOG_BIT:
                print("-------------------------------------------")
                print(f"PARENT1: {parent1.kcut}")
                print(f"PARENT2: {parent2.kcut}")
            
            ## apply crossover
            ## NOT CURRENTLY SUPPORTED BY PMX CROSSOVER (it currently generates one child)
            ## randomly selects between two children produced by crossover
            # newPath = globals()[crossover_op](parent1.path, parent2.path)[random.randint(1,2) % 2]
            ## selects first offspring of crossover
            newPath = globals()[crossover_op](parent1.path, parent2.path)[0]
            
            ## apply mutation
            if random.randint(1, int(1/mutation_chance)) == 1:
                newPath = globals()[mutation_op](newPath, mutation_k)
            newChromosome = Chromosome(newPath, self.nodeArray.distanceList)
            if newChromosome:
                newPopulation.append(newChromosome)
        self.population = newPopulation
        self.iteration += 1

    ## Generation data retrieval functions
    def getGenInfo(self):
        """Returns all generation info"""
        iteration = self.iteration
        best = self.getBestPath()
        bestFitness = best[1]
        bestPath = best[0][0]
        avgFitness = self.calcAverageFitness()
        stdDev = self.calcStandardDeviation()
        return (iteration, bestFitness, avgFitness, stdDev, bestPath)

    def getBestPath(self):
        """Returns list of best possible paths.
        In this case fitness is the shortest possible distance, so
        the lower the better.
        """
        maxDist = max( [distance for distanceList in self.nodeArray.distanceList for 
                        distance in distanceList] )
        bestFitness = maxDist * len(self.nodeArray.nodeList)
        for i in self.population:
            if i.fitness <= bestFitness:
                bestFitness = i.fitness
        
        pathList = [x.path for x in self.population if x.fitness==bestFitness]
        return (pathList, bestFitness)

    ## printing / logging functions
    def openLog(self, test_select, cnt):
        n = str(test_select)
        cnt = str(cnt)
        logPath = os.path.normpath(f"./log_files/p5_logs/TEST{n}/")
        if not os.path.exists(logPath):
            os.makedirs(logPath)
        self.logFile = open(os.path.normpath(f"{logPath}/{cnt}.txt"), 'w')
        return self.logFile

    def writeLog(self, text):
        if self.logFile != None:
            self.logFile.write(text)

    def closeLog(self):
        if self.logFile != None:
            self.logFile.close()

    def printPopulation(self):
        print("POPULATION STATS:")
        for index, k in enumerate(self.population):
            print("ID: {0}, Fitness: {1}, Path: {2}".format(index, k.fitness, k.path))

    def printBestSolutions(self, num_of_paths = 0, include_paths = False):
        best = self.getBestPath()
        best_paths = best[0]
        if num_of_paths < len(best_paths):
            best_paths = best_paths[0:num_of_paths + 1]
        
        if include_paths:
            print("## ITER = {0}, best fitness = {1}, best path(s): {2}".format(self.iteration, 
                best[1], best_paths))
        else:
            print("## ITER = {0}, best fit = {1:.4f}, avg fit = {2:.4f}, std dev = {3:.4f}".format(
                self.iteration, best[1], self.calcAverageFitness(), 
                self.calcStandardDeviation()))



def geneticAlgoGenerator(nodeArray, startNodeID):
    """Genetic algorithm generator
    :param nodeArray: array of nodes (cities)
    :param startNodeID: node ID of starting node (city) 
    :returns: Generator to interface with GraphGUI.py iterative display
    """
    ## console output bit
    console_out = False
    ## log output bit
    log_out = True
    ## select test and population size
    test_select = 97
    ## number of tests to perform (and log)
    num_of_tests = 50
    ## size of population
    popSize = 100
    ## number of generation to create for each test
    num_of_gen = 1500
    
    ### intialize first gneration
    currentGen = Generation(nodeArray)
    currentGen.initPopulation(popSize, 0, 0, startNodeID)

    def run_test(test_num, selection_op, crossover_op, mutation_op, 
        selection_k = None, 
        crossover_k = None, 
        mutation_k = None, mutation_chance = None,
        elitism = True, elitism_k = 5):
        
        if log_out:
            log.write(f"NUM OF CITIES: {os.path.basename(nodeArray.inputFileName)}\n")
            log.write(f"TEST {test_num}: pop = {popSize}, # of generations = {num_of_gen} "
                f"selection = {selection_op}, crossover = {crossover_op}, "
                f"mutation = {mutation_op}, mutation chance = {mutation_chance}\n")
            log.write("{0:<8}{1:<13}{2:<14}{3:<9}{4}\n".format("iter", "best fit",
                f"avg fit", "std dev", "best path"))
        
        for dummy in range(num_of_gen):
            currentGen.iterateGen(selection_op, crossover_op, mutation_op,
                mutation_chance = mutation_chance, mutation_k = mutation_k,
                selection_k = selection_k, crossover_k = crossover_k,
                elitism = True)   
            if console_out:
                currentGen.printBestPaths()
            best = currentGen.getGenInfo()
            if log_out:
                log.write(f"{best[0]:5d}\t{best[1]:.4f}\t{best[2]:>10.4f}\t{best[3]:>10.4f}\t{best[4]}\n")
            yield [x+1 for x in best[-1]]


    for cnt in range(num_of_tests):
        ### intialize first gneration
        currentGen = Generation(nodeArray)
        currentGen.initPopulation(popSize, 0, 0, startNodeID)
        currentGen.iteration = 0
        with currentGen.openLog(test_select, cnt) as log:
            ### Test 1
            if test_select == 1:
                time1 = time.time()
                if log_out:
                    log.write(f"NUM OF CITIES: {os.path.basename(nodeArray.inputFileName)}\n")
                    log.write("TEST 1: pop = {0}, # of generations = {1} selection = {2}, crossover = {3}, "
                        "mutation = {4}, mutation chance = {5}\n".format(popSize, num_of_gen, 
                        "select_parent_tournament", "crossover_davis_order", "mutation_inversion", 0.7))
                    log.write("{0:<8}{1:<13}{2:<14}{3:<9}{4}\n".format("iter", "best fit", "avg fit", "std dev", "best path"))
                
                for dummy in range(num_of_gen):
                    currentGen.iterateGen("select_parent_tournament", "crossover_davis_order", "mutation_inversion",
                        mutation_chance=0.7,
                        selection_k=random.randint(2, int(len(currentGen.population)/2) + 1), 
                        elitism=True)   
                    if console_out:
                        currentGen.printBestPaths()
                    best = currentGen.getGenInfo()
                    if log_out:
                        log.write(f"{best[0]:5d}\t{best[1]:.4f}\t{best[2]:>10.4f}\t{best[3]:>10.4f}\t{best[4]}\n")
                    yield [x+1 for x in best[-1]]


def main(inputFile):
    ## create first generation
    with open(inputFile) as graph_input:
        inputString = graph_input.read()
        currentGen = Generation(inputString)
        currentGen.initPopulation(50, 5)
        for x in currentGen.population:
            print(x.fitness)
    

if __name__ == "__main__":
    main(sys.argv[1])