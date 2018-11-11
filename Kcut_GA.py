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
import itertools
import Kcut
import GraphParse as gp

LOG_BIT = True
LOG_GEN_BIT = False
TEST_LOG_BIT = False

def _selectSubgraph(kcut):
    while(1):
        chosenSubgraph = random.choice(kcut)
        if len(chosenSubgraph.nodeList) > 1:
            return chosenSubgraph   

def _selectNode(kcut, curr_subgraph):
    while(1):
        chosenNode = random.choice(curr_subgraph.nodeList)
        if len(chosenNode.adjList) > 0:
            return chosenNode

def _pullNode(pullSubgraph, pushSubgraph, node):
    pullSubgraph.removeNode(node)
    pushSubgraph.addNode(node)

def _grow(kcut, graph):
    pullSubgraph = _selectSubgraph(kcut)
    # if TEST_LOG_BIT:
    #     print("GROWING SUBGRAPH")
    #     print(f"NODE LIST: {pullSubgraph.nodeList}")
    #     print(f"ADJ LIST: {pullSubgraph.adjList}")

    chosenNode = _selectNode(kcut, pullSubgraph)
    while(1):
        pushSubgraph = random.choice(kcut)
        if pushSubgraph != pullSubgraph:
            break
    # pushSubgraph = random.choice(kcut)
    _pullNode(pullSubgraph, pushSubgraph, chosenNode)

    # chosenEdge = random.choice(chosenSubgraph.adjList)
    # for subgraph in kcut:
    #     if subgraph != chosenSubgraph and chosenEdge.dest in [node.no for node in subgraph.nodeList]:
    #         chosenNode = graph.nodeList[chosenEdge.dest-1]
    #         _pullNode(subgraph, chosenSubgraph, chosenNode)


## Mutation functions
def mutation_grow(graph, kcut, grow_k = None):
    ## default number of nodes to grow
    if grow_k == None:
        grow_k = random.randint(1, len(max(kcut, key=lambda x: len(x.nodeList)).nodeList))

    newKcut = copy.deepcopy(kcut)
    for _ in range(grow_k):
        _grow(newKcut, graph)
    return newKcut


## Crossover functions
def crossover_intersection(graph, kcut1, kcut2, k = None):
    """Davis's Order Crossover (AKA Order 1 Crossover or OX1)
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
    Returns path with randomized Davis's Order crossover applied.
    """    
    if k == None:
        k = len(kcut1)

    kcut1_nodeLists = [set([node.no for node in subgraph.nodeList]) for subgraph in kcut1]
    kcut2_nodeLists = [set([node.no for node in subgraph.nodeList]) for subgraph in kcut2]
    graph_nodeList = set([node.no for node in graph.nodeList])
    
    newKcut_nodeLists = []
    curr_intersection = {}
    for nodeList1 in kcut1_nodeLists:
        best_intersection = {}
        for nodeList2 in kcut2_nodeLists:
            curr_intersection = nodeList1 & nodeList2
            if len(curr_intersection) > len(best_intersection):
                best_intersection = curr_intersection
        
        if best_intersection not in newKcut_nodeLists and best_intersection != set():
            if TEST_LOG_BIT:
                print(f"BEST INTERSECTION: {best_intersection}")
            newKcut_nodeLists.append(best_intersection)

    if TEST_LOG_BIT:
        print("FOUND BEST INTERSECTIONS")
    newKcut_nodeLists = [set(nodeList) for nodeList in newKcut_nodeLists]
    allNodes = copy.deepcopy(graph_nodeList)
    union = set()
    for kcut in newKcut_nodeLists:
        union = union | kcut
    remainingNodes = allNodes - union

    newKcut = [Kcut.Subgraph(index, [], []) for index in range(k)]
    for index, subgraph in enumerate(newKcut):
        for node_num in newKcut_nodeLists[index]:
            subgraph.addNode(graph.nodeList[node_num-1])

    remainingNodes = [graph.nodeList[index-1] for index in remainingNodes]
    remainingNodesList = list(remainingNodes)
    while remainingNodesList:
        for index, subgraph in enumerate(newKcut):
            if subgraph.adjList:
                newEdge = Kcut._findAvailableEdge(subgraph, remainingNodesList)
                if newEdge == None:
                    continue
                    
                newNode = graph.nodeList[newEdge.dest-1]
                subgraph.addNode(newNode)
                if newNode in remainingNodesList:
                    remainingNodesList.remove(newNode)
                if newEdge in subgraph.adjList:
                    subgraph.adjList.remove(newEdge)
    if TEST_LOG_BIT:
        print(f"NEW CROSSOVER KCUT: {newKcut}")
    return newKcut

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
        return "Fitness: {}, K-cuts: {}".format(self.fitness, len(self.kcut))

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
        
        # keep *elitism_k* members of current population
        if elitism:
            def min_func(x):
                return x.fitness
            sample_pop = copy.deepcopy(self.population)
            for dummy in range(elitism_k):
                test = sample_pop.pop(sample_pop.index(min(sample_pop, key = min_func)))
                newPopulation.append(test)

        ## create (self.size) new chromosomes by selecting parents and applying crossover and 
        ## mutation operators 
        # for i in range(len(self.population) - elitism_k):
        for i in range(len(self.population)):
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
            newKcut = globals()[crossover_op](self.graph, parent1.kcut, parent2.kcut)
            
            ## apply mutation
            if random.randint(1, int(1/mutation_chance)) == 1:
                newKcut = globals()[mutation_op](self.graph, newKcut, mutation_k)

            newChromosome = Chromosome(newKcut)
            for subgraph in newChromosome.kcut:
                if len(subgraph.nodeList) < 1:
                    print(newChromosome.kcut)
                    input("EMPTY SUBGRAPH PAUSE")
            if newChromosome:
                newPopulation.append(newChromosome)
            if TEST_LOG_BIT:
                print(f"{i} COMPLETE")
        self.population = newPopulation
        self.iteration += 1

    ## Generation data retrieval functions
    def getBestSolution(self):
        """Returns list of best possible paths.
        In this case fitness is the shortest possible distance, so
        the lower the better.
        """
        maxCut = sum( [x.weight for x in list(itertools.chain.from_iterable([node.adjList for node in\
            self.graph.nodeList]))] )
        bestFitness = maxCut
        for i in self.population:
            if i.fitness <= bestFitness:
                bestFitness = i.fitness
        
        kcutList = [x.kcut for x in self.population if x.fitness == bestFitness]
        return (kcutList, bestFitness)

    def getGenInfo(self):
        """Returns all generation info"""
        iteration = self.iteration
        best = self.getBestSolution()
        bestFitness = best[1]
        bestKcut = best[0]
        avgFitness = self.calcAverageFitness()
        stdDev = self.calcStandardDeviation()
        return (iteration, bestFitness, avgFitness, stdDev, bestKcut)

    ## printing / logging functions
    def openLog(self, test_select, cnt):
        n = str(test_select)
        cnt = str(cnt)
        logPath = os.path.normpath(f"./log_files/TEST{n}/")
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

    def printBestSolutions(self, num_of_solutions = 0, include_kcuts = False):
        best = self.getBestSolution()
        best_kcuts = best[0]
        if num_of_solutions < len(best_kcuts):
            best_kcuts = best_kcuts[0:num_of_solutions + 1]
        
        if include_kcuts:
            print("## ITER = {0}, best fit = {1}, best kcut(s): {2}".format(self.iteration,
                best[1], best_kcuts))
        else:
            print("## ITER = {0}, best fit = {1:.4f}, avg fit = {2:.4f}, std dev = {3:.4f}".format(
                self.iteration, best[1], self.calcAverageFitness(), 
                self.calcStandardDeviation()))



def geneticAlgoGenerator(inputString, pop_size, num_of_gen, test_select, cnt, k = 2,):
    """Genetic algorithm generator
    :param nodeArray: array of nodes (cities)
    :param startNodeID: node ID of starting node (city) 
    :returns: Generator to interface with GraphGUI.py iterative display
    """
    ## console output bit
    console_out = False
    ## log output bit
    log_out = True
    
    ### intialize first gneration
    currentGen = Generation(inputString)
    currentGen.initPopulation(pop_size, k)

    def run_test(test_num, selection_op, crossover_op, mutation_op, 
        selection_k = None, 
        crossover_k = None, 
        mutation_k = None, mutation_chance = None,
        elitism = True, elitism_k = 5):
        
        if log_out:
            log.write(f"NUM OF NODES: {len(currentGen.graph.nodeList)}, K: {k}\n")
            log.write(f"TEST {test_num}: pop = {pop_size}, # of generations = {num_of_gen} "
                f"selection = {selection_op}, crossover = {crossover_op}, "
                f"mutation = {mutation_op}, mutation chance = {mutation_chance}\n")
            log.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format("iter", "best fit",
                f"avg fit", "std dev", "best cut"))
        
        for dummy in range(num_of_gen):
            currentGen.iterateGen(selection_op, crossover_op, mutation_op,
                mutation_chance = mutation_chance, mutation_k = mutation_k,
                selection_k = selection_k, crossover_k = crossover_k,
                elitism = True)   
            if console_out:
                currentGen.printBestSolutions()
            best = currentGen.getGenInfo()
            if log_out:
                log.write(f"{best[0]:4d}\t{best[1]:.4f}\t{best[2]:>10.4f}\t{best[3]:>10.4f}\t{best[4]}\n")
            yield best
    
    with currentGen.openLog(test_select, cnt) as log:
        time1 = time.time()
        for generation in run_test(test_num = test_select,
                selection_op = "select_parent_tournament", 
                crossover_op = "crossover_intersection", 
                mutation_op = "mutation_grow", 
                selection_k = random.randint(2, int(len(currentGen.population)/10)),
                crossover_k = None, mutation_k = random.randint(2, 5),
                mutation_chance = 0.1, elitism = True, elitism_k = 5):
            yield generation   
        time2 = time.time()
        if log_out:
            log.write("SEARCH TIME = {}".format(time2 - time1))

    # for cnt in range(num_of_tests):
    #     ### intialize first gneration
    #     currentGen = Generation(nodeArray)
    #     currentGen.initPopulation(popSize, 0, 0, startNodeID)
    #     currentGen.iteration = 0
    #     with currentGen.openLog(test_select, cnt) as log:
    #         ### Test 1
    #         if test_select == 1:
    #             time1 = time.time()
    #             if log_out:
    #                 log.write(f"NUM OF CITIES: {os.path.basename(nodeArray.inputFileName)}\n")
    #                 log.write("TEST 1: pop = {0}, # of generations = {1} selection = {2}, crossover = {3}, "
    #                     "mutation = {4}, mutation chance = {5}\n".format(popSize, num_of_gen, 
    #                     "select_parent_tournament", "crossover_davis_order", "mutation_inversion", 0.7))
    #                 log.write("{0:<8}{1:<13}{2:<14}{3:<9}{4}\n".format("iter", "best fit", "avg fit", "std dev", "best path"))
                
    #             for dummy in range(num_of_gen):
    #                 currentGen.iterateGen("select_parent_tournament", "crossover_davis_order", "mutation_inversion",
    #                     mutation_chance=0.7,
    #                     selection_k=random.randint(2, int(len(currentGen.population)/2) + 1), 
    #                     elitism=True)   
    #                 if console_out:
    #                     currentGen.printBestPaths()
    #                 best = currentGen.getGenInfo()
    #                 if log_out:
    #                     log.write(f"{best[0]:5d}\t{best[1]:.4f}\t{best[2]:>10.4f}\t{best[3]:>10.4f}\t{best[4]}\n")
    #                 yield [x+1 for x in best[-1]]


def main(inputFile):
    ## create first generation
    inputString = ""
    with open(inputFile) as graph_input:
        inputString = graph_input.read()
    # currentGen = Generation(inputString)
    # currentGen.initPopulation(50, 5)
    # print("FIRST GEN")
    # for index, chromo in enumerate(currentGen.population):
    #     print("-------------------------------------------")
    #     print(f"{index} {chromo}: {chromo.kcut}")
    #     print("-------------------------------------------")
    # currentGen.iterateGen("select_parent_tournament", "crossover_intersection", "mutation_grow",
    #     mutation_chance=1, mutation_k = 1,
    #     selection_k=random.randint(2, int(len(currentGen.population)/2) + 1), 
    #     elitism=True) 
    # print("SECOND GEN")
    # for index, chromo in enumerate(currentGen.population):
    #     print("-------------------------------------------")
    #     print(f"{index} {chromo}: {chromo.kcut}")
    #     print("-------------------------------------------")
    for i in range(1, 10):
        generator = geneticAlgoGenerator(inputString, pop_size= 50, num_of_gen = 100, 
            test_select = 1, cnt = i, k = i*2)
        for _ in generator:
            pass

if __name__ == "__main__":
    main(sys.argv[1])