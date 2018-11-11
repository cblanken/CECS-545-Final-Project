"""WoC (Wisdom of Crowds)
Aggregates set of solutions (solution_set) to find a new
result (ideally one that is better than a single "expert"
in solution_set)
In the case of the k-cut,
# *solution_set* is a list of valid k-cuts to be aggregated
# 
# 
"""
import os
import sys
import copy
import time
import Kcut
import KcutResultsParser as rp

## agreement matrix, a list of 
agreement_matrix = []

## list of dictionaries containing (valid TSP solution, fitness) to be aggregated
### EXAMPLE input_list
# input_list = [
#       (fitness, cost, subgraphList),
#       EXAMPLES:
#       (2174.3236, 522.4288629089076, [[9, 5], [11], [12], [10, 8], [7, 2], [1], [4, 6], [3]])
#       (2223.9106, 538.8308585215631, [[10, 9], [4, 6], [1], [12], [11, 5], [7], [2, 8], [3]])
#       (2308.4466, 560.3736090616171, [[4], [6], [9, 10], [11, 5], [8, 12], [2, 7], [1], [3]])
#       (2326.4359, 563.6717104361878, [[7], [11, 5], [8, 2], [1], [10, 9], [12], [3], [4, 6]]) 
#       ...
# ]

def get_graph(inputString):
    return Kcut.parseInputGraphString(inputString)

def select_fittest_individuals(input_list, percent = .30):
    """Select fittest from input_list
    :param input_list: TSP solution tuple (cost, [path])
    :param percent: percent of fittest individuals to keep
    :return: 
    """
    ## sort input_list by *cost* for each path and remove lower (1-percent)% of individuals
    output_list = sorted(input_list, key=lambda x: x[0])[0:int(len(input_list) * percent)]
    return output_list    

def find_agreement(input_list, graph):
    """Return agreement matrix for given input_list
    # create matrix [i x j] that holds the value (agreement) for whether or 
    # not node (i) should belong in a subgraph with node (j)
    # the number in the matrix will be incremented for each input solution
    # where node (i) belongs in a subgraph with node (j)
    :param input_list: list of tuples (cost, [k-cut])
    :returns: agreement matrix
    """
    agreement_matrix = [[0 for x in range(len(graph.nodeList))]\
        for x in [0 for y in range(len(graph.nodeList))]]

    for solution in input_list:
        for subgraphNodeList in solution[2]:
            for node1 in subgraphNodeList:
                for node2 in subgraphNodeList:
                    agreement_matrix[node1 - 1][node2 - 1] += 1

    return agreement_matrix




def printAgreementMatrix(agreement, fittest):    
    print("\nFITTEST INDIVIDUALS:")
    for x in fittest:
        print(x)
    print("\nAGREEMENT MATRIX:")
    alignmentList = [len(str(x))+1 for x in range(len(agreement))]
    newRange = f"{' ':<{len(x)+1}}".join([str(x+1) for x in range(len(agreement))])
    print(f"{'':<6}", newRange)
    print(f"{'':<7}", end="")
    for i in range(len(newRange)):
        print("-", end="")
    print()
    for i, x in enumerate(agreement):
        # row = "|".join(str(i) for i in x)
        # print(f"{i+1}|", f"{'|':>{alignmentList[i]}}".join(str(i) for i in x))
        print(f"{i+1:>4}", end="")
        for j, y in enumerate(x):
            colStr = f"{y}"
            print(f"{colStr:>{alignmentList[j]+2}}", end="")
        print()


def main(dirPath, inputGraphFile = None):
    if inputGraphFile != None:
        graph = Kcut.parseInputGraph(inputGraphFile)
    inputList = rp.parseDir(dirPath)
    
    fittest = select_fittest_individuals(inputList, .50)
    agreement = find_agreement(fittest, graph)
    printAgreementMatrix(agreement, fittest)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ./Kcut_Woc <resultsDirPath>")
        print("or")
        print("usage: python ./Kcut_Woc <resultsDirPath, inputGraphFile>")
        exit(1)
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])