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
import itertools
import collections
import random
import Kcut
import KcutResultsParser as rp

TEST_LOG_BIT = False

## agreement matrix, a list of 
agreement_matrix = []
NodePair = collections.namedtuple('NodePair', ['agreement', 'node1', 'node2'])

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
    
    for row in range(len(agreement_matrix)):
        for col in range(len(agreement_matrix)):
            if row > col:
                agreement_matrix[row][col] = -1

    for solution in input_list:
        for subgraphNodeList in solution[2]:
            for node1 in subgraphNodeList:
                for node2 in subgraphNodeList:
                    if node1 <= node2:
                        agreement_matrix[node1 - 1][node2 - 1] += 1

    return agreement_matrix


def get_agreementList(agreement_matrix, graph):
    agreementList = []
    for i, row in enumerate(agreement_matrix):
        for j, col in enumerate(row):
            if col > 0 and i < j:
                agreementList.append(NodePair(agreement_matrix[i][j], i+1, j+1))                
    return agreementList


def _join_subgraphs(sub1, sub2):
    newSub = sub1
    for node in sub2.nodeList:
        newSub.addNode(node)
    return newSub

def _find_node(kcut, node_num):
    """:returns: (node_num, subgraph_id)"""
    for subgraph in kcut:
        if node_num in [x.no for x in subgraph.nodeList]:
            return subgraph.id
    return None

def get_aggregate(agreementList, graph, inputList):
    k = len(inputList[0][2])
    agreement_max = agreementList[0].agreement
    kcut = [Kcut.Subgraph(0, [], [])]
    ## intialize subgraph with best choices from agreementList
    first_subgraph = kcut[0]
    remainingNodes = copy.deepcopy(graph.nodeList)
    remainingNodePairs = copy.deepcopy(agreementList)
    for nodePair in agreementList:
        if nodePair.agreement != agreement_max:
            break
        else:
            firstNode = graph.nodeList[nodePair.node1-1]
            secondNode = graph.nodeList[nodePair.node2-1]
            if firstNode not in first_subgraph.nodeList:
                first_subgraph.addNode(firstNode)
                remainingNodes.remove(firstNode)
            if secondNode not in first_subgraph.nodeList:
                first_subgraph.addNode(secondNode)
                remainingNodes.remove(secondNode)
            remainingNodePairs.remove(nodePair)

    currNodePairList = [np for np in agreementList if \
        np.agreement == remainingNodePairs[0].agreement]
    if TEST_LOG_BIT:
        print(currNodePairList)
    while len(kcut) < k:
        currNodePairList = [np for np in agreementList if \
            np.agreement == remainingNodePairs[0].agreement]
        if TEST_LOG_BIT:
            for subgraph in kcut:
                print(subgraph)


        while(currNodePairList):

            if TEST_LOG_BIT:
                print(f"REMAINING NODES: {len(remainingNodes)}")
            for currNodePair, nodePair in enumerate(currNodePairList):
                ## if all remaining subgraphs available are single points
                if TEST_LOG_BIT:
                    print(f"SUBGRAPHS USED SO FAR {len(list(itertools.chain(kcut)))}")
                    print(f"K: {k}")
                if len(list(itertools.chain(kcut))) + len(remainingNodes) == k:
                    if TEST_LOG_BIT:
                        print("RAN OUT OF NODES, FILLING REST")
                    for node in remainingNodes:
                        kcut.append(Kcut.Subgraph(kcut[-1].id+1, [node], [node.adjList]))
                        currNodePairList = None
                    break
                if TEST_LOG_BIT:
                    print(f"CURR NODE: {nodePair}")
                ## if neither node is in a subgraph put both into a subgraph together
                if graph.nodeList[nodePair.node1-1] in remainingNodes and graph.nodeList[nodePair.node2-1] in remainingNodes:
                    subgraphID = kcut[-1].id+1
                    kcut.append(Kcut.Subgraph(subgraphID, [], []))
                    kcut[subgraphID].addNode(graph.nodeList[nodePair.node1-1])
                    kcut[subgraphID].addNode(graph.nodeList[nodePair.node2-1])
                    remainingNodes.remove(graph.nodeList[nodePair.node1-1])
                    remainingNodes.remove(graph.nodeList[nodePair.node2-1])


                ## if both nodes already in subgraphs check if they 
                ## are in different subgraphs and should be joined
                elif graph.nodeList[nodePair.node1-1] not in remainingNodes and\
                    graph.nodeList[nodePair.node2-1] not in remainingNodes:  
                    ## check if nodes are in different subgraphs and join them if they are
                    node1SubgraphID = 0
                    node2SubgraphID = 0
                    for i, subgraph in enumerate(kcut):
                        if graph.nodeList[nodePair.node1-1] in subgraph.nodeList:
                            node1SubgraphID = i
                            break
                    for j, subgraph in enumerate(kcut):
                        if graph.nodeList[nodePair.node2-1] in subgraph.nodeList:
                            node2SubgraphID = j
                            break
                    if node1SubgraphID != node2SubgraphID:
                        newSubgraph = _join_subgraphs(kcut[node1SubgraphID], kcut[node2SubgraphID]) 
                        ## remove (2) old subgraphs from kcut and add newSubgraph to kcut 
                        kcut.remove(kcut[node1SubgraphID])
                        if node1SubgraphID < node2SubgraphID:
                            node2SubgraphID -= 1
                        kcut.remove(kcut[node2SubgraphID])
                        kcut.append(newSubgraph)
                        ## reindex the subgraphs
                        for i, subgraph in enumerate(kcut):
                            subgraph.id = i

                ## if one node not in a subgraph find subgraph(s) for the other 
                ## node of the pair and add potentialNode to one of them (subgraph)
                else:
                    nodesToCheck = []
                    if graph.nodeList[nodePair.node1-1] in remainingNodes and graph.nodeList[nodePair.node2-1] not in remainingNodes:
                        potentialNode = nodePair.node1
                        nodesToCheck.append(nodePair.node2)
                    elif graph.nodeList[nodePair.node2-1] in remainingNodes and graph.nodeList[nodePair.node1-1] not in remainingNodes:
                        potentialNode = nodePair.node2    
                        nodesToCheck.append(nodePair.node1)
                    for subNodePair in currNodePairList[currNodePair:]:
                        if subNodePair.node1 == potentialNode:
                            nodesToCheck.append(subNodePair.node2)
                        elif subNodePair.node2 == potentialNode:
                            nodesToCheck.append(subNodePair.node1)
                    subID = _find_node(kcut, nodesToCheck[0])
                    potentialSubgraphs = [subID]
                    for node in nodesToCheck:
                        if node not in [n.no for n in kcut[subID].nodeList]:
                            potentialSubgraph = _find_node(kcut, node)
                            if potentialSubgraph not in potentialSubgraphs and potentialSubgraph != None:
                                potentialSubgraphs.append(potentialSubgraph)
                    if TEST_LOG_BIT:
                        print(f"potential: {potentialNode}")
                    if len(potentialSubgraphs) > 1:
                        kcut[random.choice(potentialSubgraphs)].addNode(graph.nodeList[potentialNode-1])
                        remainingNodes.remove(graph.nodeList[potentialNode-1])
                    else:
                        kcut[potentialSubgraphs[0]].addNode(graph.nodeList[potentialNode-1])
                        remainingNodes.remove(graph.nodeList[potentialNode-1])
                    if TEST_LOG_BIT:
                        print(f"REMAINING NODES:")
                        for i in remainingNodes:
                            print(i.no)

                remainingNodePairs.remove(nodePair)
                currNodePairList = [np for np in agreementList if \
                    np.agreement == remainingNodePairs[0].agreement]
                if TEST_LOG_BIT:
                    for subgraph in kcut:
                        print(subgraph)
                # input("AFTER\n")
    return kcut



def printAgreementMatrix(agreement, fittest):    
    print("\nFITTEST INDIVIDUALS:")
    for x in fittest:
        print(x)
    print("\nAGREEMENT MATRIX:")
    alignmentList = [len(str(x))+1 for x in range(len(agreement))]
    newRange = f"{' ':<{len(x)+1}}".join([str(x+1) for x in range(len(agreement))])
    print(f" {'':<6}", newRange)
    print(f"{'':<7}", end="")
    for i in range(len(newRange)):
        print("-", end="")
    print()
    for i, x in enumerate(agreement):
        print(f"{i+1:>4}", end="")
        for j, y in enumerate(x):
            colStr = f"{y}"
            print(f"{colStr:>{alignmentList[j]+3}}", end="")
        print()


def main(dirPath, inputGraphFile = None):
    if inputGraphFile != None:
        graph = Kcut.parseInputGraph(inputGraphFile)
    inputList = rp.parseDir(dirPath)
    
    print("\nINPUT LIST:")
    print(inputList)
    fittest = select_fittest_individuals(inputList, .50)
    agreement = find_agreement(fittest, graph)
    printAgreementMatrix(agreement, fittest)
    agreementList = sorted(get_agreementList(agreement, graph), key=lambda x: x.agreement, 
        reverse=True)
    
    for node in agreementList:
        print(node)

    kcut_aggregate = get_aggregate(agreementList, graph, inputList)
    print("\nAGGREGATE KCUT:")
    for subgraph in kcut_aggregate:
        print(subgraph)


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