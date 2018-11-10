import collections
import random
import sys
import math
import copy

import GraphParse as parse

Node = collections.namedtuple('Node', ['no', 'x', 'y', 'adjList'])
Edge = collections.namedtuple('Edge', ['src', 'dest', 'weight'])

class Graph:
    """
    """
    def __init__(self, nodeList, edgeList):
        self.nodeList = nodeList
        self.edgeList = edgeList

    def __repr__(self):
        return (
        f"----------NODE LIST-----------\n"
        f"{self.nodeList}\n"
        "----------EDGE LIST-----------\n"
        f"{self.edgeList}\n")

class Subgraph(Graph):
    """
    """
    def __init__(self, id, nodeList, edgeList, adjList):
        super().__init__(nodeList, edgeList)
        ## list of edges leaving the subgraph
        self.id = id
        self.adjList = adjList




    def _pruneAdjList(self):
        self.adjList = [edge for edge in self.adjList if edge.dest not in \
            [node.no for node in  self.nodeList]]

    def addNode(self, node):
        """Add node to subgraph and update frontier edges/nodes
        :param node: Node to be added to subgraph
        """
        self.nodeList.append(node)
        for edge in node.adjList:
            self.adjList.append(edge)
        self._pruneAdjList()
    
    def __repr__(self):
        return (f"{self.id}|\n"
        "----------NODE LIST-----------\n"
        f"{self.nodeList}\n"
        "----------EDGE LIST-----------\n"
        f"{self.edgeList}\n"
        "----------ADJ LIST------------\n"
        f"{self.adjList}\n"
        "-----------------------------------------------------------------\n"
        "-----------------------------------------------------------------\n")


def calcDistance(node1, node2):
    return math.sqrt( (node1[1] - node2[1])**2 + (node1[2] - node2[2])**2 )

def parseInputGraph(filepath):
    graph_input = parse.parse(filepath)
    node_edgeLists = [ [Edge(node[0], dest, calcDistance(node, graph_input[dest-1])) \
        for dest in node[3]] for index, node in enumerate(graph_input) ]
    nodeList = [Node(node[0], node[1], node[2], node_edgeLists[index]) for index, node \
        in enumerate(graph_input)]
    # print(nodeList)
    graph_edgeList = []
    # graph_edgeList = [edge for edge in [adjList for adjList in [node.adjList for node in node_edgeLists]]]
    return Graph(nodeList, graph_edgeList)

def kcut(graph, k):
    """Create a random k-cut on graph
    :param graph: graph on which to perform the k-cut
    :param k: k in k-cut, relating to the number of subgraphs to create
    """
    subgraphList = [Subgraph(i, [], [], []) for i in range(k)]
    
    ## initialze random starting node for k subgraphs
    firstNodes = random.sample(graph.nodeList, k)    
    for index, node in enumerate(firstNodes):
        subgraphList[index].addNode(node)
    print(f"first nodes: {firstNodes}")


    def findAvailableEdge(subgraph, remaining_nodes):
        # print("AVAILABLE")
        # print(remaining_nodes)
        # input("remaining search pause")
        for edge in subgraph.adjList:
            if edge.dest in [node.no for node in remaining_nodes]:
                return edge     
        return None


    ## expand each subgraph to fill entire graph
    remainingNodesList = copy.deepcopy(graph.nodeList)
    print(remainingNodesList)
    for node in firstNodes:
        remainingNodesList.remove(node)
    # print("REMAINING")
    # print(remainingNodesList)
    while remainingNodesList:
        for index, subgraph in enumerate(subgraphList):
            print(subgraph)
            if subgraph.adjList:
                # newEdge = random.choice(subgraph.adjList)
                newEdge = findAvailableEdge(subgraph, remainingNodesList)
                if newEdge == None:
                    continue
                    
                newNode = graph.nodeList[newEdge.dest-1]
                subgraph.addNode(newNode)
                if newNode in remainingNodesList:
                    remainingNodesList.remove(newNode)
                if newEdge in subgraph.adjList:
                    subgraph.adjList.remove(newEdge)

                print(newEdge)
                for subgraph in subgraphList:
                    outlist = [node.no for node in subgraph.nodeList]
                    print(f"{subgraph.id}")
                    print(outlist)
                input("pause")
            
    print("FINISHED K-CUT")
    for subgraph in subgraphList:
        outlist = [node.no for node in subgraph.nodeList]
        print(f"{subgraph.id}")
        print(outlist)
    # print(remainingNodesList)
    # print(subgraphList)
    return subgraphList


def getKcutFitness(graph, subgraphList):
    """Find the fitness of a particular k-cut
    :param graph: 
    :param subgraphList: List (of size k) of subgraphs for a particular k-cut
    :returns: fitness of k-cut
    """
    # unique_edges = 
    # sum([edge.fitness for edge in ])
    pass



def main(filepath):
    test_graph = parseInputGraph(filepath)
    subgraphList = kcut(test_graph, 2)
    # print(test_graph)
    # for subgraph in subgraphList:
    #     print(subgraph)    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("usage: <inputfilepath>")
