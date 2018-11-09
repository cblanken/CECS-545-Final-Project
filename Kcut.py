import collections
import random
import sys
import math

import GraphParse as parse

Node = collections.namedtuple('Node', ['no', 'x', 'y', 'adjList'])
Edge = collections.namedtuple('Edge', ['src', 'dest', 'weight'])


class Graph:
    """
    """
    def __init__(self, nodeList, edgeList):
        self.nodeList = nodeList
        self.edgeList = edgeList


class Subgraph(Graph):
    """
    """
    def __init__(self, id, nodeList = [], edgeList = [], adjList = []):
        super().__init__(nodeList, edgeList)
        ## list of edges leaving the subgraph
        self.adjList = adjList

    def _pruneAdjList(self):
        self.adjList = [edge for edge in self.adjList if edge.dest not in self.nodeList]

    def addNode(self, node):
        """Add node to subgraph and update frontier edges/nodes
        :param node: Node to be added to subgraph
        """
        self.nodeList.append(node)
        self._pruneAdjList()

def calcDistance(node1, node2):
    return math.sqrt( (node1[1] - node2[1])**2 + (node1[2] - node2[2])**2 )

def parseInputGraph(filepath):
    graph_input = parse.parse(filepath)
    edgeLists = [ [Edge(node[0], dest, calcDistance(node, graph_input[dest-1])) \
        for dest in node[3]] for index, node in enumerate(graph_input) ]
    nodeList = [Node(node[0], node[1], node[2], edgeLists[index]) for index, node \
        in enumerate(graph_input)]
    # print(nodeList)
    return nodeList

def kcut(graph, k):
    """Create a random k-cut on graph
    :param graph: graph on which to perform the k-cut
    :param k: k in k-cut, relating to the number of subgraphs to create
    """
    subgraphList = [Subgraph(id = i) for i in range(k)]
    
    ## initialze random starting node for k subgraphs
    firstNodes = random.sample(graph.nodeList, k)
    for id, subgraph in enumerate(subgraphList):
        subgraph.nodeList.append(firstNodes[id])

    ## expand each subgraph to fill entire graph
    while graph.nodeList:
        for subgraph in subgraphList:
            if subgraph.adjList:
                newEdge = subgraph.adjList.pop(random.choice(subgraph.adjList))
                subgraph.addNode(newEdge.dest)

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
    parseInputGraph(filepath)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("usage: <inputfilepath>")
