"""WoC (Wisdom of Crowds)
Aggregates set of solutions (solution_set) to find a new
result (ideally one that is better than a single "expert"
in solution_set)
In the case of the TSP,
# *solution_set* is a list of valid paths to be aggregated
# 
# 
"""
import os
import sys
import copy
import time
import NodeArray as na
import TSPParseResults as tpr

test_dir = "log_files/p5_logs/TEST56/"
## agreement matrix, a list of 
agreement_matrix = []

## list of tuples containing (valid TSP solution, fitness) to be aggregated
### EXAMPLE input_list
# input_list = [
#     (462.9242, [0, 3, 18, 12, 9, 5, 11, 7, 16, 20, 10, 15, 2, 14, 6, 21, 19, 17, 1, 13, 8, 4, 0] ),
#     (416.0946, [0, 4, 17, 21, 19, 1, 13, 8, 3, 18, 12, 9, 5, 11, 7, 16, 20, 10, 15, 2, 14, 6, 0] ),
#     (431.5676, [0, 4, 5, 11, 16, 20, 7, 9, 12, 18, 3, 8, 13, 1, 17, 21, 19, 14, 15, 10, 2, 6, 0] ),
#     (437.7973, [0, 8, 13, 1, 17, 21, 19, 14, 2, 15, 10, 20, 16, 7, 11, 5, 9, 12, 18, 3, 4, 6, 0] ),
#     (438.6074, [0, 15, 10, 20, 16, 7, 11, 5, 9, 12, 18, 3, 8, 13, 1, 17, 21, 19, 14, 2, 6, 4, 0] ),
#     (431.5676, [0, 4, 5, 11, 16, 20, 7, 9, 12, 18, 3, 8, 13, 1, 17, 21, 19, 14, 15, 10, 2, 6, 0] ),
#     (455.0395, [0, 14, 2, 15, 10, 20, 16, 7, 11, 5, 9, 12, 18, 3, 8, 17, 19, 21, 1, 13, 4, 6, 0] ),
#     (435.5487, [0, 4, 3, 18, 12, 9, 5, 11, 7, 16, 20, 10, 15, 2, 14, 21, 19, 1, 13, 8, 17, 6, 0] ),
#     (458.2955, [0, 5, 11, 16, 20, 7, 9, 12, 18, 3, 8, 13, 1, 17, 21, 19, 14, 15, 10, 2, 6, 4, 0] ),
#     (431.2060, [0, 2, 15, 10, 20, 16, 7, 11, 5, 9, 12, 18, 3, 8, 13, 1, 17, 21, 19, 14, 6, 4, 0] ),
# ]

def select_fittest_individuals(input_list, percent = .30):
    """Select fittest from input_list
    :param input_list: TSP solution tuple (cost, [path])
    :param percent: percent of fittest individuals to keep
    :return: 
    """
    ## sort input_list by *cost* for each path and remove lower (1-percent)% of individuals
    output_list = sorted(input_list, key=lambda x: x[0])[0:int(len(input_list) * percent)]
    return output_list    


def find_agreement(input_list):
    """Return agreement matrix for given input_list
    # create list that holds the value (agreement) for each possible edge
    # in the graph, e.g.) if one of the paths has a particular edge
    # then the appropriate member [x][y] will be incremented in the matrix
    :param input_list: list of tuples (cost, [path])
    :returns: agreement matrix
    """
    num_of_nodes = len(input_list[0][1]) - 1
    agreement_matrix = [[0 for x in range(num_of_nodes)] \
        for x in [0 for y in range(num_of_nodes)]]
    for solution in input_list:
        edgeList = na.calcEdgeListFromPath(solution[1])
        for edge in edgeList:
            pass
            agreement_matrix[edge[0]][edge[1]] = agreement_matrix[edge[0]][edge[1]] + 1
    return agreement_matrix


def getEdgeList(dirPath, percent = .30):
    """
    """
    input_list = tpr.parseDir(dirPath)
    selection = select_fittest_individuals(input_list, percent)
    agreement_matrix = find_agreement(selection)

    edgeList = []
    for row_index, row in enumerate(agreement_matrix):
        for col_index, agreement in enumerate(row):
            extra = 0
            ## consolidate edges that are reversed e.g.) 2->7 vs 7->2
            ## and if 7->2 exists add agreement value to 2->7
            ## only output edges where row > col
            if row_index > col_index:
                continue
            if agreement > 0:
                if agreement_matrix[col_index][row_index] >= 0:
                    extra = agreement_matrix[col_index][row_index]
                edgeList.append( ([row_index, col_index], agreement + extra) )
    return edgeList

## stitch together remaining edges and nodes
def segment_to_path(segment):
    path = []
    first_node = None
    mid_node = None
    last_node = None
    if len(segment) == 0:
        return []
    ## catch single edges
    if len(segment) == 1:
        return segment[0][0]
    ## otherwise process full edge len(edge) > 1
    for i in range(len(segment) - 1):
        ## find node connecting 2 edges *mid_node*
        for j in range(2):
            if segment[i][0][j] == segment[i+1][0][0]:
                mid_node = segment[i][0][j]
            elif segment[i][0][j] == segment[i+1][0][1]:
                mid_node = segment[i][0][j]
        ## find *first_node*
        for node in segment[i][0]:
            if node != mid_node:
                first_node = node
                break

        path.append(first_node)
        ## add final (2) nodes once end of segment reached
        if i == len(segment) - 2:
            path.append(mid_node)
            ## find *last_node*
            for node in segment[i+1][0]:
                if node != mid_node:
                    last_node = node
            path.append(last_node)
    return path



def agreementGenerator(nodeArray, startNodeID = 0, percent = .30, culling_scale = 1.0):
    """
    :param nodeArray: USED TO MAINTAIN GENERATOR CALL STRUCTURE
    :yields: list of edges and their agreements of orginal solution set 
        generated from dirPath
    """
    if percent == 0:
        percent = 0.30
    if culling_scale == 0.0:
        culling_scale = 1
    print(f"CULLING: {culling_scale}, SELECTION: {percent}")
    dirPath = os.path.normpath("log_files/p5_logs/TEST" + str(len(nodeArray.nodeList)) + "/")
    yield getEdgeList(dirPath, percent=percent) 


def aggregateGenerator(nodeArray, startNodeID = 0, percent = .30, culling_scale = 1.0):
    """
    :param nodeArray: USED TO MAINTAIN GENERATOR CALL STRUCTURE
    :yields: aggregation of edges based on orginal solution set
    """
    if percent == 0:
        percent = 0.30
    if culling_scale == 0.0:
        culling_scale = 1
    
    dirPath = os.path.normpath("log_files/p5_logs/TEST" + str(len(nodeArray.nodeList)) + "/")
    edgeList = getEdgeList(dirPath, percent=percent )
    neighbors = [[] for node in nodeArray.nodeList]
    ## eliminate edges below average agreement
    avg_edge_cost = sum( [x[1] for x in edgeList] ) / len(edgeList)
    ## check for culling scale that eliminates all edges
    if [edge for edge in edgeList if edge[1] > (avg_edge_cost * culling_scale)] == []:
        print("CULLING SCALE TOO HIGH, RESET TO 1")
        culling_scale = 1
    
    print(f"CULLING: {culling_scale}, SELECTION: {percent}")
    edgeList = [edge for edge in edgeList if edge[1] > (avg_edge_cost * culling_scale)]

    ## check edges don't overlap (three edges to or from a node) 
    ## then -> choose edge with greatest agreement
    new_edgeList = []
    for node_num, node in enumerate(nodeArray.nodeList):
        node_connections = [edge for edge in edgeList if edge[0][0] == node_num or \
            edge[0][1] == node_num]
        
        # print(f"NODE CONNs: {node_connections}")
        new_node_connections = []
        if len(node_connections) > 2:
            ## get 2 best edges leading to (or from) specified node
            for _ in range(2):
                newEdge = max(node_connections, key=lambda x: x[1])
                new_node_connections.append(newEdge)
                node_connections.remove(newEdge)
            # print(f"SHORTENED: {new_node_connections}")
            node_connections = new_node_connections

        for edge in node_connections:
            new_edgeList.append(edge)

    ## find edges connected to each individual node
    for node_num in range(len(nodeArray.nodeList)):
        for edge in edgeList:
            ## check if edge already added for another
            if node_num in edge[0]:
                for node in edge[0]:
                    if node != node_num:
                        neighbors[node_num].append(edge)

    # print(f"\nBEFORE:")
    # for i, x in enumerate(neighbors):
    #     print(f"i: {i}, {x}")

    ## remove edges when # of edges connected to a node
    ## is greater than 2
    for neighbor_edges in neighbors:
        while len(neighbor_edges) > 2:
            ## prune tree to get only 2 connections to node
            edge = min(neighbor_edges, key=lambda x: x[1])
            for ne in neighbors:
                try:
                    ne.remove(edge)
                except ValueError:
                    continue
 
    # print(f"\nAFTER:")
    # for i, x in enumerate(neighbors):
    #     print(f"i: {i}, {x}")
    
    new_edgeList = []
    for neighbor_edges in neighbors:
        for edge in neighbor_edges:
            new_edgeList.append(edge)
            # yield new_edgeList

    
    ## find segments and break cycles
    segmentList = []
    nodeList = []
    def follow_edges(start_node_num):
        curr_segment = []
        if len(neighbors[start_node_num]) == 0:
            return []
        curr_segment.append(neighbors[start_node_num][0])
        for node in neighbors[start_node_num][0][0]:
            nodeList.append(node)
            curr_node_num = node
        # print(f"CURRENT SEGMENT = {curr_segment}")
        
        ## whether a loop has been detected or not
        loop = False
        ## whether both ends of segment have been found or not
        finished_1 = False
        finished_2 = False
        while not loop and not (finished_1 and finished_2):
            ## look at last edge of curr_segment
            # print(f"curr_node_num = {curr_node_num}")
            # print(f"NEIGHBORS[curr_node_num] = {neighbors[curr_node_num]}")
            if len(neighbors[curr_node_num]) < 2 and curr_node_num != start_node_num:
                if finished_1 and not finished_2:
                    finished_2 = True
                elif not finished_1:
                    finished_1 = True
                    if len(neighbors[start_node_num]) == 2:
                        curr_segment.reverse()
                    else:
                        finished_2 = True
                curr_node_num = start_node_num
                # print(f"IN, curr_node_num = {curr_node_num}")
                continue
                
            ## look at two possible edges connected to node
            ## and choose the one that you aren't using to
            ## reach the node
            node_added = False
            for edge in neighbors[curr_node_num]:
                for node in edge[0]:
                    if node != curr_node_num and node not in nodeList:
                        curr_segment.append(edge)
                        nodeList.append(node)
                        curr_node_num = node
                        node_added = True
                        ## break just in case
                        # print(f"AFTER curr_seg: {curr_segment}")
                        break
            if not node_added:
                for edge in neighbors[curr_node_num]:
                    if edge not in curr_segment:
                        curr_segment.append(edge)
                loop = True
        # print(f"SEG: {loop}, {finished_1}, {finished_2}, {nodeList}, {curr_segment}")
        if loop:
            least_agreed_edge = min(curr_segment, key=lambda x: x[1])
            least_agreed_index = curr_segment.index(least_agreed_edge)
            if least_agreed_index > 0 and least_agreed_index < len(curr_segment)-1:
                chunk = curr_segment[:curr_segment.index(least_agreed_edge)]
                curr_segment.remove( least_agreed_edge )  
                curr_segment = curr_segment[least_agreed_index:] + chunk
            else:
                curr_segment.remove( least_agreed_edge )
        return curr_segment
    
    for node_num in range(len(nodeArray.nodeList)):
        ## follow edges of segments until end is reached in both directions
        ## break the loop if one is found
        if node_num not in nodeList:
            segmentList.append(follow_edges(node_num))
            # print(f"OUTPUT: {segmentList[-1]}")

    new_edgeList = [edge for segment in segmentList for edge in segment]
    # yield new_edgeList


            
    segment_paths = [segment_to_path(seg) for seg in segmentList]
    # segment_paths = [path for path in segment_paths]
    
    missing_nodes = []
    for node in nodeArray.nodeList:
        if node.id-1 not in [node for segment in segment_paths for node in segment]:
            segment_paths.append([node.id-1])
            missing_nodes.append([node.id-1])
            segment_paths.remove([])

    # print("SEGMENT PATHS:")
    # for path in segment_paths:
    #     print(path)
    
    stitching_edgeList = []
    ## joining segment_paths
    for _ in range(len(segment_paths)):
        segment_paths_ends = [(path[0], path[-1]) if path else () for path in segment_paths]
        shortest_new_edge = None
        best = 1000
        indexes = []
        for i, starts in enumerate(segment_paths_ends):
            for j, ends in enumerate(segment_paths_ends):
                for start in starts:
                    for end in ends:
                        dist = nodeArray.calcDistance(start, end)
                        if dist < best and start != end and not (start in segment_paths[i] and \
                            end in segment_paths[i]) and i != j:
                            best = dist
                            shortest_new_edge = [start, end]
                            indexes = [i,j]
        ## if all edges connected break from loop
        if shortest_new_edge == None:
            break
        joined_edge = segment_paths[indexes[0]] + (segment_paths[indexes[1]] if segment_paths[indexes[1]][0] \
            in shortest_new_edge else segment_paths[indexes[1]][::-1])

        removal_indexes = []
        for index, path in enumerate(segment_paths):
            if len(path) < 1:
                continue
            if path[0] == shortest_new_edge[0] or\
                path[0] == shortest_new_edge[1] or\
                path[-1] == shortest_new_edge[0] or\
                path[-1] == shortest_new_edge[1]:
                removal_indexes.insert(0, index)
        for index in removal_indexes:    
            segment_paths.pop(index)

        stitching_edgeList.append(shortest_new_edge)
        
        segment_paths.append(joined_edge)
        # print("SEGMENT PATHS:")
        # for path in segment_paths:
        #     print(path)
        # print(f"SHORTEST NEW EDGE: {shortest_new_edge}, {best}, {indexes}")


    ## collect edges into new_edgeList
    segment_paths[0].append(segment_paths[0][0])
    print(f"PATH DIST: {nodeArray.calcPathDist([x+1 for x in segment_paths[0]])}, {segment_paths[0]}")
    print(f"EDGE LIST: {[x[0] for x in new_edgeList]}")
    ## get edges used to greedily stitch edges together
    # stitching_edgeList = [(edge, -1) for edge in stitching_edgeList]
    stitching_edgeList = [(edge, -1) for edge in na.calcEdgeListFromPath(segment_paths[0]) if edge[:] not in [x[0] for x in new_edgeList] and edge[::-1] not in [x[0] for x in new_edgeList]]
    print(f"STICHING EDGE LIST: {stitching_edgeList}")
            
    ## convert segment_paths to valid edgelist to yield to GraphGUI
    new_edgeList = [(x, 255) if x not in [y[0] for y in stitching_edgeList] else (x, -1) \
        for x in na.calcEdgeListFromPath(segment_paths[0])]

    edgeList = new_edgeList
    yield edgeList


def main(dirPath):
    edgeList = getEdgeList(dirPath)

    arr = na.NodeArray("files/Random44.tsp")
    for _ in aggregateGenerator(arr, percent = 0.80):
        pass

    # for index, solution in enumerate(selection):
    #     print(f"iter:{index}-->{solution}")
    # print("AGREEMENT MATRIX:")
    # for edge in find_agreement(input_list):
    #     print(edge)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        main("log_files/p5_logs/TEST222/")
    else:
        main(sys.argv[1])