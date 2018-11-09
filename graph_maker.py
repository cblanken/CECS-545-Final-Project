import sys
import os
import random

def create_tsp_list(num_of_cities, x_range = 100, y_range = 100):
    tsp_list = [ (city_num + 1, random.random() * x_range, random.random() * y_range) for city_num in range(num_of_cities) ]
    return tsp_list

def create_edge_list(num_of_Cities, max_edges = 3):
    edge_lists = [ list(set([random.randint(1, num_of_Cities) for _ in range(random.randint(1, max_edges))]))\
        for _ in range(num_of_Cities)]
    for index, edgeList in enumerate(edge_lists):
        if index + 1 in edgeList:
            print(edgeList)
            edgeList.remove(index + 1)
            print('REMOVED {0}'.format(index+1))
    for index, edgeList in enumerate(edge_lists):
        if not edgeList:
            newNode = random.choice([x+1 for x in range(num_of_Cities) if x+1 != index])
            edgeList.append(newNode)
            print("LIST EMPTY: {0} added".format(newNode))
    return edge_lists

def main(size = 100, outputFileName = "test.tsp" ):
    path = os.path.normpath("./InputGraphs/" + outputFileName)
    outputFile = open(path, 'w')

    tsp_list = create_tsp_list(int(size))
    edge_lists = create_edge_list(int(size))
    # tsp_list = [str(city_num) + " " + str(x) + " " + str(y) for city_num, x, y in tsp_list]
    out_list = [" ".join( [str(city_num), str(x), str(y), str(edge_lists[index]) ]) for index, (city_num, x, y) in enumerate(tsp_list)]


    out_str = "\n".join(out_list)
    print(out_str)
    
    if outputFile:
        outputFile.write(out_str)
    outputFile.close()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()