"""Kcut Results File Parsing
"""
import os
import sys
import re
import ast
from Kcut import Graph, Subgraph, Node, Edge

def parseDir(dirPath):
    file_list = os.listdir(dirPath)
    info = ""
    for f in file_list:
        with open(os.path.join(dirPath, f)) as current_file:
            info += "".join(current_file.readlines()[-2:-1])
    # print(f"INFO:\n{info}")
    pattern = re.compile(r"\s\d+\s(\d+\.\d+).*(\[\[.*\]\])")
    result = re.match(pattern, info)
    result = re.findall(pattern, info)
    result = [(float(solution[0]), eval(solution[1])) for solution in result]
    return result


if __name__ == "__main__":
    resultList = parseDir(sys.argv[1])
    print(resultList)
    best = min(resultList, key = lambda x : x[0])
    print(f"\n## MIN of {sys.argv[1]}: {best}")
    worst = max(resultList, key = lambda x : x[0])
    print(f"## MAX of {sys.argv[1]}: {worst}")
    avg = sum( [x[0] for x in resultList] ) / len(resultList)
    print(f"## AVG of {sys.argv[1]}: {avg}")