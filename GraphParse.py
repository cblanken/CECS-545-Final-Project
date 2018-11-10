import os
import re
import sys

def formatNodeList(nodeList):
    # newNodeList = [node.split() for node in nodeList]
    newNodeList = [ ( int(node[0]), float(node[1]), float(node[2]), eval(node[3]) )\
        for node in nodeList]
    for node in newNodeList:
        for dest in node[3]:
            if node[0] not in newNodeList[dest-1][3]:
                newNodeList[dest-1][3].append(node[0])
                    
    return newNodeList

def getFileName():
    fileName = input("Enter file path to parse(q to quit): ")
    return fileName

def parse(filePath):
    while(not os.path.isfile(filePath)):
        fileName = getFileName()
        if fileName.lower() == "q":
            print("Shutting down.")
            sys.exit() 
        print("Invalid file name. Try again.")

    filePath = os.path.normpath(filePath)
    testFile = open(filePath, 'r')
    # RegEx to find lines with node info: e.g.)
    # 1 87.951292 2.658162
    # 2 33.466597 66.682943
    # 3 91.778314 53.807184
    # 4 20.526749 47.633290
    nodeInputRegEx = re.compile(r'^(\d+)\s(\d+\.\d+)\s(\d+\.\d+)\s(.+)$', re.MULTILINE)
    nodeList = nodeInputRegEx.findall(testFile.read())
    testFile.close()
    if nodeList != None:
        return formatNodeList(nodeList)
    else:
        return "No match found."

def main(path = None):
    # run test
    if path == None:
        path = input("Enter the name of file to parse: ")
    nodes = parse(path)
    for node in nodes:
        print(node)

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        main(sys.argv[1])
    else:
        main()
