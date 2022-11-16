from cmath import inf
from itertools import count
import math
from operator import index
import queue
from turtle import color
import graph as g
import timeit
import matplotlib.pyplot as plot


# instantaiting stack data structure
class DfaStack:
    def __init__(self):
        self.stackList = []

    def isempty(self):
        if not bool(self.stackList):
            return True
        # if len(self.stackList) == 0:
        #     return True

    def addStates(self, list_of_states):
        for state in list_of_states:
            self.stackList.append(state)
        # self.stackList.extend(list_of_states)
    
    def removeState(self):
        # if not bool(self.stackList):
        #     raise Exception("Empty frontier")
        removedState = self.stackList.pop(-1)
        return removedState

# instantiate Queue data structure
class Bfs_queue:
    def __init__(self):
        self.queueList = []
    
    def isempty(self):
        if not bool(self.queueList):
            return True
    
    def addStates(self, list_of_states):
        for state in list_of_states:
            self.queueList.append(state)
        
    
    def removeState(self):
        removedState = self.queueList.pop(0)
        return removedState

# Graph Creating function
def createGraph(textFileName):
    graph =  g.Graph()
    listOfNodes = []
    listOfEdges = []
    textFile = open(textFileName, "r")
    for textLine in textFile:
        words = textLine.split()
        listOfNodes.extend([g.Node(words[0]), g.Node(words[1])])
        listOfEdges.append((g.Node(words[0]), g.Node(words[1]), words[2]))
    
    for node in listOfNodes:
        graph.addNodeToGraph(node)
    
    for edge in listOfEdges:
        graph.addEdgeTograph(edge[0], edge[1], edge[2])
    return graph

graph = createGraph("graphData.txt")


# Depth First Search Algorithm Implementation
def depthFirstSearch(intialState, goalState):

    frontierStack = DfaStack()
    exploredStates = []

    frontierStack.addStates([intialState])
    while True:
        if not bool(frontierStack):
            raise Exception("There is no solution for this problem")
        else:
            node = frontierStack.removeState()
            if node not in exploredStates:
                exploredStates.append(node)
            if node == goalState:
                return exploredStates
            
            else:
                tobeExploredNext = []
                for e in graph.edges:
                    if e[0] == node:
                        if e[1] not in exploredStates:
                            tobeExploredNext.append(e[1])
                    elif e[1] == node:
                        if e[0] not in exploredStates:
                            tobeExploredNext.append(e[0])
                frontierStack.addStates(tobeExploredNext)


#Breadth First Search Algorithm Implementation
def bredthFirstSearch(intialState, goalState):
    frontierQueue = Bfs_queue()
    exploredQueues = []
    child_parent_dict = {}
    frontierQueue.addStates([intialState])
    while True:
        if not bool(frontierQueue):
            raise Exception("There is no solution for this problem")
        else:
            node = frontierQueue.removeState()
            if node not in exploredQueues:
                exploredQueues.append(node)
            if node == goalState:
                inputgoalState = goalState
                shortestPath = []
                while goalState != intialState:
                    for keyNode in child_parent_dict.keys():
                        if keyNode == goalState:
                            shortestPath.append(child_parent_dict[keyNode])
                            goalState = child_parent_dict[keyNode]
                shortestPath.reverse()
                shortestPath.append(inputgoalState)
                return shortestPath
            
            else:
                tobeExploredNext = []
                for e in graph.edges:
                    if e[0] == node:
                        if e[1] not in exploredQueues:
                            tobeExploredNext.append(e[1])
                            child_parent_dict.update({e[1]: node})
                    elif e[1] == node:
                        if e[0] not in exploredQueues:
                            tobeExploredNext.append(e[0])
                            child_parent_dict.update({e[0]: node})
                frontierQueue.addStates(tobeExploredNext)
    
# Dijkstra Algorithm Implementation
def DijkstraAlgorithm(startNode, destinationNode):
    visitedNodes = []
    unVisitedNodes = []
    listOfNodesInPath = []
    for node in graph.nodes:
        unVisitedNodes.append(node)


    distanceDict = {node: float(inf) for node in graph.nodes}
    distanceDict[startNode] = 0
    pathDict = {}
    distanceFromStart = queue.PriorityQueue()
    distanceFromStart.put((0, startNode))

    while distanceFromStart.qsize() != 0:
        currentNode = distanceFromStart.get()
        for e in graph.edges:
            if e[0] == currentNode[1]:
                if e[1] not in visitedNodes:
                    #distanceFromStart.put((int(e[2]), e[1]))
                    calculatedDistance = distanceDict[currentNode[1]] + int(e[2])
                    distanceFromStart.put((calculatedDistance, e[1]))
                    if calculatedDistance < distanceDict[e[1]]:
                        distanceDict[e[1]] = calculatedDistance
                        if e[1] in pathDict.keys():
                            pathDict[e[1]] = currentNode[1]
                        else:
                            pathDict.update({e[1]: currentNode[1]}) 
            elif e[1] == currentNode[1]:
                if e[0] not in visitedNodes:
                    distanceFromStart.put((int(e[2]), e[0]))
                    calculatedDistance = distanceDict[currentNode[1]] + int(e[2])
                    if calculatedDistance < distanceDict[e[0]]:
                        distanceDict[e[0]] = calculatedDistance
                        if e[0] in pathDict.keys():
                            pathDict[e[0]] = currentNode[1]
                        else:
                            pathDict.update({e[0]: currentNode[1]})
        
        visitedNodes.append(currentNode[1])
    inputDestinationNode = destinationNode
    while destinationNode != startNode:
        for keyNode in pathDict.keys():
            if keyNode == destinationNode:
                listOfNodesInPath.append(pathDict[keyNode])
                destinationNode = pathDict[keyNode]
    listOfNodesInPath.reverse()
    listOfNodesInPath.append(inputDestinationNode)
    return listOfNodesInPath

# my Heuristic function for calculating the Distance between Location
def distanceCalculateHeuristic(startNode, nextNodeConsidered):
    locationContainingFile = open("locations.txt", "r")
    first_location = []
    second_location = []
    for textLine in locationContainingFile:
        words = textLine.split()
        if words[0] == startNode: 
            first_location.extend(words)
            continue
        if words[0] == nextNodeConsidered: 
            second_location.extend(words)
            continue
    
    latitudeDifference = float(second_location[1]) - float(first_location[1])
    longitudeDifference = float(second_location[2]) - float(first_location[2])
    distanceSqured = pow(latitudeDifference, 2) + pow(longitudeDifference, 2)
    actualDistance = math.sqrt(distanceSqured)
    return actualDistance

  
# A* Star Search Algorithm Implementation
def A_Star_search_algorithm(startNode, destinationNode):
    visitedNodes = []
    unVisitedNodes = []
    listOfNodesInPath = []
    for node in graph.nodes:
        unVisitedNodes.append(node)

    distanceDict = {node: float(inf) for node in graph.nodes}
    distanceDict[startNode] = 0
    pathDict = {}
    distanceFromStart = queue.PriorityQueue()
    distanceFromStart.put((0, startNode))

    while distanceFromStart.qsize() != 0:
        currentNode = distanceFromStart.get()
        for e in graph.edges:
            if e[0] == currentNode[1]:
                if e[1] not in visitedNodes:
                    distanceFromStart.put((int(e[2]), e[1]))
                    calculatedDistance = distanceDict[currentNode[1]] + int(e[2]) + distanceCalculateHeuristic(startNode, e[1])
                    if calculatedDistance < distanceDict[e[1]]:
                        distanceDict[e[1]] = calculatedDistance
                        if e[1] in pathDict.keys():
                            pathDict[e[1]] = currentNode[1]
                        else:
                            pathDict.update({e[1]: currentNode[1]}) 
            elif e[1] == currentNode[1]:
                if e[0] not in visitedNodes:
                    distanceFromStart.put((int(e[2]), e[0]))
                    calculatedDistance = distanceDict[currentNode[1]] + int(e[2]) + distanceCalculateHeuristic(startNode, e[0])
                    if calculatedDistance < distanceDict[e[0]]:
                        distanceDict[e[0]] = calculatedDistance
                        if e[0] in pathDict.keys():
                            pathDict[e[0]] = currentNode[1]
                        else:
                            pathDict.update({e[0]: currentNode[1]})
        
        visitedNodes.append(currentNode[1])
    inputDestinationNode = destinationNode
    while destinationNode != startNode:
        for keyNode in pathDict.keys():
            if keyNode == destinationNode:
                listOfNodesInPath.append(pathDict[keyNode])
                destinationNode = pathDict[keyNode]
    listOfNodesInPath.reverse()
    listOfNodesInPath.append(inputDestinationNode)
    return listOfNodesInPath


# running dfs for every possible node combination
def dfa_run_for_all():
    path_length = 0
    for intial in graph.nodes:
        for goal in graph.nodes:
            pathlist = depthFirstSearch(intial, goal)
            path_length += len(pathlist)
    return path_length/400

# running bfs for every possible ndoe combination
def bfa_run_for_all():
    path_length = 0
    for intial in graph.nodes:
        for goal in graph.nodes:
            pathlist = bredthFirstSearch(intial, goal)
            path_length += len(pathlist)
    return path_length/400

# running dijkstra for every possible node combination
def dijkstra_run_for_all():
    path_length = 0
    for start in graph.nodes:
        for destination in graph.nodes:
            pathlist = DijkstraAlgorithm(start, destination)
            path_length += len(pathlist)
    return path_length/400 

# running A* search for every possible ndoe combination
def A_Star_run_for_all():
    path_length = 0
    for start in graph.nodes:
        for destination in graph.nodes:
            pathlist = A_Star_search_algorithm(start, destination)
            path_length += len(pathlist)
    return path_length/400

#measuring time for time graph drawing
time_taken_by_dfa = timeit.timeit(stmt= dfa_run_for_all, number= 1)
time_taken_by_bfa = timeit.timeit(stmt= bfa_run_for_all, number= 1)
time_taken_by_dijkstra = timeit.timeit(stmt= dijkstra_run_for_all, number= 1)
time_taken_by_A_Star = timeit.timeit(stmt= A_Star_run_for_all, number= 1)


dfa_pathlen = dfa_run_for_all()
bfa_pathlen = bfa_run_for_all()
dijksra_pathlen = dijkstra_run_for_all()
A_Star_pathlen = A_Star_run_for_all()

#grap drawing function for time
def drawGraph_for_time():
    searchAlgorithms = ["DFS", "BFS", "Dijkstra Algoritm", "A* Search"]

    left = [1, 2, 3, 4]
    timeOfEachAlgorithm = [time_taken_by_dfa, time_taken_by_bfa, time_taken_by_dijkstra, time_taken_by_A_Star]
    plot.bar(left, timeOfEachAlgorithm, tick_label = searchAlgorithms, width= 0.5, color = ['blue', 'green'])

    plot.xlabel('Search Algorithms')
    plot.ylabel('Time Taken')

    plot.title('Time graph')
    plot.show()

# graph drawing function for path length
def drawGraph_for_pathlen():
    searchAlgorithms = ["DFS", "BFS", "Dijkstra Algoritm", "A* Search"]
    left = [1, 2, 3, 4]

    pathlengths = [dfa_pathlen, bfa_pathlen, dijksra_pathlen, A_Star_pathlen]
    plot.bar(left, pathlengths, tick_label = searchAlgorithms, width= 0.7, color =['blue', 'green'])

    plot.xlabel('Search Algorithms')
    plot.ylabel('Average path length')

    plot.title('Path Length Graph')
    plot.show()

# AFTER THIS IS THE GROUP WORK IMPLEMENTATION

#DEGREE CALCULATING
def calculate_degree(node):
    DegreeOfNode = 0
    for edge in graph.edges:
        if edge[0] == node or edge[1] == node:
            DegreeOfNode += 1
    return DegreeOfNode

#CLOSENESS CALCULATING
def calculate_closseness(beginNode):
    total_dij_pathlen = 0
    total_a_star_pathlen = 0
    for node in graph.nodes:
        d_shortPathList = DijkstraAlgorithm(beginNode, node)
        total_dij_pathlen += len(d_shortPathList)
        A_shortPathList = A_Star_search_algorithm(beginNode, node)
        total_a_star_pathlen += len(A_shortPathList)
    dijkstra_closseness = len(graph.nodes)/total_dij_pathlen
    A_Star_closseness = len(graph.nodes)/ total_a_star_pathlen
    return dijkstra_closseness, A_Star_closseness

#BETWEENESS CALCULATING
def calculate_betweeness(inputNode):
    dij_all_shortest_paths = []
    astar_all_shortest_paths = []
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            d_shortPathList = DijkstraAlgorithm(node1, node2)
            a_shortPathList = A_Star_search_algorithm(node1, node2)
            dij_all_shortest_paths.append(d_shortPathList)
            astar_all_shortest_paths.append(a_shortPathList)
    
    d_numOf_node_appearance = 0
    for pathlist in dij_all_shortest_paths:
        for element in pathlist:
            if element == inputNode:
                d_numOf_node_appearance += 1
    a_numOf_node_appearance = 0
    for pathlist in astar_all_shortest_paths:
        for element in pathlist:
            if element == inputNode:
                a_numOf_node_appearance += 1

    
    return d_numOf_node_appearance, a_numOf_node_appearance


# DRAWING THE GRAPH OF EACH OF THE ABOVE CENTRALITIES RESPECTIVELY   
def draw_graph_for_degree():
    cities = []
    city_degree = []
    for node in graph.nodes:
        cal_degree = calculate_degree(node)
        cities.append(node)
        city_degree.append(cal_degree)
    plot.bar(cities, city_degree,color =['blue', 'green'])

    plot.xlabel('Cities')
    plot.ylabel('Degree')

    plot.title('Degree graph')
    plot.xticks(rotation = 90)
    plot.show()

def draw_graph_for_closseness():
    cities = []
    city_closseness = []
    for node in graph.nodes:
        cal_closeness = calculate_closseness(node)
        cities.append(node)
        city_closseness.append(cal_closeness[0])
    plot.bar(cities, city_closseness,color =['blue', 'green'])

    plot.xlabel('Cities')
    plot.ylabel('Closeness')

    plot.title('Closeness graph')
    plot.xticks(rotation = 90)
    plot.show()

def draw_graph_for_betweeness():
    cities = []
    city_betweeness = []
    for node in graph.nodes:
        cal_betweeness = calculate_betweeness(node)
        cities.append(node)
        city_betweeness.append(cal_betweeness[1])
    plot.bar(cities, city_betweeness,color =['blue', 'green'])

    plot.xlabel('Cities')
    plot.ylabel('Betweeness')

    plot.title('Betweeness graph')
    plot.xticks(rotation = 90)
    plot.show()


shortPath = DijkstraAlgorithm("Arad", "Urziceni")
print(shortPath)




    


