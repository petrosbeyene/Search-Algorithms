class Node:
    def __init__(self, value):
        self.value = value

class Edge:
    def __init__(self, origin, destination, edgeWeight):
        self.origin = origin
        self.destination = destination
        self.edgeWeight = edgeWeight

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    
    def addNodeToGraph(self, node):
        self.nodes.add(node.value)


    def addEdgeTograph(self, origin, destination, edgeWeight):
        
        e = Edge(origin, destination, edgeWeight)
        edgeTuple = [(e.origin.value, e.destination.value, edgeWeight)]
        self.edges.update(edgeTuple)
        


