import json
import copy

multidomain_topo = {
"graph": {

    "nodes": [
        {"id": 0, "name": "a1", "domain": "A"}, 
        {"id": 1, "name": "a2", "domain": "A"}, 
        {"id": 2, "name": "a3", "domain": "A"}, 
        {"id": 3, "name": "a4", "domain": "A"}, 
        {"id": 4, "name": "b1", "domain": "B"}, 
        {"id": 5, "name": "b2", "domain": "B"}, 
        {"id": 6, "name": "b3", "domain": "B"}, 
        {"id": 7, "name": "b4", "domain": "B"}, 
        {"id": 8, "name": "c1", "domain": "C"}, 
        {"id": 9, "name": "c2", "domain": "C"}, 
        {"id": 10, "name": "c3", "domain": "C"}, 
        {"id": 11, "name": "c4", "domain": "C"}, 
        {"id": 12, "name": "c5", "domain": "C"}, 
        {"id": 13, "name": "d1", "domain": "D"}, 
        {"id": 14, "name": "d2", "domain": "D"},
        {"id": 15, "name": "d3", "domain": "D"},
        {"id": 16, "name": "d4", "domain": "D"},
        {"id": 17, "name": "d5", "domain": "D"},
        {"id": 18, "name": "e1", "domain": "E"},
        {"id": 19, "name": "e2", "domain": "E"},
        {"id": 20, "name": "e3", "domain": "E"},
        {"id": 21, "name": "e4", "domain": "E"},
        {"id": 22, "name": "e5", "domain": "E"}],
    #   
    "links": [
        {"weight": 10, "source": 0, "target": 1, "domain":"A"}, 
        {"weight": 20, "source": 0, "target": 2, "domain":"A"}, 
        {"weight": 3, "source": 1, "target": 2, "domain":"A"},
        {"weight": 40, "source": 1, "target": 4, "domain":"AB"},
        {"weight": 50, "source": 2, "target": 3, "domain":"A"}, 
        {"weight": 34, "source": 3, "target": 4, "domain":"AB"}, 
        {"weight": 22, "source": 3, "target": 6, "domain":"AB"}, 
        {"weight": 21, "source": 4, "target": 5, "domain":"B"}, 
        {"weight": 3, "source": 4, "target": 6, "domain":"B"}, 
        {"weight": 2, "source": 4, "target": 7, "domain":"B"}, 
        {"weight": 7, "source": 5, "target": 8, "domain":"BC"}, 
        {"weight": 9, "source": 5, "target": 10, "domain":"BC"}, 
        {"weight": 19, "source": 6, "target": 7, "domain":"B"}, 
        {"weight": 34, "source": 7, "target": 12, "domain":"BC"}, 
        {"weight": 36, "source": 8, "target": 9, "domain":"C"}, 
        {"weight": 78, "source": 8, "target": 10, "domain":"C"}, 
        {"weight": 90, "source": 9, "target": 11, "domain":"C"}, 
        {"weight": 65, "source": 9, "target": 14, "domain":"CD"}, 
        {"weight": 67, "source": 10, "target": 11, "domain":"C"}, 
        {"weight": 21, "source": 11, "target": 12, "domain":"C"}, 
        {"weight": 23, "source": 11, "target": 17, "domain":"CD"}, 
        {"weight": 32, "source": 13, "target": 14, "domain":"D"}, 
        {"weight": 45, "source": 13, "target": 15, "domain":"D"}, 
        {"weight": 78, "source": 14, "target": 15, "domain":"D"}, 
        {"weight": 12, "source": 14, "target": 16, "domain":"D"}, 
        {"weight": 23, "source": 15, "target": 18, "domain":"DE"}, 
        {"weight": 8, "source": 16, "target": 17, "domain":"D"}, 
        {"weight": 4, "source": 16, "target": 20, "domain":"DE"}, 
        {"weight": 49, "source": 18, "target": 19, "domain":"E"},
        {"weight": 67, "source": 18, "target": 21, "domain":"E"},
        {"weight": 61, "source": 19, "target": 22, "domain":"E"},
        {"weight": 52, "source": 20, "target": 22, "domain":"E"},
        {"weight": 38, "source": 21, "target": 22, "domain":"E"}]
        }
}


multidomain_topo_46 = {
"graph": {

    "nodes": [
        {"id": 0, "name": "a1", "domain": "A"},{"id": 1, "name": "a2", "domain": "A"},
        {"id": 2, "name": "a3", "domain": "A"}, {"id": 3, "name": "a4", "domain": "A"}, 
        {"id": 4, "name": "b1", "domain": "B"},{"id": 5, "name": "b2", "domain": "B"},
        {"id": 6, "name": "b3", "domain": "B"},{"id": 7, "name": "b4", "domain": "B"}, 
        {"id": 8, "name": "c1", "domain": "C"},{"id": 9, "name": "c2", "domain": "C"},
        {"id": 10, "name": "c3", "domain": "C"},{"id": 11, "name": "c4", "domain": "C"},
        {"id": 12, "name": "c5", "domain": "C"}, 
        {"id": 13, "name": "d1", "domain": "D"},{"id": 14, "name": "d2", "domain": "D"},
        {"id": 15, "name": "d3", "domain": "D"},{"id": 16, "name": "d4", "domain": "D"},
        {"id": 17, "name": "d5", "domain": "D"},
        {"id": 18, "name": "e1", "domain": "E"},{"id": 19, "name": "e2", "domain": "E"},
        {"id": 20, "name": "e3", "domain": "E"},{"id": 21, "name": "e4", "domain": "E"},
        {"id": 22, "name": "e5", "domain": "E"},
        
        {"id": 23, "name": "f1", "domain": "F"},{"id": 24, "name": "f2", "domain": "F"},
        {"id": 25, "name": "f3", "domain": "F"}, {"id": 26, "name": "f4", "domain": "F"}, 
        {"id": 27, "name": "g1", "domain": "G"},{"id": 28, "name": "g2", "domain": "G"},
        {"id": 29, "name": "g3", "domain": "G"},{"id": 30, "name": "g4", "domain": "G"}, 
        {"id": 31, "name": "h1", "domain": "H"},{"id": 32, "name": "h2", "domain": "H"},
        {"id": 33, "name": "h3", "domain": "H"},{"id": 34, "name": "h4", "domain": "H"},
        {"id": 35, "name": "h5", "domain": "H"}, 
        {"id": 36, "name": "i1", "domain": "I"},{"id": 37, "name": "i2", "domain": "I"},
        {"id": 38, "name": "i3", "domain": "I"},{"id": 39, "name": "i4", "domain": "I"},
        {"id": 40, "name": "i5", "domain": "I"},
        {"id": 41, "name": "j1", "domain": "J"},{"id": 42, "name": "j2", "domain": "J"},
        {"id": 43, "name": "j3", "domain": "J"},{"id": 44, "name": "j4", "domain": "J"},
        {"id": 45, "name": "j5", "domain": "J"}       
        ],
    #   
    "links": [
        {"weight": 10, "source": 0, "target": 1, "domain":"A"}, 
        {"weight": 20, "source": 0, "target": 2, "domain":"A"}, 
        {"weight": 3, "source": 1, "target": 2, "domain":"A"},
        {"weight": 40, "source": 1, "target": 4, "domain":"AB"},
        {"weight": 50, "source": 2, "target": 3, "domain":"A"}, 
        {"weight": 34, "source": 3, "target": 4, "domain":"AB"}, 
        {"weight": 22, "source": 3, "target": 6, "domain":"AB"}, 
        {"weight": 21, "source": 4, "target": 5, "domain":"B"}, 
        {"weight": 3, "source": 4, "target": 6, "domain":"B"}, 
        {"weight": 2, "source": 4, "target": 7, "domain":"B"}, 
        {"weight": 7, "source": 5, "target": 8, "domain":"BC"}, 
        {"weight": 9, "source": 5, "target": 10, "domain":"BC"}, 
        {"weight": 19, "source": 6, "target": 7, "domain":"B"}, 
        {"weight": 34, "source": 7, "target": 12, "domain":"BC"}, 
        {"weight": 36, "source": 8, "target": 9, "domain":"C"}, 
        {"weight": 78, "source": 8, "target": 10, "domain":"C"}, 
        {"weight": 90, "source": 9, "target": 11, "domain":"C"}, 
        {"weight": 65, "source": 9, "target": 14, "domain":"CD"}, 
        {"weight": 67, "source": 10, "target": 11, "domain":"C"}, 
        {"weight": 21, "source": 11, "target": 12, "domain":"C"}, 
        {"weight": 23, "source": 11, "target": 17, "domain":"CD"}, 
        {"weight": 32, "source": 13, "target": 14, "domain":"D"}, 
        {"weight": 45, "source": 13, "target": 15, "domain":"D"}, 
        {"weight": 78, "source": 14, "target": 15, "domain":"D"}, 
        {"weight": 12, "source": 14, "target": 16, "domain":"D"}, 
        {"weight": 23, "source": 15, "target": 18, "domain":"DE"}, 
        {"weight": 8, "source": 16, "target": 17, "domain":"D"}, 
        {"weight": 4, "source": 16, "target": 20, "domain":"DE"}, 
        {"weight": 49, "source": 18, "target": 19, "domain":"E"},
        {"weight": 67, "source": 18, "target": 21, "domain":"E"},
        {"weight": 61, "source": 19, "target": 22, "domain":"E"},
        {"weight": 52, "source": 20, "target": 22, "domain":"E"},
        {"weight": 38, "source": 21, "target": 22, "domain":"E"},

        {"weight": 15, "source": 19, "target": 23, "domain":"EF"},
        {"weight": 19, "source": 21, "target": 25, "domain":"EF"},  

        {"weight": 10, "source": 23, "target": 24, "domain":"F"}, 
        {"weight": 20, "source": 23, "target": 25, "domain":"F"}, 
        {"weight": 3, "source": 24, "target": 25, "domain":"F"},
        {"weight": 40, "source": 24, "target": 27, "domain":"FG"},
        {"weight": 50, "source": 25, "target": 26, "domain":"F"}, 
        {"weight": 34, "source": 26, "target": 27, "domain":"FG"}, 
        {"weight": 22, "source": 26, "target": 29, "domain":"FG"}, 
        {"weight": 21, "source": 27, "target": 28, "domain":"G"}, 
        {"weight": 3, "source": 27, "target": 29, "domain":"G"}, 
        {"weight": 2, "source": 27, "target": 30, "domain":"G"}, 
        {"weight": 7, "source": 28, "target": 31, "domain":"GH"}, 
        {"weight": 9, "source": 28, "target": 33, "domain":"GH"}, 
        {"weight": 19, "source": 29, "target": 30, "domain":"G"}, 
        {"weight": 34, "source": 30, "target": 35, "domain":"GH"}, 
        {"weight": 36, "source": 31, "target": 32, "domain":"H"}, 
        {"weight": 78, "source": 31, "target": 33, "domain":"H"}, 
        {"weight": 90, "source": 32, "target": 34, "domain":"H"}, 
        {"weight": 65, "source": 32, "target": 37, "domain":"HI"}, 
        {"weight": 67, "source": 33, "target": 34, "domain":"H"}, 
        {"weight": 21, "source": 34, "target": 35, "domain":"H"}, 
        {"weight": 23, "source": 34, "target": 40, "domain":"HI"}, 
        {"weight": 32, "source": 35, "target": 37, "domain":"I"}, 
        {"weight": 45, "source": 35, "target": 38, "domain":"I"}, 
        {"weight": 78, "source": 37, "target": 38, "domain":"I"}, 
        {"weight": 12, "source": 37, "target": 39, "domain":"I"}, 
        {"weight": 23, "source": 38, "target": 41, "domain":"IJ"}, 
        {"weight": 8, "source": 39, "target": 40, "domain":"I"}, 
        {"weight": 4, "source": 39, "target": 43, "domain":"IJ"}, 
        {"weight": 49, "source": 41, "target": 42, "domain":"J"},
        {"weight": 67, "source": 41, "target": 44, "domain":"J"},
        {"weight": 61, "source": 42, "target": 45, "domain":"J"},
        {"weight": 52, "source": 43, "target": 45, "domain":"J"},
        {"weight": 38, "source": 44, "target": 45, "domain":"J"}


        ]
        


        }
}


def get_topo(name):
    topo = Topology()
    if name == "multidomain_topo":
        topo.set_graph(copy.deepcopy(multidomain_topo["graph"]))
    elif name == "multidomain_topo_46":
        topo.set_graph(copy.deepcopy(multidomain_topo_46["graph"]))
    else:
        return "topology not found"
        
    return topo 

class Topology:
    def __init__(self):
        # self.id=ids
        self.graph = {}

    def set_graph(self,graph):
        self.graph = graph


