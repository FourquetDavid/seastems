'''
Created on 15 nov. 2012

@author: David
inspired by Telmo Menezes's work : telmomenezes.com
'''

import random
import numpy  as np
import operator as op
import math
import graph_types.Directed_WeightedGWU as dwgwu
import graph_types.Undirected_WeightedGWU as uwgwu
import graph_types.Directed_UnweightedGWU as dugwu
import graph_types.Undirected_UnweightedGWU as uugwu

""" 
contains one main function :

*grow_network : takes a tree of decision and returns the graph that grows according to those rules
    Tree : consider an edge, returns a real number
        max depth : 3. 
        each leaf can be one of :
        "OrigId"  "TargId" "OrigInDegree" "OrigOutDegree"  "TargInDegree"  "TargOutDegree" 
        "OrigInStrength" "OrigOutStrength"  "TargInStrength"  "TargOutStrength" 
        "DirectDistance" "ReversedDistance"
        each node can be one of : + - * / exp log abs min max opp inv
    
    Growth algorithm :
        begins with an empty graph with the number of nodes  of the real network
        at each step the tree gives a probability to every possible edge
        one edge is chosen randomly
        until the number of edges equals the number of edges of the real network
        
    

"""

#avoid decorators syntax problems for line_profiling
import __builtin__
try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile

def grow_network(decision_tree, number_of_nodes, number_of_steps, graph=None):
    '''takes a tree of decision and returns the graph that grows according to those rules'''
    ''' graph has directed edges'''
    '''depending on the method used, this graph can have weighted or unweighted edges'''

    tree_type = decision_tree.getParam("tree_type")
    '''
    if tree_type == "simple" :
        return grow_simple_network(graph,decision_tree,number_of_nodes, number_of_edges)
        '''
    if tree_type == "with_constants":
        return grow_network_with_constants(decision_tree, number_of_nodes, number_of_steps, graph)
        # if tree_type == "with_constants_multi":
        # return grow_network_with_constants_multi_step(decision_tree, number_of_nodes, number_of_steps, graph)

    raise Exception("no tree_type given")


'''
Functions that grow a network according to the method of evaluation used
'''


# getattr ?
def createGraph(network_type, initial_network=None):
    if network_type == "directed_weighted":
        return dwgwu.Directed_WeightedGWU(initial_network)
    if network_type == "undirected_weighted":
        return uwgwu.Undirected_WeightedGWU(initial_network)
    if network_type == "directed_unweighted":
        return dugwu.Directed_UnweightedGWU(initial_network)
    if network_type == "undirected_unweighted":
        return uugwu.Undirected_UnweightedGWU(initial_network)
    raise Exception("network_type not given")

    '''
def grow_simple_network(graph,decision_tree,number_of_nodes, number_of_edges):
    takes a tree of decision and returns the graph that grows according to those rules
    graph can be (un)directed/(un)weighted

    #begins with an empty graph with the number of nodes  of the real network
    for i in xrange(number_of_nodes) :
        graph.add_node(i)
    #adds one edge according to its probability
    for i in xrange(number_of_edges) :
        #each edge has a probability that is the result of the tree
        probas = calc(decision_tree.getRoot(),graph)
        #we remove unnecessary edges : self loops, negative proba
        #we choose one among remaining ones
        edge,_ = choose_edge(probas, graph)
        if edge is None : #this can happen if every edge has a -infinity probability thanks to log or / or - exp...
            break
        graph.add_edge(*edge)
        
    return graph
'''


@profile
def grow_network_with_constants(decision_tree, number_of_nodes, number_of_edges, graph=None):
    '''takes a tree of decision and returns the graph that grows according to those rules'''
    '''graph can be (un)directed/(un)weighted'''
    network_type = decision_tree.getParam('network_type')


    if graph is None:
        graph = createGraph(network_type)

    if decision_tree.getParam("edge_data") is not None:
        graph.add_edge_data(decision_tree.getParam("edge_data"))
        graph.add_n_edge_data(decision_tree.getParam("n_edge_data"))

    if decision_tree.getParam("node_data") is not None:
        graph.add_node_data(decision_tree.getParam("node_data"))

    number_of_nodes_init = graph.number_of_nodes()
    for i in range(number_of_nodes):
        graph.add_node(i + number_of_nodes_init)
    # adds one edge according to its probability
    number_of_steps = 1
    old_probas = None
    while graph.number_of_edges() < number_of_edges:
        # each edge has a probability that is the result of the tree
        probas = calc_with_constants(decision_tree.getRoot(), graph)
        # if probas stay near from last step, we dobble the number of new edges created
        if near(probas, old_probas):
            number_of_steps *= 2
        else:
            number_of_steps = max(1, number_of_steps / 2)
        old_probas = probas
        list_edges = []
        for _ in range(number_of_steps):
            # we remove unnecessary edges : self loops, negative proba
            # we choose one among remaining ones
            edge, weight_value = choose_edge(probas, graph)
            if edge is None:  # this can happen if every edge has a -infinity probability thanks to log or / or - exp...
                break

            source, target = edge
            list_edges.append((source, target, {"weight": weight_value}))

        if len(
                list_edges) == 0:  # this can happen if every edge has a -infinity probability thanks to log or / or - exp...
            break
        graph.add_edges_from(list_edges)
    return graph


def near(probas, old_probas):
    if old_probas is None: return False
    try:
        diff = np.absolute(probas - old_probas)
        maxdiff = np.max(diff[np.isfinite(diff)])
        pb = np.absolute(probas)
        threshhold = np.max(pb[np.isfinite(pb)])
        if maxdiff < threshhold / 2: return True
    except ValueError:
        return False


"""
def grow_network_with_constants_multi_step(decision_tree, number_of_nodes, number_of_steps, graph=None):
    '''takes a tree of decision and returns the graph that grows according to those rules'''
    '''graph can be (un)directed/(un)weighted'''
    network_type = decision_tree.getParam('network_type')
    graph = createGraph(network_type)
    for i in range(number_of_nodes):
        graph.add_node(i)
    # adds one edge according to its probability
    for _ in range(10):
        # each edge has a probability that is the result of the tree

        # we remove unnecessary edges : self loops, negative proba
        # we choose one among remaining ones

        probas = calc_with_constants(decision_tree.getRoot(), graph)
        list_weighted_edges = choose_edges(probas, graph, number_of_steps / 10)
        for edge, weight_value in list_weighted_edges:
            if edge is None:  # this can happen if every edge has a -infinity probability thanks to log or / or - exp...
                break
            if graph.isWeighted():
                graph.add_edge(*edge, weight=weight_value)
            else:
                graph.add_edge(*edge)

    return graph
"""

'''
Functions that let us choose a random element in the matrix of probabilities
'''


@profile
def choose_edge(probas, network):
    ''' takes a matrix of probabilities and a network, 
    returns an edge (no self loop, not already present in the network) according to probabilities and its weight for the network'''
    '''the returned weight is (1+erf(proba)) /2 : because this function takes a number in R and return a number between 0 and 1'''

    # probas can contain a number + infinity, -inifinity, nan
    coord_i = np.random.randint(0, network.number_of_nodes(), network.number_of_nodes())
    coord_j = np.random.randint(0, network.number_of_nodes(), network.number_of_nodes())

    liste_probas = zip(zip(coord_i, coord_j), probas[coord_i, coord_j])

    # we list possible edge : no self loops, no existing edges, no negative probabilities
    edge = network.has_edge
    possible_edges = [x for x in liste_probas if x[1] > float('-inf') and x[0][0] != x[0][1] and not edge(*x[0])]


    # if there is no possible edge, we stop the building of the network
    if len(possible_edges) == 0:
        return (None, 0)

    # we list edges with strictly positive probabilities
    positive_edges = [x for x in possible_edges if x[1] > 0]
    # if every probability is negative, we choose one edge among the possible
    if len(positive_edges) == 0:
        edge, weight = random.choice(possible_edges)
        return (edge, normalize(weight))

    # we list edges with infinite probabilities
    infinite_edges = [x for x in positive_edges if x[1] == float('+inf')]
    # if some probability are infinite, we choose one edge among the inifinite probabilities
    if len(infinite_edges) != 0:
        weighted_edge = random.choice(infinite_edges)
        return (weighted_edge[0], 1)

    # if there is one positive probability, we choose one edge between those with positive probability
    weights_sum = sum(weighted_edge[1] for weighted_edge in positive_edges)
    rand = random.random() * weights_sum
    for edge, weight in positive_edges:
        rand -= weight
        if rand <= 0:
            return (edge, normalize(weight))
    # if weights_sum = +infty but probabilities are different from + infinity,
    # we can have this possibility
    return random.choice(possible_edges)


"""
def choose_edges(probas, network, number):
    ''' takes a matrix of probabilities and a network, 
    returns N edges (no self loop, not already present in the network) according to probabilities and its weight for the network'''
    '''the returned weight is (1+erf(proba)) /2 : because this function takes a number in R and return a number between 0 and 1'''
    # we mark impossible edge, because it faster to remove them this way instead of filtering the matrix enumerated
    # finding edges in matrices is in constant time 
    # finding edges in sequences is in linear time

    # gives -infinity as probability to self loops
    np.fill_diagonal(probas, float('-inf'))
    # gives -infinity as probability to already existing edges
    if network.isDirected():
        for edge in network.edges_iter():
            probas[edge] = float('-inf')
    else:
        'because edges are only stored once : begin-end and not end-begin'
        for target, origin in network.edges_iter():
            probas[origin, target] = float('-inf')
            probas[target, origin] = float('-inf')

    # probas can contain a number + infinity, -inifinity, nan
    liste_probas = sample(probas,network.number_of_nodes())
    # we list possible edge : no self loops, no existing edges, no negative probabilities
    possible_edges = [x for x in liste_probas if x[1] > float('-inf')]

    # we list edges with strictly positive probabilities
    positive_edges = [x for x in possible_edges if x[1] > 0]

    # we list edges with infinite probabilities
    infinite_edges = [x for x in positive_edges if x[1] == float('+inf')]
    edges_result = []
    for _ in range(number):
        weighted_edge = choose_edge_among(possible_edges, positive_edges, infinite_edges)
        edges_result.append(weighted_edge)
        try:
            possible_edges.remove(weighted_edge)
            print("possible")
            positive_edges.remove(weighted_edge)
            print("positive")
            infinite_edges.remove(weighted_edge)
            print("infinite")
        except:
            pass
    return edges_result
"""


def choose_edge_among(possible_edges, positive_edges, infinite_edges):
    ''' possible_edges and positive_edges and 
    returns an edge (no self loop, not already present in the network) according to probabilities and its weight for the network'''
    '''the returned weight is (1+erf(proba)) /2 : because this function takes a number in R and return a number between 0 and 1'''

    if len(possible_edges) == 0:
        return (None, 0)

    # if every probability is negative, we choose one edge among the possible
    if len(positive_edges) == 0:
        edge, weight = random.choice(possible_edges)
        return (edge, normalize(weight))

    # if some probability are infinite, we choose one edge among the inifinite probabilities
    if len(infinite_edges) != 0:
        weighted_edge = random.choice(infinite_edges)
        return (weighted_edge[0], 1)

    # if there is one positive probability, we choose one edge between those with positive probability
    weights_sum = sum(weighted_edge[1] for weighted_edge in positive_edges)
    rand = random.random() * weights_sum
    for edge, weight in positive_edges:
        rand -= weight
        if rand <= 0:
            return (edge, normalize(weight))
    # if weights_sum = +infty but probabilities are different from + infinity,
    # we can have this possibility
    return random.choice(possible_edges)


'''
Functions that compute the tree for each node
'''


def calc(node, graph):
    ''' takes a node of the decision tree and a graph
    computes recursively a value for each edge of the graph at the same time
    a node can be a leaf with a variable or a function
    returns a 2D array containing the value for each edge
    '''

    data = node.getData()

    # recursive computation on function nodes : we always have 2 children if not a leaf, by construction
    if node.isLeaf():
        return compute_leaf(graph, data)

    else:
        # values returned are arrays of dimension 2
        value0 = calc(node.getChild(0), graph)
        value1 = calc(node.getChild(1), graph)
        return compute_function(data, value0, value1)


def calc_with_constants(node, graph):
    ''' takes a node of the decision tree and a graph
    computes recursively a value for each edge of the graph at the same time
    returns a 2D array containing the value for each edge
    difference is that leaves of the tree contain a constant and a variable
    '''

    data = node.getData()
    # recursive computation on function nodes : we always have 2 children if not a leaf, by construction
    if node.isLeaf():
        constant, variable = data
        return constant * (compute_leaf(graph, variable))

    else:

        # values returned are arrays of dimension 2
        if data in ["H", "opp", "T", "inv", "exp", "abs", "log"]:
            value0 = calc_with_constants(node.getChild(0), graph)
            value1 = None
        else:
            value0 = calc_with_constants(node.getChild(0), graph)
            value1 = calc_with_constants(node.getChild(1), graph)
        return compute_function(data, value0, value1)


def compute_function(data, value0, value1):
    '''returns the computation of value0 data value1 
    data is an operation between two numbers
    '''
    return {
        "+": op.add,
        "-": op.sub,
        "*": op.mul,
        "min": np.minimum,
        "max": np.maximum,
        "exp": exp,
        "log": log,
        "abs": abs,
        "/": div,
        "inv": inv,
        "opp": opp,
        "H": H,
        "T": T,
        "N": N,
        ">": greater,
        "<": less,
        "=": around
    }[data](value0, value1)


def compute_leaf(graph, variable):
    """returns the computation of value0 data value1
    data is an operation between two numbers
    """
    return getattr(graph, variable)()


def div(a, b):
    'divides by 1+b to avoid dividing by 0 in most cases'
    return a / (1 + b)


def exp(a, b):
    return np.exp(a)


def log(a, b):
    'computes log(1+a) to avoid most cases where a = 0'
    return np.log(1 + a)


def abs(a, b):
    return np.absolute(a)


def inv(a, b):
    return 1 / (1 + a)


def opp(a, b):
    return -a


def H(a, b):
    # only on a , 1 if a > 0, else 0
    return (a >= 0).astype(float)


def T(a, b):
    # th(a)+1 / 2
    return (np.tanh(a) + 1) / 2


def N(a, b):
    return np.exp(-a ** 2)


def greater(a, b):
    '1 if a >b element-wise'
    return (a >= b).astype(float)


def less(a, b):
    '1 if a <b element-wise'
    return (a < b).astype(float)


def around(a, b):
    '1 if b-1<a < b+1 element-wise'
    return (np.absolute(a - b) < 1).astype(float)


def normalize(x):
    return (math.tanh(x) + 1) / 2
