'''
Created on 5 fevr. 2013

@author: davidfourquet
inspired by Telmo Menezes's work : telmomenezes.com
'''
"""
this class inherits from networkx.DiGraph and GraphWithUpdate. It stores a distance matrix and some global variables about the network. 
It allows us to update them easily instead of computing them many times.
       

"""
import networkx as nx
import igraph as ig
import numpy as np
import community as com
import GraphWithUpdate as gwu
import collections


class Undirected_UnweightedGWU(gwu.GraphWithUpdate, nx.Graph):
    #avoid decorators syntax problems for line_profiling
    import __builtin__
    try:
        __builtin__.profile
    except AttributeError:
        # No line profiler, provide a pass-through version
        def profile(func):
            return func

        __builtin__.profile = profile

    def __init__(self, graph=None):
        """ The creator of UndirectedUnweightedGraphWithUpdate Class """

        nx.Graph.__init__(self, graph)
        self.i_graphe = ig.Graph()
        self.shortest_path_dict = None
        self.shortest_path_matrix = None
        self.max_degree = None
        self.max_distance = None

    def add_edge_data(self,edge_data):
        self.edge_data =np.asarray(edge_data)

    def add_n_edge_data(self, n_edge_data):
        self.n_edge_data = np.asarray(n_edge_data)

    def add_node_data(self,node_data):
        self.targ_node_data =np.outer(np.ones(len(node_data),dtype=float), node_data)


        self.orig_node_data = np.outer(node_data, np.ones(len(node_data),dtype=float))


        self.n_targ_node_data = self.targ_node_data/np.max(self.targ_node_data)

        self.n_orig_node_data = self.orig_node_data/np.max(self.orig_node_data)



    def add_node(self, n):
        nx.Graph.add_node(self, n, attr_dict=None)
        self.i_graphe.add_vertices(1)

    def add_edge(self, u, v, **args):
        nx.Graph.add_edge(self, u, v, args)
        self.i_graphe.add_edges(u,v)

        # update info about the network : not really an update but a computation
        if self.shortest_path_dict is not None:
            self.shortest_path_dict = nx.shortest_path_length(self)
        if self.shortest_path_matrix is not None :
            self.shortest_path_matrix = np.array(self.i_graphe.shortest_paths_dijkstra())
        if self.max_degree is not None:
            self.max_degree = max(self.degree())
        if self.max_distance is not None:
            self.max_distance = max(max(self.get_shortest_path_dict().values()))
    @profile
    def add_edges_from(self,ebunch):
        nx.Graph.add_edges_from(self, ebunch)
        self.i_graphe.add_edges([(u,v) for u,v,w in ebunch])

        # update info about the network : not really an update but a computation
        if self.shortest_path_dict is not None:
            self.shortest_path_dict = nx.shortest_path_length(self)
        if self.shortest_path_matrix is not None:
            self.shortest_path_matrix = np.array(self.i_graphe.shortest_paths_dijkstra())
        if self.max_degree is not None:
            self.max_degree = max(self.degree())
        if self.max_distance is not None:
            self.max_distance = np.max(self.get_shortest_path_matrix())

    def isWeighted(self):
        return False

    def isDirected(self):
        return False

    def Targ(self, dictionnaire):
        result = np.outer(np.ones(self.number_of_nodes(),dtype=float), [dictionnaire[node] for node in self])
        return result

    def Orig(self, dictionnaire):
        result = np.outer([dictionnaire[node] for node in self], np.ones(self.number_of_nodes(),dtype=float))
        return result

    def OrigDegree(self):
        ''' returns a 2d array containing the  degree of the origin node for all edges
        '''
        return self.Orig(self.degree())

    def NormalizedOrigDegree(self):
        ''' returns a 2d array containing  degree of origin divided by max of in_degrees 
        '''
        m = max(self.degree())
        if m == 0: return self.OrigDegree()
        return self.OrigDegree() / m

    def OrigId(self):
        ''' returns a 2d array containing the identity number (0 to n=number of nodes) of the origin node for all edges
        '''
        return self.Orig(range(self.number_of_nodes()))


    def NormalizedOrigId(self):
        ''' returns a 2d array containing the identity number (0 to n=number of nodes) of the origin node for all edges divide by the total number of nodes
        '''

        return self.OrigId() / self.number_of_nodes()

    def EdgeData(self):
        return self.edge_data

    def NormEdgeData(self):
        return self.n_edge_data

    def TargNodeData(self):
        return self.targ_node_data

    def NormTargNodeData(self):
        return self.n_targ_node_data

    def OrigNodeData(self):
        return self.orig_node_data

    def NormOrigNodeData(self):
        return self.n_orig_node_data

    def TargDegree(self):
        ''' returns a 2d array containing the in degree of the target node for all edges
        '''
        return self.Targ(self.degree())

    def NormalizedTargDegree(self):
        ''' returns a 2d array containing the in degree of the target node for all edges divided by max of in_degrees
        '''
        m = max(self.degree())
        if m == 0: return self.TargDegree()
        return self.TargDegree() / self.get_max_degree()

    def TargId(self):
        ''' returns a 2d array containing the identity number of the target node for all edges
        '''
        return self.Targ(range(self.number_of_nodes()))

    def NormalizedTargId(self):
        ''' returns a 2d array containing the identity number of the target node for all edges divided by the number of nodes
        '''
        return self.TargId() / self.number_of_nodes()

    #@profile
    def OrigPagerank(self):
        ''' returns a 2d array containing the pagerank of the origin node for all edges

        probas = np.dot(
            np.array(nx.pagerank_scipy(self).values(), dtype=float).reshape(-1, 1),
            np.ones((1, self.number_of_nodes())))
        '''
        try :
            return self.Orig(self.i_graphe.pagerank())
        except :
            return self.Orig(np.ones(self.number_of_nodes(),dtype=float)/self.number_of_nodes())

    @profile
    def TargPagerank(self):
        ''' returns a 2d array containing the pagerank of the target node for all edges

        probas = np.dot(
            np.ones((self.number_of_nodes(), 1)),
            np.array(nx.pagerank_scipy(self).values(), dtype=float).reshape(1, -1)
        )
        '''
        try :
            return self.Targ(self.i_graphe.pagerank())
        except :
            return self.Targ(np.ones(self.number_of_nodes(),dtype=float)/self.number_of_nodes())



    #@profile
    def OrigCoreN(self):
        ''' returns a 2d array containing the pagerank of the origin node for all edges

        probas = np.dot(
            np.array(nx.core_number(self).values(), dtype=float).reshape(-1, 1),
            np.ones((1, self.number_of_nodes())))
        '''
        return self.Orig(nx.core_number(self))

    @profile
    def TargCoreN(self):
        """ returns a 2d array containing the pagerank of the target node for all edges

        probas = np.dot(
            np.ones((self.number_of_nodes(), 1)),
            np.array(nx.core_number(self).values(), dtype=float).reshape(1, -1)
        )
        """
        return self.Targ(nx.core_number(self))

    #@profile
    def OrigCloseness(self):
        """ returns a 2d array containing the closeness of the origin node for all edges

        probas = np.dot(
            np.array(nx.closeness_centrality(self).values(), dtype=float).reshape(-1, 1),
            np.ones((1, self.number_of_nodes())))
        """
        return self.Orig(nx.closeness_centrality(self))

    @profile
    def TargCloseness(self):
        ''' returns a 2d array containing the closeness of the target node for all edges

        probas = np.dot(
            np.ones((self.number_of_nodes(), 1)),
            np.array(nx.closeness_centrality(self).values(), dtype=float).reshape(1, -1)
        )
        '''
        return self.Targ(nx.closeness_centrality(self))


    def OrigBetweenness(self):
        ''' returns a 2d array containing the betweenness of the origin node for all edges

        probas = np.dot(
            np.array(nx.betweenness_centrality(self).values(), dtype=float).reshape(-1, 1),
            np.ones((1, self.number_of_nodes())))
        '''
        return np.outer(self.i_graphe.betweenness(),np.ones(self.number_of_nodes(),dtype=float))
        '''
        return self.Orig(nx.betweenness_centrality(self))
        '''

    @profile
    def TargBetweenness(self):
        ''' returns a 2d array containing the betweenness of the target node for all edges

        probas = np.dot(
            np.ones((self.number_of_nodes(), 1)),
            np.array(nx.betweenness_centrality(self).values(), dtype=float).reshape(1, -1)
        )
        '''
        return np.outer(np.ones(self.number_of_nodes(),dtype=float),self.i_graphe.betweenness())
        '''
        return self.Targ(nx.betweenness_centrality(self))
        '''

    @profile
    def OrigClustering(self):
        ''' returns a 2d array containing the clustering of the origin node for all edges

        probas = np.dot(
            np.array(nx.clustering(self).values(), dtype=float).reshape(-1, 1),
            np.ones((1, self.number_of_nodes())))
        '''
        return self.Orig(nx.clustering(self))

    @profile
    def TargClustering(self):
        ''' returns a 2d array containing the clustering of the target node for all edges

        probas = np.dot(
            np.ones((self.number_of_nodes(), 1)),
            np.array(nx.clustering(self).values(), dtype=float).reshape(1, -1)
        )
        '''
        return self.Targ(nx.clustering(self))

    @profile
    def OrigEccentricity(self):
        ''' returns a 2d array containing the eccentricity of the origin node for all edges
        '''
        sp = self.get_shortest_path_matrix()
        eccentricity = collections.defaultdict(lambda: float("inf"))
        for node in max(nx.connected_components(self), key=len):
            eccentricity[node] = max(sp[node])
            '''
        probas = np.dot(
            np.array(nx.eccentricity(self, sp=sp).values(), dtype=float).reshape(-1, 1),
            np.ones((1, self.number_of_nodes())))
            '''
        return self.Orig(eccentricity)

    @profile
    def TargEccentricity(self):
        ''' returns a 2d array containing the eccentricity of the target node for all edges
        '''
        sp = self.get_shortest_path_matrix()
        eccentricity = collections.defaultdict(lambda: float("inf"))
        for node in max(nx.connected_components(self), key=len):
            eccentricity[node] = max(sp[node])
            '''
        probas = np.dot(
            np.ones((self.number_of_nodes(), 1)),
            np.array(nx.eccentricity(self, sp=sp).values(), dtype=float).reshape(1, -1)
        )
        '''
        return self.Targ(eccentricity)

    @profile
    def SameCommunity(self):
        ''' returns a 2d array containing 1 when both nodes are in the same community'''
        partition = []
        if self.number_of_edges() > 3 :
            try :
                partition = com.best_partition(self)
            except ZeroDivisionError:
                print self.number_of_nodes(),self.number_of_edges()
        else :
            partition = range(self.number_of_nodes())

        probas = np.zeros((self.number_of_nodes(), self.number_of_nodes()))

        for node1 in partition:
            for node2 in partition:
                if partition[node1] == partition[node2]:
                    probas[node1, node2] = 1.

        return probas

    def Distance(self):
        ''' returns a 2d array containing the distance = shortest path length, takes weights into account'''
        ''' gives +infinity if no path'''
        '''
        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        # every path that does not exist has distance +infinity
        probas.fill(float('+inf'))

        for node1, row in self.get_shortest_path_dict().iteritems():
            for node2, length in row.iteritems():
                probas[node1, node2] = length

        return probas
        '''
        return self.get_shortest_path_matrix()
    @profile
    def CommonNeighbors(self):
        ''' returns a 2d array containing the number of common neighbours '''
        common = np.array(np.linalg.matrix_power(nx.to_numpy_matrix(self,dtype=float),2))
        return common
    @profile
    def Loop(self):
        ''' returns a 2d array containing 1 if nodes share a common neighbour, 0 else '''
        loop = np.sign(np.array(np.linalg.matrix_power(nx.to_numpy_matrix(self,dtype=float),2)))
        return loop

    def RevDistance(self):
        ''' returns a 2d array containing the distance = shortest path length, takes weights into account'''
        ''' gives +infinity if no path'''
        '''
        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        # every path that does not exist has distance +infinity
        probas.fill(float('+inf'))

        for node1, row in self.get_shortest_path_dict().iteritems():
            for node2, length in row.iteritems():
                probas[node2, node1] = length
        '''
        return np.transpose(self.get_shortest_path_matrix())


    def Reciprocity(self):
        '''returns a 2d array where a_i_j =1 if there is an edge from j to i'''
        return np.transpose(nx.to_numpy_matrix(self,dtype=float))

    def NormalizedDistance(self):
        ''' returns a 2d array containing the distance = shortest path length, takes weights into account'''
        ''' gives +infinity if no path'''
        ''' divides by distance maximal distance which is always real but can be 0 '''
        return self.Distance() / self.get_max_distance()

    def NormalizedRevDistance(self):
        ''' returns a 2d array containing the distance = shortest path length, takes weights into account'''
        ''' gives +infinity if no path'''
        ''' divides by distance maximal distance which is always real but can be 0 '''
        return self.RevDistance() / self.get_max_distance()

    def NumberOfNodes(self):
        ''' returns a 2d array filled with only one value : the number of nodes of the network'''
        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        value = self.number_of_nodes()
        probas.fill(value)
        return probas

    def NumberOfEdges(self):
        ''' returns a 2d array filled with only one value : the number of edges of the network'''
        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        value = self.number_of_edges()
        probas.fill(value)
        return probas

    def MaxDegree(self):
        ''' returns a 2d array filled with only one value : the maximal in degree among nodes in the network'''

        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        max_degree = self.get_max_degree()
        probas.fill(max_degree)
        return probas

    def AverageDegree(self):
        ''' returns a 2d array filled with only one value : the average of  in degrees in teh network'''
        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        value = sum(self.degree().values()) / self.number_of_nodes()
        probas.fill(value)
        return probas

    def MaxDistance(self):
        ''' returns a 2d array filled with only one value : the max of distances in the network'''

        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        max_distance = self.get_max_distance()
        probas.fill(max_distance)
        return probas
    """
    def AverageDistance(self):
        ''' returns a 2d array filled with only one value : the average of distances in the network'''

        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        shortest_path_dict = self.get_shortest_path_dict()
        number_of_distances = sum(len(dictionnaire.values()) for dictionnaire in shortest_path_dict.values())
        sum_of_distances = sum(sum(dictionnaire.values()) for dictionnaire in shortest_path_dict.values())
        value = sum_of_distances / number_of_distances
        probas.fill(value)
        return probas

    def TotalDistance(self):
        ''' returns a 2d array filled with only one value : the sum of distances in the network'''

        probas = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        shortest_path_dict = self.get_shortest_path_dict()
        value = sum(sum(dictionnaire.values()) for dictionnaire in shortest_path_dict.values())
        probas.fill(value)
        return probas
    """
    def Constant(self):
        ''' returns a 2d array filled with only one value : 1'''

        probas = np.ones((self.number_of_nodes(), self.number_of_nodes()))
        return probas

    def Random(self):
        ''' returns a 2d array filled with only random value between 0 and 1'''

        probas = np.random.rand(self.number_of_nodes(), self.number_of_nodes())
        return probas
    """
    def get_shortest_path_dict(self):
        ''' returns the dict od dict of shortest path lengths, if it does not exist, it creates it'''
        if self.shortest_path_dict is None:
            self.shortest_path_dict = nx.shortest_path_length(self)
        return self.shortest_path_dict
    """
    def get_shortest_path_matrix(self):
        ''' returns the dict od dict of shortest path lengths, if it does not exist, it creates it'''
        if self.shortest_path_matrix is None:
            self.shortest_path_matrix = np.array(self.i_graphe.shortest_paths_dijkstra())
        return self.shortest_path_matrix

    def get_max_degree(self):
        ''' returns the maximum of in_degrees, if it does not exist, it computes it'''
        if self.max_degree is None:
            self.max_degree = max(self.degree().values())
        return self.max_degree

    def get_max_distance(self):
        if self.max_distance is None:
            self.max_distance = np.max(self.get_shortest_path_matrix())
        return self.max_distance
