"""
Created on 15 nov. 2012

@author: David Fourquet
"""
import itertools as it
import os
import operator
import networkx as nx
import numpy as np
from lxml import etree as xml
import csv
import community

"""
contains two main function :

*get_datas_from_real_network : takes a relative path to network-type file and store relevant infos for future studies
*eval_network : takes a network and returns a number that cracterizes 
                the proximity between the network and the real network already registered
                
                methods of evaluation implemented : degree_distribution
"""

number_of_elements_by_array = 10
extension = ""


def get_datas_from_real_network(data_path, results_path, name=None, dynamic=None, evaluation_method=""):
    """takes a relative path to directory containing network files and store relevant infos for future studies
    """

    if not os.path.isfile(results_path):
        try :
            os.makedirs(os.path.dirname(results_path))
        except OSError :
            pass
        results = xml.Element('results')
        tree = xml.ElementTree(results)
    else:
        parser = xml.XMLParser(remove_blank_text=True)
        tree = xml.parse(results_path, parser)
        results = tree.getroot()
        try:
            results.remove(results.find("mesures"))
        except TypeError:
            pass

    if results.find("mesures") is None:
        graph = read_typed_file(data_path + extension)
        if graph.is_multigraph() :
            if graph.is_directed() :
                graph = nx.DiGraph(graph)
            else:
                graph = nx.Graph(graph)
        static_network = xml.SubElement(results, "mesures")
        xml.SubElement(static_network, "name", value=name)

        set_evaluation_datas(graph, static_network, evaluation_method=evaluation_method)
        print(xml.tostring(results, pretty_print=True))
        f = open(results_path, 'w')
        tree.write(f, pretty_print=True)
        f.close()


def eval_network(network, results_path, number="", **kwargs):
    eval_methods = kwargs.get("evaluation_method")
    dynamic_network = xml.parse(results_path).getroot()
    graph_xml = dynamic_network.find("mesures" + str(number))
    dictionnary_distance = {}
    dictionnary_distrib ={}
    def add(mesure, a, b): dictionnary_distance[mesure] = distance(a, b)

    for mesure in eval_methods.split('_'):
        value_test = d_method[mesure](network)
        dictionnary_distrib[mesure] = value_test
        value_goal = eval(graph_xml.find(mesure).get('value'))
        add(mesure, value_test, value_goal)

    dictionnary_distance['raw_score'] = max(dictionnary_distance.values())
    return dictionnary_distance['raw_score'],dictionnary_distance,dictionnary_distrib



    '''
    Function that help evaluating network
    '''


def hist(value, bins):
    hist = np.histogram(value, bins=bins)[0]
    hist = hist / float(sum(hist))
    return list(hist)


def get_communities(network):
    '''
    take a network and returns a vector containing sorted proportions of each community of the network
    Louvain algorithm is used to detect communities
    '''
    if network.number_of_edges()> 1:
        try :
            communities = community.best_partition(nx.Graph(network)).values()
        except ZeroDivisionError:
            communities = range(network.number_of_nodes())
            print network.size(weight = 'weight')
            print network.number_of_nodes(),network.number_of_edges()
    else :
        communities = range(network.number_of_nodes())
    communities_hist = np.bincount(communities)
    communities_distribution = communities_hist / float(len(communities))
    communities_sorted = sorted(communities_distribution, reverse=True)
    return communities_sorted


def get_hist_data(data):
    maximum = max(1, max(data))
    return hist(data, maximum)


def get_degrees(network): return get_hist_data(network.degree().values())


def get_indegrees(network): return get_hist_data(network.in_degree().values())


def get_outdegrees(network): return get_hist_data(network.out_degree().values())


def get_distances(network): return get_hist_data(list(it.chain.from_iterable(
    [dict_of_length.values() for dict_of_length
     in nx.shortest_path_length(network).values()])))


def get_clustering(network):
    return hist(nx.clustering(network.to_undirected()).values(), number_of_elements_by_array)


def get_importance(network):
    try :
        return hist(nx.eigenvector_centrality_numpy(network).values(), number_of_elements_by_array)
    except :
        return hist(np.ones(network.number_of_nodes(), dtype=float) / network.number_of_nodes(), number_of_elements_by_array)


d_method = {
    'nodes': nx.number_of_nodes,
    'density': nx.density,
    'communities': get_communities,
    'indegrees': get_indegrees,
    'outdegrees': get_outdegrees,
    'degrees': get_degrees,
    'distances': get_distances,
    'clustering': get_clustering,
    'importance': get_importance
    # 'heterogeneity':nx.number_of_nodes,
    # 'community_structure':nx.number_of_nodes,

}


def get_number_of_nodes_and_edges(results_path, name, numero=None):
    dynamic_network = xml.parse(results_path).getroot()
    if numero is not None:
        print "probleme avec les graphes dynamiques"
        static_network = dynamic_network.find(name + str(numero))
    else:
        static_network = dynamic_network.find("mesures")
    nb_nodes = static_network.find('number_of_nodes').get('value')
    nb_edges = static_network.find('number_of_edges').get('value')
    return int(nb_nodes), int(nb_edges)

def get_edge_data(edge_data_path,data_path,number_of_nodes):
    graph =nx.read_gexf(data_path)
    sublist_nodes = sorted(graph.degree_iter(),key=operator.itemgetter(1),reverse=True)[:number_of_nodes]
    try :
        graph_weights = nx.read_gexf(edge_data_path.replace(".csv",".gexf"))
    except:
        graph_weights = nx.read_weighted_edgelist(edge_data_path,delimiter=",")
        nx.write_gexf(graph_weights,edge_data_path.replace(".csv",".gexf"))
    graph_weights = nx.subgraph(graph_weights,[node for node, degree in sublist_nodes ])
    matrix = nx.to_numpy_matrix(graph_weights, dtype=float,weight="weight")
    normalized_matrix = matrix/np.max(matrix)
    return matrix,normalized_matrix

def get_node_data(node_data_path,data_path,number_of_nodes):
    graph =nx.read_gexf(data_path)
    sublist_nodes = sorted(graph.degree_iter(),key=operator.itemgetter(1),reverse=True)[:number_of_nodes]
    sublist_nodes=[node for node,degree in sublist_nodes]
    with open(node_data_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        node_data = [float(data) for node,data in reader if node in sublist_nodes]
    return node_data
'''
functions that help comparing results
'''


def distance(obj1, obj2):
    if type(obj1) is list:
        return distance_distribution_different_size(obj1, obj2)
    if type(obj1) in [float, int]:
        return distance_numbers(obj1, obj2)
    raise TypeError("Unknown type", type(obj1))


def distance_numbers(number1, number2):
    """distance de sorensen"""
    distance = abs(number1 - number2) / (number1 + number2)
    return distance


def distance_distribution_different_size(distri1, distri2):
    """city-block distance ou L1 distance"""
    length = min(len(distri1), len(distri2))
    distriaux1 = np.array(distri1[:length])
    distriaux2 = np.array(distri2[:length])
    distance = np.sum(np.absolute(distriaux1 - distriaux2)) / 2
    return distance


'''
def get_datas_from_real_network (data_path,results_path,**kwargs):
    takes a relative path to network-type file and store relevant infos for future studies
    
    
    #the function used to read the network is dependant on the type of network used
    graph = read_typed_file(data_path)
                           
    nx.draw(graph)
    plt.savefig(data_path.replace(".gml",".png"))
    plt.close()
            
    f = open(results_path, 'w')
    
    #First relevant infos are number of nodes and number of edges, 
    #should be dependant on the method used to develop the network, 
    #but until now they are necessary and always stored
    f.write(str(nx.number_of_nodes(graph)))
    f.write("\n")
    f.write(str(nx.number_of_edges(graph)))
    f.write("\n")
    #Other infos depend on the evaluation method used
    result = get_evaluation_datas_default(graph)

    f.write(result)
    f.close()
 '''

'''

    if dynamic:
        """
        print "DYNAMIC NETWORK EVALUATION NOT IMPLEMENTED"
        # number_of_networks = 0
        # for each time step, we store data about the network at this time
        for network_file in os.listdir(data_path):
            if extension in network_file:
                # number_of_networks+=1
                # only read gexf files
                graph = read_typed_file(data_path + network_file)
                # nx.draw(graph)
                # plt.savefig(data_path+name+".png")
                # plt.close()

                static_network = xml.SubElement(root, network_file.replace(extension, ""))
                set_evaluation_datas(graph, static_network, evaluation_method=evaluation_method)
                # we write the number of steps in the dynamic of the network
        # sub = xml.SubElement(dynamic_network,"number_of_timestamps")
        # sub.attrib['value'] = str(number_of_networks)
        f = open(results_path, 'w')
        xml.ElementTree(root).write(f, pretty_print=True)
        f.close()
        """
    else:
'''

'''
def eval_network(network,results_path,**kwargs):
    takes a network and returns a number that caracterizes 
    the proximity between the network and the real network already registered in results_path
    # default evaluation method is node distribution
    eval_method = kwargs.get("evaluation_method")
    network_type = kwargs["network_type"]
    if eval_method == "degree_distribution" :
        if network_type == "wundirected_weighted"  or network_type == "undirected_unweighted" :
            return eval_degree_distribution(network,results_path)
    
    
    if eval_method == "2distributions" :
        if network_type == "directed_weighted"  or network_type == "directed_unweighted" :
            return eval_2distributions_directed(network,results_path)
        if network_type == "undirected_weighted"  or network_type == "undirected_unweighted" :
            return eval_2distributions_undirected(network,results_path)
    raise Exception("no evaluation_method or network_type given")
'''

'''
def eval_network(network, results_path, number="", **kwargs):
    eval_methods = kwargs.get("evaluation_method")
    name = kwargs.get('name')
    dynamic_network = xml.parse(results_path).getroot()
    static_network = dynamic_network.find("mesures" + str(number))


    dict_proximity = {}

    if 'nodes' in eval_methods:
        proximity_nodes = eval_proximity_nodes(network, static_network)
        dict_proximity['proximity_nodes'] = proximity_nodes

    if 'vertices' in eval_methods:
        proximity_vertices = eval_proximity_vertices(network, static_network)
        dict_proximity['proximity_vertices'] = proximity_vertices

    if 'communities' in eval_methods:
        proximity_communities = eval_proximity_communities(network, static_network)
        dict_proximity['proximity_communities'] = proximity_communities

    if 'degrees' in eval_methods:
        proximity_degrees = eval_proximity_degrees(network, static_network)
        dict_proximity['proximity_degrees'] = proximity_degrees

    if 'distances' in eval_methods:
        proximity_distances = eval_proximity_distances(network, static_network)
        dict_proximity['proximity_distances'] = proximity_distances

    if 'clustering' in eval_methods:
        proximity_clustering = eval_proximity_clustering(network, static_network)
        dict_proximity['proximity_clustering'] = proximity_clustering

    if 'importance' in eval_methods:
        proximity_importance = eval_proximity_importance(network, static_network)
        dict_proximity['proximity_importance'] = proximity_importance

    if 'heterogeneity' in eval_methods:
        proximity_heterogeneity = eval_proximity_heterogeneity(network, static_network)
        dict_proximity['proximity_heterogeneity'] = proximity_heterogeneity

    if 'community_structure' in eval_methods:
        proximity_community_structure = eval_proximity_community_structure(network, static_network)
        dict_proximity['proximity_community_structure'] = proximity_community_structure

    dict_proximity['proximity_aggregated'] = min(dict_proximity.values())
    return dict_proximity
    # raise Exception("no evaluation_method given")
    '''

'''
    if  'nodes' in eval_methods:
        number_of_nodes_test = nx.number_of_nodes(network)
        goal = int(graph_xml.find('nodes').get('value'))
        add('nodes',number_of_nodes_test,goal)

    if 'density' in eval_methods:
        number_of_nodes_test = nx.density(network)
        goal = float(graph_xml.find('density').get('value'))
        add('density',number_of_nodes_test,goal)

    if 'communities' in eval_methods:
        number_of_nodes_test = nx.density(network)
        goal = float(graph_xml.find('communities').get('value'))
        add('communities',number_of_nodes_test,goal)
        proximity_communities = eval_proximity_communities(network, static_network)
        dict_proximity['proximity_communities'] = proximity_communities

    if 'degrees' in eval_methods:
        proximity_degrees = eval_proximity_degrees(network, static_network)
        dict_proximity['proximity_degrees'] = proximity_degrees

    if 'distances' in eval_methods:
        proximity_distances = eval_proximity_distances(network, static_network)
        dict_proximity['proximity_distances'] = proximity_distances

    if 'clustering' in eval_methods:
        proximity_clustering = eval_proximity_clustering(network, static_network)
        dict_proximity['proximity_clustering'] = proximity_clustering

    if 'importance' in eval_methods:
        proximity_importance = eval_proximity_importance(network, static_network)
        dict_proximity['proximity_importance'] = proximity_importance

    if 'heterogeneity' in eval_methods:
        proximity_heterogeneity = eval_proximity_heterogeneity(network, static_network)
        dict_proximity['proximity_heterogeneity'] = proximity_heterogeneity

    if 'community_structure' in eval_methods:
        proximity_community_structure = eval_proximity_community_structure(network, static_network)
        dict_proximity['proximity_community_structure'] = proximity_community_structure
    '''

"""
def eval_proximity_nodes(network, graph_xml):
    '''returns the proximity of number of nodes between synthetic network(test) and real network (goal)'''
    number_of_nodes_test = nx.number_of_nodes(network)
    number_of_nodes_goal = eval(graph_xml.find('number_of_nodes').get('value'))
    proximity = proximity_numbers(number_of_nodes_goal, number_of_nodes_test)
    return proximity


def eval_proximity_vertices(network, graph_xml):
    '''returns the proximity of proportion of vertices between synthetic network(test) and real network (goal)'''
    number_of_nodes_test = float(nx.number_of_nodes(network))
    if network.isDirected():
        proportion_edges_test = nx.number_of_edges(network) / (number_of_nodes_test * (number_of_nodes_test - 1))
    else:
        proportion_edges_test = 2. * nx.number_of_edges(network) / (number_of_nodes_test * (number_of_nodes_test - 1))

    proportion_edges_goal = eval(graph_xml.find('vertices').get('value'))
    proximity = proximity_numbers(proportion_edges_goal, proportion_edges_test)
    return proximity


def eval_proximity_communities(network, graph_xml):
    '''returns the proximity of repartition into communities between synthetic network(test) and real network (goal)'''
    # repartitions are sorted vectors that sum at 1, each items is the proportions of one community
    community_proportion_test = get_communities(network)
    number_of_equivalent_communities_test = get_equivalent_number(community_proportion_test)

    community_proportion_goal = eval(graph_xml.find('communities').get('value'))
    number_of_equivalent_communities_goal = get_equivalent_number(community_proportion_goal)

    # proximity = proximity_repartition(community_proportion_goal, community_proportion_test)
    proximity = proximity_numbers(number_of_equivalent_communities_goal, number_of_equivalent_communities_test)
    return proximity


def eval_proximity_degrees(network, graph_xml):
    '''returns the proximity of degree distributions between synthetic network(test) and real network (goal)'''
    if network.isDirected():
        degree_in_test = network.in_degree().values()
        degree_in_goal = eval(graph_xml.find('in_degrees').get('value'))
        proximity_degree_in = proximity_distributions_different_size(degree_in_goal, degree_in_test)

        degree_out_test = network.out_degree().values()
        degree_out_goal = eval(graph_xml.find('out_degrees').get('value'))
        proximity_degree_out = proximity_distributions_different_size(degree_out_goal, degree_out_test)

        proximity = (proximity_degree_in + proximity_degree_out) / 2

    else:
        degree_test = network.degree().values()
        degree_goal = eval(graph_xml.find('degrees').get('value'))
        proximity = proximity_distributions_different_size(degree_goal, degree_test)

    return proximity


def eval_proximity_distances(network, graph_xml):
    '''returns the proximity of distances distributions between synthetic network(test) and real network (goal)'''
    distances_goal = eval(graph_xml.find('distances').get('value'))
    distances_test = list(it.chain.from_iterable(
        [dict_of_length.values() for dict_of_length in nx.shortest_path_length(network).values()]))

    proximity = proximity_distributions_different_size(distances_goal, distances_test)
    return proximity


def eval_proximity_clustering(network, graph_xml):
    '''returns the proximity of clustering coefficient distributions between synthetic network(test) and real network (goal)'''
    clustering_test = nx.clustering(network.to_undirected()).values()
    clustering_goal = eval(graph_xml.find('clustering').get('value'))

    proximity = proximity_distributions_different_size(clustering_goal, clustering_test)
    return proximity


def eval_proximity_community_structure(network, graph_xml):
    '''returns the proximity of clustering coefficient distributions between synthetic network(test) and real network (goal)'''
    clustering_test = np.mean(nx.clustering(network).values())
    clustering_goal = np.mean(eval(graph_xml.find('clustering').get('value')))

    proximity = proximity_numbers(clustering_goal, clustering_test)
    return proximity


def eval_proximity_heterogeneity(network, graph_xml):
    '''returns the proximity of clustering coefficient distributions between synthetic network(test) and real network (goal)'''

    coef_test = powerlaw.Fit(nx.degree(network).values()).power_law.alpha
    coef_goal = powerlaw.Fit(eval(graph_xml.find('degree').get('value'))).power_law.alpha

    proximity = proximity_numbers(coef_goal, coef_test)
    return proximity


def eval_proximity_importance(network, graph_xml):
    '''returns the proximity of page rank scores distributions between synthetic network(test) and real network (goal)'''
    # we need to reverse the network to get a score such that the importance of a node is related to the importance of nodes that point towards it.

    if network.is_directed():
        importance_test = nx.pagerank(network).values()
    else:
        importance_test = nx.eigenvector_centrality_numpy(network).values()

    importance_goal = eval(graph_xml.find('importance').get('value'))

    proximity = proximity_distributions_different_size(importance_goal, importance_test)
    return proximity
"""

'''
def eval_degree_distribution(network,results_path):
    the bigger the result is, the worse the network is 
    #build the alist describing the distribution of degrees of the evaluated network
    hist_test = nx.degree_histogram(network)
    
    #build the list describing the distribution of degrees of the real network
    #item k is the number of nodes of degree k
    f = open(results_path, 'r')
    lines = f.readlines()
    hist_goal = eval(lines[2])
    #this function compares list and return the difference in 
    result = compare(hist_test,hist_goal )
    f.close() 
    return result

def eval_2distributions_directed(network,results_path):
    the bigger the result is, the worse the network is 
    #build the alist describing the distribution of degrees of the evaluated network
    hist_test_indegree = get_histogram(network.in_degree().values())
    hist_test_outdegree = get_histogram(network.out_degree().values())
    #this line converts dict of dict of lengths to list of lists, chain them, then get an histogram 
    hist_test_distances = get_histogram(list(it.chain.from_iterable([ dict_of_length.values() for dict_of_length in nx.shortest_path_length(network).values()])))
    
    #build the list describing the distribution of indegrees, outdegrees and distances of the real network
    #item k is the number of nodes of degree k
    f = open(results_path, 'r')
    lines = f.readlines()
    hist_goal_indegree = eval(lines[2])
    hist_goal_outdegree = eval(lines[3])
    hist_goal_distances = eval(lines[4])
    
    #this function compares list and return the difference in 
    result = compare(hist_test_indegree,hist_goal_indegree ) + compare(hist_test_outdegree,hist_goal_outdegree ) + compare(hist_test_distances,hist_goal_distances ) 
    f.close() 
    return result

def eval_2distributions_undirected(network,results_path):
    the bigger the result is, the worse the network is
    #build the alist describing the distribution of degrees of the evaluated network
    hist_test_degree = get_histogram(network.degree().values())
    #this line converts dict of dict of lengths to list of lists, chain them, then get an histogram 
    hist_test_distances = get_histogram(list(it.chain.from_iterable([ dict_of_length.values() for dict_of_length in nx.shortest_path_length(network).values()])))
    
    #build the list describing the distribution of indegrees, outdegrees and distances of the real network
    #item k is the number of nodes of degree k
    f = open(results_path, 'r')
    lines = f.readlines()
    hist_goal_degree = eval(lines[2])
    hist_goal_distances = eval(lines[3])
    
    #this function compares list and return the difference in 
    result = compare(hist_test_degree,hist_goal_degree ) + compare(hist_test_distances,hist_goal_distances ) 
    f.close() 
    return result

def eval_
                 
def compare(list1, list2):
    compares 2 arrays by appending with zeros the little one
    gives the sum of absolute differences between corresponding items of both arrays
   
    list1.extend(it.repeat(0,len(list2)-len(list1)))
    list2.extend(it.repeat(0,len(list1)-len(list2)))
    
    array1 = np.array(list1)
    array2 = np.array(list2)
    return sum(abs(array1-array2))
'''

"""
def proximity_numbers(number1, number2):
    '''returns a number between 0 and 1
    equals to 1 if networks have the same number (of nodes) in order of magnitude
    equals 0 if they have very different numbers (of nodes)'''
    maxnumber = max(number1, number2)
    minnumber = min(number1, number2)
    proximity = minnumber / maxnumber
    return proximity


'''
def proximity_repartition(array1, array2):
     returns a number between 0 and 1 : the proximity between repartitions described by arrays
    equals to 1 if repartitions are identical, 0 if repartitions are different : example : 1 concentrated (1,0,0...) and 1 balanced (1/n ; 1/n ;1/n ...)
    
    min_size = min(array1.length, array2.length)
    
    arraymin = np.minimum(np.resize(array1,min_size),np.resize(array2,min_size))
    proximity = sum(arraymin)
    return proximity
'''


def proximity_distributions_same_size(array1, array2):
    '''returns a number between 0 and 1 : the proximity between two arrays
    arrays have exactly number_of_elements_by_array elements which are proportions
    equals to 1 if arrays have the same proportions in each category
    equals to 0 if arrays have very different proportions in each category
    '''

    arraymax = np.maximum(array1, array2)
    arraymin = np.minimum(array1, array2)

    # we remove zeros from the distribution and change them to 1 in order to avoid division by zero
    arrayaux = arraymax == 0
    arraymin[arrayaux] = 1
    arraymax[arrayaux] = 1

    proximity = sum(arraymin / arraymax) / number_of_elements_by_array
    return proximity





def proximity_distributions_different_size(array1, array2):
    '''returns a number between 0 and 1 : the proximity between two arrays
    array have different number of elements which are integers
    transforms to distributions with same size and returns proximity_distributions_same_size
    '''

    # divides in 10 equal parts : parts are obtained taking the maximum of the first array as limit
    # first array becomes a normalized distribution
    # second array is not normalized : if some values are over the range of values in the first array,they are discarded

    # 2 decompositions : regular
    maxNumber = max(array1) / number_of_elements_by_array
    array1_reg, _ = np.histogram(array1, bins=maxNumber * np.arange(number_of_elements_by_array + 1))
    array1_reg = array1_reg / float(len(array1))
    array2_reg, _ = np.histogram(array2, bins=maxNumber * np.arange(number_of_elements_by_array + 1))
    array2_reg = array2_reg / float(len(array2))

    # or exponential
    constant = math.log(max(array1) + 1) / number_of_elements_by_array
    array1_exp, _ = np.histogram(array1, bins=np.exp(constant * np.arange(number_of_elements_by_array + 1)) - 1)
    array1_exp = array1_exp / float(len(array1))
    array2_exp, _ = np.histogram(array2, bins=np.exp(constant * np.arange(number_of_elements_by_array + 1)) - 1)
    array2_exp = array2_exp / float(len(array2))

    # we take the most balanced decomposition
    array1_reg_withoutzeros = array1_reg[array1_reg != 0]
    array1_exp_withoutzeros = array1_exp[array1_exp != 0]

    # balance is defined as the entropy of the distribution
    balance_regular = - np.sum(array1_reg_withoutzeros * np.log(array1_reg_withoutzeros))
    balance_exp = - np.sum(array1_exp_withoutzeros * np.log(array1_exp_withoutzeros))

    if balance_regular > balance_exp:
        proximity = proximity_distributions_same_size(array1_reg, array2_reg)
    else:
        proximity = proximity_distributions_same_size(array1_exp, array2_exp)
    return proximity

def get_communities(network):
    '''
    take a network and returns a vector containing sorted proportions of each community of the network
    Louvain algorithm is used to detect communities
    '''

    communities = community.best_partition(nx.Graph(network)).values()
    communities_hist = np.bincount(communities)
    communities_distribution = communities_hist / float(len(communities))
    communities_sorted = sorted(communities_distribution, reverse=True)
    return communities_sorted


def get_equivalent_number(proportions):
    values = np.square(proportions)
    equivalent_number_of_communities = 1 / (sum(values))
    return equivalent_number_of_communities
"""
""" 
Function that help reading file and storing datas
"""


def read_typed_file(path):
    # 3 types of network can be read
    # if ".gml" in path :
    # return nx.read_gml(path)
    if ".gexf" in path:
        return nx.read_gexf(path)
        # if ".net" in path :
        # return nx.read_pajek(path)
    raise Exception("network_format not recognized")


'''
def get_evaluation_datas(graph, **kwargs) :
    #cast to string relevant infos for the evaluation of networks
    # default_method is node_distribution
    evaluation_method = kwargs.get("evaluation_method") 
    network_type = kwargs.get("network_type")
    
    if evaluation_method == "degree_distribution" :
        if network_type == "undirected_weighted"  or network_type == "undirected_unweighted" :
            return datas_degree_distribution(graph)
        
    if evaluation_method == "2distributions" :
        if network_type == "directed_weighted"  or network_type == "directed_unweighted" :
            return datas_2distributions_directed(graph)
        if network_type == "undirected_weighted"  or network_type == "undirected_unweighted" :
            return datas_2distributions_undirected(graph)
           
    return datas_default(graph,network_type)
    
    raise Exception("no evaluation_method or network_type given")
    '''


def set_evaluation_datas(graph, graph_xml, evaluation_method=''):
    '''if no precise evaluation method is given, we compute every possible measure (wrong !!)'''

    def add_sub(name, value):
        sub = xml.SubElement(graph_xml, name)
        sub.attrib['value'] = str(value)

    # First relevant infos are number of nodes and number of edges,
    # should be dependant on the method used to develop the network,
    # but until now they are necessary and always stored
    add_sub('number_of_nodes', nx.number_of_nodes(graph))
    add_sub('number_of_edges', nx.number_of_edges(graph))

    # number of nodes
    nodes = nx.number_of_nodes(graph)

    # should be replaced by getattr(graph, variable) loop


    for mesure in evaluation_method.split('_'):
        add_sub(mesure, d_method[mesure](graph))
    '''
    if graph.is_directed():

        if 'vertices' in evaluation_method:
            add_sub('vertices', nx.number_of_edges(graph) / (nodes * (nodes - 1)))
        if 'degrees' in evaluation_method:
            ideg = graph.in_degree().values()
            add_sub('in_degrees', hist(ideg, max(ideg)))
            odeg = graph.out_degree().values()
            add_sub('out_degrees', hist(odeg, max(odeg)))
        if 'importance' in evaluation_method:
            add_sub('importance', hist(nx.pagerank(graph).values(), number_of_elements_by_array))
        if 'clustering' in evaluation_method or 'heterogeneity' in evaluation_method:
            add_sub('clustering', hist(nx.clustering(graph.to_undirected()).values(), number_of_elements_by_array))
        if 'community_structure' in evaluation_method:
            add_sub('degree', graph.degree().values())

    else:
        if 'vertices' in evaluation_method:
            add_sub('vertices', 2 * nx.number_of_edges(graph) / (nodes * (nodes - 1)))
        if 'communities' in evaluation_method:
            add_sub('communities', get_communities(graph))
        if 'degrees' in evaluation_method or 'community_structure' in evaluation_method:
            degree = graph.degree().values()
            add_sub('degrees', hist(degree, max(degree) - 1))
        if 'clustering' in evaluation_method or 'heterogeneity' in evaluation_method:
            add_sub('clustering', hist(nx.clustering(graph).values(), number_of_elements_by_array))
        if 'importance' in evaluation_method:
            add_sub('importance', hist(nx.eigenvector_centrality_numpy(graph).values(), number_of_elements_by_array))

    if 'distances' in evaluation_method:
        distance = list(it.chain.from_iterable(
            [dict_of_length.values() for dict_of_length in nx.shortest_path_length(graph).values()]))

        add_sub('distances', hist(distance, max(distance)))
    '''


'''
def get_evaluation_datas_default(network,**kwargs):
    if no precise evaluation method is given, we compute every possible measure
    
    evaluation_method = kwargs.get('evaluation_method','')
    evaluation_goals ={}
    
    #number of nodes
    nodes = nx.number_of_nodes(network)
    
    if 'nodes' in evaluation_method :
        evaluation_goals['nodes'] = str(nodes)
    
    if network.is_directed() :
        if 'vertices' in evaluation_method : 
            evaluation_goals['vertices'] = str(nx.number_of_edges(network)/(nodes*(nodes-1)))
        if 'degrees' in evaluation_method : 
            evaluation_goals['degree_in'] = str(network.in_degree().values())
            evaluation_goals['degree_out'] = str(network.out_degree().values())
        if 'importance' in evaluation_method :
            evaluation_goals['importance'] = str(nx.eigenvector_centrality_numpy(network.reverse()).values())
    else :
        if 'vertices' in evaluation_method :
            evaluation_goals['vertices'] = str(2.*nx.number_of_edges(network)/(nodes*(nodes-1)))
        if 'communities' in evaluation_method :
            evaluation_goals['communities'] = str(get_communities(network))
        if 'degrees' in evaluation_method :
            evaluation_goals['degrees'] = str(network.degree().values())
        if 'clustering' in evaluation_method :
            evaluation_goals['clustering'] = str(nx.clustering(network).values())
        if 'importance' in evaluation_method :
            evaluation_goals['importance'] = str(nx.eigenvector_centrality_numpy(network).values())
    
    if 'distances' in evaluation_method :
        evaluation_goals['distances'] = str(list(it.chain.from_iterable([ dict_of_length.values() for dict_of_length in nx.shortest_path_length(network).values()])))
    
    return evaluation_goals

def datas_degree_distribution(graph) :
    result = str(get_histogram(graph.degree().values())) 
   
    return result

def datas_2distributions_directed(graph) :
    #results are not weight-dependant because we have no weight on real edges
    indegree = get_histogram(graph.in_degree().values())
    outdegree = get_histogram(graph.out_degree().values())
    #this line converts dict of dict of lengths to list of lists, chain them, then get an histogram 
    shortest_path = get_histogram(list(it.chain.from_iterable([ dict_of_length.values() for dict_of_length in nx.shortest_path_length(graph).values()])))
    return '\n'.join([str(indegree),str(outdegree),str(shortest_path)])

def datas_2distributions_undirected(graph) :
    #results are not weight-dependant because we have no weight on real edges
    indegree = get_histogram(graph.degree().values())
    #this line converts dict of dict of lengths to list of lists, chain them, then get an histogram 
    shortest_path = get_histogram(list(it.chain.from_iterable([ dict_of_length.values() for dict_of_length in nx.shortest_path_length(graph).values()])))
    return '\n'.join([str(indegree),str(shortest_path)])
'''

'''
def get_number_of_edges(results_path,numero):
    name = results_path.split("/")[-2]
    dynamic_network = xml.parse(results_path).getroot()
    if numero is not None :
        static_network = dynamic_network.find(name+str(numero))
    else :
        static_network = dynamic_network.find(name)
    nb_edges = static_network.find('number_of_edges').get('value')
    return nb_edges

def get_histogram(degseq):
    takes a sequence of integer values and returns the distribution of values   
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq
'''
