'''
Created on 15 nov. 2012
@author: David
inspired by Telmo Menezes's work : telmomenezes.com
'''
import random
import network_development as nd
import network_evaluation as ne
import pyevolve as py
import statistics as st
import evaluation_method_options as emo
import randomization_control as rc
import pyevolve.GAllele as gall
import pyevolve.GSimpleGA as gsga
import os

"""
contains two main function :

*new_genome : takes a set of alleles and options that define initializator, mutator, crossover, evaluator 
                and returns a genome with those options
                
*evolve : takes a genome and options that define the genetic algorithm
                apply it to the genome
                returns infos about the best individual
                
"""


def new_genome(results_path, **kwargs):
    """
    *new_genome : takes a set of alleles [function_alleles, leaves_alleles]
                and options that define initializator, mutator, crossover, evaluator
                and returns a genome with those options

                possible initializator : "grow" : "grow" algorithm of network = recursive and possibly incomplete
                possible mutator : "simple" : change genomic alleles with possible alleles with probability pmut
                possible crossover :
                possible evaluator : "degree_distribution", "2distributions"
                possible network-type : "directed_weighted", "directed_unweighted", "undirected_weighted", "undirected_unweighted"
                possible tree_type : "with_constants"
     """
    evaluation_method = kwargs.get("evaluation_method")
    network_type = kwargs.get("network_type")
    data_path = kwargs.get("data_path")
    name = kwargs.get("name")
    dynamic = kwargs.get("dynamic")
    extension = kwargs.get("extension")
    number_of_nodes = kwargs.get("number_of_nodes")
    edge_data_path = kwargs.get("edge_data")




    choices = emo.get_alleles(evaluation_method, network_type)
    genome = py.GTree.GTree()

    # genome.setParams(nb_nodes=ne.get_number_of_nodes(results_path))
    # genome.setParams(nb_edges=ne.get_number_of_edges(results_path))
    genome.setParams(data_path=data_path)
    genome.setParams(results_path=results_path)
    genome.setParams(name=name)
    genome.setParams(extension=extension)
    genome.setParams(number_of_nodes=number_of_nodes)

    # external data taken into account
    if edge_data_path is not None:
        edge_data_matrix, n_edge_data_matrix = ne.get_edge_data(edge_data_path,data_path+extension,number_of_nodes)
        genome.setParams(edge_data=edge_data_matrix)
        genome.setParams(n_edge_data=n_edge_data_matrix)
        choices[1].append("EdgeData")
        choices[1].append("NormEdgeData")

    # defines alleles : one array containing possible leaves and one containing possible functions
    alleles = gall.GAlleles()
    lst = gall.GAlleleList(choices)
    alleles.add(lst)
    genome.setParams(allele=alleles)

    print alleles


    # defines the way to construct a random tree
    genome.setParams(max_depth=int(kwargs.get("max_depth", "6")))
    genome.setParams(max_siblings=int(kwargs.get("max_siblings", "2")))

    genome.setParams(tree_type=kwargs.get("tree_type", "default"))
    genome.initializator.set(tree_init)

    # defines the how to evaluate a genome
    genome.setParams(evaluation_method=evaluation_method)
    if dynamic:
        genome.evaluator.set(eval_func_dynamic)
    else:
        genome.evaluator.set(eval_func)

    # defines the crossover function - default now

    # defines the function that mutates trees
    genome.mutator.set(mutate_tree)

    # defines the network_type
    genome.setParams(network_type=network_type)

    # tree_init(genome)
    return genome


def evolve(genome, initial_graph=None, **kwargs):
    '''
     takes a genome and options that define the genetic algorithm
                apply it to the genome
                returns infos about the best individual
    '''
    # depending on the evaluation_method, we will have different goals
    goal = emo.get_goal(genome.getParam("evaluation_method"))
    multiprocessing = kwargs.get("multiprocessing", False)
    # genetic_algorithm = kwargs.get("genetic_algorithm","default_algorithm")

    algo = gsga.GSimpleGA(genome)
    algo.setMultiProcessing(multiprocessing)

    # the selector is used to choose which ones will be parents of the next generation
    algo.selector.set(py.Selectors.GRouletteWheel)

    # elitism will keep the top individuals with their score in the next generation :
    # it can help us to keep track of some of the best graph generations
    # algo.setElitism(True)
    # algo.setElitismReplacement(10)


    algo.setMinimax(py.Consts.minimaxType[goal])

    # now we do evolve with algorithm and every freq stats, we compute statistics
    freq_stats = int(kwargs.get("freq_stats", "1"))
    filename_stats = kwargs.get("stats_path")
    dot_path = kwargs.get("dot_path")
    stats_adapter = st.StatisticsInDot(filename=filename_stats,frequency = freq_stats,dot_path = dot_path)
    #stats_adapter = st.StatisticsInTxt(filename=filename_stats, frequency=freq_stats)
    algo.setDBAdapter(stats_adapter)

    number_of_generations = int(kwargs.get("nb_generations", "100"))
    algo.setGenerations(number_of_generations)

    best_individual = algo.evolve()
    st.store_best_network(best_individual)


'''
Functions that build trees
'''


def tree_init(genome,**args):
    max_depth = genome.getParam("max_depth")
    max_siblings = genome.getParam("max_siblings")
    allele = genome.getParam("allele")

    tree_type = genome.getParam("tree_type")
    if "with_constants" in tree_type:
        root = buildGTreeGrowWithConstants(0, allele[0][0], allele[0][1], max_depth)
    if tree_type == "simple":
        # default method
        root = buildGTreeGrow(0, allele[0][0], allele[0][1], max_siblings, max_depth)
    genome.setRoot(root)
    genome.processNodes()


def buildGTreeGrow(depth, value_callback, value_leaf, max_siblings, max_depth):
    ''' this function build tree with a maximal depth recursively : 
    at each step it has a probability of 0,5 of being a leaf , 0,5 of having two childs:
    a leaf contains variables, a node contains functions
    '''
    random_value = random.choice(value_callback)
    random_value_leaf = random.choice(value_leaf)
    n = py.GTree.GTreeNode(0)

    if depth == max_depth:
        n.setData(random_value_leaf)
        return n

    if py.Util.randomFlipCoin(0.5):
        child = buildGTreeGrow(depth + 1, value_callback, value_leaf, max_siblings, max_depth)
        child.setParent(n)
        n.addChild(child)

        child = buildGTreeGrow(depth + 1, value_callback, value_leaf, max_siblings, max_depth)
        child.setParent(n)
        n.addChild(child)

    if n.isLeaf():
        n.setData(random_value_leaf)
    else:
        n.setData(random_value)
    return n


def buildGTreeGrowWithConstants(depth, value_callback, value_leaf, max_depth):
    ''' this is the same as the previous function but leaves contain a constant and variable
    the constant is decided by the module of control of randomization
    '''
    random_value = random.choice(value_callback)
    random_value_leaf = [rc.new_constant(), random.choice(value_leaf)]
    n = py.GTree.GTreeNode(0)

    if depth == max_depth/2:
        n.setData(random_value_leaf)
        return n

    if py.Util.randomFlipCoin(0.5):
        child = buildGTreeGrowWithConstants(depth + 1, value_callback, value_leaf, max_depth)
        child.setParent(n)
        n.addChild(child)

        child = buildGTreeGrowWithConstants(depth + 1, value_callback, value_leaf, max_depth)
        child.setParent(n)
        n.addChild(child)

    if n.isLeaf():
        n.setData(random_value_leaf)
    else:
        n.setData(random_value)
    return n


'''
Functions that mutate trees
'''


def simple_mutate_tree(genome, **args):
    mutations = 0
    allele = genome.getParam("allele")
    for node in genome.getAllNodes():
        if py.Util.randomFlipCoin(args["pmut"]):
            mutations += 1
            if node.isLeaf():
                node.setData(random.choice(allele[0][1]))
            else:
                node.setData(random.choice(allele[0][0]))
    return mutations


def mutate_tree_with_constants(genome, **args):
    ''' this mutates the constant part, the variable part of the leaves or the function in a node
    mutations add a random number to previous constant
    mutations change function and variable randomly
    '''
    mutations = 0
    allele = genome.getParam("allele")
    for node in genome.getAllNodes():
        if node.isLeaf():
            # potentially change constant in leaf
            if py.Util.randomFlipCoin(args["pmut"]):
                mutations += 1
                constant, variable = node.getData()
                node.setData([rc.mutated_constant(constant), variable])
            # potentially change the variable in leaf
            if py.Util.randomFlipCoin(args["pmut"]):
                mutations += 1
                constant, variable = node.getData()
                node.setData([constant, random.choice(allele[0][1])])
            # potentially change leaf to tree
            if genome.getNodeDepth(node) != genome.getParam("max_depth") and py.Util.randomFlipCoin(args["pmut"] )  :
                mutations += 1
                node.setData(random.choice(allele[0][0]))
                child1 = py.GTree.GTreeNode(0)
                child1.setData([rc.new_constant(), random.choice(allele[0][1])])
                child1.setParent(node)
                node.addChild(child1)
                child2 = py.GTree.GTreeNode(0)
                child2.setData([rc.new_constant(), random.choice(allele[0][1])])
                child2.setParent(node)
                node.addChild(child2)
                genome.processNodes()
        else:
            # potentially change function in node
            if py.Util.randomFlipCoin(args["pmut"]):
                mutations += 1
                node.setData(random.choice(allele[0][0]))
            if genome.getNodeHeight(node) == 1:
                # potentially change node to leaf
                if py.Util.randomFlipCoin(args["pmut"] ):
                    mutations += 1
                    node.setData([rc.new_constant(), random.choice(allele[0][1])])
                    node.getChilds()[:] = []
                    genome.processNodes()
    return mutations


def mutate_tree(genome, **args):
    mutation_method = genome.getParam("tree_type")
    if "with_constants" in mutation_method:
        return mutate_tree_with_constants(genome, **args)
    if mutation_method == "simple":
        return simple_mutate_tree(genome, **args)
    raise Exception("no tree_type given")


'''
Functions that evaluate trees
'''

"""
def store_best_network(chromosome, nb_tests):
    # bugged and unused, unknown use
    score_max = 0
    for _ in range(nb_tests):
        print(chromosome.getParam("results_path"))
        number_of_nodes, number_of_edges = ne.get_number_of_nodes_and_edges(chromosome.getParam("results_path"))
        net = nd.grow_network(chromosome, number_of_nodes, number_of_edges)
        scores = ne.eval_network(net, chromosome.getParam("results_path"),
                                 evaluation_method=chromosome.getParam("evaluation_method"),
                                 network_type=chromosome.getParam("network_type"), name=chromosome.getParam("name"))
        if scores['proximity_aggregated'] > score_max:
            score_max = scores['proximity_aggregated']
            nx.draw(net)
            plt.savefig(chromosome.getParam('results_path').replace(".xml", ".png"))
            plt.close()

            print(scores)
            f = open(chromosome.getParam('results_path'), 'a')
            f.write(str(scores))
            f.close()
"""

def eval_func(chromosome):
    number_of_nodes, number_of_edges = ne.get_number_of_nodes_and_edges(chromosome.getParam("results_path"),
                                                                        chromosome.getParam('name'))
    static_number = min(number_of_nodes,chromosome.getParam("number_of_nodes"))
    number_of_edges = static_number * number_of_edges / number_of_nodes
    number_of_nodes =static_number

    net = nd.grow_network(chromosome, number_of_nodes, number_of_edges)
    raw_score,details,more_details = ne.eval_network(net, chromosome.getParam("results_path"),
                            evaluation_method=chromosome.getParam("evaluation_method"),
                            network_type=chromosome.getParam("network_type"),
                            name=chromosome.getParam("name"))
    chromosome.scoref = lambda: None
    setattr(chromosome.scoref, 'score', details)
    setattr(chromosome.scoref, 'distributions', more_details)
    return raw_score


def eval_func_dynamic(chromosome):
    results_path = chromosome.getParam("results_path")
    evaluation_method = chromosome.getParam("evaluation_method")
    data_path = chromosome.getParam("data_path")
    network_type = chromosome.getParam("network_type")
    extension = chromosome.getParam("extension")
    name = chromosome.getParam("name")
    number_of_networks = len([namefile for namefile in os.listdir(data_path) if extension in namefile])

    scores = {}
    # we create a graph that is similar to the initial graph

    initial_network = ne.read_typed_file(data_path + name + "0" + extension)
    net = nd.createGraph(network_type, initial_network)

    for numero in range(number_of_networks - 1):
        nodes_next, edges_next = ne.get_number_of_nodes_and_edges(results_path, numero + 1)
        nodes_now, edges_now = ne.get_number_of_nodes_and_edges(results_path, numero)
        difference_of_size = nodes_next - nodes_now
        difference_of_edge_number = edges_next - edges_now

        # at each step we add nodes and edges to match with graph at next step
        net = nd.grow_network(chromosome, difference_of_size, difference_of_edge_number, net)

        # comparison with the network at next step : returns a dict  : observable-proximity related to the observable
        scores[str(numero + 1)] = ne.eval_network(net,
                                                  results_path, numero + 1,
                                                  evaluation_method=evaluation_method,
                                                  network_type=network_type,
                                                  name=name)

        # we compute a list that contains aggregated proximity at each time step
    scores_aggregated = [scores[str(score)]['proximity_aggregated'] for score in range(1, number_of_networks)]
    return min(scores_aggregated)




'''           
def eval_func(chromosome):
    results_path = chromosome.getParam("results_path")
    evaluation_method=chromosome.getParam("evaluation_method")
    dynamic = chromosome.getParam("dynamic")
    data_path = chromosome.getParam("data_path")
    network_type = chromosome.getParam("network_type")
    extension = chromosome.getParam("extension")
    
    #if we deal with a dynamic network :
    if dynamic :
        scores ={}
        #we create a graph that is similar to the initial graph
         
        initial_network = ne.read_typed_file(data_path+"0"+extension)
        net = nd.createGraph(network_type,initial_network)
        
        
        for numero in range(number_of_networks) :
            nodes_next,edges_next = ne.get_number_of_nodes_and_edges(results_path, numero+1)
            nodes_now,edges_now = ne.get_number_of_nodes_and_edges(results_path, numero)
            difference_of_size = nodes_next-nodes_now 
            difference_of_edge_number = edges_next-edges_now
            
            # at each step we add nodes and edges to match with graph at next step
            net = nd.grow_network(net,chromosome,difference_of_size,difference_of_edge_number)
            
            #comparison with the network at next step : returns a dict  : observable-proximity related to the observable
            scores[str(numero+1)] = ne.eval_network(net,numero+1,
                                                  results_path,
                                                  evaluation_method=evaluation_method,
                                                  network_type = network_type)    
        
        #we compute a list that contains aggregated proximity at each time step                      
        scores_aggregated = [score['proximity_aggregated'] for numero,score in scores]
        return min(scores_aggregated)
    
    else :
        number_of_nodes,number_of_edges = ne.get_number_of_nodes_and_edges(results_path)
        
        #we create an empty graph of given type
        init = nd.createGraph(network_type)
        net = nd.grow_network(init,chromosome,int(number_of_nodes),int(number_of_edges))  
        scores = ne.eval_network(net,None,results_path,
                                                  evaluation_method=evaluation_method,
                                                  network_type = network_type)
        return scores['proximity_aggregated']
        '''
