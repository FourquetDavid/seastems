'''
Created on 15 nov. 2012 

@author: David
inspired by Telmo Menezes's work : telmomenezes.com
'''

""" 
this class inherits from pyevovle.DBAdapters.DBBaseAdaoter. It computes and stores statistics during genetic algorithm processing.
       

"""
import pydot
import pyevolve as py
import pyevolve.DBAdapters as db
import logging
from collections import defaultdict
import operator
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree as xml


class StatisticsInTxt(db.DBBaseAdapter):
    ''' This class inherits from DBAdpater in pyevolve, it will be called at each generation of the genetic algorithm
    and print stats in a txt file and print it on screen
    '''

    def __init__(self, filename=None, identify=None,
                 frequency=py.Consts.CDefCSVFileStatsGenFreq, reset=True):
        """ The creator of StatisticsInTxt Class """

        db.DBBaseAdapter.__init__(self, frequency, identify)
        self.filename = filename
        self.file = None
        self.reset = reset

    def __repr__(self):
        """ The string representation of adapter """
        ret = "StatisticsInTxt DB Adapter [File='%s', identify='%s']" % (self.filename, self.getIdentify())
        return ret

    def open(self, ga_engine):
        """ Open the Txt file or creates a new file
        """

        logging.debug("Opening the txt file to dump statistics [%s]", self.filename)
        if self.reset:
            open_mode = "w"
        else:
            open_mode = "a"
        self.file = open(self.filename, open_mode)

        self.file.write("name = %s \n" % ga_engine.getPopulation().oneSelfGenome.getParam("name"))
        self.file.write("number of generations = %s \n" % ga_engine.getGenerations())
        self.file.write(
            "evaluation_method = %s \n" % ga_engine.getPopulation().oneSelfGenome.getParam("evaluation_method"))
        self.file.write("network_type = %s \n" % ga_engine.getPopulation().oneSelfGenome.getParam("network_type"))
        self.file.write("tree_type = %s \n" % ga_engine.getPopulation().oneSelfGenome.getParam("tree_type"))
        if ga_engine.getPopulation().oneSelfGenome.getParam("edge_data") is not None :
            self.file.write("with_data")
        self.file.write("selector = %s \n" % ga_engine.selector)
        self.file.write("multiprocessing = %s \n" % ga_engine.getPopulation().multiProcessing[0])

        print "name = %s" % ga_engine.getPopulation().oneSelfGenome.getParam("name")
        print "number of generations = %s" % ga_engine.getGenerations()
        print "evaluation_method = %s" % ga_engine.getPopulation().oneSelfGenome.getParam("evaluation_method")
        print "network_type = %s" % ga_engine.getPopulation().oneSelfGenome.getParam("network_type")
        print "tree_type = %s" % ga_engine.getPopulation().oneSelfGenome.getParam("tree_type")
        if ga_engine.getPopulation().oneSelfGenome.getParam("edge_data") is not None:
            print "with_data"
        print "selector = %s" % ga_engine.selector
        print "multiprocessing = %s" % ga_engine.getPopulation().multiProcessing[0]

        self.file.close()

    def close(self):
        """ Closes the Txt file  """
        logging.debug("Closing the txt file [%s]", self.filename)
        self.file.close()

    def commitAndClose(self):
        """ Commits and closes """
        self.close()

    def insert(self, ga_engine):
        self.file = open(self.filename, 'a')
        """ writes population statistics and the 5 best elements"""
        self.file.write(
            "#####      Generation  {numero}   ###########\n".format(numero=ga_engine.getCurrentGeneration()))
        print "#####      Generation  {numero}   ###########".format(numero=ga_engine.getCurrentGeneration())
        print ga_engine.getStatistics()
        self.file.write(ga_engine.getStatistics().__repr__())

        pop = ga_engine.getPopulation()
        for i in xrange(25):
            self.file.write("######### Arbre num {numero} ###########\n".format(numero=i))
            print "######### Arbre num {numero} ###########".format(numero=i)
            tree = pop.bestFitness(i)
            self.file.write(str(tree.getRawScore()))
            print tree.getRawScore()
            self.file.write(getTreeString(tree))
            print getTreeString(tree)
        self.file.close()

class StatisticsInDot(db.DBBaseAdapter):
    ''' This class inherits from DBAdpater in pyevolve, it will be called at each generation of the genetic algorithm
    and print best individuals in dot format
    '''

    def __init__(self, filename=None, identify=None,
                 frequency=py.Consts.CDefCSVFileStatsGenFreq, dot_path=None, reset=True):
        """ The creator of StatisticsInTxt Class """

        db.DBBaseAdapter.__init__(self, frequency, identify)
        self.dot_path = dot_path
        self.filename = filename
        self.file = None
        self.reset = reset

    def __repr__(self):
        """ The string representation of adapter """
        ret = "StatisticsInTxt DB Adapter [File='%s', identify='%s']" % (self.filename, self.getIdentify())
        return ret

    def open(self, ga_engine):
        """ Open the Txt file or creates a new file
        """

        logging.debug("Opening the txt file to dump statistics [%s]", self.filename)
        if self.reset:
            open_mode = "w"
        else:
            open_mode = "a"
        self.file = open(self.filename, open_mode)

        self.file.write("name = %s \n" % ga_engine.getPopulation().oneSelfGenome.getParam("name"))
        self.file.write("number of generations = %s \n" % ga_engine.getGenerations())
        self.file.write(
            "evaluation_method = %s \n" % ga_engine.getPopulation().oneSelfGenome.getParam("evaluation_method"))
        self.file.write("network_type = %s \n" % ga_engine.getPopulation().oneSelfGenome.getParam("network_type"))
        self.file.write("tree_type = %s \n" % ga_engine.getPopulation().oneSelfGenome.getParam("tree_type"))
        if ga_engine.getPopulation().oneSelfGenome.getParam("edge_data") is not None :
            self.file.write("with_data")
        self.file.write("selector = %s \n" % ga_engine.selector)
        self.file.write("multiprocessing = %s \n" % ga_engine.getPopulation().multiProcessing[0])

        print "name = %s" % ga_engine.getPopulation().oneSelfGenome.getParam("name")
        print "number of generations = %s" % ga_engine.getGenerations()
        print "evaluation_method = %s" % ga_engine.getPopulation().oneSelfGenome.getParam("evaluation_method")
        print "network_type = %s" % ga_engine.getPopulation().oneSelfGenome.getParam("network_type")
        print "tree_type = %s" % ga_engine.getPopulation().oneSelfGenome.getParam("tree_type")
        if ga_engine.getPopulation().oneSelfGenome.getParam("edge_data") is not None:
            print "with_data"
        print "selector = %s" % ga_engine.selector
        print "multiprocessing = %s" % ga_engine.getPopulation().multiProcessing[0]
        self.file.close()
    def close(self):
        """ Closes the Txt file  """
        logging.debug("Closing the txt file [%s]", self.filename)
        self.file.close()

    def commitAndClose(self):
        """ Commits and closes """
        self.close()

    def insert(self, ga_engine):
        self.file = open(self.filename, "a")
        """ writes population statistics and the 5 best elements"""
        self.file.write(
            "#####      Generation  {numero}   ###########\n".format(numero=ga_engine.getCurrentGeneration()))
        print "#####      Generation  {numero}   ###########".format(numero=ga_engine.getCurrentGeneration())
        print ga_engine.getStatistics()
        self.file.write(ga_engine.getStatistics().__repr__())
        print ga_engine.getPopulation()
        writePopulationDot(ga_engine, self.dot_path, "jpeg", 0, 25)
        pop = ga_engine.getPopulation()

        for i in xrange(5):
            self.file.write("######### Arbre num {numero} ###########\n".format(numero=i))
            print "######### Arbre num {numero} ###########".format(numero=i)
            tree = pop.bestFitness(i)
            scores = {k: str(round(v,2)) for k, v in tree.scoref.score.items()}
            score = str(round(tree.getRawScore(), 2))
            score_str = score+" D:"+scores.get('degrees',"None")+" ID:"+scores.get('indegrees',"None")+" OD:"+scores.get('outdegrees',"None")+\
                        " Di:"+scores['distances']+" C:"+scores['clustering']+" I:"+scores['importance']+" Co:"+scores['communities']+"\n"
            self.file.write(score_str+"\n")
            print(score_str)
            self.file.write(getTreeString(tree))
            print getTreeString(tree)
        self.file.close()


def getTreeString(tree, start_node=None, spc=0):
    """ Returns a tree-formated string of the tree. This
        method is used by the __repr__ method of the tree
          
        :rtype: a string representing the tree
        """

    str_buff = ""
    if start_node is None:
        start_node = tree.getRoot()
        if start_node.isLeaf():
            reprint_start_node = start_node.clone()
            number,variable = reprint_start_node.getData()
            reprint_start_node.setData([round(number,2),variable])
            str_buff += "%s\n" % reprint_start_node
        else:
            str_buff += "%s\n" % start_node
    spaces = spc + 2
    if start_node.getData() in ["exp", "log", "abs", "inv", "opp", "H", "T", "N"]:
        child_node = start_node.getChild(0)
        str_buff += "%s%s\n" % (" " * spaces, child_node)
        str_buff += getTreeString(tree, child_node, spaces)
    else:
        for child_node in start_node.getChilds():
            str_buff += "%s%s\n" % (" " * spaces, child_node)
            str_buff += getTreeString(tree, child_node, spaces)
    return str_buff


def writePopulationDot(ga_engine, filename, format="jpeg", start=0, end=0):
    """ Writes to a graphical file using pydot, the population of trees


      :param ga_engine: the GA Engine
      :param filename: the filename, ie. population.jpg
      :param start: the start index of individuals
      :param end: the end index of individuals
      """
    pop = ga_engine.getPopulation()
    graph = pydot.Dot(graph_type="digraph")

    n = 0
    end_index = len(pop) if end == 0 else end
    for i in xrange(start, end_index):
        ind = pop[i]
        subg = pydot.Cluster("cluster_%d" % i, label="\"Ind. #%d - Score Raw/Fit.: %.4f/%.4f\"" % (
        i, ind.getRawScore(), ind.getFitnessScore()))
        n = writeDotGraph(ind, subg, n)
        graph.add_subgraph(subg)

    graph.write(filename, prog='dot', format=format)
    graph.write(filename.replace(".jpeg",".dot"), prog='dot', format="raw")


def writeDotGraph(tree, graph, startNode=0):
    """ Write a graph to the pydot Graph instance
      
      :param graph: the pydot Graph instance
      :param startNode: used to plot more than one individual 
      """

    count = startNode
    node_stack = []
    nodes_dict = {}
    tmp = None
    def add_node(node,count):
        newnode = pydot.Node(str(count), style="filled")
        newnode.set_color("goldenrod2")
        data = '\n'.join(map(str, node.getData()))
        newnode.set_label(data)
        nodes_dict.update({node: newnode})
        graph.add_node(newnode)
        return count +1


    node_stack.append(tree.getRoot())
    while len(node_stack) > 0:
        tmp = node_stack.pop()
        count =add_node(tmp,count)
        parent = tmp.getParent()
        if parent is not None:
            parent_node = nodes_dict[parent]
            child_node = nodes_dict[tmp]

            newedge = pydot.Edge(parent_node, child_node)
            graph.add_edge(newedge)
        if tmp.getData() in ["exp", "log", "abs", "inv", "opp", "H", "T", "N"]:
            node_stack.append(tmp.getChild(0))
        if tmp.getData() in ["+", "-", "*", "/", "min", "max", ">", "<", "="]:
            node_stack.extend([tmp.getChild(1),tmp.getChild(0)])

    return count

'''
Function that plts and stores stats from best individual
'''
def store_best_network(chromosome) :
    eval_methods = chromosome.getParam("evaluation_method")
    results_path = chromosome.getParam("results_path")
    dynamic_network = xml.parse(results_path).getroot()
    graph_xml = dynamic_network.find("mesures")

    def plot(mesure) :
        value_test = chromosome.scoref.distributions[mesure]
        value_goal = eval(graph_xml.find(mesure).get('value'))
        fig, ax = plt.subplots()
        model = ax.bar(2 * np.arange(len(value_test)), value_test, 0.8, color='#ccffcc')
        real = ax.bar(2 * np.arange(len(value_goal)) + 0.8, value_goal, 0.8, color='#ff9999')
        ax.set_xticks(0.8 + 2 * np.arange(max(len(value_goal), len(value_test))))
        ax.set_xticklabels(1 + np.arange(max(len(value_goal), len(value_test))))
        ax.set_title("Distribution of " + mesure)
        ax.legend((real, model), ('Real', 'Model'))
        plt.savefig(results_path.replace("results.xml", "") + mesure + ".png")
        plt.clf()



    def save(mesure):
        parser = xml.XMLParser(remove_blank_text=True)
        tree = xml.parse(results_path, parser)
        results = tree.getroot()
        static_network = results.find("model")
        try:
            results.remove(results.find(mesure))
        except TypeError:
            pass
        xml.SubElement(static_network, mesure, value=str(chromosome.scoref.distributions[mesure]))
        f = open(results_path, 'w')
        tree.write(f, pretty_print=True)
        f.close()

    def save_tree():
        parser = xml.XMLParser(remove_blank_text=True)
        tree = xml.parse(results_path, parser)
        results = tree.getroot()
        try:
            results.remove(results.find("model"))
        except TypeError:
            pass
        static_network = xml.SubElement(results, "model")
        xml.SubElement(static_network, "tree", value=getTreeString(chromosome))
        f = open(results_path, 'w')
        tree.write(f, pretty_print=True)
        f.close()

    save_tree()
    for mesure in eval_methods.split('_'):
        save(mesure)
        plot(mesure)





list_of_functions_number = defaultdict(int)
list_of_functions_sum = defaultdict(int)
count = 0


class StatisticsQualityInTxt(py.DBAdapters.DBBaseAdapter):
    ''' This class inherits from DBAdpater in pyevolve, it will be called at each generation of the genetic algorithm
    and print stats in a txt file and print it on screen
    '''

    def __init__(self, current_variable, filename=None, identify=None,
                 frequency=py.Consts.CDefCSVFileStatsGenFreq, reset=True):
        """ The creator of StatisticsInTxt Class """
        global count

        py.DBAdapters.DBBaseAdapter.__init__(self, frequency, identify)
        self.filename = filename
        self.file = None
        self.reset = reset

    def __repr__(self):
        """ The string representation of adapter """
        ret = "StatisticsQualityInTxt DB Adapter [File='%s', identify='%s']" % (self.filename, self.getIdentify())
        return ret

    def open(self, ga_engine):
        """ Open the Txt file or creates a new file
        """

        logging.debug("Opening the txt file to dump statistics [%s]", self.filename)
        if self.reset:
            open_mode = "w"
        else:
            open_mode = "a"
        self.file = open(self.filename, open_mode)

    def close(self):
        """ Closes the Txt file  """
        logging.debug("Closing the txt file [%s]", self.filename)
        global list_of_functions_number
        global list_of_functions_sum
        global count

        count += 1
        if count == 10:
            count = 0
            list_of_functions_quality = {}
            list_of_functions_product = {}
            list_of_functions_rapport = {}

            for key in list_of_functions_number:
                list_of_functions_quality[key] = list_of_functions_sum[key] / list_of_functions_number[key]

            maximum = max(list_of_functions_quality.values())
            for key in list_of_functions_quality:
                list_of_functions_quality[key] = maximum - list_of_functions_quality[key]
                list_of_functions_product[key] = list_of_functions_quality[key] * list_of_functions_number[key]
                list_of_functions_rapport[key] = list_of_functions_number[key] * list_of_functions_number[key] / \
                                                 list_of_functions_sum[key]

            sorted_quality = sorted(list_of_functions_quality.iteritems(), key=operator.itemgetter(1))[-8:]
            sorted_product = sorted(list_of_functions_product.iteritems(), key=operator.itemgetter(1))[-8:]
            sorted_rapport = sorted(list_of_functions_rapport.iteritems(), key=operator.itemgetter(1))[-8:]

            print "\n### Sorted by Quality ####"
            for (key, _) in sorted_quality:
                print " ".join(
                    ["variable :", key, "number of apparitions =", str(list_of_functions_number[key]), "quality =",
                     str(list_of_functions_quality[key])])
            print "\n#####Sorted by Quantity*Quality#########"
            for (key, _) in sorted_product:
                print " ".join(
                    ["variable :", key, "number of apparitions =", str(list_of_functions_number[key]), "quality =",
                     str(list_of_functions_quality[key])])
            print "\n######Sorted by Rate######"
            for (key, _) in sorted_rapport:
                print " ".join(
                    ["variable :", key, "number of apparitions =", str(list_of_functions_number[key]), "quality =",
                     str(list_of_functions_quality[key])])

            self.file.write("\n### Sorted by Quality ####\n")
            for (key, _) in sorted_quality:
                self.file.write(" ".join(
                    ["variable :", key, "number of apparitions =", str(list_of_functions_number[key]), "quality =",
                     str(list_of_functions_quality[key]), "\n"]))
            self.file.write("\n#####Sorted by Quantity*Quality#########\n")
            for (key, _) in sorted_product:
                self.file.write(" ".join(
                    ["variable :", key, "number of apparitions =", str(list_of_functions_number[key]), "quality =",
                     str(list_of_functions_quality[key]), "\n"]))
            self.file.write("\n######Sorted by Rate######\n")
            for (key, _) in sorted_rapport:
                self.file.write(" ".join(
                    ["variable :", key, "number of apparitions =", str(list_of_functions_number[key]), "quality =",
                     str(list_of_functions_quality[key]), "\n"]))

            list_of_functions_number.clear()
            list_of_functions_sum.clear()
        self.file.close()

    def commitAndClose(self):
        """ Commits and closes """
        self.close()

    def insert(self, ga_engine):
        global list_of_functions_number
        global list_of_functions_sum

        pop = ga_engine.getPopulation()
        for element in pop:
            score = element.getRawScore()
            for node in element.getAllNodes():
                if node.isLeaf():
                    variable = node.getData()[1]
                    list_of_functions_number[variable] += 1
                    list_of_functions_sum[variable] += score
                    # else :
                    # variable = node.getData()
                    # list_of_functions_number[variable]+=1
                    # list_of_functions_sum[variable]+=score
