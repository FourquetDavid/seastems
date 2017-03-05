import os
import networkx as nx
import community as com
from collections import defaultdict
import subprocess
from itertools import combinations
import math
import csv
import sys

def write_motifs(path_to_file):
    print "start"
    subprocess.call(["mfinder1.21/mfinder", path_to_file + ".txt",
                     "-nd", "-omat", "-s", "3", "-f", path_to_file + "_3", "-q"])
    print "3 done"
    subprocess.call(["mfinder1.21/mfinder", path_to_file + ".txt",
                     "-nd", "-omat", "-s", "4", "-f", path_to_file + "_4", "-q"])
    print "4 done"

def write_edgelist(path_to_file):
    graphe = nx.read_gexf(path_to_file + ".gexf")
    for source, target in graphe.edges():
        if source == target :
            graphe.remove_edge(source,target)
        else:
            graphe[source][target].clear()
            graphe[source][target]['weight'] = 1

    nx.write_weighted_edgelist(nx.convert_node_labels_to_integers(graphe), path_to_file + ".txt")

def analyse_motifs(path_to_file):
        motifs_dict = {}
        with open(path_to_file + "_3_MAT.txt") as mat_file_3:
            for line in mat_file_3:
                linesplit = line.split(' ')
                if float(linesplit[2]) != 0:
                    numero = int(linesplit[0])
                    nreal = int(linesplit[1])
                    nrand = float(linesplit[2])
                    srp = (nreal - nrand) / (nreal + nrand + 4)
                    motifs_dict[numero] = srp

        with open(path_to_file + "_4_MAT.txt") as mat_file_4:
            for line in mat_file_4:
                linesplit = line.split(' ')
                if float(linesplit[2]) != 0:
                    numero = int(linesplit[0])
                    nreal = int(linesplit[1])
                    nrand = float(linesplit[2])
                    srp = (nreal - nrand) / (nreal + nrand + 4)
                    motifs_dict[numero] = srp
        total = math.sqrt(sum([i ** 2 for i in motifs_dict.values()]))
        for key in motifs_dict:
            motifs_dict[key] /= total
        return motifs_dict.iteritems()


def effective_diameter(net):
    #90% des paires de noeuds connectes sont a distance inferieure ou egale au diametre effectif
    distances=defaultdict(int)
    sp = nx.shortest_path_length(net)
    for n1,n2 in combinations(net.nodes(),2):
        if n2 in sp[n1]:
            distances[sp[n1][n2]]+=1
    cumsum=0
    total = sum(distances.values())
    for dist in range(max(distances.values())+1):
        cumsum+=distances[dist]
        if cumsum > 0.9*total:
            return dist


resultats={}
fieldnames =[]
def add(name,value) :
    fieldnames.append(name)
    resultats[name]=value

file = sys.argv[1]
res_file= sys.argv[2]

if ".gexf" in file:
        net= nx.read_gexf(file)
        number_nodes = net.number_of_nodes()
        number_edges = net.number_of_edges()
        density = float(number_edges)/number_nodes
        add("nodes",number_nodes)
        add("edges", number_edges)
        add("density", density)

        N = net.order()
        degrees = net.degree().values()
        max_d = max(degrees)
        centralization = float((N * max_d - sum(degrees))) / (N - 1) ** 2
        add("centralization",centralization)

        clust= nx.average_clustering(net)
        add("clustering", centralization)


        mcc = max(nx.connected_component_subgraphs(net),key=len)
        ecc = nx.eccentricity(mcc)
        min_ecc = min(ecc.values())
        size_center = sum([1 for node in mcc if ecc[node]<=min_ecc+1])
        diameter = max(ecc.values())
        ed =effective_diameter(net)
        add("diameter", diameter)
        add("effective_diameter",ed)
        add("size_center", size_center)

        dendo =com.generate_dendogram(net)
        dic_com_nodes=defaultdict(list)
        for node,community in dendo[0].iteritems():
            dic_com_nodes[community].append(node)
        mod = com.modularity(dendo[0],net)
        add("nb_communities", len(dic_com_nodes))
        add("modularity", mod)

        path_to_file = file.replace(".gexf","")
        write_edgelist(path_to_file)
        write_motifs(path_to_file)
        for motif, score in analyse_motifs(path_to_file):
            add("motif_"+str(motif), score)

if not os.path.exists("results_a/"):
    os.makedirs("results_a/")

with open(res_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow(resultats)
