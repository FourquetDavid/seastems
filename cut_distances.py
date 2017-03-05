import networkx as nx
import os
for filename in os.listdir("dwt_weighted_network_distance_year"):
    print filename
    if ".gexf" in filename :
        graphe = nx.read_gexf("dwt_weighted_network_distance_year/"+filename)
        year=filename.split('_')[4].replace(".gexf","")
        edge_data_path="distances/maritime_distance_matrix_{}.csv".format(year)
        distances = nx.read_weighted_edgelist(edge_data_path)
        subdist=nx.subgraph(distances,graphe.nodes())
        nx.write_gexf(subdist,"distances_d/"+filename.replace("weighted","distance"))
        print graphe.number_of_nodes(),subdist.number_of_nodes()