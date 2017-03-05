"""
Created on 27 nov. 2012

@author: David
inspired by Telmo Menezes's work : telmomenezes.com
"""

""" 
contains two main function :

*get-goal() : takes an evaluation method and returns the goal that is associated : minimize if we have to minimize the evaluation_funcion
        
*get-alleles() : takes an evaluation method and returns the alleles that are associated : 
 
    

"""


def get_goal(evaluation_method):
    if evaluation_method == "degree_distribution":
        return "minimize"
    if evaluation_method == "2distributions":
        return "minimize"

    return "minimize"
    # raise Exception("no evaluation_method")


def get_alleles(evaluation_method, network_type):
    if evaluation_method == "degree_distribution":
        return [["+", "-", "*", "min", "max"],
                ["OrigId", "TargId", "OrigInDegree", "TargInDegree", "OrigOutDegree", "OrigInDegree"]]

    if evaluation_method == "2distributions":
        if network_type == "directed_weighted":
            return [["+", "-", "*", "/", "min", "max", "exp", "log", "abs", "inv", "opp"],
                    ["TargId", "OrigId", "OrigInDegree", "TargInDegree", "OrigOutDegree", "TargOutDegree",
                     "DirectDistance", "ReversedDistance",
                     "NumberOfEdges", "Constant"
                     ]]
        if network_type == "undirected_weighted":
            return [["+", "-", "*", "/", "min", "max", "exp", "log", "abs", "inv", "opp"],
                    ["TargId", "OrigId", "OrigDegree", "TargDehree", "Distance", "Distance",
                     "NumberOfEdges", "Constant"
                     ]]
        if network_type == "undirected_unweighted":
            return [["+", "-", "*", "/", "min", "max", "exp", "log", "abs", "inv", "opp"],
                    ["TargId", "OrigId", "OrigDegree", "TargDegree", "Distance",
                     "NumberOfEdges", "Constant"
                     ]]
        if network_type == "directed_unweighted":
            return [["+", "-", "*", "/", "min", "max", "exp", "log", "abs", "inv", "opp"],
                    ["TargId", "OrigId", "OrigInDegree", "TargInDegree", "OrigOutDegree", "TargOutDegree",
                     "DirectDistance", "ReversedDistance",
                     "NumberOfEdges", "Constant"
                     ]]

    else:
        if network_type == "directed_weighted" or network_type == "undirected_weighted":
            return [["+", "-", "*", "/", "min", "max", "exp", "log", "abs", "inv", "opp", "H", "T", "N", ">", "<", "="],
                    ["NormalizedTargId", "TargId", "OrigId", "NormalizedOrigId", "OrigInStrength", "TargInStrength",
                     "OrigOutStrength", "TargOutStrength", "DirectDistance", "ReversedDistance",
                     "NormalizedOrigInStrength", "NormalizedTargInStrength", "NormalizedOrigOutStrength",
                     "NormalizedTargOutStrength", "NormalizedDirectDistance", "NormalizedReversedDistance",
                     "AverageInStrength", "AverageOutStrength", "AverageWeight", "AverageDistance", "NumberOfNodes",
                     "NumberOfEdges", "MaxInStrength", "MaxOutStrength", "MaxWeight", "MaxDistance",
                     "TotalDistance", "TotalWeight", "Constant", "Random"
                     ]]
        if network_type == "undirected_unweighted":
            return [["+", "-", "*", "/", "min", "max", "exp", "log", "abs", "inv", "opp", "H", "T", "N", ">", "<", "="],
                    ["NormalizedTargId", "TargId", "OrigId", "NormalizedOrigId",
                     "OrigDegree", "TargDegree", "NormalizedOrigDegree", "NormalizedTargDegree",
                     "OrigPagerank", "TargPagerank", "OrigCloseness", "TargCloseness", "OrigBetweenness",
                     "TargBetweenness",
                     "OrigClustering", "TargClustering", "OrigCoreN", "TargCoreN",
                     "OrigEccentricity","TargEccentricity",
                     "Distance", "NormalizedDistance","RevDistance","NormalizedRevDistance",
                     "SameCommunity","Loop","CommonNeighbors",
                     "NumberOfEdges", "Constant", "Random"
                     ]]
        if network_type == "directed_unweighted" :
            return [["+", "-", "*", "/", "min", "max", "exp", "log", "abs", "inv", "opp", "H", "T", "N", ">", "<", "="],
                    ["NormalizedTargId", "TargId", "OrigId", "NormalizedOrigId",
                     "OrigInDegree", "TargInDegree", "NormalizedOrigInDegree", "NormalizedTargInDegree",
                     "OrigOutDegree", "TargOutDegree", "NormalizedOrigOutDegree", 'NormalizedTargOutDegree',
                     "OrigPagerank", "TargPagerank", "OrigCloseness", "TargCloseness", "OrigBetweenness",
                     "TargBetweenness",
                     "OrigClustering", "TargClustering", "OrigCoreN", "TargCoreN",
                     "OrigEccentricity","TargEccentricity",
                     "Distance", "NormalizedDistance","RevDistance","NormalizedRevDistance",
                     "SameCommunity","Reciprocity",  "FeedForwardLoop","FeedBackLoop","SharedConsequence","SharedCause",
                     "NumberOfEdges", "Constant", "Random"
                     ]]

    raise Exception("no evaluation_method or network_type given")
    # if evaluation_method == "weighted" :
    #    return [["+","-","*","/","min","max","exp","log","abs","inv"],
    #                   ["NormalizedTargId","NormalizedOrigId","OrigInStrength","TargInStrength","TargOutStrength","OrigInStrength","DirectDistance","ReversedDistance"]]
