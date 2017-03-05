import os
import csv
import re
import numpy as np

def arrondi(phrase):
    result = phrase
    for number in re.findall("\d+\.\d+", phrase):
        result = result.replace(number, str(round(float(number), 2)))
    return result


def joli(phrase):
    res = re.sub("GTreeNodeBase \[Childs=\d\] - ", "", phrase)
    res = re.sub("[\[\,\]\']", "", res)
    return res


reduced_name = {"NormalizedTargId": 0, "TargId": 0, "OrigId": 0, "NormalizedOrigId": 0,
                "OrigDegree": 1, "TargDegree": 1, "NormalizedOrigDegree": 1, "NormalizedTargDegree": 1,
                "OrigPagerank": 2, "TargPagerank": 2, "OrigCloseness": 3, "TargCloseness": 3, "OrigBetweenness": 4,
                "TargBetweenness": 4,
                "OrigClustering": 5, "TargClustering": 5, "OrigCoreN": 6, "TargCoreN": 6,
                "OrigEccentricity": 7, "TargEccentricity": 7,
                "Distance": 8, "NormalizedDistance": 8, "RevDistance": 8, "NormalizedRevDistance": 8,
                "SameCommunity": 9, "Loop": 10, "CommonNeighbors": 11,
                "NumberOfEdges": 12, "Constant": 13, "Random": 14,

                'NormalizedTargInDegree': 24, 'TargInDegree': 24,
                'NormalizedTargOutDegree': 17, 'TargOutDegree': 17,
                'NormalizedOrigInDegree': 18, 'OrigInDegree': 18,
                'NormalizedOrigOutDegree': 16, 'OrigOutDegree': 16,
                "FeedForwardLoop": 19, "FeedBackLoop": 20,
                "SharedCause": 21, "SharedConsequence": 22, "Reciprocity": 23,
                "EdgeData":15,"NormEdgeData":15}
rnames={"Id":0,"Degree":1,"Pagerank":2,"Closeness": 3, "Betweenness": 4,
                "Clustering": 5,  "CoreN": 6,
                "Eccentricity": 7,
                "Distance": 8,
                "SameCommunity": 9, "Loop": 10, "CommonNeighbors": 11,
                "NumberOfEdges": 12, "Constant": 13, "Random": 14,"GeoDistance":15}
op = ["+", "-", "*", "/", "min", "max", "exp", "log", "abs", "inv", "opp", "H", "T", "N", ">", "<", "="]

def tree_to_vect(tree):
    line = [st for st in tree.pop(0).split(" ") if st != ""]
    if len(line) == 2:
        value = float(line[0])
        variable = line[1]
        res = np.zeros(size)
        res[reduced_name[variable]] += value
        return res
    if len(line) == 1:
        op = line[0]
        if op in ["exp", "log", "abs", "H", "T", "N"]:
            res = tree_to_vect(tree)
            return res
        if op in ["inv", "opp"]:
            res = -tree_to_vect(tree)
            return res
        if op in ["+", "*"]:
            vect1 = tree_to_vect(tree)
            vect2 = tree_to_vect(tree)
            res = vect1 + vect2
            return res
        if op in ["-", "/"]:
            vect1 = tree_to_vect(tree)
            vect2 = tree_to_vect(tree)
            res = vect1 - vect2
            return res

        if op in ["max"]:
            vect1 = tree_to_vect(tree)
            vect2 = tree_to_vect(tree)
            res = vect1 + vect2
            # print res
            return res
        if op in ["min"]:
            vect1 = tree_to_vect(tree)
            vect2 = tree_to_vect(tree)
            res = -vect1 - vect2
            # print res
            return res
        if op in ["="]:
            vect1 = tree_to_vect(tree)
            vect2 = tree_to_vect(tree)
            res = np.zeros(size)
            return res
        if op in ["<"]:
            vect1 = tree_to_vect(tree)
            vect2 = tree_to_vect(tree)
            res = np.zeros(size)
            return res
        if op in [">"]:
            vect1 = tree_to_vect(tree)
            vect2 = tree_to_vect(tree)
            res = np.zeros(size)
            return res

        else:
            print "problem" + op
    else:
        print "problem" + str(line)


def main_tree_to_vect(tree):
    vect = tree_to_vect([lines for lines in tree.splitlines() if lines != ""])
    v = sum(abs(vect))
    if v == 0:
        return vect
    else:
        return vect / v

folder = "results"
dossier = folder
size = 16
with open(dossier + "2.csv", 'w') as vectorfile:
    with open(folder + '.csv', 'w') as csvfile:
        fieldnames = ['date', 'type', 'name','tree0', 'tree1', 'tree2', 'tree3','tree4','fitness0','fitness1','fitness2','fitness3','fitness4']+rnames.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for file in os.listdir(dossier):

            dico = {}
            dico['name'] = file
            dico['date'] = file.split("_")[3]
            dico['type']= file.split("_")[4]
            print dico['name']
            f = open(dossier + "/" + file + "/stats.txt")
            lines = f.readlines()
            found_generation = False
            found_tree = -1
            tree = ""
            vector = np.zeros(size)
            score_total = 0
            for line in lines:
                if line == "#####      Generation  35   ###########\n":
                    found_generation = True

                if found_generation and line == "######### Arbre num 0 ###########\n":
                    found_tree = 0
                    tree = ""

                if found_generation and line == "######### Arbre num 1 ###########\n":
                    found_tree = 1

                    score = 1 - float(tree.splitlines()[1].split()[0])
                    tree = joli(arrondi('\n'.join(tree.split('\n')[2:])))
                    vector += score * main_tree_to_vect(tree)
                    score_total += score
                    dico['tree0'] = tree
                    dico['fitness0']=round(1 - score,2)
                    tree = ""

                if found_generation and line == "######### Arbre num 2 ###########\n":
                    found_tree = 2
                    score = 1 - float(tree.splitlines()[1].split()[0])
                    tree = joli(arrondi('\n'.join(tree.split('\n')[2:])))
                    vector += score * main_tree_to_vect(tree)
                    score_total += score
                    dico['tree1'] = tree
                    dico['fitness1'] = round(1 - score,2)
                    tree = ""

                if found_generation and line == "######### Arbre num 3 ###########\n":
                    found_tree = 3
                    score = 1 - float(tree.splitlines()[1].split()[0])
                    tree = joli(arrondi('\n'.join(tree.split('\n')[2:])))
                    vector += score * main_tree_to_vect(tree)
                    score_total += score
                    dico['tree2'] = tree
                    dico['fitness2'] = round(1 - score,2)
                    tree = ""

                if found_generation and line == "######### Arbre num 4 ###########\n":
                    found_tree = 4
                    score = 1 - float(tree.splitlines()[1].split()[0])
                    tree = joli(arrondi('\n'.join(tree.split('\n')[2:])))
                    vector += score * main_tree_to_vect(tree)
                    score_total += score
                    dico['tree3'] = tree
                    dico['fitness3'] = round(1 - score,2)
                    tree = ""

                if found_generation and found_tree >= 0:
                    tree += line

            score = 1 - float(tree.splitlines()[1].split()[0])
            tree = joli(arrondi('\n'.join(tree.split('\n')[2:])))
            vector += score * main_tree_to_vect(tree)
            score_total += score
            dico['tree4'] = tree
            dico['fitness4'] = round(1 - score,2)
            tree = ""
            for variable in rnames:
                dico[variable]=round(vector[rnames[variable]]/ score_total,2)
            vectorfile.write(dico['name'].replace("_", "-") + "; " + '; '.join(map(str, vector / score_total)) + "\n")


            writer.writerow(dico)
            print 10 * vector / score_total
