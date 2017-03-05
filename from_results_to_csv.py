import os
import csv
import matplotlib.pyplot as plt

def plot(name,data):
    fig, ax = plt.subplots()
    ax.plot(data,'o-')
    ax.set_ylabel('Normalized Z-score')
    plt.ylim(-0.75, 0.75)
    plt.xlim(-0.5,len(data))
    plt.xticks([])
    plt.yticks([-0.5,0,0.5])
    plt.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_position(('outward', 50))
    #plt.show()
    plt.savefig("results_p/"+name+'.png')
    plt.close()
    #print name+".png"

res = []
fieldnames=['nodes','edges','density','modularity','nb_communities','centralization','size_center','diameter','effective_diameter','motif_78','motif_238','motif_4382','motif_4698','motif_4958','motif_13260','motif_13278','motif_31710']
f2 = ['date','type','nodes','edges','density','modularity','nb_communities','centralization','size_center','diameter','effective_diameter','motif_78','motif_238','motif_4382','motif_4698','motif_4958','motif_13260','motif_13278','motif_31710']
if not os.path.exists("results_p/"):
    os.mkdir("results_p/")
for filename in os.listdir("results_a"):
    try :

        with open("results_a/"+filename) as csvfile:
            file_res = csv.DictReader(csvfile)
            for line in file_res :

                modified_line={field:line[field] for field in fieldnames}
                date = filename.split('_')[-2]
                type = filename.split('_')[-1]
                modified_line["date"]=date
                modified_line["type"]=type

                res.append(modified_line)
                def r(a): return round(float(modified_line.get("motif_"+str(a), '0')), 2)
                plot(filename,map(r,[78,238,4382,4698,4958,13260,13278,31710]))

    except Exception,e:
        print e,filename
        #os.remove("results_a/"+filename)
with open("complete.csv",'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames=f2)
    writer.writeheader()
    for line in res:
        writer.writerow(line)