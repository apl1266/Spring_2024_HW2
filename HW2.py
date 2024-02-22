import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import mlrose_hiive
import time
import random

np.random.seed(812)


def frames(name,rat):
    pd_fruits=pd.read_csv(name)
    if name=="fruits_data.csv":
        pd_fruits.loc[pd_fruits["Class"] == "BERHI", "Class"] = 0
        pd_fruits.loc[pd_fruits["Class"] == "DEGLET", "Class"] = 1
        pd_fruits.loc[pd_fruits["Class"] == "DOKOL", "Class"] = 2
        pd_fruits.loc[pd_fruits["Class"] == "IRAQI", "Class"] = 3
        pd_fruits.loc[pd_fruits["Class"] == "ROTANA", "Class"] = 4
        pd_fruits.loc[pd_fruits["Class"] == "SAFAVI", "Class"] = 5
        pd_fruits.loc[pd_fruits["Class"] == "SOGAY", "Class"] = 6

    scaler=preprocessing.MinMaxScaler()
    np_fruits=pd_fruits.values
    np_fruits_scaled=scaler.fit_transform(np_fruits)
    np.random.shuffle(np_fruits_scaled)
    if name == "fruits_data.csv":
        np_fruits_scaled[:,-1]=np_fruits_scaled[:,-1]*6
    else:
        np_fruits_scaled[:, -1] = np_fruits_scaled[:, -1] * 3
    split=int(np.shape(np_fruits_scaled)[0]*rat//1)
    np_fruits_scaled[:,-1] = np_fruits_scaled[:,-1].astype(int)
    return np_fruits_scaled[split:,:-1],np_fruits_scaled[split:,-1].flatten(),np_fruits_scaled[:split,:-1],np_fruits_scaled[:split,-1].flatten()

def plotter(x,y,label,name,x_label="number of iterations"):
    plt.plot(x, y , "-o", label=label)
    plt.xlabel(x_label)
    plt.ylabel("score")
    plt.savefig(name)
    plt.clf()

def plot_11(train_f_score, validation_f_score, x_axis, title,x_label,file_name=False, extra_label=False):
    plt.plot(x_axis, train_f_score , "-o", label="train")
    plt.plot(x_axis, validation_f_score, "-o", label="validation")
    if extra_label:
        plt.plot([], [], ' ', label="Best fit parameters")
        plt.plot([], [], ' ', label=extra_label)
    plt.xlabel(x_label)
    plt.ylabel("f_score")
    plt.title(title)
    plt.legend()
    if file_name:
        plt.savefig(file_name)
    plt.clf()

def queeens_fun(board):
    val = 0
    for i in range(len(board) - 1):
        for j in range(i + 1, len(board)):
            if (board[j] != board[i]) and (board[j] != board[i] - (j - i)) and (board[j] != board[i] + (j - i)):
                val += 1
    return val

fun = mlrose_hiive.CustomFitness(queeens_fun)
max_attack_evade = 0
queens = 70
for i in range(queens-1):
    for j in range(i+1,queens):
        max_attack_evade+=1
print("---------------------")
print("For ",queens," max number of evadable atacks is ",max_attack_evade)
print("---------------------")

problem_set = mlrose_hiive.DiscreteOpt(queens,fun,max_val = queens)

if 0:
    t0=time.time()
    queens_RHC = mlrose_hiive.random_hill_climb(problem_set, max_attempts=5000, max_iters=10000, restarts=0,init_state=None, curve=True, random_state=812)
    dt=time.time()-t0
    print("Queens RHC for ",queens," run for ", dt," seconds ", "with the best score of ",  queens_RHC[1])
    plotter(queens_RHC[2][:,1], queens_RHC[2][:,0],"RHC for "+str(queens)+" queens", "Queens_RHC.png")
    print("---------------------")

if 0:
    t0=time.time()
    queens_SA = mlrose_hiive.simulated_annealing(problem_set, schedule=mlrose_hiive.GeomDecay(), max_attempts=1000,max_iters=10000, init_state=None, curve=True,random_state=812)
    dt=time.time()-t0
    print("Queens SA for ",queens," run for ", dt," seconds ", "with the best score of ",  queens_SA[1])
    plotter(queens_SA[2][:,1], queens_SA[2][:,0],"SA for "+str(queens)+" queens", "Queens_SA.png")
    print("---------------------")

if 0:
    t0=time.time()
    queens_GA = mlrose_hiive.genetic_alg(problem_set, pop_size=200, mutation_prob=0.1, max_attempts=20,max_iters=20000, curve=True, random_state=812)
    dt=time.time()-t0
    print("Queens GA for ",queens," run for ", dt," seconds ", "with the best score of ",  queens_GA[1])
    plotter(queens_GA[2][:,1], queens_GA[2][:,0],"GA for "+str(queens)+" queens", "Queens_GA.png","total population number")
    print("---------------------")

if 0:
    t0=time.time()
    queens_MIMIC = mlrose_hiive.mimic(problem_set, pop_size=500, keep_pct=0.5, max_attempts=10,max_iters=20000, curve=True, random_state=812)
    dt=time.time()-t0
    print("Queens MIMIC for ",queens," run for ", dt," seconds ", "with the best score of ",  queens_MIMIC[1])
    plotter(queens_MIMIC[2][:,1], queens_MIMIC[2][:,0],"MIMIC for "+str(queens)+" queens", "Queens_MIMIC.png","total population number")
    print("---------------------")


#FlipFlop
flips=400
problem_set=mlrose_hiive.FlipFlopOpt(length=flips)

if 0:
    t0=time.time()
    FlipFlop_RHC = mlrose_hiive.random_hill_climb(problem_set, max_attempts=5000, max_iters=10000, restarts=0,init_state=None, curve=True, random_state=812)
    dt=time.time()-t0
    print("FlipFlop RHC for ",flips," run for ", dt," seconds ", "with the best score of ",  FlipFlop_RHC[1])
    plotter(FlipFlop_RHC[2][:,1], FlipFlop_RHC[2][:,0],"RHC for "+str(flips)+" FlipFlops", "FlipFlop_RHC.png")
    print("---------------------")

if 0:
    t0=time.time()
    FlipFlop_SA = mlrose_hiive.simulated_annealing(problem_set, schedule=mlrose_hiive.GeomDecay(), max_attempts=2000,max_iters=100000, init_state=None, curve=True,random_state=812)
    dt=time.time()-t0
    print("FlipFlop SA for ",flips," run for ", dt," seconds ", "with the best score of ",  FlipFlop_SA[1])
    plotter(FlipFlop_SA[2][:,1], FlipFlop_SA[2][:,0],"SA for "+str(flips)+" FlipFlops", "FlipFlop_SA.png")
    print("---------------------")

if 0:
    t0=time.time()
    FlipFlop_GA = mlrose_hiive.genetic_alg(problem_set, pop_size=200, mutation_prob=0.1, max_attempts=200,max_iters=200000, curve=True, random_state=812)
    dt=time.time()-t0
    print("FlipFlop GA for ",flips," run for ", dt," seconds ", "with the best score of ",  FlipFlop_GA[1])
    plotter(FlipFlop_GA[2][:,1], FlipFlop_GA[2][:,0],"GA for "+str(flips)+" FlipFlops", "FlipFlop_GA.png","total population number")
    print("---------------------")

if 0:
    t0=time.time()
    FlipFlop_MIMIC = mlrose_hiive.mimic(problem_set, pop_size=20, keep_pct=0.5, max_attempts=10,max_iters=50, curve=True, random_state=812)
    dt=time.time()-t0
    print("FlipFlop MIMIC for ",flips," run for ", dt," seconds ", "with the best score of ",  FlipFlop_MIMIC[1])
    plotter(FlipFlop_MIMIC[2][:,1], FlipFlop_MIMIC[2][:,0],"FlipFlop for "+str(flips)+" FlipFlops", "FlipFlop_MIMIC.png","total population number")
    print("---------------------")


#TSP
np.random.seed(812)
num_towns=200
grid=2*num_towns
towns=[]
for i in range(num_towns):
    towns.append((int(np.random.randint(0,grid)),int(np.random.randint(0,grid))))
cords=mlrose_hiive.TravellingSales(coords=towns)
problem_set = mlrose_hiive.TSPOpt(length = num_towns, fitness_fn = cords, maximize=False)

if 0:
    t0=time.time()
    TSP_RHC = mlrose_hiive.random_hill_climb(problem_set, max_attempts=5000, max_iters=100000, restarts=0,init_state=None, curve=True, random_state=812)
    dt=time.time()-t0
    print("TSP RHC for ",num_towns," run for ", dt," seconds ", "with the best score of ",  TSP_RHC[1])
    plotter(TSP_RHC[2][:,1], TSP_RHC[2][:,0],"RHC for "+str(num_towns)+" towns", "TSP_RHC.png")
    print("---------------------")

if 0:
    t0=time.time()
    TSP_SA = mlrose_hiive.simulated_annealing(problem_set, schedule=mlrose_hiive.GeomDecay(), max_attempts=2000,max_iters=100000, init_state=None, curve=True,random_state=812)
    dt=time.time()-t0
    print("TSP SA for ",num_towns," run for ", dt," seconds ", "with the best score of ",  TSP_SA[1])
    plotter(TSP_SA[2][:,1], TSP_SA[2][:,0],"SA for "+str(num_towns)+" towns", "TSP_SA.png")
    print("---------------------")

if 0:
    t0=time.time()
    TSP_GA = mlrose_hiive.genetic_alg(problem_set, pop_size=200, mutation_prob=0.1, max_attempts=10,max_iters=500, curve=True, random_state=812)
    dt=time.time()-t0
    print("TSP GA for ",num_towns," run for ", dt," seconds ", "with the best score of ",  TSP_GA[1])
    plotter(TSP_GA[2][:,1], TSP_GA[2][:,0],"GA for "+str(num_towns)+" towms", "TSP_GA.png","total population number")
    print("---------------------")

if 0:
    t0=time.time()
    #TSP_MIMIC = mlrose_hiive.mimic(problem_set, pop_size=100, keep_pct=0.1, max_attempts=5,max_iters=30, curve=True, random_state=812)
    TSP_MIMIC = mlrose_hiive.MIMICRunner(problem=problem_set,
                                   experiment_name="MMC_Exp",
                                   seed=44,
                                   iteration_list=[100],
                                   max_attempts=10,
                                   population_sizes=[20, 50, 100],
                                   keep_percent_list=[0.25, 0.5, 0.75],
                                   use_fast_mimic=True)
    mmc_run_stats, mmc_run_curves = TSP_MIMIC.run()
    dt=time.time()-t0
    #print("TSP MIMIC for ",num_towns," run for ", dt," seconds ", "with the best score of ",  TSP_MIMIC[1])
    plotter(mmc_run_stats, mmc_run_curves,"TSP for "+str(num_towns)+" towns", "TSP_MIMIC.png","total population number")
    print("---------------------")

#Continious Peaks
peak_len=1000
problem_set=mlrose_hiive.DiscreteOpt(length=peak_len,fitness_fn=mlrose_hiive.ContinuousPeaks(),maximize=True,max_val=2)

if 0:
    t0=time.time()
    CC_RHC = mlrose_hiive.random_hill_climb(problem_set, max_attempts=5000, max_iters=500000, restarts=0,init_state=None, curve=True, random_state=812)
    dt=time.time()-t0
    print("CC RHC for ",peak_len," run for ", dt," seconds ", "with the best score of ",  CC_RHC[1])
    plotter(CC_RHC[2][:,1], CC_RHC[2][:,0],"CC for "+str(peak_len)+" problem length", "CC_RHC.png")
    print("---------------------")

if 0:
    t0=time.time()
    CC_SA = mlrose_hiive.simulated_annealing(problem_set, schedule=mlrose_hiive.GeomDecay(), max_attempts=2000,max_iters=500000, init_state=None, curve=True,random_state=812)
    dt=time.time()-t0
    print("CC SA for ",peak_len," run for ", dt," seconds ", "with the best score of ",  CC_SA[1])
    plotter(CC_SA[2][:,1], CC_SA[2][:,0],"SA for "+str(peak_len)+" problem length", "CC_SA.png")
    print("---------------------")

if 0:
    t0=time.time()
    CC_GA = mlrose_hiive.genetic_alg(problem_set, pop_size=200, mutation_prob=0.1, max_attempts=100,max_iters=5000, curve=True, random_state=812)
    dt=time.time()-t0
    print("CC GA for ",peak_len," run for ", dt," seconds ", "with the best score of ",  CC_GA[1])
    plotter(CC_GA[2][:,1], CC_GA[2][:,0],"GA for "+str(peak_len)+" problem length", "CC_GA.png","total population number")
    print("---------------------")

if 0:
    t0=time.time()
    CC_MIMIC = mlrose_hiive.mimic(problem_set, pop_size=20, keep_pct=0.5, max_attempts=10,max_iters=50, curve=True, random_state=812)
    dt=time.time()-t0
    print("CC MIMIC for ",peak_len," run for ", dt," seconds ", "with the best score of ",  CC_MIMIC[1])
    plotter(CC_MIMIC[2][:,1], CC_MIMIC[2][:,0],"CC for "+str(peak_len)+" problem length", "CC_MIMIC.png","total population number")
    print("---------------------")

#knap
knap_num=500
problem_set = mlrose_hiive.KnapsackOpt(weights=np.random.uniform(20,40,knap_num), values=np.random.uniform(20,30,knap_num),max_weight_pct=0.9)

if 0:
    t0=time.time()
    KS_RHC = mlrose_hiive.random_hill_climb(problem_set, max_attempts=5000, max_iters=5000, restarts=0,init_state=None, curve=True, random_state=812)
    dt=time.time()-t0
    print("KS RHC for ",knap_num," run for ", dt," seconds ", "with the best score of ",  KS_RHC[1])
    plotter(KS_RHC[2][:,1], KS_RHC[2][:,0],"KS for "+str(knap_num)+" problem length", "KS_RHC.png")
    print("---------------------")

if 0:
    t0=time.time()
    KS_SA = mlrose_hiive.simulated_annealing(problem_set, schedule=mlrose_hiive.GeomDecay(), max_attempts=5000,max_iters=500000, init_state=None, curve=True,random_state=812)
    dt=time.time()-t0
    print("KS SA for ",knap_num," run for ", dt," seconds ", "with the best score of ",  KS_SA[1])
    plotter(KS_SA[2][:,1], KS_SA[2][:,0],"KS for "+str(knap_num)+" problem length", "KS_SA.png")
    print("---------------------")

if 0:
    t0=time.time()
    KS_GA = mlrose_hiive.genetic_alg(problem_set, pop_size=200, mutation_prob=0.1, max_attempts=100,max_iters=5000, curve=True, random_state=812)
    dt=time.time()-t0
    print("KS GA for ",knap_num," run for ", dt," seconds ", "with the best score of ",  KS_GA[1])
    plotter(KS_GA[2][:,1], KS_GA[2][:,0],"KS for "+str(knap_num)+" problem length", "KS_GA.png","total population number")
    print("---------------------")

if 0:
    t0=time.time()
    KS_MIMIC = mlrose_hiive.mimic(problem_set, pop_size=20, keep_pct=0.5, max_attempts=10,max_iters=50, curve=True, random_state=812)
    dt=time.time()-t0
    print("KS MIMIC for ",knap_num," run for ", dt," seconds ", "with the best score of ",  KS_MIMIC[1])
    plotter(KS_MIMIC[2][:,1], KS_MIMIC[2][:,0],"KS for "+str(knap_num)+" problem length", "KS_MIMIC.png","total population number")
    print("---------------------")


#Four peaks
peaks_num=1000
problem_set = mlrose_hiive.DiscreteOpt(length = peaks_num, fitness_fn = mlrose_hiive.FourPeaks(t_pct=0.15), maximize=True,max_val=2)
if 0:
    t0=time.time()
    FP_RHC = mlrose_hiive.random_hill_climb(problem_set, max_attempts=5000, max_iters=50000, restarts=0,init_state=None, curve=True, random_state=812)
    dt=time.time()-t0
    print("FP RHC for ", peaks_num, " run for ", dt, " seconds ", "with the best score of ", FP_RHC[1])
    plotter(FP_RHC[2][:, 1], FP_RHC[2][:, 0], "FP for " + str(peaks_num) + " problem length", "FP_RHC.png")
    print("---------------------")

if 0:
    t0=time.time()
    FP_SA = mlrose_hiive.simulated_annealing(problem_set, schedule=mlrose_hiive.GeomDecay(), max_attempts=50000,max_iters=2000000, init_state=None, curve=True,random_state=812)
    dt=time.time()-t0
    print("FP SA for ",knap_num," run for ", dt," seconds ", "with the best score of ",  FP_SA[1])
    plotter(FP_SA[2][:,1], FP_SA[2][:,0],"FP for "+str(knap_num)+" problem length", "FP_SA.png")
    print("---------------------")

#KColor
np.random.seed(812)
num_edges=400
num_dots=100
num_colors=3
edges=[]
for i in range(num_edges):
    edges.append([int(np.random.randint(num_dots)),int(np.random.randint(num_dots))])

problem_set = mlrose_hiive.DiscreteOpt(length = num_dots, fitness_fn = mlrose_hiive.MaxKColor(edges), maximize=True,max_val=num_colors)

if 0:
    t0=time.time()
    KC_RHC = mlrose_hiive.random_hill_climb(problem_set, max_attempts=5000, max_iters=50000, restarts=0,init_state=None, curve=True, random_state=812)
    dt=time.time()-t0
    print("KC RHC for ", num_edges, " run for ", dt, " seconds ", "with the best score of ", KC_RHC[1])
    plotter(KC_RHC[2][:, 1], KC_RHC[2][:, 0], "KC for " + str(num_edges) + " problem length", "KC_RHC.png")
    print("---------------------")

if 0:
    t0=time.time()
    KC_SA = mlrose_hiive.simulated_annealing(problem_set, schedule=mlrose_hiive.GeomDecay(), max_attempts=5000,max_iters=50000, init_state=None, curve=True,random_state=812)
    dt=time.time()-t0
    print("KC SA for ",num_edges," run for ", dt," seconds ", "with the best score of ",  KC_SA[1])
    plotter(KC_SA[2][:,1], KC_SA[2][:,0],"FP for "+str(num_edges)+" problem length", "KC_SA.png")
    print("---------------------")

if 0:
    t0=time.time()
    KC_GA = mlrose_hiive.genetic_alg(problem_set, pop_size=500, mutation_prob=0.5, max_attempts=100,max_iters=15000, curve=True, random_state=812)
    dt=time.time()-t0
    print("KC GA for ",num_edges," run for ", dt," seconds ", "with the best score of ",  KC_GA[1])
    plotter(KC_GA[2][:,1], KC_GA[2][:,0],"KS for "+str(num_edges)+" number of edges", "KC_GA.png","total population number")
    print("---------------------")

if 0:
    t0=time.time()
    KC_MIMIC = mlrose_hiive.mimic(problem_set, pop_size=400, keep_pct=0.8, max_attempts=20,max_iters=200, curve=True, random_state=812)
    dt=time.time()-t0
    print("KC MIMIC for ",num_edges," run for ", dt," seconds ", "with the best score of ",  KC_MIMIC[1])
    plotter(KC_MIMIC[2][:,1], KC_MIMIC[2][:,0],"KC for "+str(num_edges)+" number of edges", "KC_MIMIC.png","total population number")
    print("---------------------")


#Nueral nets optimizers
fruits_train_x,fruits_train_y,fruits_test_x,fruits_test_y= frames("fruits_data.csv",0.2)
if 0:
    Classifier=MLPClassifier(tol=0.005,hidden_layer_sizes=[25,25],activation='relu',learning_rate="constant",learning_rate_init=0.01, random_state=812)
    Classifier_RHC = mlrose_hiive.NeuralNetwork(hidden_nodes = [25,25], activation = 'relu', algorithm ="random_hill_climb",
                                     learning_rate = 0.02, random_state = 812,pop_size=200,mutation_prob=0.02,max_iters=100000)
    Classifier_SA = mlrose_hiive.NeuralNetwork(hidden_nodes = [25,25], activation = 'relu', algorithm ="simulated_annealing", schedule=mlrose_hiive.GeomDecay(init_temp=1, decay=0.99, min_temp=0.001),
                                     learning_rate = 0.02, random_state = 812,pop_size=200,mutation_prob=0.02,max_iters=100000)
    Classifier_GA = mlrose_hiive.NeuralNetwork(hidden_nodes = [25,25], activation = 'relu', algorithm = 'genetic_alg',
                                     learning_rate = 0.02, random_state = 812,pop_size=50,mutation_prob=0.01,max_iters=5000)
    t0=time.time()
    Classifier.fit(fruits_train_x, pd.get_dummies(fruits_train_y))
    dt_class=time.time()-t0

    t0=time.time()
    #Classifier_RHC.fit(fruits_train_x, pd.get_dummies(fruits_train_y))
    dt_class_RHC=time.time()-t0

    t0=time.time()
    Classifier_SA.fit(fruits_train_x, pd.get_dummies(fruits_train_y))
    dt_class_SA=time.time()-t0

    t0=time.time()
    Classifier_GA.fit(fruits_train_x, pd.get_dummies(fruits_train_y))
    dt_class_GA=time.time()-t0

    pred_y_test_class = Classifier.predict(fruits_test_x)
    pred_y_test_class_RHC = Classifier_RHC.predict(fruits_test_x)
    pred_y_test_class_SA = Classifier_SA.predict(fruits_test_x)
    pred_y_test_class_GA = Classifier_GA.predict(fruits_test_x)
    print("Regular classifier perfomance trained in ", dt_class, " seconds")
    print(sklearn.metrics.classification_report(pd.get_dummies(fruits_test_y), pred_y_test_class, digits=4))
    print("----------------")
    print("RHC classifier perfomance trained in ", dt_class_RHC, " seconds")
    print(sklearn.metrics.classification_report(pd.get_dummies(fruits_test_y), pred_y_test_class_RHC, digits=4))
    print("----------------")
    print("SA classifier perfomance trained in ", dt_class_SA, " seconds")
    print(sklearn.metrics.classification_report(pd.get_dummies(fruits_test_y), pred_y_test_class_SA, digits=4))
    print("----------------")
    print("GA classifier perfomance trained in ", dt_class_GA, " seconds")
    print(sklearn.metrics.classification_report(pd.get_dummies(fruits_test_y), pred_y_test_class_GA, digits=4))

Classifier=MLPClassifier(tol=0.005,hidden_layer_sizes=[25,25],activation='relu',learning_rate="constant",learning_rate_init=0.01, random_state=812)
Classifier_RHC = mlrose_hiive.NeuralNetwork(hidden_nodes = [25,25], activation = 'relu', algorithm ="random_hill_climb",
                                 learning_rate = 0.02, random_state = 812,pop_size=200,mutation_prob=0.02,max_iters=100000)
Classifier_SA = mlrose_hiive.NeuralNetwork(hidden_nodes = [25,25], activation = 'relu', algorithm ="simulated_annealing", schedule=mlrose_hiive.GeomDecay(init_temp=1, decay=0.99, min_temp=0.001),
                                 learning_rate = 0.02, random_state = 812,pop_size=200,mutation_prob=0.02,max_iters=100000)
Classifier_GA = mlrose_hiive.NeuralNetwork(hidden_nodes = [25,25], activation = 'relu', algorithm = 'genetic_alg',
                                 learning_rate = 0.02, random_state = 812,pop_size=50,mutation_prob=0.01,max_iters=5000)

t0=time.time()
a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier, fruits_train_x,pd.get_dummies(fruits_train_y), train_sizes=np.linspace(0.25,1,10),scoring='f1_weighted', cv=2, n_jobs=-1)
plot_11(train_score.mean(axis=1), test_score.mean(axis=1), np.linspace(0.25,1,10) * len(fruits_train_x), "NN_learning_curve","sample size", "NN_learning_curve.png")
print(time.time()-t0," seconds")

#a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier_RHC, fruits_train_x,pd.get_dummies(fruits_train_y), train_sizes=np.linspace(0.25,1,10),scoring='f1_weighted', cv=2, n_jobs=-1)
#plot_11(train_score.mean(axis=1), test_score.mean(axis=1), np.linspace(0.25,1,10) * len(fruits_train_x), "NN_RHC_learning_curve","sample size", "NN_RHC_learning_curve.png")
#print(time.time()-t0," seconds")

#a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier_SA, fruits_train_x,pd.get_dummies(fruits_train_y), train_sizes=np.linspace(0.25,1,10),scoring='f1_weighted', cv=2, n_jobs=4)
#plot_11(train_score.mean(axis=1), test_score.mean(axis=1), np.linspace(0.25,1,10) * len(fruits_train_x), "NN_SA_learning_curve","sample size", "NN_SA_learning_curve.png")
#print(time.time()-t0," seconds")

a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier_GA, fruits_train_x,pd.get_dummies(fruits_train_y), train_sizes=np.linspace(0.25,1,10),scoring='f1_weighted', cv=2, n_jobs=4)
plot_11(train_score.mean(axis=1), test_score.mean(axis=1), np.linspace(0.25,1,10) * len(fruits_train_x), "NN_GA_learning_curve","sample size", "NN_GA_learning_curve.png")
print(time.time()-t0," seconds")