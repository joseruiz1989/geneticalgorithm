"""





"max_num_iteration": int -> number of iterations ou generations
"population_size": int -> number of individuals in each generation
"mutation_probability": float between 0 and 1, where 0 is 0% and 1 is 100%
    values below 0 and above 1 will be replaced to 0 and 1 respectively
"elit_ratio": float or int -> number of individual to keep in the next generation
    values below 0 will be replaced to 0, this means that no individual will be kept for the next generation
    values between 0+ and 1 will be replaced to 1, this means that only one individual will be kept for the next generation
    values above 1 will be rounded and that amount of individuals will be kept for the next generation
        if this value is higer than the population, will be kept only the population size individuals
"crossover_probability": 1.0,
"parents_portion": 0.3,
"crossover_type": "uniform",
"max_iteration_without_improv": 20,
"multiple_cpu": False,

"""

### libraries from our GA code
from utils_ga import *

### common libraries
import pandas as pd
import random, math
import matplotlib.pyplot as plt

### multiprocessing libraries
from multiprocessing import Pool, cpu_count
import tqdm, datetime, time



# TODO: reunir todos los individuos de todas las generaciones, una sola lista y adicionar el numero de la generación 
# TODO: buscar si el individuo ya fue simulado y saltar la simulación si ya lo fue, creo que dentro de la simulación tiene que verificar si existe el valor
# TODO: implementar los graficos y estadisticas de las simulaciones 




class GeneticAlgorithm():
    def __init__(self, function, dimension,	variable_type, \
        variable_boundaries, algorithm_parameters={'max_num_iteration': 50,\
                'population_size':100,\
                'mutation_probability':0.1,\
                'elit_ratio': 0.01,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'uniform',\
                'max_iteration_without_improv':None, \
                'multiple_cpu': True, \
                'separate_simulation_fortran': True},\
                live_plot = False):
        
        self.folder_path = "C:/PD/sims/"
        self.function = function
        self.dimension = dimension
        self.variable_type = variable_type
        self.variable_boundaries = variable_boundaries
        self.algorithm_parameters = algorithm_parameters
        self.max_num_iteration = int(algorithm_parameters["max_num_iteration"])
        self.population_size = int(algorithm_parameters["population_size"])
        self.multiple_cpu = algorithm_parameters["multiple_cpu"]
        self.live_plot = live_plot
        if "separate_simulation_fortran" in algorithm_parameters:
            self.separate_simulation_fortran = algorithm_parameters["separate_simulation_fortran"]
        else:
            self.separate_simulation_fortran = False
        
        if "otimization" in algorithm_parameters:
            self.otimization_value = algorithm_parameters["otimization"]
        else:
            self.otimization_value = ""

        
        if self.multiple_cpu:
            self.cores=cpu_count()

        assert dimension == len(variable_boundaries)

        ###################################################
        # "mutation_probability": 0.1,
        # "elit_ratio": 0.01,
        # "crossover_probability": 0.5,
        # "parents_portion": 0.3,
        # "crossover_type": "uniform",
        # "max_iteration_without_improv": 20,
        # "multiple_cpu": False,
        ###################################################

        if algorithm_parameters["crossover_probability"] > 1:
            self.cross_rate = 1
        elif algorithm_parameters["crossover_probability"] < 0:
            self.cross_rate = 0
        else:
            self.cross_rate = algorithm_parameters["crossover_probability"]
        
        if algorithm_parameters["mutation_probability"] > 1:
            self.mutation_rate = 1
        elif algorithm_parameters["mutation_probability"] < 0:
            self.mutation_rate = 0
        else:
            self.mutation_rate = algorithm_parameters["mutation_probability"]

        if algorithm_parameters["elit_ratio"] <= 0:
            self.elit_ratio = 0
        elif algorithm_parameters["elit_ratio"] <= 1:
            self.elit_ratio = 1
        else:
            self.elit_ratio = round(algorithm_parameters["elit_ratio"])

        if self.elit_ratio >= self.population_size:
            self.elit_ratio = self.population_size

        
        self.bits_individual=0
        self.bits_bound=[0]
        for bounds in variable_boundaries:
            bits = math.ceil(math.log2(bounds[1]-bounds[0]+1))
            self.bits_bound.append(self.bits_bound[-1]+bits)
            self.bits_individual += bits

        self.csv_df_document = pd.DataFrame([["geracao", "individuo", "aptidao", "estrutura",\
                                            "OscStr max", "OscStr energy [meV]", \
                    
                                            "PC max", "PC max energy [meV]", \
                                            "PC min", "PC min energy [meV]", \
                                            "PC abs", "PC abs energy [meV]", \
                                            "origem", "pai1", "pai2",\
                                            "left wells", "qw left [nm]", "qb left [nm]", \
                                            "defect [nm]", \
                                            "right wells", "qw right [nm]", "qb right [nm]", \
                                            "estrutura python"]])

        
        # print(bin2int([1, 0, 1, 1, 0]))
        
        # print(int2bin(15))

        
        # # print(self.bits_individual)
        # # print(self.bits_bound)








        ###################################################
        self.ini_sort_pop(method="lineal_norm")
        self.actual_generation=0

    def run(self):
        print("starting...")

        # create and evaluate first generation
        self.first_generation()
        self.evaluate_generation(self.pop)
        self.order_result()
        self.all_generations=[self.pop_res.copy()]
        # for p in self.pop_res:
            # print(p)
        self.best_ind = self.pop_res[-1]
        if self.live_plot:
            self.plot_ga()
        print("best ind generation", 1, self.best_ind)
        # print("----\n",self.pop_res)
        
        
        
        # individuito = self.sort_pop(self.pop_res)[0]
        # print("individuo org: ", individuito)
        # print("individuo bin: ", int2bin(individuito[0]-self.variable_boundaries[0][0]),int2bin(individuito[1]-self.variable_boundaries[1][0]),int2bin(individuito[2]-self.variable_boundaries[2][0]))
        # individuito = self.intind2binind(individuito)
        # print("individuo jun: ", individuito)
        # individuito = self.binind2intind(individuito)
        # print("individuo or2: ", individuito)
        
        
        # self.pop = self.create_new_generation()
        # # print("\n\n\n\n***************")
        # # for p in self.pop:
        # # 	print(p)
        # # print("\n\n\n\n***************")
        # self.evaluate_generation(self.pop)
        # self.order_result()
        # self.best_ind = self.pop_res[-1]
        # print("melhor", 2, self.best_ind)


        # create and evaluate second generation
        for i in range(self.max_num_iteration-1):
            self.pop = self.create_new_generation()
            self.evaluate_generation(self.pop)
            self.order_result()
            self.all_generations.append(self.pop_res.copy())
            self.best_ind = self.pop_res[-1]
            if self.live_plot:
                self.plot_ga()
            print("best ind generation", i+2, self.best_ind)
        print("finish run all generations")
        
        
        

    def first_generation(self):
        if self.variable_type == "int":
            self.pop=[]
            if hasattr(self, 'init_pop'):
                print("init population")
                for individual in self.init_pop:
                    valid_ind = False
                    same_dimension = False

                    if len(individual) == self.dimension:
                        same_dimension = True

                    if same_dimension:
                        if hasattr(self, 'ind_condition'):
                            valid_ind = self.ind_condition(individual)
                            if valid_ind == False:
                                print("not valid individual!!! ind_condition not satisfied", individual)
                        else:
                            valid_ind = True

                        for b, bound in enumerate(self.variable_boundaries):
                            if individual[b] < bound[0] or individual[b] > bound[1]:
                                valid_ind = False
                                print("not valid individual!!! values out of boundaries", individual)

                        if valid_ind:
                            self.pop.append([individual, ["init", [0,0,0], [0,0,0]]])

            
            while len(self.pop) < self.population_size:
                valid_ind = False
                while not valid_ind:
                    individual = []
                    for bound in self.variable_boundaries:
                        individual.append(random.randint(bound[0], bound[1]))
                    if hasattr(self, 'ind_condition'):
                        valid_ind = self.ind_condition(individual)
                    else:
                        valid_ind = True
                self.pop.append([individual, ["rand", [0,0,0], [0,0,0]]])
        self.actual_generation=1

    def evaluate_generation(self, pop):
        if self.multiple_cpu:
            if self.separate_simulation_fortran:
                self.pop_res=[]


                ##############################################################
                ##############################################################

                # para crear os arquivos em fortran e o .exe
                print("\nCriando lista de simulações...", self.actual_generation, "geração")
                to_run = []
                for par in pop:

                    values = par[0]
                    structure = set_structure_values(values)

                    parameters=[structure, self.folder_path, self.otimization_value]
                    to_run.append(parameters)
                
                


                print("Criando arquivos .f90...", self.actual_generation, "geração")
                with Pool(processes=self.cores) as p:
                    with tqdm.tqdm(total=len(to_run)) as pbar:
                        for i, result in enumerate(p.imap_unordered(sim_for_ga_step1, to_run)):
                            pbar.update()
                            
                time.sleep(2)
                paralelo = True
                if paralelo:
                    print("Compilando os arquivos .f90 paralelamente...", self.actual_generation, "geração")
                    with Pool(processes=self.cores) as p:
                        with tqdm.tqdm(total=len(to_run)) as pbar:
                            for i, _ in enumerate(p.imap_unordered(sim_for_ga_step2, to_run)):
                                pbar.update()

                print("Rodando as simulações em paralelo...", self.actual_generation, "geração")
                # print(self.otimization_value)
                with Pool(processes=self.cores) as p:
                    with tqdm.tqdm(total=len(to_run)) as pbar:
                        for i, result in enumerate(p.map(sim_for_ga_step3, to_run)):
                            self.pop_res.append([result[1], [pop[i]]])
                            pbar.update()

                # ##############################################################
                # ##############################################################
                # with Pool(processes=cores) as p:
                # 	with tqdm.tqdm(total=self.population_size) as pbar:
                # 		# for i, result in enumerate(p.imap_unordered(self.function, pop)):
                # 		for i, result in enumerate(p.map(self.function, pop)):
                # 			self.pop_res.append([result[1], [pop[i]]])
                # 			pbar.update()
                # 			# pass
            else:
                self.pop_res=[]
                with Pool(processes=cores) as p:
                    with tqdm.tqdm(total=self.population_size) as pbar:
                        # for i, result in enumerate(p.imap_unordered(self.function, pop)):
                        for i, result in enumerate(p.map(self.function, pop)):
                            self.pop_res.append([result[1], [pop[i]]])
                            pbar.update()
                            # pass
                
                
        else:
            self.pop_res=[]
            for i in range(len(pop)):
                #return [values, [sum(values[0]), 10, 20]]
                result = self.function(pop[i])
                self.pop_res.append([result[1], [pop[i]]])
                # print(self.pop_res[-1])

    def order_result(self):
        sub_li = self.pop_res
        l = len(sub_li)
        for i in range(0, l):
            for j in range(0, l-i-1):
                if (sub_li[j][0][0] > sub_li[j + 1][0][0]):
                    tempo = sub_li[j]
                    sub_li[j]= sub_li[j + 1]
                    sub_li[j + 1]= tempo
        self.pop_res = sub_li.copy()
        
    def create_new_generation(self):
        pop_org = self.pop_res
        individuito = self.sort_pop(self.pop_res)[0]
        new_pop = []
        if self.elit_ratio == 1:
            new_pop=[[self.pop_res[-1][1][0][0].copy(), ["best", [1,1,1], [1,1,1]]]]
        elif self.elit_ratio > 1:
            new_pop=[[self.pop_res[-1][1][0][0].copy(), ["best", [1,1,1], [1,1,1]]]]
            for i in range(self.elit_ratio-1):
                new_pop.append([self.pop_res[-i-2][1][0][0].copy(), ["elit", [1,1,1], [1,1,1]]])

        # seleciona 2 individuos para cruzamento
        while len(new_pop) < (self.population_size*self.cross_rate):
            valid_ind = False
            while not valid_ind:
                son1, son2, par1, par2 = self.cross_individual()

                if hasattr(self, 'ind_condition'):
                    if self.ind_condition(son1) and self.ind_condition(son2):
                        valid_ind = True
                else:
                    valid_ind = True
            new_pop.append([son1, ["son-", par1, par2]])
            if len(new_pop) < (self.population_size*self.cross_rate):
                new_pop.append([son2, ["son-", par1, par2]])
        # print(len(new_pop))
        while len(new_pop) < (self.population_size):
            valid_ind = False
            while not valid_ind:
                individual = []
                for bound in self.variable_boundaries:
                    individual.append(random.randint(bound[0], bound[1]))
                if hasattr(self, 'ind_condition'):
                    valid_ind = self.ind_condition(individual)
                else:
                    valid_ind = True
                new_pop.append([individual, ["rand", [0,0,0], [0,0,0]]])
        self.actual_generation += 1
        
        return new_pop

    def cross_individual(self, points=1):
        parent1 = self.sort_pop(self.pop_res)[1][0][0]
        parent2 = self.sort_pop(self.pop_res)[1][0][0]
        while parent1 == parent2:
            parent2 = self.sort_pop(self.pop_res)[1][0][0]
        parent1bin = self.intind2binind(parent1)
        parent2bin = self.intind2binind(parent2)
        if points == 1:
            valid_son=False
            while not valid_son:
                cross_point = random.randint(1, len(parent1bin))
                new_son1=parent1bin[0:cross_point] + parent2bin[cross_point:]
                new_son2=parent2bin[0:cross_point] + parent1bin[cross_point:]
                # print("aaaaaa")
                if random.random()<self.mutation_rate:
                    new_son1 = self.mutation_individual(new_son1)
                if random.random()<self.mutation_rate:
                    new_son2 = self.mutation_individual(new_son2)

                new_son1_int = self.binind2intind(new_son1)
                new_son2_int = self.binind2intind(new_son2)
                # print(parent1, parent2)
                # print(new_son1_int, new_son2_int)
                valid_son = self.son_validation([new_son1_int, new_son2_int])
        return new_son1_int, new_son2_int, parent1, parent2

    def mutation_individual(self, son):
        mutation_point = random.randint(1, len(son)-1)
        if son[mutation_point] == 1:
            son[mutation_point] = 0
        else:
            son[mutation_point] = 1
        return son

    def son_validation(self, sons):
        valid_son = True
        for son in sons:
            for i in range(len(self.variable_boundaries)):
                if son[i] >= self.variable_boundaries[i][0] and son[i] <= self.variable_boundaries[i][1]:
                    pass
                else:
                    valid_son = False
        return valid_son

    def ini_sort_pop(self, method="lineal_norm"):
        """
        proporcional
        por torneios
        com truncamento
        por normalização linear = lineal_norm
        por normalização exponencial
        """
        if method=="lineal_norm":
            self.prob_list = []
            for i in range(self.population_size):
                self.prob_list.extend([i]*(i+1))

    def sort_pop(self, pop, method="lineal_norm"):
        """
        retorna um individuo da população sorteado segundo seu fittnes

        proporcional
        por torneios
        com truncamento
        por normalização linear = lineal_norm
        por normalização exponencial
        """
        if method=="lineal_norm":
            return pop[rand_list(self.prob_list)]

    def intind2binind(self, individual:list):
        bin_individual = []
        for i in range(len(individual)):
            new_bin = int2bin(individual[i]-self.variable_boundaries[i][0])
            correct_len_bin = self.bits_bound[i+1] - self.bits_bound[i]
            org_len_bin = len(new_bin)
            if correct_len_bin > org_len_bin:
                new_bin = complete_bin(new_bin, correct_len_bin)
            bin_individual.extend(new_bin)
        return bin_individual

    def binind2intind(self, individual:list):
        int_individual = []
        for i in range(len(self.bits_bound)-1):
            bin_value = individual[self.bits_bound[i]:self.bits_bound[i+1]]
            int_individual.append(bin2int(bin_value)+self.variable_boundaries[i][0])
        return int_individual

    def plot_ga(self, path_file='temp_files/0001.png', prints=False):
        fig, ax = plt.subplots()
        for gen in range(len(self.all_generations)):
            gen_ = gen+1
            all_ind_ger = []
            for ind in range(len(self.all_generations[gen])):
                all_ind_ger.append(self.all_generations[gen][ind][0][0])
                ax.plot(gen_, self.all_generations[gen][ind][0][0], marker=".", color="#aaaaaa", markersize=4)
            ax.plot(gen_, self.all_generations[gen][-1][0][0], marker=".", color="g", markersize=8)
            # plt.plot(gen_, self.all_generations[gen][0][0][0], marker=".", color="r")
            ax.plot(gen_, sum(all_ind_ger)/len(all_ind_ger), marker=".", color="b", markersize=4)
            ax.set(xlim=[-0.5, self.max_num_iteration+1])
            if prints: print("gen:", gen_, " de:", len(self.all_generations), end="\r")
            
        if prints: print()
        if prints: print("gen:", gen_, " de:", len(self.all_generations))
        
        if prints: print("********** starting to save")
        plt.savefig(path_file, dpi=250)
        
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 

