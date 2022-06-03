from GeneticAlgorithm import GeneticAlgorithm as GA
from GeneticAlgorithm import *

if __name__ == "__main__":
    cores=cpu_count()
    print("cores: ", cores)


    # 1 = test GA

    run_option = 1


    if run_option == 1:
        print("running option 1...")

        ga_config = {
            "max_num_iteration": 5,
            "population_size": 100,
            "mutation_probability": 0.01,
            "elit_ratio": 2,
            "crossover_probability": 0.8,
            "parents_portion": 0.2,
            "crossover_type": "uniform",
            "max_iteration_without_improv": 20,
            "multiple_cpu": False,
        }

        class GA_test(GA):
            def fitness_function(self, values):
                return sum(values)

            def individual_condition(self, individual):
                if individual[1] > individual[0]*2:
                    return True
                else:
                    return False
        
        varbound=([[0,10000]]*3)
        # varbound=([[100,1000], [10,100], [10,2000]])


        model = GA_test(dimension=3,\
                        variable_type='int',\
                        variable_boundaries=varbound,\
                        ga_config=ga_config,\
                        live_plot = False)
        
        model.init_pop = [[0,9999,999],
                        [0,1111,505],
                        [785,2654,1896]]

        model.run()
        
        print("\n********** starting to plot")
        model.plot_ga(prints=True)
        
        print("best result fitness:", model.all_generations[-1][-1][0])
        print("best result individual", model.all_generations[-1][-1][1][0])