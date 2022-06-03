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

        varbound=([[0,10000]]*3)
        # varbound=([[100,1000], [10,100], [10,2000]])

        class GA_test(GA):
            def fitness_function(self, values):
                return sum(values)

            def individual_condition(self, individual):
                if individual[1] > individual[0]*2:
                    return True
                else:
                    return False

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
        
        print("best result", model.all_generations[-1][-1])

        # save csv
        if False:
            for i in range(len(model.all_generations)):
                for j in range(len(model.all_generations[i])):
                    ger = i + 1

                    fit = model.all_generations[i][j][0][0]
                    ene = model.all_generations[i][j][0][1]
                    osc = model.all_generations[i][j][0][2]

                    ind = model.all_generations[i][j][1][0][0]

                    org = model.all_generations[i][j][1][0][1][0]
                    pa1 = model.all_generations[i][j][1][0][1][1]
                    pa2 = model.all_generations[i][j][1][0][1][2]
                

                    # print(ger,"-", ind, "-", fit, "-", org)
            
                    model.csv_df_document = add_to_df(model.csv_df_document, [ger, ind, fit, ene, osc, org, pa1, pa2])
            
            now = datetime.datetime.now()
            csv_name = "C:/PD/zz_test_GA_{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}_{:05d}_otim.csv".format(now.year, now.month, now.day, now.hour, now.minute, now.second, random.randint(10000, 99999))
            model.csv_df_document.to_csv (csv_name, index= False, header= False, line_terminator='\n')