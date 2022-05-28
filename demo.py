from GeneticAlgorithm import GeneticAlgorithm as GA
from GeneticAlgorithm import *

def test_ga_function(values):
	# print(values)
	sumita = 1
	# for i in range(50):
		# sumita = sumita **1.0001
	return [values, [sum(values[0]), 10, 20]]
	
	

if __name__ == "__main__":
    cores=cpu_count()
    print("cores: ", cores)


    # 1 = test GA
    # 2 = run ga fortran

    run_option = 1


    if run_option == 1:
        print("run option 1")
        varbound=([[0,10000]]*3)
        # varbound=([[100,1500], [100,1500], [10,1500]])
        
        # 100	128		7 
        # 300	512		9
        # 50	64		6
        # 				22
        #									1	1	0	0	1	0	0
        #	1	1	1	1	1	10	9	8	7	6	5	4	3	2	1	
        #							256	128	64	32	16	8	4	2	1
        
        algorithm_param = {
            "max_num_iteration": 30,
            "population_size": 15,
            "mutation_probability": 0.3,
            "elit_ratio": 13.9,
            "crossover_probability": 1.0,
            "parents_portion": 0.3,
            "crossover_type": "uniform",
            "max_iteration_without_improv": 20,
            "multiple_cpu": False,
        }



        model = GA(function=test_ga_function,\
                            dimension=3,\
                            variable_type='int',\
                            variable_boundaries=varbound,\
                            algorithm_parameters=algorithm_param,\
                            live_plot = True)
        
        # set condition
        def check_ind(ind):
            if ind[1] > ind[2] + 5:
                return True
            else:
                return False
        
        model.ind_condition = check_ind

        model.init_pop = [[0,9999,999],
                        [0,1111,505],
                        [0,9888,9099]]

        model.run()
        
        print("\n********** starting plot")
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