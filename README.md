# geneticalgorithm

Genetic Algorithm library.

Our GA library is configured to maximize the fitness function

demo.py contains the example described here

### Run a optimization

In this simple example, we will optimize (maximize) an individual composed of three values between 0 and 10000, the fitness_function is the sum of the three values

To run a Genetic Algorithm optimization is necessary to clone the python repository https://github.com/joseruiz1989/geneticalgorithm

Into a new python file we need to import the library

```
from GeneticAlgorithm import GeneticAlgorithm as GA
from GeneticAlgorithm import *
```

The next step is create a config GA dictionary

```
ga_config = {
    "max_num_iteration": 50,
    "population_size": 100,
    "mutation_probability": 0.01,
    "elit_ratio": 2,
    "crossover_probability": 0.8,
    "parents_portion": 0.2,
    "crossover_type": "uniform",
    "max_iteration_without_improv": 20,
    "multiple_cpu": False,
}
```
*Parameters definitions at the end of this document

The next step is define a class based on our GA class, in the class is necessary define the fitness_function, aditionally and optional it is posible to define a individual_condition to set individual restrictions

```
class GA_test(GA):
    def fitness_function(self, values):
        return sum(values)

    def individual_condition(self, individual):
        if individual[1] > individual[0]*2:
            return True
        else:
            return False
```

>The **fitness_function** must to return a number value, in this example, the fitness function is the sum of the individual values
The **individual_condition** must to return a boolean value, in our case, the restriction is that the second number must be at less twice the first number

The next step is to define the limits of the values of the individual, they must be defined as a list of lists containing the minimum and the maximum. For our case of an individual with 3 values it would be:

[ [min0, max0], [min1, max1], [min2, max2] ]

```
varbound = [[0, 10000],[0, 10000],[0, 10000]]
```

With this, it is possible to create an instance of the class created above

```
model = GA_test(dimension=3,\
                variable_type='int',\
                variable_boundaries=varbound,\
                ga_config=ga_config,\
                live_plot = False)
```
*Parameters definitions at the end of this document

As an additional step, we can create an initial population for our genetic algorithm.

The individuals of the initial population have to be defined as a list of lists.

In the first generation they will be included as long as they are within the limits of the established values and that they meet the individual_condition if it was previously established.


```
model.init_pop = [[0,9999,999],
                [0,1111,505],
                [785,2654,1896]]
```

To run the GA:

```
model.run()
```

To save the graphic of the GA in the file 'temp_files/0001.png'

```
model.plot_ga()
```



__________________

## Additional Information
### A. Config GA dictionary parameters:

- "max_num_iteration": int -> number of iterations ou generations
- "population_size": int -> number of individuals in each generation
- "mutation_probability": float between 0 and 1, where 0 is 0% and 1 is 100% -> probability to modify each gen in the individual
  - values below 0 and above 1 will be replaced to 0 and 1 respectively
  - probability of each bit in each individual to be replaced
- "elit_ratio": float or int -> number of individual to keep in the next generation
  - values below 0 will be replaced to 0, this means that no individual will be kept for the next generation
  - values between 0+ and 1 will be replaced to 1, this means that only one individual will be kept for the next generation
  - values above 1 will be rounded and that amount of individuals will be kept for the next generation
    - if this value is higer than the population, will be kept only the population size individuals
- "parents_portion": 0.3, -> the portion of new generation filled by the members of the previous generation, chosen by the ga sorted method
- "crossover_probability": 1.0, -> float between 0 and 1, where 0 is 0% and 1 is 100%
  - values below 0 and above 1 will be replaced to 0 and 1 respectively
  - percent of new population to be filled by sons 
- "multiple_cpu": bool -> if the simulations are performed in only one core or in multiples cores (total in the machine)

to be implemented:
- "crossover_type": "uniform",
- "max_iteration_without_improv": 20,

#### Example for elit_ratio, parents_portion and crossover_probability
population:100
- elit_ratio = 2
- parents_portion = 0.1
- crossover_probability = 0.9

then the new generation will be:
- 2 - best individuals of las generation, total 2
- 8 - selected individuals of the last generation, total 10 
- 80 - new individuals generated by crossing the selected parents, total 90
- 10 - random individuals, total 100


### B. GA class parameters:

- dimension: int -> is the number of variables that our individual has
- variable_type='int' -> for now we only support integer values
- variable_boundaries: list -> list of individual value boundaries
- ga_config:dictionary -> GA settings that were previously defined
- live_plot: Bool -> Defines if you want the GA plot to be generated on each generation

