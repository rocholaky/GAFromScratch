#Importing the libraries
import numpy as np
from Tree import *

class GeneticProgramming:
    def __init__(self, N_Generations, inputs, population_size, depth, mutation_rate, desired_output):
        var_names = ['x', 'y', 'z', 'w', 't', 'h']
        self.N_generations   = N_Generations      # The number of iterations the algortihm have for find the solution.
        self.population_size = population_size    # The number of tress the algorithm try per iterations.
        self.mutation_rate   = mutation_rate      # The probability of do the mutation operation on a tree.
        self.chiffres        = True               # The parameter is set to True for assuming the problem to solve, unless otherwise specified
        if desired_output == None:                # The condition for solve the other problem is verified , if is the case, the corresponding parameters are specified.
            inputs              = inputs.astype('float32')
            self.chiffres       = False           # The parameter is set to false.
            self.inputs         = inputs[:, 1:].T # When "Desired_output == None" the input is a matrix with the shape (n,m) with n the number of target point, and m the number of variables of a point.
            self.desired_output = inputs[:, 0]    # The desired output is set to the first column of the input, due to in the first column are stored the target values.
            self.tree_factory   = tree_factory(var_names[:self.inputs.shape[0]], depth)
        else:
            self.inputs             = inputs
            self.desired_output     = desired_output
            self.tree_factory       = tree_factory(var_names[:len(self.inputs)], depth)


    def generate_population(self):                      # The function is in charged of create the trees wich represent the initials programs.
        population = list()                             # The list that will contain the programs is created.
        for gen in range(self.population_size):         # The trees are created, finally we get "Population_size" number of tress.
            atree = self.tree_factory.generate_tree()   #One tree is created.
            self.tree_factory.set_tree_index(atree)     # The index of the tree is set up to the initials index.
            population.append(atree)                    # The tree just created, is stored in the initial list.
        return population                               # We return the whole list with the trees.

    def fitness(self, population):                      # The function is in charged of measure how fit is a given individual of a population, return the entire list with all the fits measures of all the individuals of the population.
        fit_list = list()                               # The list that will contain the meazures is created.
        for individual in population:                   # The entire population is traversed individual per individual.
            result = individual.evaluate(self.inputs, mode_chiffres=self.chiffres)    # We evaluate the result of a single individual, taking care of the evaluation was made for the "Chiffres" situation or the "Functions and point" situation.
            error  = np.square(np.subtract(result, self.desired_output)).mean()       # The measure of the RMSE is made.
            if error < 0:                               # A print if made if the error just calculated is inconsistent.
                print("Error: error menor a cero")
            fit_list.append(error)                      # The error calculated is added to the initial list of errors.
        fit_list = np.array(fit_list)                   # The final list is transformed to a numpyu array.
        return fit_list                                 # We return the final fit list.

    def selection(self, population, fitness):                                 # The function is in charged of selecting the individuals with the higher fitnes score, and return them.
        selected_indiviudals = list()                                         # The list that will contain the fitness scores is created.
        amount_selected      = int(len(population)/2)                         # The amount of selected individuals is set up.
        # get the best fit:
        best                 = np.argmin(fitness)                             # The index of the best current score is selected.
        selected_indiviudals.append(population[best])                         # We get the individual that correspond to the index just selected.
        fit_correction       = 1/(fitness+0.01)                               # The fitness score is redefined, consecuently to want a metric in which a higher score represent a better individual.
        sum_fit              = np.sum(fit_correction)                         # We add the scores, and then apply the operation.
        prob                 = fit_correction/sum_fit                         # The probabilities of selection are set up.
        random_prob          = np.random.random((amount_selected-1, 1))       # We create an array of random probabilities.
        for random in random_prob:                                            # Then, we apply the naive algorithm seen in class, to do the "Roulette" method.
            actual_value = 0
            for i, individual in enumerate(population):
                actual_value += prob[i]
                if actual_value >= random:
                    selected_indiviudals.append(individual)
                    break
        return selected_indiviudals                                           # The selected individuals are selected.



    def cross_over(self, selected_individuals):                                                  # The function is in charged of implemented the crossover operation.
        limit = self.population_size                                                             # Some useful parameters are created.
        sons  = list()
        sons.append(selected_individuals[0])
        while len(sons) < limit:                                                                 # The process of making the crossover follow the rules seen in classes, the method start now.
            r1                   = random.randint(0, len(selected_individuals)-1)                # A random index is chosen for selected an individual randomly.
            r2                   = random.randint(0, len(selected_individuals)-1)
            parent1              = selected_individuals[r1]                                      # The "parent1" is selected.
            change_index_1       = random.choice(parent1.possible_index())                       # The index where the crossover is made over the tree is set up.
            parent2              = selected_individuals[r2]                                      # The "parent2" is selected.
            change_index_1_level = parent1.get_level(change_index_1)                             # We get the level of the "parent1" for search a valid index to do the crossover in the "parent2".
            index2_valid         = parent2.is_valid_index(change_index_1_level)                  # We get the list of all the index where the crossover operation can be made into the "parent2".
            change_index_2       = random.choice(index2_valid)                                   # The index where the operation of crossover is made over the "parent2" is set up.
            sub_tree             = parent2.get_tree_at_index(change_index_2)                     # The extraction of the subtree corresponding to the index above is made.
            son                  = parent1.replace_tree_at_index(change_index_1, sub_tree)       # The replacement of the tress in the positions found above is made.
            self.tree_factory.set_tree_index(son)                                                # The new indexes and level of the new tree are fixed.
            self.tree_factory.set_tree_levels(son)
            sons.append(son)                                                                     # We added the "Crossovered" tree to the initial list.
        return sons                                                                              # Returns all the "Crossoverd" trees.

    def mutation(self, crossovered_trees):                                                       # The function is in charged of implemented the mutation operation.
        max_index              = self.tree_factory.depth + 1
        number_of_mutations    = int(len(crossovered_trees) * self.mutation_rate)                # The number of individuals to mutate is defined according the amount of individuals in the generation and the mutation rate.
        function_possibilities = self.tree_factory.internal_function_posibility                  # We defined the list with all the possibles functions in a node.
        aleatorios             = np.random.randint(0, len(crossovered_trees), size=(number_of_mutations,))  # A numpy array with random values is created, this is used to chose the selected crossovered trees.
        for i in range(number_of_mutations):                                                                # The process of making the mutations follow the rules seen in classes, the method start now.
            the_tree     = crossovered_trees[aleatorios[i]]                                                 # A tree is selected.
            random_index = random.choice(the_tree.possible_index(True))                                     # We extract all the possibles index where we can do the mutation operation.
            if the_tree.get_level(random_index) == 1:                                                       # If the index randomly chosen is a leaf, we only change the field "component".
                amutated_tree = the_tree.copy_tree()                                                        # A copy of the tree is made.
                amutated_tree.set_component(random.choice(function_possibilities))                          # The set up of the component mentioned above is made.
            else:
                level_at_index = the_tree.get_level(random_index)                                           # We get the level associated to the index passed before, this is for controlling the depth of the tress.
                if level_at_index == max_index:                                                             # If the variable take a value that is not possible to apply, the value is fixed with the maximum value possible.
                    random_level = max_index
                else:
                    random_level = random.randint(level_at_index, max_index)                                # If there isn't any problem with the index value, we used the index for finding the place to do the replacement of the trees.
                amutated_tree = crossovered_trees[aleatorios[i]].replace_tree_at_index(random_index,\
                                                            self.tree_factory.generate_tree(random_level))  # The replacement between the trees is made
            self.tree_factory.set_tree_index(amutated_tree)                                                 # The new indexes and levels are set up in the new tree.
            self.tree_factory.set_tree_levels(amutated_tree)
            crossovered_trees[aleatorios[i]] = amutated_tree  # We do the mutation
        return crossovered_trees



    #run genetic algorithm:
    def run(self, verbose=True):                                                            # The function is in charged of running the principal program, applying the genetic operations, and getting the performance metrics.
        #create list for later plotting:
        mean_fit_per_generation  = list()                                                   # The lists that will contains the fitness scores are created.
        worst_fit_per_generation = list()
        best_fit_per_generation  = list()

        # Initialice individuals:
        generation = self.generate_population()                                             # The initial generation is created.
        # Run algorithm for the number of generations:
        for generation_number in range(0, self.N_generations):                              # The program is executed "N_generations" times, i.e:
            if verbose:
                print("working in generation " + str(generation_number))

            # Calculate generation fit:
            fit_of_generation = self.fitness(generation)                                    # Starting with flow of the usual genetic programs, the fitness is
                                                                                            # measure, then we get the measurements for plotting its after, and
                                                                                            # continue with the flow.
            # Get the mean, worst, best fits:
            mean_fit_per_generation.append(np.mean(fit_of_generation))
            worst_fit_per_generation.append(np.amax(fit_of_generation))
            best_fit_per_generation.append(np.amin(fit_of_generation))
            if verbose:
                print("the best fit is: " + str(best_fit_per_generation[-1]))

            # Check if theres any fit that matches the desired fit if theres any:
            if 0 in fit_of_generation:                                                     # If we get the score = 0, is because the genetic program found a tree that commits no error in the prediction of the function that we search for.
                index      = np.argmin(fit_of_generation)                                  # If the case above, we take the index of the element with less fitness, i.e. with no errors.
                individual = generation[index]                                             # Then, we take the individual of that index
                return individual, mean_fit_per_generation, worst_fit_per_generation, best_fit_per_generation  # If the case, we return the tree that represent
                                                                                                               # the searched function and all the related operations over the calculated metrics.
            # Select the individuals:
            selected_individuals    = self.selection(generation, fit_of_generation)        # If isn´t the case above, the flow of the genetic program continue
                                                                                           # The best individuals of the generation are selected.
            # Reproduct individuals:
            reproducted_individuals = self.cross_over(selected_individuals)                # With the selected individuals, we make the crossover.

            # mutate individuals:
            mutated_individuals     = self.mutation(reproducted_individuals)               # Then we apply the mutation operation over the new individuals.

            #unite new generation:
            new_generation = mutated_individuals                                           # We defined an auxiliary variable for save the generation with all the operations.
            #update generation:
            generation     = new_generation                                                # And set the ultimate generation of the current iterations.
        #get the final results from the final generation:
        last_fit = self.fitness(generation)                                                # When the iterations are over, we added the last fitness made by all the above process.
        #get the mean, worst, best fits:
        mean_fit_per_generation.append(np.mean(last_fit))                                  # We save the last metrics over the fitness.
        worst_fit_per_generation.append(np.amax(last_fit))
        best_fit_per_generation.append(np.amin(last_fit))

        #return best individual of last generation and the progress:
        best_fit_index = np.argmin(last_fit)                                               # We extract the index that represent the best fitness.
        best_individual = generation[best_fit_index]                                       # We extract the best individual found by the index.
        return best_individual, mean_fit_per_generation, worst_fit_per_generation, best_fit_per_generation # We return the best individual and all the metrics.
