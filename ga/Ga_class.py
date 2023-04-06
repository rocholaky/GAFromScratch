import numpy as np
import random

# GA class that given a problem solves it using genetic algorithm ###
# in order to use this script you need to provide the following variables and functions to this class
# You need to give: 
# 1. number of generations (int non negative number)
# 2. individuals per generation (int non negative number)
# 3. mutation rate (float non negative number between 0 and 1)
# 4. posibilities: the possible espectrum of values that the genes of a chromosome can take
# 4. individual_factory function (this function should receive the amount of individuals per generation)
# 5. fit function (a function that receives the generation and returns the fit of each individual)
# 6. selection label: you need to define if the selection algortihm is roulette or tournament (default normal)
# 7. cross_over label: you need to define if the crossover is a normal or ordered crossover (default normal)
# 8. mutation label: you need to define if the mutation is a normal or swap mutation (default normal)
# 9. desired fitness ((float non negative number, that describes a goal we want to reach)


class GA():
    def __init__(self, number_of_generations, individuals_per_generations, gen_length, mutation_rate, possibilities,                                        individual_factory_function, fit_function, selection='roulette', cross_over='normal', mutation='normal',                                                                                                     desired_fitness=None):             
        #algorithms parameters:
        self.number_of_generations = number_of_generations
        self.individuals_per_generation = individuals_per_generations
        self.mutation_rate = mutation_rate
        self.desired_fitness = desired_fitness
        self.gen_length = gen_length
        self.possibilities = possibilities
        
        #algorithms functions:
        self.individual_factory_function = individual_factory_function
        self.fit_function = fit_function
        
        #labels for selection, 
        self.elements_per_tournament=None
        if selection.lower() =='tournament':
            self.elements_per_tournament = int(input('How many individuals per tournament?'))
        self.selection_label = selection.lower()
        self.cross_over_label = cross_over.lower()
        self.mutation_label = mutation.lower()
   
    #run genetic algorithm:
    def run(self, verbose=True):
        #create list for later plotting:
        mean_fit_per_generation = list()
        worst_fit_per_generation = list()
        best_fit_per_generation = list()
        
        #initialice individuals:
        generation = self.individual_factory_function(self.individuals_per_generation, self.gen_length)
        #run algorithm for the number of generations:
        for generation_number in range(0, self.number_of_generations):
            if verbose:
                print("working in generation " + str(generation_number))
            
            #calculate generation fit:
            fit_of_generation = self.fit_function(generation)
            
            
            #get the mean, worst, best fits:
            mean_fit_per_generation.append(np.mean(fit_of_generation))
            worst_fit_per_generation.append(np.amin(fit_of_generation))
            best_fit_per_generation.append(np.amax(fit_of_generation))
            if verbose:
                print("the best fit is: " + str(best_fit_per_generation[-1]))
            
            #check if theres any fit that matches the desired fit if theres any:
            if self.desired_fitness in fit_of_generation and self.desired_fitness is not None:
                index = np.argmax(fit_of_generation)
                individual = generation[index]
                return individual, mean_fit_per_generation, worst_fit_per_generation, best_fit_per_generation
                
            #select the individuals:
            if self.selection_label == 'roulette':
                selected_individuals = self.roulette_selection(generation, fit_of_generation)
            elif self.selection_label == 'tournament':
                selected_individuals = self.tournament_selection(generation, fit_of_generation, self.elements_per_tournament)
            
            
            # reproduct individuals:
            if self.cross_over_label == 'normal':
                reproducted_individuals = self.normal_cross_over(selected_individuals)
            elif self.cross_over_label == 'ordered':
                reproducted_individuals = self.ordered_cross_over(selected_individuals)
            
            # mutate individuals:
            if self.mutation_label == 'normal':
                mutated_individuals = self.normal_mutation(reproducted_individuals, self.mutation_rate, self.possibilities)
            elif self.mutation_label == 'swap':
                mutated_individuals = self.swap_mutation(reproducted_individuals, self.mutation_rate, self.possibilities)
            
           
            #unite new generation:
            new_generation = mutated_individuals
            
            #update generation:
            generation = new_generation
            
        #get the final results from the final generation:
        last_fit = self.fit_function(generation)
        #get the mean, worst, best fits:
        mean_fit_per_generation.append(np.mean(last_fit))
        worst_fit_per_generation.append(np.amin(last_fit))
        best_fit_per_generation.append(np.amax(last_fit))
        
        #return best individual of last generation and the progress:
        best_fit_index = np.argmax(last_fit)
        best_individual = generation[best_fit_index]
        return best_individual, mean_fit_per_generation, worst_fit_per_generation, best_fit_per_generation
    
    
    
    #Roulette_selection: Using the roulette method to create selection:
    def roulette_selection(self, individuals, fitness_of_generation):
        #create list for storaging:
        individuals_selected = list()
        #storaging indexes to only store new individuals:
        individuals_selected_index = list()
        #store the best:
        best_of_generation = individuals[np.argmax(fitness_of_generation)]
        individuals_selected.append(best_of_generation)
        
        #select the amount of individuals to be selected:
        selected_amount_of_individuals = int(0.5*individuals.shape[0])-1
        fitness_sum = np.sum(fitness_of_generation)
        #get fittness of generation:
        fitness_proportion = (fitness_of_generation/fitness_sum)
        random_values = np.random.uniform(0, 1, size=(selected_amount_of_individuals,))
        for random_top in random_values:
            #define starting value:
            sum_of_proportions = 0
            for individual_index in range(individuals.shape[0]):
                #sum till the number is bigger than the corresponding random value:
                sum_of_proportions += fitness_proportion[individual_index]
                #if the sum is bigger and the element is not in the selected add it.
                if sum_of_proportions >= random_top and individual_index not in individuals_selected_index:
                    individuals_selected_index.append(individual_index)
                    individuals_selected.append(individuals[individual_index])
                    break
        individuals_selected = np.vstack(individuals_selected)
        return individuals_selected

    #tournament_selection:
    #tournament:
    def tournament_selection(self, individuals, fitness_of_generation, elements_per_tournament):
        #create list for storaging:
        individuals_selected = list()
        #define the amount of individuals to select
        selected_amount_of_individuals = int(0.5*individuals.shape[0])
        
        #get the best fit:
        best = np.argmax(fitness_of_generation)
        best = individuals[best]
        
        n_of_individuals = individuals.shape[0]
        #we put the bet individual first: 
        individuals_selected.append(best)
        for time in range(selected_amount_of_individuals):
            #select players:
            indexes_selected = np.random.randint(0, n_of_individuals , size=(elements_per_tournament,))
            fitness_selected = fitness_of_generation[indexes_selected]
            #get the best index of the selected index
            best_index = np.argmax(fitness_selected)
            best_selected = individuals[indexes_selected[best_index]]
            #put the winner inside:
            individuals_selected.append(best_selected)
        individuals_selected = np.vstack(individuals_selected)
        return individuals_selected


    #Normal crossover, divides arrays in two and exchanges the genes:
    def normal_cross_over(self, selected_individuals):
        generated_sons = list()
        #get the best to the next generation
        best_of_generation = selected_individuals[0]

        generated_sons.append(best_of_generation)

        for index in range(selected_individuals.shape[0]):
            #get the first parent
            parent1 = selected_individuals[index]
            #get the couple
            random_match = np.random.randint(selected_individuals.shape[0])
            #if its the same throw again
            if random_match==index:
                random_match = np.random.randint(selected_individuals.shape[0])

            parent2 = selected_individuals[random_match]
            index_till = np.random.randint(1, selected_individuals.shape[1])
            #parent1 genes:
            parent1_genes1 = parent1[:index_till]
            parent1_genes2 = parent1[index_till:]

            #parent2 genes:
            parent2_genes1 = parent2[:index_till]
            parent2_genes2 = parent2[index_till:]

            #son generation:
            son1 = np.concatenate((parent1_genes1, parent2_genes2), axis=0)
            son2 = np.concatenate((parent2_genes1, parent1_genes2), axis=0)
            generated_sons.append(son1)
            generated_sons.append(son2)
        generated_sons.pop(-1)
        generated_sons = np.vstack(generated_sons)
        return generated_sons
    
    
    #ordered cross_over:
    def ordered_cross_over(self, selected_individuals):
        generated_sons = list()
        best_of_generation = selected_individuals[0]
        generated_sons.append(best_of_generation)
        gen_length=selected_individuals.shape[1]
        for index in range(selected_individuals.shape[0]):
            parent1 = selected_individuals[index]
            #get the couple if they are the same throw again:
            random_match = np.random.randint(selected_individuals.shape[0])
            if random_match==index:
                random_match = np.random.randint(selected_individuals.shape[0])
            
            #we get the positions to be inserted in the parent 2
            parent2 = selected_individuals[random_match]
            index1_son1 = np.random.randint(0, gen_length)
            index2_son1 = np.random.randint(0, gen_length)
            if index2_son1 == index1_son1:
                while index2_son1 == index1_son1:
                    index2_son1 = np.random.randint(0, gen_length)
            #get the position to be inserted in parent 1:
            index1_son2 = np.random.randint(0, gen_length)
            index2_son2 = np.random.randint(0, gen_length)
            if index2_son2 == index1_son2:
                #throw till the elements are different
                while index2_son2 == index1_son2:
                    index2_son2 = np.random.randint(0, gen_length)
            
            #get elements from parent 1
            element1 = parent1[index1_son1]
            element2 = parent1[index2_son1]
            #get the parent 2 to a new list to aboid mutation (python)
            son1 = list(parent2)
            #because they all use the same letters, we remove this ones to avoid replications
            son1.remove(element1)
            son1.remove(element2)
            #we insert the same letters but in different places.
            son1.insert(index1_son1, element1)
            son1.insert(index2_son1, element2)

            # we repeate the same with parent 2:
            element1 = parent2[index1_son2]
            element2 = parent2[index2_son2]
            son2 = list(parent1)
            son2.remove(element1)
            son2.remove(element2)
            son2.insert(index1_son2, element1)
            son2.insert(index2_son2, element2)

            generated_sons.append(son1)
            generated_sons.append(son2)
        generated_sons.pop(-1)
        generated_sons = np.vstack(generated_sons)
        return generated_sons
            
    #get normal mutations:
    def normal_mutation(self, sons, mutation_rate, possibility):
        amount_of_mutations = int(sons.shape[0]*mutation_rate)
  
        #get the amount of individuals
        amount_of_individuals = sons.shape[0]
        for time in range(amount_of_mutations):
            #pick a individual
            random_index_of_extraction = np.random.randint(0, sons.shape[0])
            #pick a posible gen
            mutated_gen = random.choices(possibility, k=1)[0]
            
            #pick a gen index in son
            random_index = np.random.randint(0, sons.shape[1])
            #mutate son
            sons[random_index_of_extraction, random_index]= mutated_gen
        return sons


    # swap mutation:
    def swap_mutation(self, sons, mutation_rate, posibilities):
        amount_of_mutations = int(sons.shape[0]*mutation_rate)
        amount_of_individuals = sons.shape[0]
        gen_length= sons.shape[1]
        for time in range(amount_of_mutations):
            #pick a individual:
            random_index_of_extraction = np.random.randint(0, amount_of_individuals)
            #pick two indexes for swapping
            random_index_swapping1 = np.random.randint(0, gen_length)
            random_index_swapping2 = np.random.randint(0, gen_length)
            if random_index_swapping1 == random_index_swapping2:
                #make sure that they are different
                while random_index_swapping1 == random_index_swapping2:
                    random_index_swapping2 = np.random.randint(0, gen_length)
            #swap elements
            element_at_1 = sons[random_index_of_extraction, random_index_swapping1]
            element_at_2 = sons[random_index_of_extraction, random_index_swapping2]
            sons[random_index_of_extraction, random_index_swapping1]= element_at_2
            sons[random_index_of_extraction, random_index_swapping2]= element_at_1
        return sons
            
            
            
        

