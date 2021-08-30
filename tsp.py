# tsp.py

#
# Helper functions for the Traveling Salesman Problem
#

import math
import random
import time
import getopt
import sys

# write a route to a file
def write_route_to_file(route, filename):
	text_file = open(filename, "w")
	text_file.write(str_lst(route) + '\n')
	text_file.close()

# write a population to a file
def write_population_to_file(population, filename):
	text_file = open(filename, "w")
	for route in population:
		text_file.write(str_lst(route) + '\n')
	text_file.close()

# read a population from a file and returns the population
def read_population_from_file(filename):
	f = open(filename)
	pupulation = []
	for line in f:
		data = [int(s) for s in line.split(' ')]
		population.append(data)
	f.close()
	return population


# returns true if lst is a permutation of the ints 1 to len(lst), and false
# otherwise
def is_good_perm(lst):
  	return sorted(lst) == list(range(1, len(lst) + 1))

# returns a list of pairs of coordinates, where the first pair is the location
# of the first city, the second pair is the location of second city, and so on
def load_city_locs(fname):
	f = open(fname)
	result = []
	for line in f:
		data = [int(s) for s in line.split(' ')]
		result.append(tuple(data[1:]))
	return result

# returns the distance between points (x1,y1) and (x2,y2)
def dist(x1, y1, x2, y2):
	dx = x1 - x2
	dy = y1 - y2
	return math.sqrt(dx*dx + dy*dy)

# c1 and c2 are integer names of cities, randing from 1 to len(city_locs)
# city_locs is a list of pairs of coordinates (e.g. from load_city_locs)
def city_dist(c1, c2, city_locs):
	return dist(city_locs[c1-1][0], city_locs[c1-1][1],
		        city_locs[c2-1][0], city_locs[c2-1][1])

def test1():
	city_locs = load_city_locs('cities5.txt')
	print(city_locs)
	for i in range(1,6):
		for j in range(1,6):
			d = city_dist(i, j, city_locs)
			print(f'dist between cities {i} and {j}: {d}')

# city_perm is a list of 1 or more city names
# city_locs is a list of pairs of coordinates for every city
def total_dist(city_perm, city_locs):
	assert is_good_perm(city_perm), f'city_perm={city_perm}'
	n = len(city_perm)
	total = city_dist(city_perm[0], city_perm[n-1], city_locs)
	for i in range(1, n):
		total += city_dist(city_perm[i-1], city_perm[i], city_locs)
	return total

def str_lst(lst):
	return ' '.join(str(i) for i in lst)

def first_n(n, lst):
	result = str_lst(lst[:n])
	if len(lst) > n:
		result += ' ...'
	return result

def rand_perm(n):
	result = list(range(1, n+1))
	random.shuffle(result)
	return result

# return the shortest of max_iter random permutations
def rand_best(fname, max_iter):
	city_locs = load_city_locs(fname)
	n = len(city_locs)

	best = rand_perm(n)
	best_score = total_dist(best, city_locs)	
	#print(f'New best perm ({n}): {best}')
	#print(f'              Score: {best_score}\n')
	assert is_good_perm(best)

	curr = best[:]
	curr_score = total_dist(curr, city_locs)
	assert is_good_perm(best)
	assert is_good_perm(curr)

	for i in range(max_iter):
		if curr_score < best_score:
			best = curr
			best_score = curr_score
			#print(f'New best perm ({n}): {first_n(5, best)}')
			#print(f'              Score: {best_score}\n')
		random.shuffle(curr)
		curr_score = total_dist(curr, city_locs)

	print(f'After {max_iter} random tries, this is the best tour:')
	print(best)
	print(f'score = {best_score}')
	print()


###########################################################################################
#
#	 Crossover Operators
#
###########################################################################################

#
# The following explanation of PMX is from this paper:
#
# https://www.researchgate.net/publication/245746380_Genetic_Algorithm_Solution_of_the_TSP_Avoiding_Special_Crossover_and_Mutation
#
# Parents are s=5713642 and t=4627315, and | is a randomly chosen crossover
# point:
#
#    s=571|3642
#    t=462|7315
#
# Two offspring will be created by the crossover.
#
# First offspring: make a copy of s (call it s'), and write s' above t (t
# won't change). Then go through the first 3 cities of s and swap them (in s')
# with the city underneath in t. For example:
#
#       swap        swap        swap
#       +-----+     +---+       +----+
#       |     |     |   |       |    |
#   s'= 571|3642   471|3652   461|3752   462|3751
#   t = 462|7315   462|7315   462|7315   462|7315
#
# So 4627315 is the first offspring.
#
# The second offspring is created similarly, but this time s stays the same
# and a copy t' of t is made:
# 
#   s = 571|3642   571|3642   571|3642   571|3642
#   t'= 462|7315   562|7314   572|6314   571|6324
#       |      |    |  |        |   |
#       +------+    +--+        +---+
#       swap        swap        swap
#
# Thus 5716324 is the second offspring.
#
def pmx(s, t):
	assert is_good_perm(s)
	assert is_good_perm(t)
	assert len(s) == len(t)
	n = len(s)
	
	# choose crossover point at random
	c = random.randrange(1, n-1) # c is index of last element of left part

	# first offspring
	first = s[:]
	i = 0
	while i <= c:
		j = first.index(t[i])
		first[i], first[j] = first[j], first[i]
		i += 1	

	# second offspring
	second = t[:]
	i = 0
	while i <= c:
		j = second.index(s[i])
		second[i], second[j] = second[j], second[i]
		i += 1

	assert is_good_perm(first)
	assert is_good_perm(second)

	return first, second

def pmx_test():
	s = [5,7,1,3,6,4,2]
	t = [4,6,2,7,3,1,5]
	print(f's={s}\nt={t}')
	first, second = pmx(s, t)
	print(f'  {first}  first offspring')
	print(f'  {second} second offspring')


# referenced from: 
# https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
def breed_operator1(parent1, parent2):
	child = []
	childP1 = []
	childP2 = []
    
	geneA = int(random.random() * len(parent1))
	geneB = int(random.random() * len(parent1))
    
	startGene = min(geneA, geneB)
	endGene = max(geneA, geneB)

	for i in range(startGene, endGene):
		childP1.append(parent1[i])

	childP2 = [item for item in parent2 if item not in childP1]

	child = childP1 + childP2
	return child

# Ordered crossover
# referenced from: https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
def ox(ind1, ind2):
	"""
	Executes an ordered crossover (OX) on the input
	individuals. The two individuals are modified in place. This crossover
	expects :term:`sequence` individuals of indices, the result for any other
	type of individuals is unpredictable.
	:param ind1: The first individual participating in the crossover.
	:param ind2: The second individual participating in the crossover.
	:returns: A tuple of two individuals.
	Moreover, this crossover generates holes in the input
	individuals. A hole is created when an attribute of an individual is
	between the two crossover points of the other individual. Then it rotates
	the element so that all holes are between the crossover points and fills
	them with the removed elements in order. For more details see
	[Goldberg1989]_.
	This function uses the :func:`~random.sample` function from the python base
	:mod:`random` module.
	.. [Goldberg1989] Goldberg. Genetic algorithms in search,
	optimization and machine learning. Addison Wesley, 1989
	"""
	size = min(len(ind1), len(ind2))
	a, b = random.sample(range(size), 2)
	if a > b:
		a, b = b, a
	holes1, holes2 = [True] * size, [True] * size
	for i in range(size):
		if i < a or i > b:
			holes1[ind2[i] - 1] = False
			holes2[ind1[i] - 1] = False

	# We must keep the original values somewhere before scrambling everything
	child1, child2 = ind1[:], ind2[:]
	k1, k2 = b + 1, b + 1
	for i in range(size):
		if not holes1[ind1[(i + b + 1) % size] - 1]:
			child1[k1 % size] = ind1[(i + b + 1) % size]
			k1 += 1

		if not holes2[ind2[(i + b + 1) % size] - 1]:
			child2[k2 % size] = ind2[(i + b + 1) % size]
			k2 += 1

	# Swap the content between a and b (included)
	for i in range(a, b + 1):
		child1[i], child2[i] = child2[i], child1[i]

	return child1, child2


def ox_test():
	#s = [5,7,1,3,6,4,2]
	#t = [4,6,2,7,3,1,5]
	#s = [9,8,4,5,6,7,1,3,2,0]
	s = [1,2,3,4,5,6,7,8,9]
	t = [4,5,2,1,8,7,6,9,3]
	print(f's={s}\nt={t}')
	first, second = ox(s, t)
	print(f'  {first}  first offspring')
	print(f'  {second} second offspring')
	print(f's={s}\nt={t}')


###########################################################################################
#
#	Mutation operators
#
###########################################################################################

# random swap
def do_rand_swap(lst):
	n = len(lst)
	i, j = random.randrange(n), random.randrange(n)
	lst[i], lst[j] = lst[j], lst[i]  # swap lst[i] and lst[j]
	return lst


# the cim and rsm operators are referenced from:
# https://arxiv.org/pdf/1203.3097.pdf

# Centre inverse mutation
def cim(lst):
	n = len(lst)
	i = random.randrange(1, n-1)
	lst[0:i] = reversed(lst[0:i])
	lst[i:] = reversed(lst[i:])
	return lst

# Reverse Sequence Mutation
def rsm(lst):
	n = len(lst)
	i = random.randrange(0, n-1)
	j = random.randrange(i+1, n)
	lst[i:j+1] = reversed(lst[i:j+1])
	return lst 

# a modified mutator based on 2-opt, but for efficiency we only swap once and stop
def two_opt_mutator(lst, city_locs):
	cur_dist = total_dist(lst, city_locs)
	n = len(lst)
	for i in range(1, n - 1):
		for k in range(i + 1, n):
			city_k_next = lst[0] if k + 1 == n else lst[k+1]
			dist1 = city_dist(lst[i-1], lst[i], city_locs) + city_dist(lst[k], city_k_next, city_locs)
			dist2 = city_dist(lst[i-1], lst[k], city_locs) + city_dist(lst[i], city_k_next, city_locs)
			
			if dist1 > dist2:
				lst[i:k+1] = reversed(lst[i:k+1])
				return lst
	return lst



###########################################################################################
#
#		2-opt for TSP
#
# We use 2-opt to help select a good initial population for our genetic algorithm framework
#
# The following implementation of 2-opt is referenced from and modified based on:
#
#   https://en.wikipedia.org/wiki/2-opt
#   https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
###########################################################################################

def do_2opt_swap(route, i, k):
	assert i >= 0 and i < (len(route) - 1)
	assert k > i and k < len(route)
	route[i:k+1] = reversed(route[i:k + 1])
	return route

# city_route is a list of 1 or more city names
# city_locs is a list of pairs of coordinates for every city
# max_improve_iter is the maximum number of iteration of try when no improvement can be found
def do_2opt(cur_route, city_locs):
	best_route = cur_route[:]
	best_dist = total_dist(best_route, city_locs)
	n = len(best_route)
	iteration = 0
	improvement = True
	while improvement:
		improvement = False
		for i in range(1, n - 1):
			for k in range(i + 1, n):
				city_k_next = best_route[0] if k + 1 == n else best_route[k+1]
				dist1 = city_dist(best_route[i-1], best_route[i], city_locs) + city_dist(best_route[k], city_k_next, city_locs)
				dist2 = city_dist(best_route[i-1], best_route[k], city_locs) + city_dist(best_route[i], city_k_next, city_locs)
			
				if dist1 > dist2:
					best_route = do_2opt_swap(best_route, i, k)
					improvement = True
					break
			if improvement:
				break
	assert is_good_perm(best_route)
	return best_route	

###########################################################################################
#
#		Find nearest neighor solution for TSP
#
# We use nerst neighbor solution to help select a good initial generation for our genetic algorithm framework
#
# The following implementation of nearest neighbor solution is referenced and modified based on:
#
# https://github.com/rshipp/tsp/blob/master/nearestneighbor.py
#
###########################################################################################

def get_nearest_neighbor(rest_cities, city_name, city_locs):
	dmin = city_dist(rest_cities[0], city_name, city_locs);
	closest_city = rest_cities[0]
	for i in rest_cities:
		d = city_dist(i, city_name, city_locs)
		if d < dmin:
			dmin = d
			closest_city = i
	return closest_city
		

def get_route_by_nearest_neighbor(start_city, city_locs):
	n = len(city_locs)
	rest_cities = list(range(1,n+1))
	rest_cities.remove(start_city)
	route = [start_city]

	next_city = start_city
	while(len(route) < n):
		closest_city = get_nearest_neighbor(rest_cities, next_city, city_locs)	
		route.append(closest_city)
		rest_cities.remove(closest_city)

	assert is_good_perm(route)
	return route 

###########################################################################################
#
#   The implentation of our genetirc algorithm for TSP class
#
###########################################################################################

class GA_TSP(object):

	def __init__(self, fname):
		self.fname = fname
		self.city_locs = load_city_locs(fname)
	
	def run_GA(self, init_pop_operator, crossover_operator, mutation_operator, max_gen = 10000, pop_size = 20, known_population = []):
		self.init_pop_operator = init_pop_operator
		self.crossover_operator = crossover_operator
		self.mutation_operator = mutation_operator
		self.pop_size = pop_size
		self.max_gen = max_gen
		self.best_route = []
		self.running_time = 0

		if len(known_population) > 0:
			for route in known_population:
				assert len(route) == len(self.city_locs)

		self.cur_gen = known_population[:]


		start_time = time.time()
		self.cur_gen = self.__gen_init_population()
		for i in range(max_gen):
			if int(i % (max_gen / 10)) == 0:
				print(f'{int(i * 100 / max_gen)}% is completed...')
			self.cur_gen = self.__next_generation(self.cur_gen)
		end_time = time.time()

		self.running_time = end_time - start_time

		self.best_route = self.__get_best_route(self.cur_gen)
		
		# a 2opt post-process here
		print("Post processing with 2opt...")
		self.best_route = do_2opt(self.best_route, self.city_locs)
		
		assert len(self.best_route) == len(self.city_locs)

		self.__print_result()

		return self.best_route

	def write_best_route(self, filename):
		write_route_to_file(self.best_route, filename)
		print(f'The best solution is written to {filename}')
		print()

	def write_log(self, filename):
		log = open(filename, "a")
		log.write(f'GA completed in {self.running_time}s, with operators:\n')
		log.write(f'\tinit_pop_generator = {self.init_pop_operator}\n')
		log.write(f'\tcrossover_operator = {self.crossover_operator}\n')
		log.write(f'\tmutation_operator = {self.mutation_operator}\n')
		log.write(f'\tmax_gen = {self.max_gen}\n')
		log.write(f'\tpop_size = {self.pop_size}\n')
		log.write(f'Best score = {self.__get_total_dist(self.best_route)}\n')
		log.write(str_lst(self.best_route) + '\n\n')
		log.close()
	
	def __gen_init_population(self):
		log.close()

	def write_generation(self, filename):
		self.cur_gen = self.__rank_routes(self.cur_gen)
		write_population_to_file(self.cur_gen, filename)
		print(f'The current population is written to {filename}')
		print()

	def __print_result(self):
		"""
		print(f'\tinit_pop_generator = {self.init_pop_operator}')
		print(f'\tcrossover_operator = {self.crossover_operator}')
		print(f'\tmutation_operator = {self.mutation_operator}')
		print(f'\tmax_gen = {self.max_gen}')
		print(f'\tpop_size = {self.pop_size}')
		"""
		print(f'GA completed in {self.running_time}s, with operators:')
		print(f'Best score = {self.__get_total_dist(self.best_route)}')
		print(str_lst(self.best_route) + '\n')
	
	def __gen_init_population(self):
		n = len(self.city_locs)
		remain_pop_size = self.pop_size - len(self.cur_gen)
		population = []

		# if a known best route is specified, append it to the initial population, so reduce the remain population size

		if self.init_pop_operator == "random":
			population = [rand_perm(n) for i in range(remain_pop_size)]

		elif self.init_pop_operator == "random_2opt":
			random_pop = [rand_perm(n) for i in range(remain_pop_size)]
			count = 0
			for route in random_pop:
				print(f'generating route {count}...')
				count += 1
				new_route = do_2opt(route, self.city_locs)
				population.append(new_route)

		elif self.init_pop_operator == "nearest_neighbor_2opt":
			count = 0
			for i in range(remain_pop_size):
				print(f'generating route {count}...')
				count += 1
				
				start_city = random.randrange(1, n+1)
				new_route = get_route_by_nearest_neighbor(start_city, self.city_locs)
				new_route = do_2opt(new_route, self.city_locs)
				population.append(new_route)
		else:
			print(f'Error: invalid init_pop_operator: {self.init_pop_operator}')
			print(f'Valid init_pop_operator includes: random, random_2opt, nearest_neighbor_2opt')
			sys.exit(1)

		# if a known best route is specified, append it to the initial population
		self.cur_gen = self.cur_gen + population

		assert len(self.cur_gen) == self.pop_size	
		return self.cur_gen
		
	def __next_generation(self, cur_gen, mutation_rate = 0.4):
		self.weights_cached = False  

		n = len(cur_gen)
		cur_gen = self. __rank_routes(cur_gen)

		top_half = cur_gen[:int(n/2)]
		next_gen = top_half[:]
		#print(f'best score = {self.__get_total_dist(cur_gen[0])}')
		while len(next_gen) < self.pop_size:
			parent1, parent2 = self.__selection(cur_gen)
			child1, child2 = self.__crossover(parent1, parent2)

			if random.uniform(0, 1) < mutation_rate:
				self.__mutate(child1)
				self.__mutate(child2)

			next_gen.append(child1)
			next_gen.append(child2)

		next_gen = next_gen[:self.pop_size]
		return next_gen

	def __selection(self, cur_gen):
		factor = self.__get_total_dist(cur_gen[0])          # multiply by a factor, otherwise the weight will be too small
		fitness_fn = lambda s: factor / self.__get_total_dist(s)         # weight is (factor * 1.0 / dist)	
		return self.__random_weighted_selections(cur_gen, 2, fitness_fn)

	# The following implentation is referened and modified based on AIMA-Python code:
	# http://aima.cs.berkeley.edu/python/search.html
	def __random_weighted_selections(self, seq, n, weight_fn):
		"""Pick n elements of seq, weighted according to weight_fn.
		That is, apply weight_fn to each element of seq, add up the total.
		Then choose an element e with probability weight[e]/total.
		Repeat n times, with replacement. """
		# if weights are not cached, then compute the weights
		# cache the weights for cur_gen so that we don't need to compute every time while generating nex 
		if self.weights_cached == False:
			self.weights = []
			runningtotal = 0
			for item in seq:
				runningtotal += weight_fn(item)
				self.weights.append(runningtotal)
			self.weights_cached = True
		else:
			assert len(self.weights) == len(seq)

		selections = []
		for s in range(n):
			r = random.uniform(0, self.weights[-1])
			for i in range(len(seq)):
				if self.weights[i] > r:
					selections.append(seq[i])
					break
		assert len(selections) == n
		return selections


	def __crossover(self, parent1, parent2):
		if self.crossover_operator == "pmx":
			return pmx(parent1, parent2)
		elif self.crossover_operator == "ox":
			return ox(parent1, parent2)
		else:
			print(f'Error: invalid crossover_operator: {self.crossover_operator}')
			print(f'Valid crossover_operator includes: pmx, ox')
			sys.exit(1)
	

	def __mutate(self, child):
		if self.mutation_operator == "rand_swap":
			child = do_rand_swap(child)
		elif self.mutation_operator == "cim":
			child = cim(child)
		elif self.mutation_operator == "rsm":
			child = rsm(child)
		elif self.mutation_operator == "2opt":
			child = two_opt_mutator(child, self.city_locs)
		else:
			print(f'Error: invalid mutation_operator: {self.mutation_operator}')
			print(f'Valid mutation_operators includes: rand_swap, cim, rsm, 2opt')
			sys.exit(1)
		return child

	def __get_total_dist(self, route):
		return total_dist(route, self.city_locs)

	def __get_best_route(self, cur_gen):
		ranked_gen = self.__rank_routes(cur_gen)
		assert len(ranked_gen[0]) == len(self.city_locs)
		assert is_good_perm(ranked_gen[0])
		return ranked_gen[0]

	def __rank_routes(self, population):
		curr_gen = [(self.__get_total_dist(p), p) for p in population]
		curr_gen.sort()
		population = [p[1] for p in curr_gen[:]]
		return population


###########################################################################################
#
#	Some tests for the above different genetic algorithms
#
###########################################################################################

def run_opt_test(fname):
	city_locs = load_city_locs(fname)
	n = len(city_locs)
	cur_route = list(range(1,n+1))
	cur_route = rand_perm(n)
	route_2opt = do_2opt(cur_route[:], city_locs, max_iter)
	print(f'2-opt search completed, best_route is:')
	print(route_2opt)
	print(f'score = {total_dist(route_2opt, city_locs)}')

	route_3opt = do_3opt(cur_route[:], city_locs, max_iter)
	print(f'3-opt search completed, best_route is:')
	print(route_3opt)
	print(f'score = {total_dist(route_3opt, city_locs)}')

	route_nearsest_neighbor =  get_route_by_nearest_neighbor(random.randrange(1, n), city_locs)
	print(f'nearest_neighbor search completed, best_route is:')
	print(route_nearsest_neighbor)
	print(f'score = {total_dist(route_nearest_neighbor, city_locs)}')

def run_genetic_algorithm_test(fname, max_iter=1000, pop_size=20, population = []):
	print('Start testing the GA_TSP class ...')

	tsp = GA_TSP(fname)
	"""
	tsp.run_GA(init_pop_operator, crossover_operator, mutation_operator, max_iter, pop_size, population)

	Avaliable options:
	init_pop_operator: random, random_2opt, nearest_neighbor_2opt'
	crossover_operator: pmx, ox'
	mutation_operators: rand_swap, cim, rsm, 2opt
	"""

	log_ofile = 'log.txt'

	best1 = tsp.run_GA("random", "pmx", "rand_swap", max_iter, pop_size, population)
	tsp.write_log(log_ofile)
	
	best2 = tsp.run_GA("random_2opt", "pmx", "rand_swap", max_iter, pop_size, population)
	tsp.write_log(log_ofile)

	best3 = tsp.run_GA("nearest_neighbor_2opt", "pmx", "rand_swap", max_iter, pop_size, population)
	tsp.write_log(log_ofile)
	
	best4 = tsp.run_GA("random", "ox", "rand_swap", max_iter, pop_size, population)
	tsp.write_log(log_ofile)

	best5 = tsp.run_GA("random", "pmx", "rand_swap", max_iter, pop_size, population)
	tsp.write_log(log_ofile)

	best6 = tsp.run_GA("random", "pmx", "rand_swap", max_iter, pop_size, population)
	tsp.write_log(log_ofile)

	best7 = tsp.run_GA("random", "pmx", "cim", max_iter, pop_size, population)
	tsp.write_log(log_ofile)

	best8 = tsp.run_GA("random", "pmx", "rsm", max_iter, pop_size, population)
	tsp.write_log(log_ofile)

	best9 = tsp.run_GA("random", "pmx", "2opt", max_iter, pop_size, population)
	tsp.write_log(log_ofile)

def printHelpManual():
	print()
	print("tsp.py [option] [argument]")
	print("Available options and arguments include:")
	print("-h [--help]: Print command line interface help manual")
	print("-i [--city_ifile] $arg: set the input city dataset filename")
	print("-o [--solution_ofile] $arg: set the output filename for writing the best solution.")
	print("\t\t\t $arg has a default value: best_solution.txt")
	print("-s [--init_pop_generator] $arg: set the initial population generator for GA")
	print("\t\t\t valid options of $arg are: random, random_2opt, nearest_neighbor_2opt")
	print("\t\t\t $arg has a default value: random")
	print("-c [--crossover] $arg: set the crossover operator for GA")
	print("\t\t\t valid options of $arg are: pmx, ox")
	print("\t\t\t $arg has a default value: pmx")
	print("-m [--mutation] $arg: set the mutation operator for GA")
	print("\t\t\t valid options of $arg are: rand_swap, cim, rsm, 2opt")
	print("\t\t\t $arg has a default value: rsm")
	print("-n [--max_gen] $arg: set the maximum number of generation for GA")
	print("\t\t\t $arg has a default value: 10000")
	print("-k [--pop_size] $arg: set the population size for GA")
	print("\t\t\t $arg has a default value: 20")
	print("-p [--pop_ifile] $arg: set the input filename for reading a known population.")
	print("\t\t\t This allow us to run the genetic algorithm based on an existing good population")
	print("-g [--pop_ofile] $arg: set the output filename for writing the final generation after running the genetic algorithm.")
	print("-l [--log_ofile] $arg: set the output filename for writing the log.")
	print()	

def main(argv):
	city_ifile = ''
	pop_ifile = ''
	pop_ofile = ''
	solution_ofile = 'best_solution.txt'
	log_ofile = ''
	init_pop_generator = 'random'
	crossover_operator = 'pmx'
	mutation_operator = 'rsm'
	max_gen = 10000
	pop_size = 20
	try:
		opts, args = getopt.getopt(argv, "hi:o:p:g:s:c:m:n:k:l:", \
			     ["help", "city_ifile", "solution_ofile", "pop_ifile","pop_ofile" \
			     "init_pop_generator", "crossover", "mutation", "max_gen", "pop_size", "log_ofile"])
			
	except getopt.GetoptError:
		print('Invalid command line input.')
		printHelpManual()
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			printHelpManual()
			sys.exit()
		elif opt in ("-i", "--city_ifile"):
			city_ifile = arg
		elif opt in ("-o", "--solution_ofile"):
			solution_ofile = arg
		elif opt in ("-p", "--pop_ifile"):
			pop_ifile = arg
		elif opt in ("-g", "--pop_ofile"):
			pop_ofile = arg
		elif opt in ("-s", "--init_pop_generator"):
			init_pop_generator = arg
		elif opt in ("-c", "--crossover"):
			crossover_operator = arg
		elif opt in ("-m", "--mutation"):
			mutation_operator = arg
		elif opt in ("-n", "--max_gen"):
			max_gen = int(arg)
		elif opt in ("-k", "--pop_size"):
			pop_size = int(arg)
		elif opt in ("-l", "--log_ofile"):
			log_ofile = arg
		else:
			print("Invalid option: {opt}")
			printHelpManual()
			sys.exit(1)

	if city_ifile == '':
		print()
		print("Input dataset if not specified, please specify the city location file with -i option")
		print("e.g., python3 tsp.py -i cities1000.txt")
		print("You can use -h to see the help manual")
		print()
		sys.exit(1)

	print()
	print('Running genetic algorithm framework with following arguments:')
	print(f'\tinit_pop_generator = {init_pop_generator}')
	print(f'\tcrossover = {crossover_operator}')
	print(f'\tmutation = {mutation_operator}')
	print(f'\tmax_gen = {max_gen}')
	print(f'\tpop_size = {pop_size}')

	population = []
	if pop_ifile != '':
		population = read_population_from_file(pop_ifile)
		print(f'\tpop_ifile = {pop_ifile}')
	print()

	tsp = GA_TSP(city_ifile)
	tsp.run_GA(init_pop_generator, crossover_operator, mutation_operator, max_gen, pop_size, population)
	
	# write the population to a file	
	if pop_ofile != '':
		tsp.write_generation(pop_ofile)

	# write the best route solution to a file
	tsp.write_best_route(solution_ofile)

	# write log to a file
	if log_ofile != '':
		tsp.write_log(log_ofile)

if __name__ == '__main__':
	# the main function
	main(sys.argv[1:])

	# Run the test functions
	#run_opt_test('cities20.txt')
	#run_genetic_algorithm_test('cities20.txt', 100, 20)
	#run_genetic_algorithm_test('cities1000.txt', 10000, 20)

