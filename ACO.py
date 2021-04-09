import numpy as np
from threading import Thread
#import pygame 

#pygame.init()
#np.random.seed(1)
pheromone_const = 5
# COLORS = [[[(r*42,g*42,b*42) for b in range(6)]for g in range(6)]for r in range(6)]
# COLORS = np.reshape(COLORS,(216,3))

class Egde_IDPC(object):
	def __init__(self,u, v, w, d):
		super(Egde_IDPC, self).__init__()
		self.u = u
		self.v = v
		self.w = w
		self.d = d
		self.pheromone = 0
		self.ants = []
		#self.color = COLORS[d]

	def update_pheromone(self, pheromone_evaporation):
		self.pheromone = (1-pheromone_evaporation)*self.pheromone
		for ant in self.ants:
			self.pheromone += pheromone_const*ant.tour_complete
		self.ants = []
	def print(self):
		print('({}->{}), color: {}, cost: {}'.format(self.u, self.v, self.d, self.w))

class Graph_IDPC():
	def __init__(self, N, D, s, t):
		self.N = N
		self.D = D
		self.s = s
		self.t = t
		self.edges = [[[] for v in range(N+1)] for u in range(N+1)]

	def print(self):
		for u in range(N+1):
			for v in range(N+1):
				print('({}->{})'.format(u, v))
				for edge in self.edges[u][v]:
					print('color: {}, cost: {}'.format(edge.d, edge.w))


class Ant():
	def __init__(self, name, graph, alpha, beta):
		super(Ant, self).__init__()
		self.name = name
		self.graph = graph			
		self.route = [graph.s]
		self.domains_traveled = [-1]
		self.distance_traveled = 0.0
		self.location = graph.s
		self.alpha = alpha
		self.beta = beta
		self.tour_complete = -1
			
	def run(self):
		while self.location != graph.t:
			move = self.move()
			if not move:
				return
		self.tour_complete = 1

	def move(self):
		def compute_attractiveness(edge):
			attractiveness = -edge.w
			#attractiveness *= 10 if edge.d == self.domains_traveled[-1] else 1
			return attractiveness

		candidate_edges = [edge for edge in self.graph.adj[self.location] 
			if not (edge.d != self.domains_traveled[-1] and edge.d in self.domains_traveled)
			and edge.v not in self.route]
		if len(candidate_edges) == 0:
			return False

		edges_prob = [np.exp(edge.pheromone*self.alpha + (compute_attractiveness(edge))*self.beta)
						for edge in candidate_edges] 

		edges_prob = edges_prob/np.sum(edges_prob)
		#print(np.max(edges_prob))
		chosen_edge = np.random.choice(candidate_edges, p=edges_prob)
		self.route.append(chosen_edge.v)
		self.domains_traveled.append(chosen_edge.d)
		self.distance_traveled += chosen_edge.w
		self.location = chosen_edge.v
		chosen_edge.ants.append(self)
		return True

	def back(self):
		self.route = [self.graph.s]
		self.domains_traveled = [-1]
		self.distance_traveled = 0.0
		self.location = self.graph.s
		self.tour_complete = -1

	def states(self):
		print()
		print('name',self.name)
		print('route',self.route)
		print('domains_traveled',self.domains_traveled)
		print('distance_traveled',self.distance_traveled)
		print('location',self.location)
		print('tour_complete', self.tour_complete)


class ACO(object):
	def __init__(self, graph, num_ants, alpha, beta, pheromone_evaporation):
		super(ACO, self).__init__()
		self.graph = graph
		self.num_ants = num_ants
		self.alpha = alpha              	
		self.beta = beta
		self.pheromone_evaporation = pheromone_evaporation
		self.ants = []
		for name in range(num_ants):
			self.ants.append(Ant(name, graph, alpha, beta))
		self.shortest_distance = 1e9
		self.optimal_route = None
		self.optimal_path = None

	def run(self, iterations):
		print('source:',self.graph.s)
		print('terminal:',self.graph.t)
		ctr = 0
		for _ in range(iterations):
			ctr += 1
			print()
			print('iteration:',ctr)
			for ant in self.ants:
				ant.run()
				#ant.states()

				if ant.distance_traveled < self.shortest_distance:
					self.shortest_distance = ant.distance_traveled
					self.optimal_route = ant.route
					self.optimal_path = ant.domains_traveled

			#update pheromone
			for adj_u in self.graph.adj:
				for edge in adj_u:
					#print([ant.name for ant in edge.ants])
					edge.update_pheromone(self.pheromone_evaporation)

			for ant in self.ants:
				ant.back()

			print()
			print('optimal_route:',self.optimal_route)
			print('shortest_distance:', self.shortest_distance)
			print('optimal_path:', self.optimal_path)

class NSGA_II(object):
	"""docstring for NSGA_II"""
	def __init__(self, arg):
		super(NSGA_II, self).__init__()
		self.arg = arg
		
					
def gen_solution(graph):
	nodes = list(range(1,graph.N+1))
	s = graph.s
	t = graph.t
	path = []
	nodes.remove(s)
	cur_node = s
	while cur_node != t:
		while True:
			next_node = np.random.choice(nodes)
			if len(graph.edges[cur_node][next_node])!=0:
				nodes.remove(next_node)
				break
		path.append(np.random.choice(graph.edges[cur_node][next_node]))
		cur_node = next_node
	for edge in path:
		edge.print()
	return path

def constraint_violation(path):
	cur_domain = -1
	domains_traveled = set()
	cs = 0
	for edge in path:
		if edge.d in domains_traveled and edge.d != cur_domain:
			cs += 1

		domains_traveled.add(edge.d)
		cur_domain = edge.d
	print(domains_traveled)
	return cs

def traveled_distance(path):
	return sum([edge.w for edge in path])

def change_edges(path, graph, mutation_rate):
	new_path = []
	for edge in path:
		if np.random.rand() > mutation_rate:
			u = edge.u
			v = edge.v
			new_path.append(np.random.choice(graph.edges[u][v]))
		else:
			new_path.append(edge)
	return new_path

def change_nodes(path, graph, mutation_rate):
	nodes = set([edge.u for edge in path]+[graph.t])
	node = np.random.choice(list(set(range(1,graph.N+1))-nodes))
	idx = np.random.choice(range(len(path)))
	edge = np.random.choice(graph.edges[path[idx].v][path[idx+1].u])
	return path[:idx] + [edge] + path[idx:]

def merge_edges(mom, dad, graph):
	edges = set(mom) | set(dad)
	s = graph.s 
	t = graph.t
	cur_node = s 
	child = []
	while cur_node != t:
		chosen_edge = np.random.choice([edge for edge in edges if edge.u == cur_node])
		child.append(chosen_edge)
		cur_node = chosen_edge.v
	if child == mom or child == dad:
		return []
	print_path(child)
	return [child]

def merge_nodes(mom, dad, graph):
	while True:
		mom_point = np.random.choice(range(len(mom)))
		dad_point = np.random.choice(range(len(dad)))
		u_mom = mom[mom_point].u
		v_mom = mom[mom_point].v
		u_dad = dad[dad_point].u
		v_dad = dad[dad_point].v
		if len(graph.edges[u_mom][v_dad]) != 0 and len(graph.edges[u_dad][v_mom]) != 0:
			break
	childs = [mom[:mom_point], dad[:dad_point]]
	childs[0].append(np.random.choice(graph.edges[u_mom][v_dad]))
	childs[1].append(np.random.choice(graph.edges[u_dad][v_mom]))
	childs[0] += dad[mom_point+1:]
	childs[1] += mom[mom_point+1:]
	print('child 1')
	print_path(childs[0])
	print('child 2')
	print_path(childs[1])
	return childs

def print_path(path):
	for edge in path:
		edge.print()



with open('IDPC-DU/set1/idpc_10x5x425.idpc','r') as f:
	lines = f.read().splitlines()
	# number of nodes and domains
	N, D = lines[0].split()
	# source node and terminal node
	s, t = lines[1].split()
	# edge (u,v) has weight w and belong to domain d.
	N, D, s, t = int(N), int(D), int(s), int(t)
	graph = Graph_IDPC(N, D, s, t)
	check = []
	for line in lines[2:]:
		u, v, w, d = line.split()
		u, v, w, d = int(u), int(v), int(w), int(d)
		found = False
		for edge in graph.edges[u][v]:
			if edge.d == d:
				found = True
				if edge.w > w:
					edge.w = w
				break
		if not found:
			graph.edges[u][v].append(Egde_IDPC(u, v, w, d))

# aco = ACO(graph, num_ants=100, alpha=1, beta=0.5, pheromone_evaporation=0.5)
# aco.run(100)

#graph.print()
print('mom')
mom = gen_solution(graph)
print('dad')
dad = gen_solution(graph)
print('child')
merge_edges(mom, dad, graph)

merge_nodes(mom, dad, graph)

# cs = constraint_violation(path)
# length = traveled_distance(path)
# print(cs)
# print(length)

#new_path = mutation(path, graph, 0.4)

class NearestFirst(object):
	"""docstring for NearestFirst"""
	def __init__(self):
		super(NearestFirst, self).__init__()

	def p(self, edges):
		p = np.array([np.exp(-edge.w) for edge in edges])
		return p / np.sum(p)

	def gen_solution(self, graph):
		nodes = list(range(1,graph.N+1))
		s = graph.s
		t = graph.t
		path = []
		nodes.remove(s)
		cur_node = s

		while cur_node != t:
			cur_edges = []
			for node in nodes:
				cur_edges += graph.edges[cur_node][node]

			chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges))
			path.append(chosen_edge)

			cur_node = chosen_edge.v
			nodes.remove(cur_node)

		return Path(path)

	def crossover(self, mom, dad, graph, crossover_rate=0.9):
		if np.random.rand() > crossover_rate:
			return []
		edges = set(mom.edges) | set(dad.edges)
		s = graph.s
		t = graph.t
		nodes = list(range(1,graph.N+1))
		cur_node = s
		nodes.remove(s)
		path = []
		while cur_node != t:
			cur_edges = [edge for edge in edges if edge.u == cur_node and edge.v in nodes]
			if len(cur_edges) != 0:
				chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges))
			else:
				cur_edges = []
				for node in nodes:
					cur_edges += graph.edges[cur_node][node]

				chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges))

			nodes.remove(chosen_edge.v)
			cur_node = chosen_edge.v
			path.append(chosen_edge)

		if path == mom.edges or path == dad.edges:
			return []

		return [Path(path)]

	def mutation(self, path, graph, mutation_rate=0.1):
		if np.random.rand() > mutation_rate:
			return path

		nodes = set([edge.u for edge in path.edges]+[graph.t])
		candidate = list(set(range(1,graph.N+1))-nodes)
		if len(candidate) == 0:
			return path
		node = np.random.choice(candidate)
		idx = np.random.choice(range(len(path.edges)))
		edges = graph.edges[path.edges[idx].u][node]
		edge1 = np.random.choice(edges, p=self.p(edges))
		edges = graph.edges[node][path.edges[idx].v]
		edge2 = np.random.choice(edges, p=self.p(edges))
		return Path(path.edges[:idx] + [edge1, edge2] + path.edges[idx+1:])


class ConstaintFirst(object):
	"""docstring for ConstaintFirst"""
	def __init__(self):
		super(ConstaintFirst, self).__init__()

	def p(self, edges, domains_traveled):
		p = []
		for edge in edges:
			if edge.d in domains_traveled:
				if edge.d == domains_traveled[-1]:
					p.append(20)
				else:
					p.append(1)
			else:
				p.append(10)
		return p / np.sum(p)

	def gen_solution(self, graph):
		nodes = list(range(1,graph.N+1))
		s = graph.s
		t = graph.t
		path = []
		nodes.remove(s)
		domains_traveled = [-1]
		cur_node = s

		while cur_node != t:
			cur_edges = []
			for node in nodes:
				cur_edges += graph.edges[cur_node][node]

			chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges, domains_traveled))
			domains_traveled.append(chosen_edge.d)
			path.append(chosen_edge)
			cur_node = chosen_edge.v
			nodes.remove(cur_node)

		return Path(path)

	def crossover(self, mom, dad, graph, crossover_rate=0.9):
		edges = set(mom) | set(dad)
		s = graph.s
		t = graph.t
		nodes = list(range(1,graph.N+1))
		cur_node = s
		nodes.remove(s)
		domains_traveled = [-1]
		path = []
		while cur_node != t:
			cur_edges = [edge for edge in edges if edge.u == cur_node and edge.v in nodes]
			if len(cur_edges) != 0:
				chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges, domains_traveled))
			else:
				cur_edges = []
				for node in nodes:
					cur_edges += graph.edges[cur_node][node]

				chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges, domains_traveled))

			domains_traveled.append(chosen_edge.d)
			nodes.remove(chosen_edge.v)
			cur_node = chosen_edge.v
			path.append(chosen_edge)

		return path


class NodePriority(object):
	"""docstring for NodePriority"""
	def __init__(self, graph):
		super(NodePriority, self).__init__()
		self.node_priority = [0]
		for node in range(1, graph.N+1):
			conect_edges = []
			for edges in graph.edges[node]:
				conect_edges += edges
			self.node_priority.append(np.exp(-np.mean([edge.w for edge in conect_edges])))

	def p(self, nodes):
		p = [self.node_priority[node] for node in nodes]
		return p / np.sum(p)

	def pnf(self, edges):
		p = np.array([np.exp(-edge.w) for edge in edges])
		return p / np.sum(p)

	def gen_solution(self, graph):
		nodes = list(range(1,graph.N+1))
		s = graph.s
		t = graph.t
		path = []
		nodes.remove(s)
		cur_node = s
		while cur_node != t:
			while True:
				next_node = np.random.choice(nodes, p=self.p(nodes))
				if len(graph.edges[cur_node][next_node])!=0:
					nodes.remove(next_node)
					break
			path.append(np.random.choice(graph.edges[cur_node][next_node], p=self.pnf(graph.edges[cur_node][next_node])))
			cur_node = next_node
		return Path(path)

	def crossover(self, mom, dad, graph, crossover_rate=0.9):
		edges = set(mom) | set(dad)
		s = graph.s
		t = graph.t
		nodes = list(range(1,graph.N+1))
		cur_node = s
		nodes.remove(s)
		path = []
		while cur_node != t:
			cur_edges = [edge for edge in edges if edge.u == cur_node and edge.v in nodes]
			if len(cur_edges) != 0:
				chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges))
			else:
				cur_edges = []
				for node in nodes:
					cur_edges += graph.edges[cur_node][node]

				chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges))

			nodes.remove(chosen_edge.v)
			cur_node = chosen_edge.v
			path.append(chosen_edge)

		return path


class RandomMove(object):
	"""docstring for RandomMove"""
	def __init__(self, arg):
		super(RandomMove, self).__init__()
		self.arg = arg