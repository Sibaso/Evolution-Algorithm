import numpy as np
from initialization import pop_init, choose_path
import data_reader
class Egde_IDPC(object):
	def __init__(self,u, v, w, d): # edgu (u, v), weight = w, domain = d
		super(Egde_IDPC, self).__init__()
		self.u = u
		self.v = v
		self.w = w
		self.d = d


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
		num_edges = 0
		for u in range(N+1):
			for v in range(N+1):
				print('({}->{})'.format(u, v))
				for edge in self.edges[u][v]:
					print('color: {}, cost: {}'.format(edge.d, edge.w))
					num_edges += 1
		print('number of edges:', num_edges)


class Path(object):
	"""docstring for Path"""
	def __init__(self, edges, skill_factor):
		super(Path, self).__init__()
		self.edges = edges
		self.cost = (self.traveled_distance(), self.constraint_violation()) #[self.constraint_violation(), self.traveled_distance()]
		self.rank = [1, 1]
		self.skill_factor = skill_factor
		self.scalar_fitness = 0

	def print(self, name='path'):
		print('__{}__: constraint violation = {}, traveled distance = {}, rank = {}, skill factor = {}, scalar fitness = {}'.format(
			name, self.cost[1], self.cost[0], self.rank, self.skill_factor, self.scalar_fitness))
		for edge in self.edges:
			edge.print()

	def constraint_violation(self):
		cur_domain = -1
		domains_traveled = set()
		cs = 0
		for edge in self.edges:
			if edge.d in domains_traveled and edge.d != cur_domain:
				cs += 1

			domains_traveled.add(edge.d)
			cur_domain = edge.d
		return cs

	def traveled_distance(self):
		return sum([edge.w for edge in self.edges])

	def update(self):
		#self.fitness = (self.constraint_violation(), self.traveled_distance())
		self.skill_factor = np.argmin(self.rank)
		self.scalar_fitness = 1.0/np.min(self.rank)



class DistanceFirst(object):
	"""docstring for DistanceFirst"""
	def __init__(self, graph):
		super(DistanceFirst, self).__init__()
		self.graph = graph

	def p(self, edges, nodes):
		graph = self.graph
		p = []
		for edge in edges:
			near_v = []
			for i in nodes:

				near_v += graph.edges[edge.v][i]

			if len(near_v) != 0:
				p_v = np.min([j.w for j in near_v])
			elif edge.v == graph.t:
				p_v = 10
			else:
				p_v = 100
			p.append(edge.w+p_v)
		p = np.array(p)
		p = 1/p
		return p / np.sum(p)

	def gen_solution(self):
		graph = self.graph
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

			chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges, nodes))
			path.append(chosen_edge)

			cur_node = chosen_edge.v
			nodes.remove(cur_node)

		return Path(path, 1)

	def crossover(self, _mom, _dad, crossover_rate=0.9):
		if np.random.rand() > crossover_rate:
			return []

		mom = _mom.edges
		dad = _dad.edges
		graph = self.graph

		if mom == dad:
			return [self.mutation(_mom, mutation_rate=1)]

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
				chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges, nodes))
			else:
				cur_edges = []
				for node in nodes:
					cur_edges += graph.edges[cur_node][node]

				chosen_edge = np.random.choice(cur_edges, p=self.p(cur_edges, nodes))

			nodes.remove(chosen_edge.v)
			cur_node = chosen_edge.v
			path.append(chosen_edge)

		if path == mom or path == dad:
			return []

		return [Path(path, 1)]

	def mutation(self, _path, mutation_rate=0.1):
		if np.random.rand() > mutation_rate:
			return _path

		path = _path.edges
		graph = self.graph
		nodes = set([edge.u for edge in path]+[graph.t])
		candidate = list(set(range(1,graph.N+1))-nodes)
		if len(candidate) == 0:
			return _path
		node = np.random.choice(candidate)
		idx = np.random.choice(range(len(path)))
		edges = graph.edges[path[idx].u][node]
		edge1 = np.random.choice(edges, p=self.p(edges, nodes))
		edges = graph.edges[node][path[idx].v]
		edge2 = np.random.choice(edges, p=self.p(edges, nodes))
		return Path(path[:idx] + [edge1, edge2] + path[idx+1:], 1)


class ConstaintFirst(object):
	"""docstring for ConstaintFirst"""
	def __init__(self, graph):
		super(ConstaintFirst, self).__init__()
		self.graph = graph

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

	def gen_solution(self):
		graph = self.graph
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

		return Path(path, 0)

	def crossover(self, _mom, _dad, crossover_rate=0.9):
		if np.random.rand() > crossover_rate:
			return []

		mom = _mom.edges
		dad = _dad.edges
		graph = self.graph

		if mom == dad:
			return [self.mutation(_mom, mutation_rate=1)]

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

		return [Path(path, 0)]

	def mutation(self, _path, mutation_rate=0.1):

		path = _path.edges
		graph = self.graph
		new_path = []
		for edge in path:
			if np.random.rand() > mutation_rate:
				u = edge.u
				v = edge.v
				new_path.append(np.random.choice(graph.edges[u][v]))
			else:
				new_path.append(edge)
		return Path(new_path, 0)


def gen_solution(graph):
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

		chosen_edge = np.random.choice(cur_edges)
		path.append(chosen_edge)

		cur_node = chosen_edge.v
		nodes.remove(cur_node)

	return Path(path, 1)

def crossover(_mom, _dad, graph, crossover_rate=1):
	mom = _mom.edges
	dad = _dad.edges
	if np.random.rand() > crossover_rate:
		return []

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
			chosen_edge = np.random.choice(cur_edges)
		else:
			cur_edges = []
			for node in nodes:
				cur_edges += graph.edges[cur_node][node]

			chosen_edge = np.random.choice(cur_edges)

		nodes.remove(chosen_edge.v)
		cur_node = chosen_edge.v
		path.append(chosen_edge)

	if path == mom or path == dad:
		return [mutation(Path(path, 1), graph, mutation_rate=1)]

	return [Path(path, 1)]

def mutation(_path, graph, mutation_rate=0.2):
	if np.random.rand() > mutation_rate:
		return _path

	path = _path.edges
	nodes = set([edge.u for edge in path]+[graph.t])
	candidate = list(set(range(1,graph.N+1))-nodes)
	if len(candidate) == 0:
		return _path
	node = np.random.choice(candidate)
	idx = np.random.choice(range(len(path)))
	edges = graph.edges[path[idx].u][node]
	edge1 = np.random.choice(edges)
	edges = graph.edges[node][path[idx].v]
	edge2 = np.random.choice(edges)
	return Path(path[:idx] + [edge1, edge2] + path[idx+1:], 1)


def GA(h, corpus_size=200):
	corpus = []
	# _,paths = pop_init(corpus_size, 'IDPC-DU/set1/idpc_20x20x8000.idpc')
	# corpus = [Path(path, 1) for path in paths]
	print('init ...')
	while len(corpus) < corpus_size:
		path = gen_solution(graph)
		ok = True
		for i in corpus:
			if i.edges == path.edges:
				ok = False
				break
		if ok:
			corpus.append(path)

	corpus = sorted(corpus, key=lambda path: path.cost)
	# for path in corpus:
	# 	path.print()
	for generation in range(300):
		childs = []
		for _ in range(corpus_size):
			paths = np.random.choice(corpus, size=11, replace=False)
			mom, dads = paths[0], list(paths[1:])
			while len(dads) > 2:
				dad1, dad2 = np.random.choice(dads, size=2, replace=False)
				if dad1.cost > dad2.cost:
					dads.remove(dad1)
				elif dad1.cost < dad2.cost:
					dads.remove(dad2)
				else:
					dads.remove(dad2)
					dads.remove(dad1)

			if len(dads) == 0:
				continue
			else:
				dad = dads[0]

			child = crossover(mom, dad, graph)
			child = [mutation(i, graph) for i in child]
			childs += child

		corpus += childs
		print(len(corpus))
		corpus = sorted(corpus, key=lambda path: path.cost)
		corpus = corpus[:corpus_size//2] + corpus[-corpus_size//2:]
		print('generation: {}'.format(generation))
		for path in corpus[:1]:
			path.print('best solution')

	# for path in corpus:
	# 	path.print()

	return

def MFEA(graph, corpus_size=25, rmp=0.3):
	h = [DistanceFirst(graph), ConstaintFirst(graph)]
	p = np.array([np.log(corpus_size*2 - i + 1) for i in range(corpus_size*2)])
	p = p / np.sum(p)
	corpus = []
	while len(corpus) < corpus_size:
		path = h1.gen_solution()
		ok = True
		for i in corpus:
			if i.edges == path.edges:
				ok = False
				break
		if ok:
			corpus.append(path)

	while len(corpus) < corpus_size*2:
		path = h2.gen_solution()
		ok = True
		for i in corpus:
			if i.edges == path.edges:
				ok = False
				break
		if ok:
			corpus.append(path)

	corpus = sorted(corpus, key=lambda path: (path.cost[1], path.cost[0]))
	for i in range(corpus_size*2):
		corpus[i].rank[1] = i+1

	corpus = sorted(corpus, key=lambda path: path.cost)
	for i in range(corpus_size*2):
		corpus[i].rank[0] = i+1

	for path in corpus:
		path.update()

	for generation in range(50):
		childs = []
		for _ in range(corpus_size*2):
			mom, dad = np.random.choice(corpus, size=2, replace=False, p=p)
			if mom.skill_factor == dad.skill_factor or np.random.rand() < rmp:
				sf = mom.skill_factor if np.random.rand() < 0.5 else dad.skill_factor
				child = h[sf].crossover(mom, dad)
				child = [h[sf].mutation(i) for i in child]
				childs += child

			else:
				h[mom.skill_factor].mutation(mom, mutation_rate=1).print()
				childs.append(h[mom.skill_factor].mutation(mom, mutation_rate=1))
				childs.append(h[dad.skill_factor].mutation(dad, mutation_rate=1))

		corpus += childs

		corpus = sorted(corpus, key=lambda path: (path.cost[1], path.cost[0]))
		for i in range(corpus_size*2):
			corpus[i].rank[1] = i+1

		corpus = sorted(corpus, key=lambda path: path.cost)
		for i in range(corpus_size*2):
			corpus[i].rank[0] = i+1

		for path in corpus:
			path.update()
			# print(path.rank)
			# path.print()

		corpus = corpus[:corpus_size*2]
		best = corpus[0]
		print('generation: {}'.format(generation))
		best.print('best solution')

	return 


print('loading data ...')
with open('IDPC-DU/set1/idpc_25x25x15625.idpc','r') as f:
	lines = f.read().splitlines()
	# number of nodes and domains
	N, D = lines[0].split()
	# source node and terminal node
	s, t = lines[1].split()
	# edge (u,v) has weight w and belong to domain d.
	N, D, s, t = int(N), int(D), int(s), int(t)
	graph = Graph_IDPC(N, D, s, t)
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

graph.print()
h1 = DistanceFirst(graph)
h2 = ConstaintFirst(graph)

# GA(h1)
# MFEA(graph)