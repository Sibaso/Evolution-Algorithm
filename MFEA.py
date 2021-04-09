import numpy as np

class Egde_IDPC(object):
	def __init__(self,u, v): # edgu (u, v), weight = w, domain = d
		super(Egde_IDPC, self).__init__()
		self.u = u
		self.v = v
		self.cost = []


	def print(self, idx):
		print('({}->{}): cost {}'.format(self.u, self.v, self.cost[idx%len(self.cost)]))

class Graph_IDPC():
	def __init__(self, N, D, s, t):
		self.N = N
		self.D = D
		self.s = s
		self.t = t
		self.edges = [[Egde_IDPC(u, v) for v in range(N+1)] for u in range(N+1)]

	def print(self):
		num_edges = 0
		for u in range(N+1):
			for v in range(N+1):
				self.edges[u][v].print()
				num_edges += 1
		print('number of edges:', num_edges)


class Problem(object):
	"""docstring for Problem"""
	def __init__(self, nodes, graph):
		super(Problem, self).__init__()
		self.nodes = nodes
		self.graph = graph
		self.best_path = None
		self.edges = []

		cur = graph.s
		for i, node in enumerate(nodes):
			self.edges.append(graph.edges[cur][node])
			cur = node
			if cur == graph.t:
				self.pos_t = i
				break
		# print(nodes, self.pos_t)
	def print(self):
		for i, edge in enumerate(self.edges):
			edge.print(self.best_path.code[i])

	

class Path(object):
	"""docstring for Path"""
	def __init__(self, code, skill_factor):
		super(Path, self).__init__()
		self.code = code
		self.cost = self.score(skill_factor) #dict([(problem, self.score(problem) )for problem in problems])
		self.rank = 1e8 #dict()
		self.skill_factor = skill_factor
		self.scalar_fitness = 1e-8

	def score(self, problem):
		if problem == None:
			return
		cur_domain = -1
		domains_traveled = set()
		cs = 0
		length = 0
		for i, edge in enumerate(problem.edges):
			idx = self.code[i]%len(edge.cost)
			w, d = edge.cost[idx]
			length += w
			if d in domains_traveled and d != cur_domain:
				cs += 1

			domains_traveled.add(d)
			cur_domain = d

		return (length, cs)


# for problem
def PMX_crossover(mom, dad, rate=0.9):
	if rate < np.random.rand():
		return []

	p1, p2 = np.random.choice(range(N-1), size=2, replace=False)
	p1, p2 = sorted([p1, p2])
	child1, child2 = [], []
	for i in range(N-1):
		if i >= p1 and i < p2:
			child1.append(dad.nodes[i])
			child2.append(mom.nodes[i])
			continue

		cur = mom.nodes[i]
		while cur in dad.nodes[p1: p2]:
			for j in range(p1, p2):
				if cur == dad.nodes[j]:
					cur = mom.nodes[j]
		child1.append(cur)

		cur = dad.nodes[i]
		while cur in mom.nodes[p1: p2]:
			for j in range(p1, p2):
				if cur == mom.nodes[j]:
					cur = dad.nodes[j]
		child2.append(cur)

	# print('mom:', mom.nodes, len(set(mom.nodes)))
	# print('dad:', dad.nodes, len(set(dad.nodes)))
	# print('child1:', child1, len(set(child1)))
	# print('child2:', child2, len(set(child2)))

	child1 = swap_mutation(Problem(child1, mom.graph))
	child2 = swap_mutation(Problem(child2, dad.graph))
	if child1.nodes[:child1.pos_t] == mom.nodes[:mom.pos_t] or child1.nodes[:child1.pos_t] == dad.nodes[:dad.pos_t]:
		if child2.nodes[:child2.pos_t] == mom.nodes[:mom.pos_t] or child2.nodes[:child2.pos_t] == dad.nodes[:dad.pos_t]:
			return []
		else:
			return [child2]
	else:
		if child2.nodes[:child2.pos_t] == mom.nodes[:mom.pos_t] or child2.nodes[:child2.pos_t] == dad.nodes[:dad.pos_t]:
			return [child1]
		else:
			return [child1, child2]

def swap_mutation(path, rate=0.1):
	if rate < np.random.rand():
		return path
	p1 = np.random.choice(range(path.pos_t+1), replace=False)
	p2 = np.random.choice(range(N-1), replace=False)
	while p1 == p2:
		p2 = np.random.choice(range(N-1), replace=False)

	temp = path.nodes[p1]
	path.nodes[p1] = path.nodes[p2]
	path.nodes[p2] = temp
	return Problem(path.nodes, graph)

# for path
def TPX_crossover(mom, dad, rate=0.9, rmp=0.5):
	if rate < np.random.rand():
		return []
	if not (mom.skill_factor == dad.skill_factor or rmp > np.random.rand()):
		return [change_mutation(mom, rate=1), change_mutation(dad, rate=1)]

	p1, p2 = np.random.choice(range(N-1), size=2, replace=False)
	p1, p2 = sorted([p1, p2])

	child1 = mom.code[:p1] + dad.code[p1:p2] + mom.code[p2:]
	child2 = dad.code[:p1] + mom.code[p1:p2] + dad.code[p2:]
	if mom.skill_factor == dad.skill_factor:
		child1 = Path(child1, mom.skill_factor)
		child2 = Path(child2, mom.skill_factor)
	else :
		child1 = Path(child1, mom.skill_factor if np.random.rand()<0.5 else dad.skill_factor)
		child2 = Path(child2, mom.skill_factor if np.random.rand()<0.5 else dad.skill_factor)
		
	return [change_mutation(child1), change_mutation(child2)]

def change_mutation(path, rate=0.1):
	if rate < np.random.rand():
		return path
	pos = np.random.randint(len(path.skill_factor.edges))
	e = path.code[pos]
	cand = list(range(MAX_OUT))
	cand.remove(e)
	path.code[pos] = np.random.choice(cand)
	return Path(path.code, path.skill_factor)


def GA(graph, corpus_size=5, num_gen=20):

	corpus = []
	s = graph.s
	t = graph.t
	factor = 0
	while len(corpus) < corpus_size:
		nodes = list(range(1, N+1))
		nodes.remove(s)
		np.random.shuffle(nodes)
		ok = True
		for i in corpus:
			if i.nodes == nodes:
				ok = False
				break
		if ok:
			corpus.append(Problem(nodes, graph))

	for generation in range(num_gen):
		print('__GA generation {}'.format(generation))
		corpus = MFEA(corpus)
		corpus = sorted(corpus, key=lambda x: x.best_path.cost)[:corpus_size]
		print('best:',corpus[0].best_path.cost, corpus[0].best_path.score(corpus[0]))
		print(corpus[0].nodes)
		corpus[0].print()
		# corpus[1].print()
		# print(corpus[1].nodes)
		p = np.array([1/(i+1) for i in range(corpus_size)])
		p = p/sum(p)
		childs = []
		while len(childs) < corpus_size:
			mom, dad = np.random.choice(corpus, size=2, replace=False, p=p)
			childs += PMX_crossover(mom, dad)



def MFEA(problems, corpus_size=20, num_gen=10):

	corpus = []
	while len(corpus) < corpus_size:
		path = list(np.random.choice(range(MAX_OUT), size=N-1, replace=True))
		ok = True
		for i in corpus:
			if i.code == path:
				ok = False
				break
		if ok:
			corpus.append(Path(path, None))

	for problem in problems:
		for path in corpus:
			path.cost = path.score(problem)
		corpus = sorted(corpus, key=lambda path: path.cost)
		for i, path in enumerate(corpus):
			if path.rank > i+1:
				path.rank = i+1
				path.skill_factor = problem
				path.scalar_fitness = 1/path.rank

	for path in corpus:
		path.cost = path.score(path.skill_factor)

	p = np.array([path.scalar_fitness for path in corpus])
	p = p/sum(p)

	for generation in range(num_gen):

		# print('____MFEA generation {}'. format(generation))
		for problem in problems:
			can = [path for path in corpus if path.skill_factor==problem]
			if len(can) == 0:
				continue
			can = sorted(can, key=lambda path: path.cost)
			print(can[0].cost, can[0].score(problem))
			problem.best_path = can[0]
			
			for i, path in enumerate(can):
				path.rank = i+1
				path.scalar_fitness = 1/path.rank

		corpus = sorted(corpus, key=lambda path: path.scalar_fitness, reverse=True)[:corpus_size]
		p = np.array([path.scalar_fitness for path in corpus])
		p = p/sum(p)

		childs = []

		while len(childs) < corpus_size:
			mom, dad = np.random.choice(corpus, size=2, replace=False, p=p)
			childs += TPX_crossover(mom, dad)

		corpus += childs


	return problems


def GA2():
	pass
		
if __name__ == '__main__':

	print('loading data ...')
	with open('IDPC-DU/set1/idpc_15x15x3375.idpc','r') as f:
		lines = f.read().splitlines()
		# number of nodes and domains
		N, D = lines[0].split()
		# source node and terminal node
		s, t = lines[1].split()
		# edge (u,v) has weight w and belong to domain d.
		N, D, s, t = int(N), int(D), int(s), int(t)
		graph = Graph_IDPC(N, D, s, t)
		for line in lines[2:]:
			found = False
			u, v, w, d = line.split()
			u, v, w, d = int(u), int(v), int(w), int(d)
			for i, (ew, ed) in enumerate(graph.edges[u][v].cost):
				if ed == d:
					found = True
					if w < ew:
						graph.edges[u][v].cost[i][0] = w
					break
			if not found:
				graph.edges[u][v].cost.append([w, d])

	MAX_OUT = max([max([len(graph.edges[u][v].cost) for v in range(N+1)]) for u in range(N+1)])
	print('number of nodes:', N)
	print('number of domains:', D)
	print('max out edges from one node:', MAX_OUT)
	print('start node:', graph.s)
	print('terminate node', graph.t)

	# graph.print()
	GA(graph)

