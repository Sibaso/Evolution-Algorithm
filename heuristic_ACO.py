import numpy as np 
import copy
import time
import _thread as thread
import multiprocessing as mp

class Egde_IDPC(object):
	def __init__(self,u, v, w, d):
		super(Egde_IDPC, self).__init__()
		self.u = u
		self.v = v
		self.w = w
		self.d = d
		self.pheromone = 1

	def pheromone_drop(self):
		self.pheromone = (1-rho)*self.pheromone

	def print(self):
		print('({}->{}), color: {}, cost: {}'.format(self.u, self.v, self.d, self.w))

class Node_IDPC(object):
	"""docstring for Node_IDPC"""
	def __init__(self, idx):
		super(Node_IDPC, self).__init__()
		self.idx = idx
		self.pheromone = 1


class Graph_IDPC():
	def __init__(self, N, D, s, t):
		self.N = N
		self.D = D
		self.s = s
		self.t = t
		self.edges = [[[] for v in range(N+1)] for u in range(N+1)]
		self.nodes = []

	def print(self):
		for u in range(N+1):
			for v in range(N+1):
				print('({}->{})'.format(u, v))
				for edge in self.edges[u][v]:
					print('color: {}, cost: {}'.format(edge.d, edge.w))

alpha = 1
beta = 1
rho = 0.05
Q = 1
b = 5

class Ant1(object):
	"""docstring for Ant"""
	def __init__(self, graph):
		super(Ant1, self).__init__()
		self.graph = graph

	def prob(self, edges, nodes):
		p = []
		for edge in edges:
			v = edge.v
			if v == self.graph.t:
				p.append(edge.pheromone**alpha * (max_w+edge.w)**(-beta))
				continue
			cand = []
			for node in nodes:
				cand += self.graph.edges[v][node]
			if len(cand) != 0:
				min_out = min([e.w for e in cand])
			else:
				min_out = max_w
			p.append(edge.pheromone**alpha * (min_out+edge.w)**(-beta))

		p = np.array(p)/sum(p)
		# print(p)
		return p

	def move(self):
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

			chosen_edge = np.random.choice(cur_edges, p=self.prob(cur_edges, nodes))
			path.append(chosen_edge)

			cur_node = chosen_edge.v
			nodes.remove(cur_node)

		return Path(path)

class Ant2(object):
	"""docstring for Ant"""
	def __init__(self, graph):
		super(Ant2, self).__init__()
		self.graph = graph

	def prob(self, edges, domains_traveled):
		p = []
		for edge in edges:
			if edge.d in domains_traveled:
				if edge.d == domains_traveled[-1]:
					p.append(edge.pheromone**alpha * (1)**beta)
				else:
					p.append(edge.pheromone**alpha * (0.1)**beta)
			else:
				p.append(edge.pheromone**alpha * (0.5)**beta)
		return np.array(p)/sum(p)

	def move(self):
		graph = self.graph
		nodes = list(range(1,graph.N+1))
		s = graph.s
		t = graph.t
		path = []
		nodes.remove(s)
		cur_node = s
		domains = [-1]

		while cur_node != t:
			cur_edges = []
			for node in nodes:
				cur_edges += graph.edges[cur_node][node]

			chosen_edge = np.random.choice(cur_edges, p=self.prob(cur_edges, domains))
			path.append(chosen_edge)
			domains.append(chosen_edge.d)
			cur_node = chosen_edge.v
			nodes.remove(cur_node)

		return Path(path)


class Ant3(object):
	"""docstring for Ant"""
	def __init__(self, graph):
		super(Ant3, self).__init__()
		self.graph = graph

	def prob(self, edges):
		p = [edge.pheromone/edge.w for edge in edges]
		return np.array(p)/sum(p)

	def move(self):
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

			chosen_edge = np.random.choice(cur_edges, p=self.prob(cur_edges))
			path.append(chosen_edge)
			cur_node = chosen_edge.v
			nodes.remove(cur_node)

		return Path(path)

def score(edges):
	domains_traveled = [-1]
	cs = 0
	length = 0
	for edge in edges:
		w, d = edge.w, edge.d
		length += w
		if d in domains_traveled and d != domains_traveled[-1]:
			cs += 1

		domains_traveled.append(d)

	return (length, cs)

class Path(object):
	"""docstring for Path"""
	def __init__(self, edges, nodes=None):
		super(Path, self).__init__()
		self.edges = edges
		self.nodes = nodes
		self.cost = score(edges)
		self.L = self.cost[0]#+self.cost[1]

	def update_pheromone(self):
		for edge in self.edges:
			edge.pheromone += Q#/self.L
			# if edge in best_path.edges:
			# 	edge.pheromone += b*Q/best_path.L

		if self.nodes!=None:
			for node in self.nodes:
				node_pheromone[node] += Q#/self.L
		
	def print(self):
		for edge in self.edges:
			edge.print()

class Ant:
	def __init__(self, graph):
		self.graph = graph

	def node_prob(self, nodes):
		p = []
		for node in nodes:
			p.append(node_pheromone[node]**alpha * priority[node]**(-beta))
		return np.array(p)/sum(p)

	def edge_prob(self, edges):
		p = []
		for edge in edges:
			p.append(edge.pheromone**alpha * edge.w**(-beta))
		return np.array(p)/sum(p)

	def move(self):
		graph = self.graph
		nodes = []
		s = graph.s
		t = graph.t
		path = []
		nodes.append(s)
		all_nodes = np.arange(1,N+1)

		while nodes[-1] != t:
			chosen_node = np.random.choice(all_nodes, p=self.node_prob(all_nodes))
			while chosen_node in nodes:
				chosen_node = np.random.choice(all_nodes, p=self.node_prob(all_nodes))

			edges = graph.edges[nodes[-1]][chosen_node]
			chosen_edge = np.random.choice(edges , p=self.edge_prob(edges))
			path.append(chosen_edge)
			nodes.append(chosen_node)

		return Path(path, nodes)

def ACO(graph, num_ants=100, num_iters=20):
	ant1 = Ant1(graph)
	ant2 = Ant2(graph)
	ant3 = Ant3(graph)

	best_path = ant3.move()
	c0 = time.time()
	for i in range(num_iters):
		try:
			print('iter:',i)
			paths = []
			for _ in range(num_ants):
				path = ant3.move()
				if best_path.cost[0] > path.cost[0]:
					best_path = path
				paths.append(path)

			for u in range(1, N+1):
				for v in range(1, N+1):
					for edge in graph.edges[u][v]:	
						edge.pheromone = (1-rho)*edge.pheromone
			for path in paths:
				path.update_pheromone()

			print(best_path.cost)
			best_path.print()
			c1 = time.time()
			print('Rs:', c1-c0)
		except:
			break
	

def ACO_parallel(graph, num_ants=100, num_iters=20):
	
	ant3 = Ant3(graph)
	best_path = ant3.move()
	best_path.print()
	pool = mp.Pool(mp.cpu_count())
	for i in range(num_iters):
		print('iter:',i)
		paths = pool.starmap_async(ant3.move, [() for _ in range(num_ants)]).get()
		print(len(paths))
		for path in paths:
			# print('path')
			# path.print()
			# print(path.cost)
			if best_path.cost[0] > path.cost[0]:
				best_path = path

		for u in range(1, N+1):
			for v in range(1, N+1):
				for edge in graph.edges[u][v]:	
					edge.pheromone = (1-rho)*edge.pheromone

		# pool.starmap_async(edge.pheromone_drop, [[[() for edge in graph.edges[u][v]] for v in range(1,N+1)] for u in range(1,N+1)])

		# pool.starmap_async(path.update_pheromone, [() for path in paths])
		for path in paths:
			path.update_pheromone()

		print(best_path.cost)
		best_path.print()
		# break
	pool.close()
	for path in paths:
		for e in path.edges:
			print(e.pheromone)

def ACO2(graph, num_ants=100, num_iters=20):
	ant = Ant(graph)
	
	best_path = ant.move()
	for i in range(num_iters):
		# try:
			print('iter:',i)
			for _ in range(num_ants):
				path = ant.move()
				if best_path.L > path.L:
					best_path = path
				paths.append(path)

			for u in range(1, N+1):
				node_pheromone[u] = (1-rho)*node_pheromone[u]
				for v in range(1, N+1):
					for edge in graph.edges[u][v]:	
						edge.pheromone = (1-rho)*edge.pheromone
			for path in paths:
				path.update_pheromone(best_path)


			print(best_path.cost)
			best_path.print()
		# except:
		# 	break


if __name__ == '__main__':
	name = '80x40x175762'
	num_iters= 50
	num_ants = 1000

	with open('IDPC-DU/set2/idpc_{}.idpc'.format(name), 'r') as f:
		lines = f.read().splitlines()
		# number of nodes and domains
		N, D = lines[0].split()
		# source node and terminal node
		s, t = lines[1].split()
		# edge (u,v) has weight w and belong to domain d.
		N, D, s, t = int(N), int(D), int(s), int(t)
		graph = Graph_IDPC(N, D, s, t)
		max_w = 0
		for line in lines[2:]:
			u, v, w, d = line.split()
			u, v, w, d = int(u), int(v), int(w), int(d)
			if w > max_w:
				max_w = w
			found = False
			for edge in graph.edges[u][v]:
				if edge.d == d:
					found = True
					if edge.w > w:
						edge.w = w
					break
			if not found:
				graph.edges[u][v].append(Egde_IDPC(u, v, w, d))

	print('number of nodes:', N)
	print('number of domains:', D)
	print('start node:', graph.s)
	print('terminate node:', graph.t)
	print('max weight:', max_w)

	priority = [1]
	for u in range(1, N+1):
		edges_out = []
		for v in range(1, N+1):
			edges_out += graph.edges[u][v]
		min_out = min([edge.w for edge in edges_out])

		edges_in = []
		for v in range(1, N+1):
			edges_in += graph.edges[v][u]
		min_in = min([edge.w for edge in edges_in])

		priority.append(min_out)

	# priority = []
	node_pheromone = [1 for _ in range(N+1)]
	print(priority)
	c0 = time.time()
	ACO(graph, num_ants=num_ants, num_iters=num_iters)
	c1 = time.time()
	print('Rs:', c1-c0)


