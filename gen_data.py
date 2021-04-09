import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1)

def gen_graph(N, D, num_edges):
	nodes = []
	choices = list(range(0, 20, int(20//(np.sqrt(N))) ))
	while len(nodes) < N:
		x = np.random.choice(choices)		
		y = np.random.choice(choices)
		if (x, y) not in nodes:
			nodes.append((x, y))

	print(nodes)
	edges = []
	while len(edges) < num_edges:
		try:
			i, j = np.random.choice(range(N), size=2, replace=False)
			u, v = nodes[i], nodes[j]
			u, v = sorted([u,v])
			print(u, v)
			c = np.random.randint(20)
			A = np.array([[u[0]**2, u[0]], [v[0]**2, v[0]]])
			B = np.array([u[1]-c, v[1]-c])
			a, b = np.linalg.inv(A).dot(B)
			x = np.linspace(u[0], v[0], 100)
			plt.plot(x, a*x**2 + b*x + c)

			break
		except:
			continue

plt.show()


gen_graph(10, 10, 30)