# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph
# class Graph():

# def __init__(self, vertices):
# 	self.V = vertices
# 	self.graph = [[0 for column in range(vertices)]
# 				for row in range(vertices)]

def printSolution(dist):
	print("Vertex \t Distance from Source")
	for node in range(n):
		print(node, "\t\t", dist[node])

# A utility function to find the vertex with
# minimum distance value, from the set of vertices
# not yet included in shortest path tree
def minDistance(dist, sptSet):

	# Initialize minimum distance for next node
	min = 1e7

	# Search not nearest vertex not in the
	# shortest path tree
	for v in range(n):
		if dist[v] < min and sptSet[v] == False:
			min = dist[v]
			min_index = v

	return min_index

# Function that implements Dijkstra's single source
# shortest path algorithm for a graph represented
# using adjacency matrix representation
def dijkstra(src):

	dist = [1e7] * n
	dist[src] = 0
	sptSet = [False] * n

	for cout in range(n):

		# Pick the minimum distance vertex from
		# the set of vertices not yet processed.
		# u is always equal to src in first iteration
		u = minDistance(dist, sptSet)

		# Put the minimum distance vertex in the
		# shortest path tree
		sptSet[u] = True

		# Update dist value of the adjacent vertices
		# of the picked vertex only if the current
		# distance is greater than new distance and
		# the vertex in not in the shortest path tree
		for v in range(n):
			if (graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + graph[u][v]):
				dist[v] = dist[u] + graph[u][v]

	printSolution(dist)

# Driver program
n = 9
# g = Graph(9)
graph = [[0, 1, 0, 0, 0, 0, 0, 1, 0],
		[1, 0, 1, 0, 0, 0, 0, 1, 0],
		[0, 1, 0, 1, 0, 1, 0, 0, 1],
		[0, 0, 1, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 0, 1, 0, 0, 0],
		[0, 0, 1, 1, 1, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 1, 1],
		[1, 1, 0, 0, 0, 0, 1, 0,1],
		[0, 0, 1, 0, 0, 0, 1, 1, 0]]

dijkstra(0)

# This code is contributed by Divyanshu Mehta
