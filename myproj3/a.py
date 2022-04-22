# Python3 program to find minimum edge
# between given two vertex of Graph
import queue

# function for finding minimum
# no. of edge using BFS
def minEdgeBFS(edges, u, v, n):
	print("edges:", edges)
	# visited[n] for keeping track
	# of visited node in BFS
	visited = [0] * n

	# Initialize distances as 0
	distance = [0] * n

	# queue to do BFS.
	Q = queue.Queue()
	distance[u] = 0

	Q.put(u)
	visited[u] = 1
	while (not Q.empty()):
		x = Q.get()
		
		for i in range(len(edges[x])):
			if (visited[edges[x][i]]):
				continue

			# update distance for i
			distance[edges[x][i]] = distance[x] + 1
			Q.put(edges[x][i])
			visited[edges[x][i]] = 1
			print("dist: ", distance)
			print("visited: ", visited)
			print("-------")
	return distance[v]

# function for addition of edge
def addEdge(edges, u, v):
	edges[u].append(v)
	edges[v].append(u)

# Driver Code
if __name__ == '__main__':

	# To store adjacency list of graph
	n = 9
	edges = [[] for i in range(n)]
	addEdge(edges, 0, 1)
	addEdge(edges, 0, 7)
	addEdge(edges, 1, 7)
	addEdge(edges, 1, 2)
	addEdge(edges, 2, 3)
	addEdge(edges, 2, 5)
	addEdge(edges, 2, 8)
	addEdge(edges, 3, 4)
	addEdge(edges, 3, 5)
	addEdge(edges, 4, 5)
	addEdge(edges, 5, 6)
	addEdge(edges, 6, 7)
	addEdge(edges, 7, 8)
	u = 3
	v = 7
	print(minEdgeBFS(edges, u, v, n))

# This code is contributed by PranchalK
