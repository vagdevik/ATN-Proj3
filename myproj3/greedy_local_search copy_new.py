'''
1. take input no of nodes
2. generate random coordinates
3. compute geometric distances between each pair of coordinate points
4. constructNetwork with each node of atleast 3 neighbours and the whole diameter of atmost 4
5. compute cost
6. print the cost

'''

import random
import math
import queue
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



#generate coordinates
def generateCoordinates(N):
	coordinates = []
	for _ in range(N):
		x = random.randint(1,50)
		y = random.randint(1,50)
		coordinates.append([x,y])
	return coordinates

# compute euclidean distance between 2 coordinate points
def getEuclidean(a,b):
	# l = pow((a[0]-b[0]),2)
	# r = pow((a[1]-b[1]),2)
	y = np.linalg.norm(np.array(a)-np.array(b))
	return y

def computeDistances(coordinates):
	dists_matrix = [[0 for _ in range(N)] for _ in range(N)]
	for i in range(len(coordinates)):
		for j in range(len(coordinates)):
			dists_matrix[i][j] = getEuclidean(coordinates[i],coordinates[j])
			dists_matrix[j][i] = dists_matrix[i][j]
	return dists_matrix

def getCurrLinksCount(vertex, adjaceny_matrix):
	curr_links = adjaceny_matrix[vertex].count(1)
	return curr_links

def reconstructNetwork():
	pass

def dfs(vertice, adjacency_list, visited):        
    visited.add(vertice)
    
    for adj_vertice in adjacency_list[vertice]:
        if adj_vertice not in visited:
            dfs(adj_vertice, adjacency_list, visited)   


def isConnected(adjaceny_matrix):
    edges = []
    for i in range(N):
        for j in range(N):
            if adjaceny_matrix[i][j]==1:
                s = [i,j]
                s.sort()
                if s not in edges:	
                    edges.append(s)
    count = 0
    
    visited = set()

    adjacency_list = []
    
    for i in range(N):
        adjacency_list.append([])
        
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    # print("edges:",edges)   
    # adjacency_list = [[1,7],[0,2,7],[1,3,5,8],[2,4,5],[3,5],[2,3,4,6],[5,7,8],[0,1,6,8],[2,6,7]]
    # print("adjacency_list: ",adjacency_list)
    for vertice in range(N):            
        if vertice not in visited:
            dfs(vertice, adjacency_list, visited)
            count += 1
    
    if count == 1:
    	return True
    return False

# def connectTheGraph(dists_matrix):
# 	pass

def constructNetwork(dists_matrix, k):
	curr_links = 0
	for i in range(N):
		curr_links = getCurrLinksCount(i, adjaceny_matrix)
		# print("i-> curr_links:", i,curr_links)
		# print(adjaceny_matrix[i])
		row_sorted_distances = sorted(dists_matrix[i])
		col = 0
		count = 0
		# print("sortedrow: ", row_sorted_distances[:4])
		for col in range(0, N):
			# print(i, col,"$$",row_sorted_distances.index(dists_matrix[i][col]))
			if curr_links < k and row_sorted_distances.index(dists_matrix[i][col]) < k+1:
				if i!=col and adjaceny_matrix[i][col]!=1:
					count+=1
					v = dists_matrix[i][col]
					adjaceny_matrix[i][col], adjaceny_matrix[col][i] = 1, 1
					curr_links += 1 
			col += 1
	v = True
	nei = True
	c = True
	for u in range(N):
		c = isConnected(adjaceny_matrix)
		# if c>1:
		# 	connectTheGraph(dists_matrix)
		k = validateNetwork(u, N)
		x = len([element for element in k if element > 4])
		# print("--count: ",x)
		if x>0:
			v = False
		for i in range(N):
			if adjaceny_matrix[i].count(1)<3:
				nei = False
		if not nei or not v or not c:
			# print("ffffffffffffFalse")
			return False
	# print("Trueeeeeee")
	return True
	

		# if k==False:
		# 	print("False")
			# reconstructNetwork(N, )


def printSolution(dist):
	pass
	# print("Vertex \t Distance from Source")
	# for node in range(N):
		# print(node, "\t\t", dist[node])

def minDistance(dist, sptSet, n):

	# Initialize minimum distance for next node
	min = 1e7
	min_index = ''

	# Search not nearest vertex not in the
	# shortest path tree
	for v in range(n):
		if dist[v] < min and sptSet[v] == False:
			min = dist[v]
			min_index = v

	return min_index

def dijkstra(src, n):
	# print("src: ",src)

	dist = [1e7] * n
	dist[src] = 0
	sptSet = [False] * n

	for cout in range(n):

		# Pick the minimum distance vertex from
		# the set of vertices not yet processed.
		# u is always equal to src in first iteration
		u = minDistance(dist, sptSet, n)

		# Put the minimum distance vertex in the
		# shortest path tree
		# print(cout, "type(u): ", type(u),u)
		sptSet[u] = True
		# print("zzzzzz: ",n)

		# Update dist value of the adjacent vertices
		# of the picked vertex only if the current
		# distance is greater than new distance and
		# the vertex in not in the shortest path tree
		for v in range(n):
			if (adjaceny_matrix[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + adjaceny_matrix[u][v]):
				dist[v] = dist[u] + adjaceny_matrix[u][v]
	printSolution(dist)
	return dist


# def isValidDiameter(u, N, adjaceny_matrix_row):	
# 	visited = [0 for _ in range(N)]
# 	distance = [0 for _ in range(N)]
# 	Q = queue.Queue()
# 	distance[u] = 0
# 	Q.put(u)
# 	visited[u] = 1
# 	while (not Q.empty()):
# 		x = Q.get() 
# 		for i in range(N):
# 			if visited[i] == 1:
# 				continue
# 			distance[i] = distance[x] + 1
# 			Q.put(i)
# 			visited[i] = 1
# 	if max(distance)<=4:
# 		return True
# 	return False

def isValidDiameter(u, N, adjaceny_matrix_row):	
	pass



def validateNetwork(u, N):	
	k = dijkstra(u, N)
	return k
		# print("k:",k)
		# if k==False:
		# 	reconstructNetwork()

def networkCost(dists_matrix):
	cost = 0
	n = len(dists_matrix)
	for row in range(n):
		for col in range(n):
			if adjaceny_matrix[row][col]==1:
				cost = cost+dists_matrix[row][col]
	return cost

def showResult(cost, N):
	print("Cost with ", N, "nodes: ", cost)

def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    mylabels = {}
    for node in range(N):
    	mylabels[node] = node
    nx.draw(gr, labels=mylabels, with_labels=True)
    plt.show()


# take input no of nodes
print("Enter the number of nodes:")
N = int(input())

# generate N random 2D coordinates
coordinates = generateCoordinates(N)

# compute geometric distances between each pair of coordinate points
dists_matrix = computeDistances(coordinates)
adjaceny_matrix = [[0 for _ in range(N)] for _ in range(N)]

# generate network and validate with conditions
# k = 3
# for i in range(N-2):
# 	a = constructNetwork(dists_matrix, k)
# 	print("***** came back: ")
# 	# print("a: ", a)
# 	if a:
# 		break
# 	k+=1

k = 3
a = constructNetwork(dists_matrix, k)
k = k + 1
while a==False:
	a = constructNetwork(dists_matrix, k)
	print("***** came back: ")
	k+=1


# calculate cost
cost = networkCost(dists_matrix)
showResult(cost, N)

print("Finally: ")
max_n = -1
for i in range(N):
	max_n = max(max_n, max(dijkstra(i, N)))
print("max_n ",max_n )

show_graph_with_labels(np.array(adjaceny_matrix))

