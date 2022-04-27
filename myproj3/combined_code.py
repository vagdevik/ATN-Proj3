'''
1. take input no of nodes
2. generate random coordinates
3. compute geometric distances between each pair of coordinate points
4. constructNetwork with each node of atleast 3 neighbours and the whole diameter of atmost 4
5. compute cost
6. print the cost

'''
import time
startTime = time.time()

# import modules
import random
import math
import queue
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# coordinate points to vertex name mappings
coordinates_names_map = {}

# edges of graph
edges = {}

# edges to remove
edges_to_remove = []

# generate coordinates and also simultaneously store the coordinates_names_map mappings
def generateCoordinates(N):
	coordinates = []
	for c in range(N):
		x = random.randint(1,50)
		y = random.randint(1,50)
		coordinate = [x,y]
		coordinates.append(coordinate)
		coordinates_names_map[c] = coordinate
	return coordinates

# compute euclidean distance between 2 coordinate points
def getEuclidean(a,b):
	y = np.linalg.norm(np.array(a)-np.array(b))
	return y

# compute distances matrix for Algorithm1
def computeDistances_1(coordinates):
	dists_matrix = [[0 for _ in range(N)] for _ in range(N)]
	for i in range(len(coordinates)):
		for j in range(len(coordinates)):
			dists_matrix[i][j] = getEuclidean(coordinates[i],coordinates[j])
			dists_matrix[j][i] = dists_matrix[i][j]
	return dists_matrix

# compute distances matrix for Algorithm2
def computeDistances_2(coordinates):
	dists_matrix = [[0 for _ in range(N)] for _ in range(N)]
	for i in range(len(coordinates)):
		for j in range(len(coordinates)):
			weight = getEuclidean(coordinates[i],coordinates[j])
			dists_matrix[i][j] = weight
			dists_matrix[j][i] = dists_matrix[i][j]
			coord = tuple(sorted([i,j]))
			if coord not in edges:
				if coord[0]!=coord[1]:
					edges[coord] = weight
	return dists_matrix

# get number of current links to a vertex
def getCurrLinksCount(vertex, adjaceny_matrix):
	curr_links = adjaceny_matrix[vertex].count(1)
	return curr_links

# depth first search
def dfs(vertice, adjacency_list, visited):        
    visited.add(vertice)
    for adj_vertice in adjacency_list[vertice]:
        if adj_vertice not in visited:
            dfs(adj_vertice, adjacency_list, visited)   

# returrn index of unvisited vertex of minimum distance 
def minDistance(dist, spt_set, n):
	min = 1e7
	min_index = ''
	for v in range(n):
		if dist[v] <= min and spt_set[v] == False:
			min = dist[v]
			min_index = v
	return min_index


# function to check if graph is connected
def isConnected():
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

    for vertice in range(N):            
        if vertice not in visited:
            dfs(vertice, adjacency_list, visited)
            count += 1
   
    if count == 1:
    	return True
    return False

# function to check if vertices of the graph of Algorithm 1 has atleast 3 neighbours
def atleast3neighbours_1():
	nei = True
	for i in range(N):
			if adjaceny_matrix[i].count(1)<3:
				nei = False
	return nei

# function to check if the given vertices of the graph of Algorithm 2 has atleast 3 neighbours
def atleast3neighbours_2(u, v):
	nei1 = True
	nei2 = True
	if adjaceny_matrix[u].count(1)<3:
		nei1 = False
	if adjaceny_matrix[v].count(1)<3:
		nei1 = False
	return nei1 and nei2

# function to check the diameter of graph of Algorithm 1
def atmost4Diameter_1(u, N):	
	k = dijkstra(u)
	return k

# function to check the diameter of graph of Algorithm 2
def atmost4Diameter_2():
	r = True
	max_d = -1
	for i in range(N):
		k = dijkstra(i)
		max_d = max(max_d, max(k))
		if max(k)>4:
			return False
	return r

# removes edges from complete graph of Algorithm 2 using greedy global search
def reconstructNetwork():
	global edges_to_remove
	temp_edges_to_remove = []
	global rec_cons_count
	for edge in sorted_edges:
		c, k, nei = False, False, False
		x = edge[0]
		y = edge[1]
		if adjaceny_matrix[x][y] and adjaceny_matrix[y][x]:
			adjaceny_matrix[x][y], adjaceny_matrix[y][x] = 0, 0
			c = isConnected()
			k = atmost4Diameter_2()
			nei = atleast3neighbours_2(x, y)
			if not c or not k or not nei:
				adjaceny_matrix[x][y], adjaceny_matrix[y][x] = 1, 1
			else:
				rec_cons_count+=1
				if rec_cons_count%10==0:
					cost = networkCost()
					print("Cost for", rec_cons_count, "iteration is:", cost)
				temp_edges_to_remove.append(edge)

	for k in temp_edges_to_remove:
		del sorted_edges[tuple(k)]
	edges_to_remove+=temp_edges_to_remove

	# check if the graph is valid after reconstruction
	max_n = -1
	d = True
	for i in range(N):
		max_n = max(max_n, max(dijkstra(i)))
	if max_n>4:
		d = False
	bbb = True
	for i in range(N):
				if adjaceny_matrix[i].count(1)<3:
					bbb = False
	if bbb and d and isConnected():
		return True
	return False

# check if the graph is valid
def isNetworkValid():
	v = True
	nei = True
	c = True
	for u in range(N):
		c = isConnected()

		k = atmost4Diameter_1(u, N)
		x = len([element for element in k if element > 4])

		if x>0:
			v = False
		nei = atleast3neighbours_1()
		if not nei or not v or not c:
			return False
	return True
	
# construct the graph for Algorithm 1 using greeedy local search
def constructNetwork(dists_matrix, k):
	curr_links = 0
	for i in range(N):
		curr_links = getCurrLinksCount(i, adjaceny_matrix)
		row_sorted_distances = sorted(dists_matrix[i])
		col = 0
		count = 0
		for col in range(0, N):
			if curr_links < k and row_sorted_distances.index(dists_matrix[i][col]) < k+1:
				if i!=col and adjaceny_matrix[i][col]!=1:
					count+=1
					v = dists_matrix[i][col]
					adjaceny_matrix[i][col], adjaceny_matrix[col][i] = 1, 1

					curr_links += 1 
			col += 1
	nv = isNetworkValid()
	if not nv:
		return False
	return True

# dijkstra function
def dijkstra(src):
	dist = [1e7] * N
	dist[src] = 0
	spt_set = [False] * N

	for cout in range(N):
		u = minDistance(dist, spt_set, N)
		spt_set[int(u)] = True

		for v in range(N):
			if (adjaceny_matrix[u][v] > 0 and spt_set[v] == False and dist[v] > dist[u] + adjaceny_matrix[u][v]):
				dist[v] = dist[u] + adjaceny_matrix[u][v]
	return dist

# prune weights to optimise the graph constructed using Algorithm 1
def pruneNetwork():
	global count_its2
	for i in range(N):
		row_sorted_dist = sorted(dists_matrix[i], reverse=True)
		for j in range(N):
			curr_dist = row_sorted_dist[j]
			ind = dists_matrix[i].index(curr_dist)
			if adjaceny_matrix[i][ind]==1 and adjaceny_matrix[ind][i]==1:
				adjaceny_matrix[i][ind], adjaceny_matrix[ind][i] = 0, 0
				nv = isNetworkValid()
				if not nv:
					adjaceny_matrix[i][ind], adjaceny_matrix[ind][i] = 1, 1
				else:
					count_its2+=1
					if count_its2%10==0:
						cost = networkCost()
						print("@@ **** Cost for", count_its2, "iteration is:", cost)
	nv = isNetworkValid()
	if not nv:
		return False
	return True

# compute the network cost
def networkCost():
	cost = 0
	n = len(dists_matrix)
	for row in range(n):
		for col in range(n):
			if adjaceny_matrix[row][col]==1:
				cost = cost+dists_matrix[row][col]
	return cost

# generate the final graph using the networkx library of Python
def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    t_edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(t_edges)
    mylabels = {}
    for node in range(N):
    	mylabels[node] = tuple(coordinates_names_map[node])
    nx.draw(gr, labels=mylabels, with_labels=True)
    plt.show()

############################ Main Function ############################


N_vals = [17,24,32,47,56]

# take input no of nodes
for ntimes in N_vals:
	N = ntimes
	print("The number of nodes:")
	print(N)
	# print("Enter the number of nodes:")
	# N = int(input())

# generate N random 2D coordinates
coordinates = generateCoordinates(N)


############################ Algo 1 ############################

print("########################### Algo 1 ##########################")

# compute geometric distances between each pair of coordinate points
dists_matrix = computeDistances_1(coordinates)
adjaceny_matrix = [[0 for _ in range(N)] for _ in range(N)]


print("Construction Stage:")
k = 3
a = constructNetwork(dists_matrix, k)

k = k + 1
count_its = 1
cost = networkCost()
print("Cost for", count_its, "iteration is:", cost)

while a==False:
	count_its+=1
	a = constructNetwork(dists_matrix, k)
	k+=1
	cost = networkCost()
	print("Cost for", count_its, "iteration is:", cost)

count_bef=0
for i in range(N):
	count_bef = count_bef + adjaceny_matrix[i].count(1)

count_its2 = 0
print("Pruning Stage:")
cost = networkCost()
a = pruneNetwork()
while a==False:
	count_its2 += 1
	a = pruneNetwork()
	if count_its2%10==0:
		cost = networkCost()
		print(" Cost for", count_its2, "iteration is:", cost)


algo1_costs = []
print("Finally: ")
cost = networkCost()
algo1_costs.append(cost)

print("$$$$$$$ Cost for", N, "nodes with iterations", count_its2,"  is:", cost)
bbb = True
count=0
for i in range(N):
	count = count + adjaceny_matrix[i].count(1)
	if adjaceny_matrix[i].count(1)<3:
		bbb = False
# print("After Prune No of edges:", count/2)

max_n = -1
for i in range(N):
	max_n = max(max_n, max(dijkstra(i)))
print("Graph Diameter: ",max_n )
print("isConnected: ",isConnected())

print("atleast 3 neighbours: ", bbb)

# executionTime = (time.time() - startTime)
# print('Execution time in seconds: ' + str(executionTime))
# show_graph_with_labels(np.array(adjaceny_matrix))

############################################ Algo 2 ############################################

print("########################################## Algo 2 #########################################")

startTime = time.time()

dists_matrix = computeDistances_2(coordinates)
adjaceny_matrix = [[1 for _ in range(N)] for _ in range(N)]
for i in range(N):
	adjaceny_matrix[i][i] = 0

sorted_edges = dict(sorted(edges.items(), key=lambda x: x[1], reverse=True))

rec_cons_count = 0
a = reconstructNetwork()
cost = networkCost()
while not a:
	a = reconstructNetwork()
	rec_cons_count+=1
	cost = networkCost()
	print("Cost for", rec_cons_count, "iteration is:", cost)

cost = networkCost()
algo2_costs = []
algo2_costs.append(cost)

print("Finally: ")
print("$$$$$$$ Cost for", N, "nodes with iterations", rec_cons_count, "  is:", cost)
max_n = -1
for i in range(N):
	max_n = max(max_n, max(dijkstra(i)))
print("Graph Diameter: ",max_n )
print("isConnected: ",isConnected())
bbb = True
for i in range(N):
			if adjaceny_matrix[i].count(1)<3:
				bbb = False
print("atleast 3 neighbours: ", bbb)



y1 = np.array(algo1_costs)
y2 = np.array(algo2_costs)

plt.plot(y1, label='Algo 1')
plt.plot(y2, label='Algo 2')

plt.plot(N_vals, y1, label="algo1", marker='o')
plt.plot(N_vals, y2, label="algo2", marker='o')

plt.title('Number of nodes vs Cost of the network')
plt.xlabel('number of nodes')
plt.ylabel('cost')
plt.legend()
plt.show()



# executionTime = (time.time() - startTime)
# print('Execution time in seconds: ' + str(executionTime))

# show_graph_with_labels(np.array(adjaceny_matrix))

