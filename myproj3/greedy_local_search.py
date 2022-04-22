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

def constructNetwork(dists_matrix):
	curr_links = 0
	for i in range(N):
		curr_links = getCurrLinksCount(i, adjaceny_matrix)
		# print("i-> curr_links:", i,curr_links)
		print(adjaceny_matrix[i])
		row_sorted_distances = sorted(dists_matrix[i])
		col = 0
		count = 0
		# print("sortedrow: ", row_sorted_distances[:4])
		for col in range(0, N):
			# print(i, col,"$$",row_sorted_distances.index(dists_matrix[i][col]))
			if curr_links < 3 and row_sorted_distances.index(dists_matrix[i][col]) < 4:
				if i!=col and adjaceny_matrix[i][col]!=1:
					count+=1
					v = dists_matrix[i][col]
					adjaceny_matrix[i][col], adjaceny_matrix[col][i] = 1, 1
					curr_links += 1 
			col += 1
	for u in range(N):
		k = validateNetwork(u, N, adjaceny_matrix[u])
		if k==False:
			print("False")
			reconstructNetwork(N)

		# print(adjaceny_matrix[i])
		# print('count', count)
		# print("curr_links:",adjaceny_matrix[i].count(1))
		# print("----")

def isValidDiameter(u, N, adjaceny_matrix_row):	
	visited = [0 for _ in range(N)]
	distance = [0 for _ in range(N)]
	Q = queue.Queue()
	distance[u] = 0
	Q.put(u)
	visited[u] = 1
	while (not Q.empty()):
		x = Q.get() 
		for i in range(N):
			if visited[i] == 1:
				continue
			distance[i] = distance[x] + 1
			Q.put(i)
			visited[i] = 1
	if max(distance)<=4:
		return True
	return False


def validateNetwork(u, N, u_edges):	
	k = isValidDiameter(u, N, u_edges)
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
constructNetwork(dists_matrix)


# calculate cost
cost = networkCost(dists_matrix)
showResult(cost, N)
show_graph_with_labels(np.array(adjaceny_matrix))

