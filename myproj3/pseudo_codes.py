'''
########################### Algorithm 1 #####################################

# algorithm----------------------------------------------------------------

1. take the input no of nodes
2. generate random coordinates
3. compute geometric distances between each pair of coordinate points and store the matrix
4. initialize an empty adjacency matrix 
5. initialize k = 3
6. construct a Network:
         a. for each vertex add k edges such that their edge weight is in the top 3 minimum distances of that vertex
         b. check if the thus constructed graph meets 3 conditions: it is a connected graph, 
         has at-most diameter of 4 hops, and each vertex of at least 3 neighbors. Return True is yes, else return False
7. if step 6 returns False, increment k by 1 and repeat step 4.
8. prune the network:
		 a. for each vertex v, sort its neighbor edges in decrasing order.
			 a1. for each sorted edge of vertex v, remove it from adjacency and check if the graph is still valid. If no, add the edge back
		 b. repeat 8a for each edge of each vertex and continue till no more edges can be dropped
9. compute the cost of the network
10. print the cost




# pseudo code----------------------------------------------------------------
def algorithm_1:
	dists_matrix = computeDistances_1(coordinates)
	adjaceny_matrix = [[0 for _ in range(N)] for _ in range(N)]

	k = 3
	a = constructNetwork(dists_matrix, k)
	while a==False:
		a = constructNetwork(dists_matrix, k)
		k+=1

	b = pruneNetwork()
	while b==False:
		b = pruneNetwork()

	cost = networkCost()
	print("Cost for", N, "nodes"  is:", cost)
	
# construct the network
def constructNetwork(dists_matrix, k):
	for each vertex v:
		getCurrLinksCount(v)
		x = get the vertices in the ascending order of distances from v
		for each xi in x:
			if curr_links < k:
				if no edge between xi and v:
					add an edge between xi and v
					curr_links += 1 
	nv = isNetworkValid()
	if not nv:
		return False
	return True

# prune the network
def pruneNetwork():
	for each vertex v:
		row_sorted_dist = distances of other vertices sorted in descending order
		for each neighbour nei of v: 
				remove their edge from graph
				# check if the new graph after removing link is still valid
				nv = isNetworkValid()
				if not nv:
					add their edge back to the graph
	nv = isNetworkValid()
	if not nv:
		return False
	return True

# check if the graph is valid
# isConnected: returns true if the entire graph is connected
# atmost4Diameter: returns True if the maximum diameter of the graph is 4
# atleast3neighbours: returns True if each of the vertices in the graph has atleast 3 neightbours
def isNetworkValid():
	if isConnected() and atmost4Diameter() and atleast3neighbours():
		return True
	return False


########################### Algorithm 2 #####################################

# algorithm----------------------------------------------------------------


1. make a complete graph for the given coordinates (edges weight=eucledian distance)
2. sort the edges in decreasing order of their eucledian distance weights
3. for each edge in sorted edges:
        a. change the graph by removing that edge
        b. if the new graph is not connected or not atmost4diameter or not atleast3neighbours):
                 add the edge back
4. while the network is not valid, keep reconstructing by step 3
5. compute the cost of the network
6. print the cost

# pseudo code----------------------------------------------------------------

def algorithm_2:
	create an adjacency matrix adjaceny_matrix for N-complete graph for the coordinates
	sorted_edges = list of all the edges of the graph sorted based on their weights in decreasing order

	a = reconstructNetwork()
	while not a:
		a = reconstructNetwork()
		
	cost = networkCost()
	show_graph_with_labels(np.array(adjaceny_matrix))

def reconstructNetwork():
	temp_edges_to_remove = []
	global rec_cons_count
	for edge in sorted_edges:
		remove the `edge` from the adjacency_matrix of the graph and obtain the new graph g 
		if g not isNetworkValid():
			add the edge back to the graph
		else:
			temp_edges_to_remove.append(edge)

	for k in temp_edges_to_remove:
		del sorted_edges[k]
	

# check if the graph is valid
def isNetworkValid():
	if isConnected() and atmost4Diameter() and atleast3neighbours():
		return True
	return False

# network cost
def networkCost():
	cost = 0
	for row in range(n):
		for col in range(n):
			if adjaceny_matrix[row][col]==1:
				cost = cost+dists_matrix[row][col]
	return cost

'''



