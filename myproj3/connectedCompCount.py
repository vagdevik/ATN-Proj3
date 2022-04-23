
# def countComponents():
#     count = 0
#     # graph = [[] for _ in range(n)]
#     seen = [False for _ in range(n)]   
#     # print(graph)
#     def dfs(node):
#         print("@ ",node)
#         for adj in adj_mat[node]:
#             print("adj: ",adj,adj_mat[node])
#             print("**")
#             if not seen[adj]:
#                 print("not seen")
#                 seen[adj] = True
#                 dfs(adj)

#     print("seen: ", seen)

#     for i in range(n):
#         print(i)
#         if not seen[i]:
#             count += 1
#             seen[i] = True
#             dfs(i)      
#     return count

# # n = 9
# n=5

# graph = [[0, 1, 0, 0, 0, 0, 0, 1, 0],
# 		[1, 0, 1, 0, 0, 0, 0, 1, 0],
# 		[0, 1, 0, 1, 0, 1, 0, 0, 1],
# 		[0, 0, 1, 0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 1, 0, 1, 0, 0, 0],
# 		[0, 0, 1, 1, 1, 0, 1, 0, 0],
# 		[0, 0, 0, 0, 0, 1, 0, 1, 1],
# 		[1, 1, 0, 0, 0, 0, 1, 0,1],
# 		[0, 0, 1, 0, 0, 0, 1, 1, 0]]

# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

# # def show_graph_with_labels(adjacency_matrix):
# #     rows, cols = np.where(adjacency_matrix == 1)
# #     edges = zip(rows.tolist(), cols.tolist())
# #     gr = nx.Graph()
# #     gr.add_edges_from(edges)
# #     mylabels = {}
# #     for node in range(9):
# #     	mylabels[node] = node
# #     nx.draw(gr, labels=mylabels, with_labels=True)
# #     plt.show()
# # show_graph_with_labels(np.array(graph))
# # adj_mat = []

# # for i in range(n):
# # 	for j in range(n):
# # 		if graph[i][j]==1:
# # 			s = [i,j]
# # 			s.sort()
# # 			if s not in adj_mat:	
# # 				adj_mat.append(s)
# # adj_mat = [[1], [0, 2], [1], [4], [3]]
# # adj_mat = [[0, 1], [0, 7], [1, 2], [1, 7], [2, 3], [2, 5], [2, 8], [3, 4], [3, 5], [4, 5], [5, 6], [6, 7], [6, 8], [7, 8]]
# adj_mat=[[1], [0, 2], [1], [4], [3]]
# print(adj_mat)
# # graph = [[0,1,0],[1,0,0],[0,0,0]]

# print(countComponents())

# # This code is contributed by rutvik_56



def countComponents(n, edges):
    
    count = 0
    
    visited = set()

    adjacency_list = []
    
    for i in range(n):
        adjacency_list.append([])
        
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    print("edges:",edges)   
    # adjacency_list = [[1,7],[0,2,7],[1,3,5,8],[2,4,5],[3,5],[2,3,4,6],[5,7,8],[0,1,6,8],[2,6,7]]
    print("adjacency_list: ",adjacency_list)
    for vertice in range(n):            
        if vertice not in visited:
            dfs(vertice, adjacency_list, visited)
            count += 1
    
    return count


def dfs(vertice, adjacency_list, visited):        
    visited.add(vertice)
    
    for adj_vertice in adjacency_list[vertice]:
        if adj_vertice not in visited:
            dfs(adj_vertice, adjacency_list, visited)   
   
# edges = [[0,1],[1,2],[3,4]]
n = 9

graph = [[0, 1, 0, 0, 0, 0, 0, 1, 0],
		[1, 0, 1, 0, 0, 0, 0, 1, 0],
		[0, 1, 0, 1, 0, 1, 0, 0, 1],
		[0, 0, 1, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 0, 1, 0, 0, 0],
		[0, 0, 1, 1, 1, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 1, 1],
		[1, 1, 0, 0, 0, 0, 1, 0,1],
		[0, 0, 1, 0, 0, 0, 1, 1, 0]]

edges = []
for i in range(n):
	for j in range(n):
		if graph[i][j]==1:
			s = [i,j]
			s.sort()
			if s not in edges:	
				edges.append(s)

print(edges)
print("**")
# edges = [[1,7],[0,2,7],[1,3,5,8],[2,4,5],[3,5],[2,3,4,6],[5,7,8],[0,1,6,8],[2,6,7]]
# print(edges)
print(countComponents(n, edges))