import numpy as np

def refine_mesh(nodes, edges_to_split):
    new_nodes=[]

    for edge in edges_to_split:
        a,b=edge
        center=(nodes[a]+nodes[b])/2
        new_nodes.append(center)

    if new_nodes:
        new_nodes = np.array(new_nodes)
        nodes=np.vstack([nodes, new_nodes])

    return nodes
    