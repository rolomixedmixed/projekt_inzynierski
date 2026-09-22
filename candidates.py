import numpy as np

def candidates(edges, quadrants):

    candidate_edges=[]

    for edge_index, (a,b) in edges:
        if abs(quadrants[a]-quadrants[b])==2:
            candidate_edges.append(edge_index)

    return np.array(candidate_edges, dtype=np.int64)