import numpy as np
from collections import Counter

def regions(triangles, candidate_edges):

    candidate_triangles=[]
    candidate_set=set(tuple(e) for e in candidate_edges)

    for triangle in triangles:
        a,b,c=triangle
        edge1=tuple(sorted([a,b]))
        edge2=tuple(sorted([b,c]))
        edge3=tuple(sorted([c,a]))
        if edge1 in candidate_set or edge2 in candidate_set or edge3 in candidate_set:
            candidate_triangles.append((edge1, edge2, edge3))

    candidate_regions=[]

    for e1,e2,e3 in candidate_triangles:
        candidate_regions.extend([e1,e2,e3])

    edge_counts=Counter(candidate_regions)
    boundary_edges=[edge for edge, count in edge_counts.items() if count==1]

    edges_to_split=list(edge_counts)

    return boundary_edges, edges_to_split
