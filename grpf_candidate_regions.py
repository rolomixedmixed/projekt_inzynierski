import numpy as np
from collections import Counter

def regions(triangles, candidate_edges, build_boundary=True):

    candidate_triangles=[]
    candidate_set=set(tuple(sorted(e)) for e in candidate_edges)

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
    edges_to_split=list(edge_counts)

    boundary_regions=[]
    if build_boundary:
        remaining=candidate_triangles.copy()
        while remaining:
            region_edges=list(remaining.pop())
            region_set=set(region_edges)
            changed=True

            while changed:
                changed=False
                for triangle in remaining.copy():
                    if any(edge in region_set for edge in triangle):
                        region_edges.extend(triangle)
                        region_set.update(triangle)
                        remaining.remove(triangle)
                        changed=True

            counts=Counter(region_edges)
            boundary=[edge for edge, count in counts.items() if count ==1]
            boundary_regions.append(boundary)

    return boundary_regions, edges_to_split
