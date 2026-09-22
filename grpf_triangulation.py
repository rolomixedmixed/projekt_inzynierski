import numpy as np
from scipy.spatial import Delaunay

def triangulate(nodes):

    triangulation = Delaunay(nodes)
    triangles = triangulation.simplices
    edges=set()

    for triangle in triangles:
        a, b, c = triangle

        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((c, b))))
        edges.add(tuple(sorted((a, c))))

    edges=np.array(list(edges), dtype=np.int64)

    return triangles, edges
    

