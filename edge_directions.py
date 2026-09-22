import numpy as np

def directions(nodes, edges):

    angles=[]

    for edge in edges:
        id_a, id_b=edge
        a=nodes[id_a]
        b=nodes[id_b]

        dx=b[0] - a[0]
        dy=b[1] - a[1]

        angle=np.arctan2(dy, dx)

        if angle<0:
            angle+=np.pi

        if np.abs(angle-np.pi)<1e-3:
            angle=0.0

        angles.append(angle)

    angles=np.array(angles, dtype=np.float64)
    round_angles=np.round(angles, decimals=3)
    unique_angles=np.unique(round_angles)

    edges_by_angle=[]
    for ang in unique_angles:
        mask=(round_angles == ang)
        edges_by_angle.append(edges[mask])

    edges_by_angle.sort(key=len, reverse=True)

    if len(edges_by_angle)<3:
        raise ValueError("not enough unique angles")

    e_alpha=edges_by_angle[0]
    e_beta=edges_by_angle[1]
    e_gamma=edges_by_angle[2]

    return e_alpha, e_beta, e_gamma

