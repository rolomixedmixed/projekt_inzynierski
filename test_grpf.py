from Grover import grover_one_dir
from rect_mesh import rect_grid
from triangulation import triangulate
from edge_directions import directions
from evaluate_quadrants import evaluate_quadrants
from fun import fun
from visualisation import vis
import numpy as np


if __name__=="__main__":
    #grid generation
    nodes=rect_grid(x_min=-2, x_max=2, y_min=-2, y_max=2, res=0.5)

    #triangulation, edges
    triangles, edges=triangulate(nodes)

    #dividing into directions
    e_alpha, e_beta, e_gama=directions(nodes, edges)

    #calculating quadrants
    quadrants=evaluate_quadrants(nodes, fun)

    #applying grover for each direction
    candidates_alpha=grover_one_dir(e_alpha, quadrants, "alpha")
    candidates_beta=grover_one_dir(e_beta, quadrants, "beta")
    candidates_gama=grover_one_dir(e_gama, quadrants, "gama")

    all_candidates=candidates_alpha+candidates_beta+candidates_gama

    print("Found: "+str(len(all_candidates))+" candidates")

    #visualise
    phase_diff=np.abs(quadrants[edges[:,0]]-quadrants[edges[:,1]]) 
    vis(nodes, edges, (quadrants+1).reshape(-1,1), phase_diff, 1, True, all_candidates)
    