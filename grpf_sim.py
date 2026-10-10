from grpf_Grover import grover_one_dir
from grpf_rect_mesh import rect_grid
from grpf_triangulation import triangulate
from grpf_edge_directions import directions
from grpf_evaluate_quadrants import evaluate_quadrants
from grpf_fun import fun
from grpf_visualisation import vis
from grpf_candidate_regions import regions
from grpf_refine_mesh import refine_mesh
from grpf_postprocessing import find_singularities
import numpy as np
import cupy as cp


if __name__=="__main__":
    #grid generation
    nodes=rect_grid(x_min=-8, x_max=8, y_min=-8, y_max=8, res=1)

    tolerance=1e-9
    max_iterations=100
    iterations=0
    start_event=cp.cuda.Event()
    end_event=cp.cuda.Event()

    while True:

        iterations+=1

        #triangulation, edges
        triangles, edges=triangulate(nodes)

        #calculating quadrants
        quadrants=evaluate_quadrants(nodes, fun)

        start_event.record()

        if iterations==1:
            #dividing into directions
            e_alpha, e_beta, e_gama=directions(nodes, edges)

            #applying grover for each direction
            candidates_alpha=grover_one_dir(e_alpha, quadrants, "alpha")
            candidates_beta=grover_one_dir(e_beta, quadrants, "beta")
            candidates_gama=grover_one_dir(e_gama, quadrants, "gama")

            all_candidates=candidates_alpha+candidates_beta+candidates_gama

        else:
            phase_diff=np.mod(quadrants[edges[:,0]]-quadrants[edges[:,1]], 4)
            all_candidates=edges[phase_diff==2].tolist()

        end_event.record()
        end_event.synchronize()

        if len(all_candidates)==0:
            print("No candidates found")
            break

        max_edge_length=0
        for edge in all_candidates:
            a,b = edge
            length=np.linalg.norm(nodes[a] - nodes[b])
            if length>max_edge_length:
                max_edge_length=length

        if max_edge_length<tolerance:
            print("reached desired tolerance")
            break
        if iterations>=max_iterations:
            print("reached max operations")
            break

        boundary_edges,edges_to_split=regions(triangles, all_candidates)

        nodes=refine_mesh(nodes, edges_to_split)

    print("Found: "+str(len(all_candidates))+" candidates")

    zeros, poles=find_singularities(nodes, all_candidates, fun)

    print("zeros: ")
    for z, mult in zeros:
        print("real: "+str(z.real)+" imag: "+str(z.imag)+" order: "+str(mult))

    print("poles: ")
    for z, mult in poles:
        print("real: "+str(z.real)+" imag: "+str(z.imag)+" order: "+str(mult))

    elapsed_time_ms=cp.cuda.get_elapsed_time(start_event, end_event)
    print("Elapsed time: "+str(elapsed_time_ms)+" ms")

    #visualise
    phase_diff=np.abs(quadrants[edges[:,0]]-quadrants[edges[:,1]]) 
    vis(nodes, edges, (quadrants+1).reshape(-1,1), phase_diff, 1, True, all_candidates)
    