import numpy as np

def find_singularities(nodes, boundary_edges, quadrants):
    found_zeros=[]
    found_poles=[]
    neighbours={}

    for a,b in boundary_edges:
        neighbours.setdefault(a,[]).append(b)
        neighbours.setdefault(b,[]).append(a)
    
    for node, adjacent in neighbours.items():
        if len(adjacent)!=2:
            raise ValueError(f"Węzeł {node}, pozycja {nodes[node]}, sąsiedzi {adjacent}")

    remaining=set(neighbours)

    while remaining:
        start=remaining.pop()
        contour=[start]
        previous=None
        current=start
        while True:
            following=neighbours[current][0]
            if following==previous:
                following=neighbours[current][1]
            contour.append(following)
            if following==start:
                break
            remaining.remove(following)
            previous=current
            current=following

        phase_sum=0
        area=0.0
        origin=nodes[start]

        for i in range(len(contour)-1):
            a,b=contour[i],contour[i+1]

            change=np.mod(quadrants[b]-quadrants[a], 4)
            if change==2:
                raise ValueError("Jumped 2 quadrants at once, need to refine mesh")
            if change==3:
                change=-1

            phase_sum+=change

            u=nodes[a]-origin
            v=nodes[b]-origin

            area+=u[0]*v[1]-u[1]*v[0]

        if area==0:
            raise ValueError("contour has zero area")
        winding=phase_sum//4
        if area<0:
            winding=-winding

        centre=nodes[contour[:-1]].mean(axis=0)
        z=complex(centre[0], centre[1])

        if winding>0:
            found_zeros.append((z, winding))
        elif winding<0:
            found_poles.append((z, -winding))



    return found_poles, found_zeros
