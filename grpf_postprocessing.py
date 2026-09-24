import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate

def find_singularities(nodes, candidate_edges, fun, Tol=1e-6):
    found_zeros=[]
    found_poles=[]

    def f_pole(vars):
        z=complex(vars[0], vars[1])
        try:
            w=fun(z)
            if w==0: return[1e9, 1e9]
            w_inv=1.0/w
            return [w_inv.real, w_inv.imag]
        except ZeroDivisionError:
            return [0.0, 0.0]

    def f_zero(vars):
            z=complex(vars[0], vars[1])
            try:
                w=fun(z)
                return[w.real, w.imag]
            except ZeroDivisionError:
                return [1e9, 1e9]

    def winding_numbers(z_center, radius=1e-4):
        def f_prime(z):
            h=1e-6
            return (fun(z+h)-fun(z-h))/(2*h)
        def integrand(t):
            z=z_center+radius*np.exp(1j*t)
            dz=1j*radius*np.exp(1j*t)
            return (f_prime(z)/fun(z))*dz

        integral_real,_=integrate.quad(lambda t: integrand(t).real, 0, 2*np.pi)
        integral_imag, _ = integrate.quad(lambda t: integrand(t).imag, 0, 2*np.pi)

        integral=complex(integral_real, integral_imag)
        w=integral/(2*np.pi*1j)
        return int(np.round(w.real))

    for edge in candidate_edges:
        a,b = edge
        node_a=nodes[a]
        node_b=nodes[b]

        z_guess=complex((node_a[0]+node_b[0])/2, (node_a[1]+node_b[1])/2)
        guess_vars=[z_guess.real, z_guess.imag]

        sol_z=opt.root(f_zero, guess_vars, method='hybr')
        if sol_z.success and np.linalg.norm(sol_z.fun)<Tol:
            found_zeros.append(complex(sol_z.x[0], sol_z.x[1]))
            continue

        sol_p = opt.root(f_pole, guess_vars, method='hybr')
        if sol_p.success and np.linalg.norm(sol_p.fun)<Tol:
            found_poles.append(complex(sol_p.x[0], sol_p.x[1]))

    def remove_duplicates(points):
        unique=[]
        for p in points:
            if not any(np.abs(p-u)<Tol for u in unique):
                unique.append(p)
        return unique
    
    unique_zeros=remove_duplicates(found_zeros)
    unique_poles=remove_duplicates(found_poles)

    zeros_with_mult=[(z, winding_numbers(z)) for z in unique_zeros]
    poles_with_mult=[(p, abs(winding_numbers(p))) for p in unique_poles]

    return zeros_with_mult, poles_with_mult
