from grpf_Grover import GroverAlgorithm
import math
import numpy as np
import cupy as cp
import time

n_qubits = 15
marked_state = 6767

simulator=GroverAlgorithm(n_qubits)

simulator.applyH_n()

print(simulator)

k=math.floor((np.pi/4)*np.sqrt(2**n_qubits))

start=time.time()

for step in range(k):
    simulator.grover_step(marked_state)

end=time.time()

print(simulator)

cp.cuda.runtime.deviceSynchronize()

result=simulator.measure_most_probable()
print(result)

print("time:", end-start)

simulator.destroy()