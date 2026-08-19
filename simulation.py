from Grover import GroverAlgorithm
import math
import numpy as np
import cupy as cp

n_qubits = 20
marked_state = 67

simulator=GroverAlgorithm(n_qubits)

simulator.applyH_n()

simulator.console_print()

k=math.floor((np.pi/4)*np.sqrt(2**n_qubits))

for step in range(k):
    simulator.grover_step(marked_state)

simulator.console_print()

cp.cuda.runtime.deviceSynchronize()

result=simulator.measure_most_probable()
print(result)

simulator.destroy