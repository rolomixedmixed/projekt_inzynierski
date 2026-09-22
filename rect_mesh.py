import numpy as np
import matplotlib.pyplot as plt

def rect_grid(x_min, x_max, y_min, y_max, res):
    x_min = x_min
    x_max = x_max
    y_min = y_min
    y_max = y_max
    res = res
    X=x_max-x_min
    Y=y_max-y_min

    n=np.ceil(Y/res+1)
    dy=Y/(n-1)
    m=np.ceil(X/np.sqrt(res**2-dy**2/4)+1)
    dx=X/(m-1)

    vx=np.linspace(x_min, x_max, int(m))
    vy=np.linspace(y_min, y_max, int(n))

    x,y = np.meshgrid(vx,vy)

    temp = np.ones((int(n), 1))
    temp[-1] = 0
    y = y + 0.5 * dy * np.kron((1 + (-1)**np.arange(1, m + 1)) / 2, temp)

    x = np.reshape(x, (int(m * n), 1))
    y = np.reshape(y, (int(m * n), 1))
    tx = ((np.arange(2, m + 1, 2) - 1) * dx + x_min).reshape(-1, 1)
    ty = np.zeros_like(tx) + y_min
    x = np.vstack([x, tx])
    y = np.vstack([y, ty])

    NewNodesCoord = np.concatenate([x, y], axis=1)

    return NewNodesCoord





