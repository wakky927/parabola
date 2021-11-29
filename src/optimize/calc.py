import numpy as np
import pandas as pd
import torch


def test_func(vel, x, y, U, m, x0, y0):
    # calc velocity
    _u = m * (x - x0) / ((x - x0) ** 2 + (y - y0) ** 2)
    _v = - U + m * (y - y0) / ((x - x0) ** 2 + (y - y0) ** 2)

    # concatenate velocity
    _vel_pred = torch.cat([_u, _v], axis=1)

    # calc error
    error = (vel - _vel_pred).norm() / vel.norm()

    return error


def my_adam(x, y, func, param, out_file, lr=0.01):
    # remove NaN
    x = torch.tensor(x[~np.isnan(x)])
    y = torch.tensor(y[~np.isnan(y)])

    # set params and optimizer
    p = torch.tensor(param, requires_grad=True)
    params = [p]
    optimizer = torch.optim.Adam(params, lr)

    # optimization
    outputs = None
    p_list = []
    i_list = []
    e_list = []

    for i in range(100000):
        print(f"Step: {i}")
        optimizer.zero_grad()
        outputs = func(x, y, *params)
        outputs.backward()
        optimizer.step()
        p_list.append(p.item())
        i_list.append(i)
        e_list.append(outputs.item())

        if outputs < 1e-4:
            break

    # save results
    columns = ["p", "error"]
    df = pd.concat([pd.Series(p_list), pd.Series(e_list)], axis=1)
    df.columns = columns
    df.to_csv(out_file)

    return outputs
