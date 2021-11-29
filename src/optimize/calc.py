import numpy as np
import pandas as pd
import torch


def my_adam(x, y, out_file=None, lr=0.01):
    def _parabola(_x, _y, _c, _b, _a):
        _y_p = _c + _b * _x + _a * _x ** 2

        return (_y_p - _y).norm() / _y.norm()

    # remove NaN
    x = torch.tensor(x[~np.isnan(x)])
    y = torch.tensor(y[~np.isnan(y)])

    # set params and optimizer
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    c = torch.tensor(1.0, requires_grad=True)
    pp = [c, b, a]
    optimizer = torch.optim.Adam(pp, lr)

    # optimization
    outputs = None
    a_list = []
    b_list = []
    c_list = []
    i_list = []
    e_list = []

    for i in range(100000):
        optimizer.zero_grad()
        outputs = _parabola(x, y, *pp)
        outputs.backward()
        optimizer.step()
        a_list.append(a.item())
        b_list.append(b.item())
        c_list.append(c.item())
        i_list.append(i)
        e_list.append(outputs.item())
        # print(f"Step: {i} error = {outputs}")

        if outputs < 1e-4:
            break

    # save results
    if out_file is not None:
        columns = ["a", "error"]
        df = pd.concat([pd.Series(a_list), pd.Series(e_list)], axis=1)
        df.columns = columns
        df.to_csv(out_file)

    idx = e_list.index(max(e_list))
    result = [c_list[idx], b_list[idx], a_list[idx]]

    return result
