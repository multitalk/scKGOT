import torch
import numpy as np
from ot.lp import emd


@torch.no_grad()
def gwgrad_partial_gpu(c1, c2, T):
    col_vec = torch.ones(c2.shape[0], dtype=c2.dtype, device=c2.device).reshape(-1, 1)
    row_vec = torch.ones(c1.shape[0], dtype=c1.dtype, device=c1.device).reshape(1, -1)
    const_c1 = torch.matmul(c1 ** 2 / 2, torch.matmul(T, col_vec))
    const_c2 = torch.matmul(torch.matmul(row_vec, T), c2 ** 2 / 2)
    const_c = const_c1 + const_c2
    A = torch.matmul(-torch.matmul(c1, T), c2.transpose(0, 1))
    tens = const_c + A
    return tens * 2


@torch.no_grad()
def gwloss_partial_gpu(c1, c2, T):
    g = gwgrad_partial_gpu(c1, c2, T) * 0.5
    return torch.sum(g * T)


@torch.no_grad()
def partial_gromov_wasserstein_gpu(c1, c2, p, q, m=None, nb_dummies=1, G0=None,
        thres=1, numItermax=1000, tol=1e-7, early_stop=True, **kwargs):
    """
    expected inputs:
    c1: 
    c2:
    p: numpy.ndarray
    q: numpy.ndarray
    """
    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    if G0 is None:
        G0 = np.outer(p, q)
        G0 = torch.from_numpy(G0).to('cuda:0')

    cpt = 0
    err = 1
    log = {'err': []}
    if isinstance(c1, np.ndarray):
        c1 = torch.from_numpy(c1).to('cuda:0')
    if isinstance(c2, np.ndarray):
        c2 = torch.from_numpy(c2).to('cuda:0')
    with torch.no_grad():
        if early_stop:
            patient = 3
            best_loss = np.inf
        while (err > tol and cpt < numItermax):
            Gprev = torch.clone(G0)
            M = gwgrad_partial_gpu(c1, c2, G0)  # GPU accelerated
            M_emd = torch.zeros(dim_G_extended)
            M_emd[:len(p), :len(q)] = M
            M_emd[-nb_dummies:, -nb_dummies:] = torch.max(M) * 1e2
            M_emd = np.asarray(M_emd, dtype=np.float64)
            # ot.emd is implemented with C++
            Gc, logemd = emd(p_extended, q_extended, M_emd, log=True, **kwargs)

            if logemd['warning'] is not None:
                raise ValueError("Error in the EMD resolution: try to increase the"
                                 " number of dummy points")
            G0 = Gc[:len(p), :len(q)]
            G0 = torch.from_numpy(G0).to('cuda:0')
            err = torch.linalg.norm(G0 - Gprev)
            err = float(err)
            if early_stop:
                if err < best_loss:
                    best_loss = err
                else:
                    patient -= 1
                if patient == 0:
                    break

            if log:
                log['err'].append(err)

            deltaG = G0 - Gprev
            a = gwloss_partial_gpu(c1, c2, deltaG)  # GPU accelerated
            b = 2 * torch.sum(M * deltaG)
            if b > 0:  # due to numerical precision
                gamma = 0
                cpt = numItermax
            elif a > 0:
                gamma = min(1, torch.divide(-b, 2.0 * a))
            else:
                if (a + b) < 0:
                    gamma = 1
                else:
                    gamma = 0
                    cpt = numItermax

            G0 = Gprev + gamma * deltaG
            cpt += 1
        if isinstance(c1, np.ndarray):
            c1 = torch.from_numpy(c1)
        if isinstance(c2, np.ndarray):
            c2 = torch.from_numpy(c2)
        if isinstance(G0, np.ndarray):
            G0 = torch.from_numpy(G0)
        c1 = c1.to('cuda:0')
        c2 = c2.to('cuda:0')
        G0 = G0.to('cuda:0')
        gw_dist = gwloss_partial_gpu(c1, c2, G0)
        gw_dist = gw_dist.cpu().numpy()
        G0 = G0.cpu().numpy()
        log['partial_gw_dist'] = gw_dist
        log['T'] = G0[:len(p), :len(q)]
    return gw_dist, log
