import torch

def getCov(X):
    X_mean = X - torch.mean(X, dim=0, keepdim=True)
    cov = torch.mm(X_mean.T, X_mean) / (X.shape[0] - 1)
    return cov

def getDiffNN(f, Z):
    # Convert Z to tensor if it's not already one, assuming Z is a numpy array or a list
    Z = torch.argmax(torch.tensor(Z, dtype=torch.long), dim=1) # one-hot to normal label
    f = torch.tensor(f, dtype=torch.float32)
    
    Covf = getCov(f)
    alphabetZ = torch.unique(Z)
    g = torch.zeros_like(f)
    for z in alphabetZ:
        l = Z == z
        fl = f[l, :]
        Ef_z = torch.mean(fl, dim=0)
        g[l] = Ef_z

    Covg = getCov(g)
    dif = torch.trace(torch.mm(torch.pinverse(Covf, rcond=1e-15), Covg))
    return dif

def getDiffNNCov(f, inverse, Z):
    # Convert inputs to torch tensors if they are not already
    f = torch.tensor(f, dtype=torch.float32)
    inverse = torch.tensor(inverse, dtype=torch.float32)
    Z = torch.tensor(Z, dtype=torch.long)  # Assuming Z is a class index array

    alphabetZ = torch.unique(Z)
    g = torch.zeros_like(f)
    for z in alphabetZ:
        l = Z == z
        fl = f[l, :]
        Ef_z = torch.mean(fl, dim=0)
        g[l] = Ef_z

    Covg = getCov(g)  # getCov needs to be implemented using torch
    dif = torch.trace(torch.mm(inverse, Covg))
    return dif

import numpy as np

def getCov_np(X):
    X_mean = X-np.mean(X, axis=0, keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1)
    return cov


def getDiffNN_np(f, Z):
    #Z=np.argmax(Z, axis=1)
    Covf = getCov_np(f)
    alphabetZ = list(set(Z))
    g = np.zeros_like(f)
    for z in alphabetZ:
        l = Z == z
        fl = f[Z == z, :]
        Ef_z = np.mean(fl, axis=0)
        g[Z == z] = Ef_z

    Covg = getCov_np(g)
    dif = np.trace(np.dot(np.linalg.pinv(Covf, rcond=1e-15), Covg))
    return dif


def getDiffNNCov_np(f, inverse, Z):
    #Z=np.argmax(Z, axis=1)

    alphabetZ = list(set(Z))
    g = np.zeros_like(f)
    for z in alphabetZ:
        l = Z == z
        fl = f[Z == z, :]
        Ef_z = np.mean(fl, axis=0)
        g[Z == z] = Ef_z

    Covg = getCov_np(g)
    dif = np.trace(np.dot(inverse, Covg))
    return dif

if __name__ == '__main__':
    import time

    f = torch.rand(100, 10)
    Z = torch.randint(0, 5, (100,))
    f_np, Z_np = f.numpy(), Z.numpy()
    # Test the torch implementation
    time1 = time.time()
    Covf = getCov(f)
    inverse = torch.pinverse(Covf, rcond=1e-15)
    score1 = getDiffNNCov(f, inverse, Z)
    time2 = time.time()

    score = getDiffNN(f, Z)
    time3 = time.time()
    
    print("Torch implementation:")
    print(score, score1)
    assert score.item() == score1.item()
    print("getDiffNNCov time:", time2-time1, "getDiffNN time:", time3-time2)

    # Test the numpy implementation
    time1 = time.time()
    Covf_np = getCov_np(f_np)
    inverse_np = np.linalg.pinv(Covf_np, rcond=1e-15)
    score1_np = getDiffNNCov_np(f_np, inverse_np, Z_np)
    time2 = time.time()
    score_np = getDiffNN_np(f_np, Z_np)
    time3 = time.time()

    print("Numpy implementation:")
    print(score_np, score1_np)
    assert score_np == score1_np
    print("getDiffNNCov time:", time2-time1, "getDiffNN time:", time3-time2)

    print("Difference between torch and numpy implementations:")
    print(score.item() - score_np)
    print(score1.item() - score1_np)



