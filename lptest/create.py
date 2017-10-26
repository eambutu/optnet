#!/usr/bin/env python3
"""
Generates random LPs of the form:
   argmin_{z} p^T * z
   where z >= 0, and Az = b
p is the input, while A, b are randomly generated

Note that since all the constraints are equality, we should have the number of
constraints be a lot less than number of variables. In sudoku, the number of
constraints = n^2 while number of variables = n^3.
"""

import argparse
import numpy as np
import numpy.random as npr
import scipy.optimize
import torch

import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numConstraint', type=int, default=4)
    parser.add_argument('--numVariable', type=int, default=8)
    parser.add_argument('--numSamples', type=int, default=10000)
    parser.add_argument('--data', type=str, default='data')
    args = parser.parse_args()

    npr.seed(0)

    save = os.path.join(args.data,
        str(args.numConstraint) + '_' + str(args.numVariable))
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)

    A, b = generate_random_LP(args.numConstraint, args.numVariable)

    P = []
    Z = []
    for i in range(args.numSamples):
        p, z = sample(A, b, args.numVariable)
        P.append(p)
        Z.append(z)

    P = np.array(P)
    Z = np.array(Z)

    for loc,arr in (('features.pt', P), ('labels.pt', Z)):
        fname = os.path.join(save, loc)
        with open(fname, 'wb') as f:
            torch.save(torch.Tensor(arr), f)
        print('Created {}'.format(fname))

# Returns a sample p, z given A, b
def sample(A, b, numV):
    p = npr.randn(numV)
    temp = scipy.optimize.linprog(c=p, A_eq=A, b_eq=b)
    return p, temp.x

# Generates a random satisfiable LP, with only equality constraints
# Creates a full rank matrix A. Selects all coordinates from normal distribution
# with mean = 0, variance = 1
def generate_random_LP(numC, numV):
    A = np.ndarray(shape=(numC, numV))
    for i in range(numC):
        A[i] = np.append(np.zeros(i), npr.randn(numV - i))
    # In the rare instance that one of the pivots was 0
    assert np.linalg.matrix_rank(A) == min(numC, numV)
    # Creates b by making sure there is at least one positive solution
    x = npr.randn(numV) + 1
    return A, np.dot(A, x)

if __name__ == '__main__':
    main()
