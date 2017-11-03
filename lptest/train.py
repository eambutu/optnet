#!/usr/bin/env python3

import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--numVar', type=int, default=8)
    parser.add_argument('--numConst', type=int, default=4)
    parser.add_argument('--Qpenalty', type=float, default=0.1)
    parser.add_argument('--batchSz', type=int, default=150)
    parser.add_argument('--testBatchSz', type=int, default=200)
    parser.add_argument('--numEpoch', type=int, default=100)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    args = parser.parse_args()

    p = str(args.numConst) + '_' + str(args.numVar)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.save is None:
        args.save = args.work

    with open('./data/{}/features.pt'.format(p), 'rb') as f:
        X = torch.load(f)
    with open('./data/{}/labels.pt'.format(p), 'rb') as f:
        Y = torch.load(f)

    N = X.size(0)

    nTrain = int(N * (1. - args.testPct))
    nTest = N - nTrain

    trainX = X[:nTrain]
    trainY = Y[:nTrain]
    testX = X[nTrain:]
    testY = Y[nTrain:]

    assert(nTrain % args.batchSz == 0)
    assert(nTest % args.testBatchSz == 0)

    save = args.save
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)

    print('Building model')
    model = models.OptNetEq(args.numVar, args.numConst, args.Qpenalty,
                            not args.no_cuda)

    if args.cuda:
        model = model.cuda()

    lr = 1e-1
    optimizer = optim.Adam(model.parameters(), lr=lr)

    test(args, model, testX, testY)
    for epoch in range(1, args.numEpoch):
        train(args, epoch, model, optimizer, trainX, trainY)
        test(args, model, testX, testY)
        torch.save(model, os.path.join(args.save, 'latest.pth'))

def test(args, model, testX, testY):
    batchSz = args.testBatchSz

    test_loss = 0
    batch_data_t = torch.FloatTensor(batchSz, testX.size(1))
    batch_targets_t = torch.FloatTensor(batchSz, testY.size(1))
    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()
    batch_data = Variable(batch_data_t, volatile=True)
    batch_targets = Variable(batch_targets_t, volatile=True)

    for i in range(0, testX.size(0), batchSz):
        print('Testing model: {}/{}'.format(i, testX.size(0)), end='\r')
        batch_data[:] = testX[i:i + batchSz]
        batch_targets.data[:] = testY[i:i + batchSz]
        output = model(batch_data)
        test_loss += nn.MSELoss()(output, batch_targets)

    numBatches = testX.size(0) / batchSz
    test_loss = test_loss.data[0] / numBatches
    print('Test average loss: {:.4f}'.format(test_loss))

def train(args, epoch, model, optimizer, trainX, trainY):
    batchSz = args.testBatchSz

    batch_data_t = torch.FloatTensor(batchSz, trainX.size(1))
    batch_targets_t = torch.FloatTensor(batchSz, trainY.size(1))
    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()
    batch_data = Variable(batch_data_t, requires_grad=False)
    batch_targets = Variable(batch_targets_t, requires_grad=False)

    for i in range(0, trainX.size(0), batchSz):
        print('Testing model: {}/{}'.format(i, trainX.size(0)), end='\r')
        batch_data[:] = trainX[i:i + batchSz]
        batch_targets.data[:] = trainY[i:i + batchSz]

        optimizer.zero_grad()
        output = model(batch_data)
        loss = nn.MSELoss()(output, batch_targets)
        loss.backward()
        optimizer.step()
        
        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
            epoch, i + batchSz, trainX.size(0),
            float(i + batchSz) / trainX.size(0) * 100, loss.data[0]))

if __name__ == '__main__':
    main()
