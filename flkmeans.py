#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import timeit
import time
import scipy.cluster
import scipy.cluster.vq
import math
from numpy.random import randint
from numpy import shape, zeros, sqrt, argmin, minimum, array, \
     newaxis, arange, compress, equal, common_type, single, double, take, \
     std, mean
import pyflann

SAVE_CODE_BOOKS = False

# wrapper taken from
# https://github.com/scipy/scipy/blob/master/scipy/cluster/vq.py

flann = pyflann.FLANN()


def save_index(filename):
  flann.save_index(filename)


def load_index(filename):
  flann.load_index(filename)


def fl_vq(obs, code_book):
  global flann
  result, dists = flann.nn_index(obs, num_neighbors=1)
  return result, np.sqrt(dists)


def py_vq(obs, code_book):
    # n = number of observations
    # d = number of features
    (n, d) = np.shape(obs)

    code = np.zeros(n, dtype=int)
    min_dist = np.zeros(n)
    for i in range(n):
        dist = np.sum((obs[i] - code_book) ** 2, 1)
        code[i] = np.argmin(dist)
        min_dist[i] = dist[code[i]]
    return code, np.sqrt(min_dist)


def _kmeans(obs, guess, thresh=1e-5):
    code_book = np.array(guess, copy=True)
    avg_dist = []
    diff = thresh + float(1)
    while diff > thresh:
        nc = code_book.shape[0]
        #compute membership and distances between obs and code_book
        global flann
        flann.build_index(code_book, target_precision=0, build_weight=10)
        #obs_code2, distort2 = py_vq(obs, code_book)
        obs_code, distort = fl_vq(obs, code_book)
        avg_dist.append(np.mean(distort, axis=-1))
        #recalc code_book as centroids of associated obs
        if(diff > thresh):
            has_members = []
            for i in np.arange(nc):
                cell_members = np.compress(np.equal(obs_code, i), obs, 0)
                if cell_members.shape[0] > 0:
                    code_book[i] = np.mean(cell_members, 0)
                    has_members.append(i)
            #remove code_books that didn't have any members
            code_book = np.take(code_book, has_members, 0)
        if len(avg_dist) > 1:
            diff = avg_dist[-2] - avg_dist[-1]
    return code_book, avg_dist[-1]


def flkmeans(obs, k_or_guess, iter=20, thresh=1e-5):
    if int(iter) < 1:
        raise ValueError('iter must be at least 1.')
    if type(k_or_guess) == type(np.array([])):
        guess = k_or_guess
        if guess.size < 1:
            raise ValueError("Asked for 0 cluster ? initial book was %s" % \
                             guess)
        result = _kmeans(obs, guess, thresh=thresh)
    else:
        #initialize best distance value to a large value
        best_dist = np.inf
        No = obs.shape[0]
        k = k_or_guess
        if k < 1:
            raise ValueError("Asked for 0 cluster ? ")
        for i in range(iter):
            #the intial code book is randomly selected from observations
            guess = take(obs, randint(0, No, k), 0)
            book, dist = _kmeans(obs, guess, thresh=thresh)
            if dist < best_dist:
                best_book = book
                best_dist = dist
        result = best_book, best_dist
    return result


if __name__ == "__main__":
  print "starting benchmarking and error checking"

  dimensions = 128
  nclusters = 256

  thresh = 1e-2  # try [1,1e-1,1e-2,1e-5]:
  rounds = 1  # for timeit

  for i in range(1):
  #for i in range(51,75):
    points = 512 * i
    points = 50000
    data = np.random.randn(points, dimensions).astype(np.float32)
    print "points", points, "  dimensions", dimensions, "  nclusters", nclusters, "  rounds", rounds
    clusters = data[:nclusters]
    print 'numpyC', timeit.timeit(lambda: scipy.cluster.vq.kmeans(data, nclusters, iter=1), number=rounds)
    print 'flann', timeit.timeit(lambda: flkmeans(data, clusters, iter=1, thresh=thresh), number=rounds)
    for j in range(3):
      cpu_book, cpu_dist = scipy.cluster.vq.kmeans(data, clusters)
      fln_book, fln_dist = flkmeans(data, clusters, thresh=thresh)
      rtol = 0.001
      errorsCPU_FLA = nclusters * dimensions - sum(1 for a, b in zip(fln_book.ravel(), cpu_book.ravel()) if (abs(a - b) <= (rtol * abs(b))))

      print errorsCPU_FLA, cpu_dist, fln_dist,

      if SAVE_CODE_BOOKS:
        np.savetxt('cpu_book-' + str(i) + '-' + str(points) + '.txt', cpu_book, '%f')
        np.savetxt('fln_book-' + str(i) + '-' + str(points) + '.txt', fln_book, '%f')
