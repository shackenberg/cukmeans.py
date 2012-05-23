cukmeans.py  -  Kmeans in PyCUDA
===============================

This is an implemenations of the [clustering algorithm k-means][2] in [PyCUDA][3]. As the interface was taken from [Scipy K-means][1], one can easily replace all usages.

Sample usage
------------
    from cukmeans import cukmeans
    from numpy.random import randn
    nclusters = 100
    npoints = 10000
    dimensions = 2
    data = randn(npoints, dimensions)
    gpu_book, gpu_dist = cukmeans(data, nclusters)


Demo and benchmark modus
------------------------
If you run the script directly it starts into a demo modus benchmarking itself against the `C` based Scipy implementation of kmeans and a kmeans implementation using flann (if available).


Contents 
--------
- `cukmeans.py` the kmeans implementation
- `flkmeans.py` a flann based kmeans implemantation as additional benchmark 


Requirements
------------
- PyCUDA: http://mathema.tician.de/software/pycuda
- Flann (optional*): https://github.com/mariusmuja/flann

  

\* Flann is only needed as an aditional kmeans implementation to benchmark to.
[1]: https://github.com/scipy/scipy/blob/master/scipy/cluster/vq.py
[2]: https://en.wikipedia.org/wiki/K-means_clustering
[3]: http://mathema.tician.de/software/pycuda