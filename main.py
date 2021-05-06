from sample_utils import normalize, N_sample_n_sphere, N_epsilon, sgn_det_jac, sort_orientation
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import jax.numpy as np
from scipy.spatial import distance_matrix as dmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def f1(x):
    return np.array([-x[0], -x[1]])

def f3(x):
    return np.array([-x[0]])

def f2(x):
    return np.array([np.sin(2*x[0]), np.cos(4*x[1])])

def get_cluster_distance(data, eps):
    if data.shape[0] == 0: return 0
    result = ripser(data, maxdim=0)
    deaths = result['dgms'][0][:,1]
    deaths = deaths[deaths < 1E308]
    diffs = np.diff(deaths,axis=0)
    max_index = np.argmax(diffs)
    radius = deaths[max_index+1]/4
    if radius < eps: radius = 1E6
    return radius

def get_clusters_(data, radius):
    N = data.shape[0]
    dist_matrix = dmat(data, data, p=2)
    labels = np.zeros(data.shape[0])
    adj_matrix = np.where(dist_matrix <= radius, 1, 0)
    graph = csr_matrix(adj_matrix)
    n_comp, labels = connected_components(graph, directed=False, return_labels=True)
    return n_comp, labels

def get_clusters(H_m, H_p, r_m, r_p):
    if H_m.shape[0] > 0:
        n_m, clusters_m = get_clusters_(H_m, r_m)
    else:
        n_m = 0
        clusters_m = []
    if H_p.shape[0] > 0:
        n_p, clusters_p = get_clusters_(H_p, r_p)
        clusters_p += n_m
    else:
        n_p = 0
        clusters_p = []

    clusters = [[] for i in range(n_m+n_p)]
    for i in range(len(clusters_m)):
        clusters[clusters_m[i]].append(H_m[i])
    for i in range(len(clusters_p)):
        clusters[clusters_p[i]].append(H_p[i])
    
    print("number of clusters: {}".format(n_p+n_m))
    index = n_p - n_m

    return clusters, index


    


d = 2
N = 100000
e = 0.01
radius = 1
x = N_sample_n_sphere(d, radius,N)
y = 1/np.sqrt(d)*np.ones(d)
print('regular value: {}'.format(y))
data = N_epsilon(normalize(f1), x, y, epsilon=e)
local_degree = sgn_det_jac(normalize(f1), data)
print('approximate preimage samples: {}'.format(data.shape))
H_m, H_p = sort_orientation(data, local_degree)
r_m = get_cluster_distance(H_m, e)
r_p = get_cluster_distance(H_p, e)
print('H_- samples: {}, H_+ samples: {}'.format(H_m.shape, H_p.shape))
#print(r_m, r_p)
clusters, index = get_clusters(H_m, H_p, r_m, r_p)
print('index number: {}'.format(index))
#plt.scatter(data[:,0],data[:,1])
#plt.show()
