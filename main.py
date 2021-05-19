from sample_utils import normalize, N_sample_n_sphere, N_epsilon, sgn_det_jac, sort_orientation
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import jax.numpy as np
from scipy.spatial import distance_matrix as dmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

#from mpl_toolkits import plot3d

import jax
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
def f1(x):
    return np.array([-np.sin(2*x[0]), -x[1]])

def f2(x):
    return np.array([-x[0] + x[0]*x[1], -x[1]])

def f3(x):
    return np.array([x[0]**2-x[1]**2, 2*x[0]*x[1]])

def f4(x):
    return np.array([-x[0],-x[1],-x[2]])

def f5(x):
    return np.array([-x[0]**2, -x[1]**2])


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
    
    #print(n_p, n_m)
    print("number of clusters: {}".format(n_p+n_m))
    index = n_p - n_m

    return clusters, index

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_clusters_2D(clusters, data, x, y):
    n = len(clusters)
    cmap = ['r','b','g','y','c','black','m','tab:pink','coral']
    #cmap = get_cmap(n)
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(r'$f(x_1,x_2) = \frac{1}{||f(x_1,x_2)||}(-\sin(2 x_1), -x_2)$', fontsize=12)

    ax1.set_title(r'Initial Samples and Preimage')
    ax1.scatter(x[:,0], x[:,1], label=r'$S_{\varepsilon}(x_*)$ samples',c='grey')
    ax1.scatter(data[:,0], data[:,1], label=r'$N_{\delta}(y)$ approx. preimage', c='b')
    ax1.scatter(y[0],y[1], label=r'$y$ target regular value',c='m')
    #f_vmap = jax.vmap(f)
    #y = f_vmap(x)
    #print(y.shape)
    #ax1.scatter(y[:,0],y[:,1])
    ax1.set_xlim(-1.25,1.25)
    ax1.set_ylim(-1.25,1.25)
    ax1.set_xlabel(r'$x_1$',fontsize=12)
    ax1.set_ylabel(r'$x_2$',fontsize=12)
    ax1.legend()
    ax2.scatter(x[:,0], x[:,1], c='grey')
    for i in range(n):
        cluster_i = np.array(clusters[i])
        ax2.scatter(cluster_i[:,0], cluster_i[:,1],c=cmap[i],label=r'cluster ${}$'.format(i))
    ax2.set_title(r'$H_0$-based Clustering')
    ax2.set_xlim(-1.25,1.25)
    ax2.set_ylim(-1.25,1.25)
    ax2.set_xlabel(r'$x_1$',fontsize=12)
    ax2.set_ylabel(r'$x_2$',fontsize=12)
    ax2.legend()
    
    plt.show()

def plot_clusters_3D(clusters, data, x, y):
    n = len(clusters)
    cmap = ['r','b','g','y','c','black','m','tab:pink', 'coral']
    #cmap = get_cmap(n)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    fig.suptitle(r'$f(x_1,x_2,x_3) = \frac{1}{||f(x_1,x_2, x_3)||}(-x_1, -x_2, -x_3)$', fontsize=12)

    ax1.set_title(r'Initial Samples and Preimage')
    ax1.scatter(x[:,0], x[:,1], x[:,2], label=r'$S_{\varepsilon}(x_*)$ samples',c='grey',s=1)
    ax1.scatter(data[:,0], data[:,1], data[:,2], label=r'$N_{\delta}(y)$ approx. preimage', c='b',s=20)
    ax1.scatter(y[0],y[1], y[2], label=r'$y$ target regular value',c='m',s=20)
    #f_vmap = jax.vmap(f)
    #y = f_vmap(x)
    #print(y.shape)
    #ax1.scatter(y[:,0],y[:,1])
    ax1.set_xlim(-1.25,1.25)
    ax1.set_ylim(-1.25,1.25)
    ax1.set_xlabel(r'$x_1$',fontsize=12)
    ax1.set_ylabel(r'$x_2$',fontsize=12)
    ax1.legend()
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.scatter(x[:,0], x[:,1], x[:,2],c='grey', s=1)
    
    for i in range(n):
        cluster_i = np.array(clusters[i])
        ax2.scatter(cluster_i[:,0], cluster_i[:,1], cluster_i[:,2], c=cmap[i],label=r'cluster ${}$'.format(i), s=20)
    ax2.set_title(r'$H_0$-based Clustering')
    ax2.set_xlim(-1.25,1.25)
    ax2.set_ylim(-1.25,1.25)
    ax2.set_xlabel(r'$x_1$',fontsize=12)
    ax2.set_ylabel(r'$x_2$',fontsize=12)
    ax2.legend()
    
    plt.show()




d = 2
N = 1000
e = 0.1
radius = 1.0
function = f1
norm_function = normalize(function)
x = N_sample_n_sphere(d, radius, N)
#y = np.array([-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)])
y = np.array([-1/np.sqrt(2), -1/np.sqrt(2)])
print('regular value: {}'.format(y))
#print(np.linalg.norm(x[0]))

data = N_epsilon(norm_function, x, y, epsilon=e)
print('data set size: {}'.format(data.shape[0]))
local_degree = sgn_det_jac(norm_function, data)
print('approximate preimage samples: {}'.format(data.shape))
H_m, H_p = sort_orientation(data, local_degree)
r_m = get_cluster_distance(data, e)
r_p = get_cluster_distance(H_p, e)

print('H_- radius: {}, H_+ radius: {}'.format(r_m, r_p))
print('H_- samples: {}, H_+ samples: {}'.format(H_m.shape, H_p.shape))

clusters, index = get_clusters(H_m, H_p, r_m, r_p)
print('index number: {}'.format(index))
#plot_clusters_2D(clusters, data, x, y)
plot_clusters_2D(clusters, data, x,y)
