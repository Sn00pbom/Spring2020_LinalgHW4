import numpy as np
from numpy import array, ndarray
from numpy import linalg as LA
import struct
from matplotlib import pyplot as plt
from datetime import datetime


class kMeans(object):
    """k-means algorithm where k is number of groups and x is vector st x_i is n-vector"""

    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.n = len(x[0]) # number of entries in each n-vector
        self.N = len(x) # number of n-vectors

        self.c = np.random.randint(0,k,self.N) # init c_i element tells which group x_i belong to
        self.z = array([np.random.rand(self.n)*255 for _ in range(k)]) # init reps. group i

    def __next__(self):
        self._update_groups()
        self._update_reps()
        return self.Jclust()

    def _G(self, j):
        return array([self.x[i] for i in range(self.N) if self.c[i] == j])

    def _update_groups(self):
        for i in range(self.N):
            jmin = np.argmin(array([self.dist(self.x[i], self.z[j]) for j in range(self.k)]))
            self.c[i] = jmin

    def _update_reps(self):
        for j in range(self.k):
            Gj = self._G(j)
            self.z[j] = np.zeros(self.n) if len(Gj) == 0 else np.mean(Gj, axis=0)

    def Jclust(self):
        tot = 0.0
        for i in range(self.N):
            xi = self.x[i]
            zci = self.z[self.c[i]]
            tot += self.dist(xi, zci)
        tot /= self.N
        return tot

    @staticmethod
    def dist(a, b):
        return LA.norm(a-b)


if __name__ == "__main__":

    # load MNIST into x (array of vectors)
    with open('/home/zach/DATASET/train-images-idx3-ubyte', 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        x = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        x = x.reshape((size, nrows*ncols))

    # define k groups
    k = 20

    km = kMeans(x, k)

    def gen_imgs():
        fig, axes = plt.subplots(5, 4, subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(hspace=0.5)
        for ax, zi, j in zip(axes.flat, km.z, range(k)):
            ax.imshow(zi.reshape((nrows, ncols)))
            ax.set_title("z_{}".format(j))
        plt.savefig(datetime.now().strftime('./img/%M-%S.png'), format='png')

    last = -1
    curr = 0
    count = 0
    while curr != last:
        gen_imgs()
        print(km.Jclust())
        last = curr
        curr = next(km)
    gen_imgs()
