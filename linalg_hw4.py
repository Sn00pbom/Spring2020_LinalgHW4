import struct
from datetime import datetime

import numpy as np
from numpy import array, ndarray
from numpy import linalg as LA
from matplotlib import pyplot as plt


class kMeans(object):
    """
    k-Means algorithm object.

    Steps are computed using 'next'

    Attributes:
    x: N-vector of n-vectors
    k: number of groups
    n: size of xᵢ
    N: size of x
    c: grouping such that cᵢ = Group(xᵢ)
    z: group representatives z₁,....,zₖ
    """

    def __init__(self, x, z):
        """Initialize k-means on vectors xᵢ with k groups"""
        self.x = x
        self.k = len(z)  # k depends on initial z that is passed
        self.n = len(x[0])  # number of entries in each n-vector
        self.N = len(x)  # number of n-vectors

        self.c = np.random.randint(0,self.k,self.N)  # init random group assignments
        self.z = z

    def __next__(self):
        """Update step - groups then reps. Returns computed error Jᶜˡᶸˢᵗ"""
        self._update_groups()  # put xᵢ into group by cᵢ by smallest distance in z
        self._update_reps()  # set each group rep zⱼ to mean of it's group or 0 if empty group
        return self.Jclust()  # compute and return error

    def _G(self, j):
        """Return array of vectors in group j → {xᵢ | cᵢ = j}, ∀ i ∈ N"""
        return array([self.x[i] for i in range(self.N) if self.c[i] == j])

    def _update_groups(self):
        """Set each cᵢ to argmin(||xᵢ - z₁||,...,||xᵢ - zⱼ||), ∀ i ∈ N"""
        for i in range(self.N):
            # Compute argmin(||xᵢ - z₁||,...,||xᵢ - zⱼ||)
            jmin = np.argmin(array([self.dist(self.x[i], self.z[j]) for j in range(self.k)]))
            # Set cᵢ
            self.c[i] = jmin

    def _update_reps(self):
        """Set each zⱼ to avg(Gⱼ) if Gⱼ not empty else 0ₙ, ∀ j ∈ k"""
        for j in range(self.k):
            # Get group Gⱼ
            Gj = self._G(j)
            # Set zⱼ
            self.z[j] = np.zeros(self.n) if len(Gj) == 0 else np.mean(Gj, axis=0)

    def Jclust(self):
        """Compute and return sum of square distances over N, ∀ i ∈ N → ∑(||xᵢ - zcᵢ||²)/N"""
        tot = 0.0
        for i in range(self.N):
            xi = self.x[i]
            zci = self.z[self.c[i]]
            tot += self.dist(xi, zci)**2
        tot /= self.N
        return tot

    @staticmethod
    def dist(a, b):
        """Compute and return ||a - b|| = √((a₁-b₁)² + ... + (aN-bN)²)"""
        return LA.norm(a-b)


if __name__ == "__main__":
    # Load MNIST into x array of n-vectors
    with open('train-images-idx3-ubyte', 'rb') as f:
        # As per specification on the MNIST source site
        # first four bytes are magic number
        # second four bytes are size of array
        _, size = struct.unpack(">II", f.read(8))
        # third four bytes are nrows
        # fourth four bytes are ncols
        nrows, ncols = struct.unpack(">II", f.read(8))
        print(nrows*ncols, 'entries')
        # Rest is data, load directly into numpy
        x = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        # Reshape into "size" nrows*ncols-vectors
        x = x.reshape((size, nrows*ncols))

    # Load z initial state from csv
    with open('z.csv', 'r') as f:
        z = f.read()  # read string
        z = z.split('\n')  # split string into array by newline
        z = [s.split(',') for s in z]  # split strings in array into arrays by comma
        z = array(z, dtype='float')  # convert to numpy array

    # Init k-means object with vector array x and k groups
    km = kMeans(x, z)

    def gen_imgs():
        """Save reps to image"""
        fig, axes = plt.subplots(5, 4, subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(hspace=0.5)
        for ax, zi, j in zip(axes.flat, km.z, range(len(z))):
            ax.imshow(zi.reshape((nrows, ncols)))
            ax.set_title("z_{}".format(j))
        plt.savefig(datetime.now().strftime('./img/%M-%S.png'), format='png')

    # loop until 2 consecutive runs have same value
    last = -1
    curr = 0
    count = 0
    while curr != last:
        gen_imgs()
        print('Jclust', count, km.Jclust())
        last = curr
        curr = next(km)
        count += 1
    gen_imgs()
