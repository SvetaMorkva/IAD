import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

x = np.array([[22, 20, 31, 16, 14,  7, 18],
       [17, 22, 40, 12, 36, 42, 13],
       [29,  1, 42, 35, 47, 40, 37],
       [ 8, 42,  0, 37, 25, 26, 47],
       [29, 38, 13,  0, 46, 31, 47],
       [ 5,  6, 24, 39, 48, 48, 25],
       [41,  2,  5,  9, 13,  2, 22],
       [ 7, 28, 16, 16, 23, 34,  2],
       [14, 31, 38,  1, 49,  7,  3],
       [19, 21, 47, 36,  7, 32, 48],
       [24, 33, 22, 18, 46, 35, 34],
       [19, 19, 48,  8, 22, 36, 30],
       [13, 23, 13, 24, 40, 33, 14],
       [ 6, 24, 43, 49, 16, 23, 10],
       [27, 22, 11, 19, 27, 30,  1],
       [17, 46, 39, 43, 23,  5,  6],
       [20, 28,  8,  3, 31, 36, 26],
       [ 5, 42, 16, 41, 25,  2, 48],
       [11,  0,  4, 17, 46, 16, 30],
       [30,  7, 48, 12, 32, 24,  7]])

m, n = x.shape


def euklid_norm():
    D = np.zeros((m, m + 1))
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = np.linalg.norm(x[i] - x[j])
            D[j, i] = D[i, j]
    return D


def get_dist(x, vecc, row, col):
    x[row] = (x[row] + x[col]) / 2 - abs((x[row] - x[col]) / 2)
    x[:, row] = (x[:, row] + x[:, col]) / 2 - abs((x[:, row] - x[:, col]) / 2)
    x = np.delete(x, col, 0)
    x = np.delete(x, col, 1)
    x[row, row] = 0
    vecc[row] = np.max(vecc) + 1
    vecc = np.delete(vecc, col)
    return x, vecc


result = euklid_norm()
link = np.zeros((m - 1, 4))
claster = np.arange(x.shape[0])

i = 0

while len(result) > 1:
    min_ = np.min(result[np.nonzero(result)])
    temp = np.where(result == min_)[0]
    link[i] = [claster[temp[0]], claster[temp[1]], min_, 2]
    result, claster = get_dist(result, claster, temp[0], temp[1])
    i = i + 1


print(link)
plt.figure(figsize=(25, 10))
plt.title('Dendrogram')
plt.xlabel('clusters')
plt.ylabel('distance')
dendrogram(
    link,
    leaf_rotation=90.,
    leaf_font_size=8.,
)

plt.show()



