import numpy as np

grid1 = np.arange(5)
grid2 = np.arange(4, 8)
grid_xs, grid_ys = np.meshgrid(grid1, grid2)

print(grid_xs)
print(grid_ys)


a = np.random.randn(3,1)
print(a.shape)

print(np.concatenate([a,a], axis=1).shape)
b = np.concatenate([a,a], axis=1)

c = np.repeat(b, 2)
print(c.shape)