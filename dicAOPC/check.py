import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
a_arg = np.argsort(-a[:, 0, :])

a_sort = np.take_along_axis(a, a_arg[None, :, :], axis=2)

print('w')
