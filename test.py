import numpy as np

a = np.array([[[[3, 1, 4], [1, 5, 9], [2, 6, 5]], [[3, 1, 4], [1, 5, 9], [2, 6, 5]]]])
print(a.shape)

XX, YY = np.meshgrid(np.arange(a.shape[2]), np.arange(a.shape[3]))
# print(XX, YY)

a = a.reshape(1, 2, 9, 1)

b = np.zeros([1, 2, 9])
print(b)
b = b + XX.ravel()
b = np.expand_dims(b, axis=-1)
print(b.shape)
print(b)
print(XX.ravel().shape)

c = np.zeros([1, 2, 9]) + YY.ravel()
c = np.expand_dims(c, axis=-1)
print(c)
print(c.shape)

z = np.concatenate([a, b, c], axis=-1)
# z = z.transpose([0, 1, 3, 2])
print(z.shape)

print("aaaaa" + str(z))

print("########################################################")

deck = np.array([[[6., 2.],
                  [10., 1.],
                  [5., 1.],
                  [9., 2.],
                  [4., 1.],
                  [3., 2.],
                  [11., 2.]],

                 [[6., 2.],
                  [2., 2.],
                  [2., 3.],
                  [11., 1.],
                  [11., 3.],
                  [5., 3.],
                  [4., 4.]]])

sortval = deck[:, :, 0] * 4 + deck[:, :, 1]
print(deck.shape)
print(sortval.shape)
sortval *= -1  # if you want largest first
sortedIdx = np.argsort(sortval)
print(sortedIdx.shape)
print(np.arange(len(deck))[:, np.newaxis].shape)
deck = deck[np.arange(len(deck))[:, np.newaxis], sortedIdx]
print(deck)

a_2d = np.array([[20, 3, 100], [1, 200, 30], [300, 10, 2]])
# print(a_2d.shape)
# print(np.argsort(a_2d[:, 0]).shape)
a_2d_sort_col_num = a_2d[np.argsort(a_2d[:, 0])]
# print(a_2d_sort_col_num)

a = np.array([[[[8, 1, 4], [1, 5, 9], [2, 6, 5]], [[2, 1, 1], [1, 1, 1], [3, 1, 1]]],
              [[[2, 1, 1], [3, 1, 1], [4, 1, 1]], [[6, 1, 4], [5, 5, 9], [4, 6, 5]]]])
print("a = " + str(a[:, :, :, :].shape))
# print(a)

print(np.argsort(a[:, :, :, 0]).shape)
print(np.argsort(a[:, :, :, 0]))
# a_test = np.take_along_axis(a, np.argsort(a[:, :, :, 0]), axis=0)


print("++++++++++++++++++++++++++++++++++++++")

psh = np.array(
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
     [[3, 2, 1], [6, 5, 4], [9, 8, 7]]]
)
ph_idx = np.array([[0, 1, 2], [2, 1, 0]])

print(psh.shape)
print(ph_idx[:, None, :].shape)

print(np.take_along_axis(psh, ph_idx[:, None, :], axis=2))

# list = np.array([0, 1])[:, np.newaxis, np.newaxis], np.array([0, 1])[:, np.newaxis], np.argsort(
#    a[:, :, :, 0])
# a_test = a[list]


a = np.array([[[[8, 1, 4], [1, 5, 9], [2, 6, 5]], [[2, 1, 1], [1, 1, 1], [3, 1, 1]]],
              [[[2, 1, 1], [3, 1, 1], [4, 1, 1]], [[6, 1, 4], [5, 5, 9], [4, 6, 5]]]])

print("shape\n" + str(a[:, :, :, 1]))

# これだああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああ
a_test = np.take_along_axis(a, np.argsort(a[:, :, :, 0])[:, :, :, None], axis=2)

print(a_test)

#################################

nn = np.array([1, 2, 3, 4])
nn[2] = 10000
print(nn)


def test():
    i = 0
    while True:
        i += 1
        if i == 10:
            return
        yield i


for x in test():
    print(x)

#################################
from ImageDataGenerator import ImageDataGenerator

generator = ImageDataGenerator()

for x in generator.flow_from_personal_directory("/mnt/data2/img/20191207/00002", 128, 128, 3, 100):
    print(x)

b = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]])
print(b.mean(axis=(1, 2)).shape)
print(b - b.mean(axis=(1, 2)))

a = np.random.rand(101, 256, 1, 3, 1, 10)
b = np.random.rand(101, 20, 10)

b = b[:, 10, 0:5]
print(b.shape)

#################################

haha = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(haha[2:8])
