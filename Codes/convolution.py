import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 50)
x = 10 * np.sin((2 * np.pi * 50 * t) + np.pi / 2)
b = np.linspace(0, 25)
h = np.sinc(2 * np.pi * 0.03 * b)
xloc = 0
hloc = 0
x1 = np.array(x).reshape((len(x), 1))
h1 = np.array(h).reshape((1, len(h)))
mat = np.matmul(x1, h1)
ans = np.zeros(len(x) + len(h) - 1)
for i in range(len(x)):
    for j in range(len(h)):
        ans[i + j] += mat[i][j]
a = np.zeros_like(ans)
for i in range(len(x)):
    a[i] = i - xloc - hloc

plt.subplot(3, 2, 1)
plt.stem(t, x)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('sin(10,50,90deg)')

plt.subplot(3, 2, 2)
plt.stem(b, h)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('sinc(x)')

plt.subplot(3, 2, 5)
plt.stem(a, ans)
plt.ylabel('x[n]*h[n]')
plt.title('using own algorithm')

plt.subplot(3, 2, 6)
plt.stem(a, np.convolve(x, h))
plt.title('using np.convolve function')

c = 3
for i in range(len(mat)):
    shift = np.zeros((len(x) + len(h) - 1)).tolist()
    plt.subplot(3, 2, c)
    for j in range(len(mat[i])):
        shift[i + j] = mat[i][j]
        plt.stem(a, shift)
    c += 1

plt.tight_layout(pad=0.1)
plt.show()