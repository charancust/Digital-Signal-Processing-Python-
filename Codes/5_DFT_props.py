import numpy as np
import matplotlib.pyplot as plt

plt.subplot(5, 2, 1)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("s1")
t1 = np.arange(-32, 32, 1)
s1 = 10 * np.cos((2 * np.pi * (10 / 640) * t1) + (np.pi / 4))
plt.stem(t1, s1)

plt.subplot(5, 2, 2)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("s2")
s2 = 15 * np.cos((2 * np.pi * (30 / 640) * t1) + (np.pi / 2))
plt.stem(t1, s2)

plt.subplot(5, 2, 3)
plt.title("s3")
plt.xlabel("Time")
plt.ylabel("Amplitude")
s3 = s1+s2
plt.stem(t1, s3)

plt.subplot(5, 2, 4)
h = [5, -7, 10, -20, 15, -30]
hn = np.arange(0, len(h), 1)
plt.title("h[n]")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.stem(hn, h)

plt.subplot(5, 2, 5)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("x*h convolution using inbuilt")
conres = np.convolve(s3, h)
conax = np.arange(0, len(conres), 1)
plt.stem(conax, conres)

plt.subplot(5, 2, 6)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("x*h convolution using own algorithm")
ylen = len(s3) + len(h) - 1
s3p = np.pad(s3, (0, ylen - len(s3)), 'constant')
hp = np.pad(h, (0, ylen - len(h)), 'constant')
conv = np.zeros(ylen)
for i in range(ylen):
    for j in range(len(h)):
        if i - j >= 0:
            conv[i] += s3p[i - j] * hp[j]
cax = np.arange(0, len(conv), 1)
plt.stem(cax, conv)

plt.subplot(5, 2, 9)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Y[n] using own algorithm" )
ft1 = np.zeros(ylen, dtype=complex)
ft2 = np.zeros(ylen, dtype=complex)
for i in range(ylen):
    for j in range(ylen):
        ft1[i] += s3p[j] * np.exp(-2j * np.pi * j * i / ylen)
        ft2[i] += hp[j] * np.exp(-2j * np.pi * j * i / ylen)
resft = ft1 * ft2
res = np.zeros(ylen, dtype=complex)
for i in range(ylen):
    for j in range(ylen):
        res[i] += resft[j] * np.exp(2j * np.pi * j * i / ylen)
res /= ylen
plt.stem(np.arange(0, len(res)), np.real(res))

plt.subplot(5, 2, 10)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Y2[n] using inbuilt")
ft3 = np.fft.fft(s3p)
ft4 = np.fft.fft(hp)
resft3 = np.fft.ifft(ft3 * ft4)
plt.stem(np.arange(0, len(resft3)), np.real(resft3))

freq1 = np.arange(-ylen / 2, ylen / 2, 1)
plt.subplot(5, 2, 7)
plt.title("s3[K]")
plt.xlabel("f")
plt.ylabel("Magnitude")
plt.stem((640 / 32) * freq1, np.fft.fftshift(np.abs(ft3)))

plt.subplot(5, 2, 8)
plt.title("h[k]")
plt.ylabel("Magnitude")
plt.xlabel("f")
plt.stem((640 / 32) * freq1, np.fft.fftshift(np.abs(ft4)))

plt.show()
