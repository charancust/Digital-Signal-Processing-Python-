import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

def fft(x):
    N = len(x)
    if N <= 4:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        X = np.concatenate([X_even + factor[:N//2] * X_odd,
                            X_even + factor[N//2:] * X_odd], axis=0)  
        return X

s = 1024
np1 = np.arange(s)
x1p = 5 * np.sin(2 * np.pi * 1000 * np1 / s + np.deg2rad(90))  
x2p = 10 * np.cos(2 * np.pi * 2000 * np1 / s + np.deg2rad(45))  
x3p = x1p + x2p
f = 1000  
bw = 20  
h = np.ones(s)
for k in range(s):
    if abs(k - f) <= bw or abs(k + f) <= bw:
        h[k] = 0

x3fp = np.fft.ifft(fft(x3p) * h)

fs = 8000
N = 64
t = np.arange(N)
n = np.arange(N)
x1 = 5 * np.sin(2 * np.pi * 1000/fs * t + np.pi * 0.5)
x2 = 10 * np.cos(2 * np.pi * 2000/fs * t + np.pi * 0.25)
x3 = x1 + x2
freq = np.fft.fftfreq(N, 1/fs)

x3ft = dft(x3)
for fr in range(len(x3ft)):
    freq = fr * fs / N
    if freq <= 1000:
        x3ft[fr] = 0 
        x3ft[-fr] = 0

x3f = idft(x3ft)

plt.figure(figsize=(12, 6))
plt.subplot(3, 2, 1)
plt.plot(np1, x1p)
plt.title('x1(n)')
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 2)
plt.plot(np1, x2p)
plt.title('x2(n)')
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 3)
plt.plot(np1, x3p)
plt.title('x3(n)')
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 4)
plt.plot(np1, x3fp.real)  
plt.title('Filtered signal, x3f(n)')
plt.xlabel('n')
plt.ylabel('Amplitude')

freq1= np.fft.fftfreq(len(x3ft), 1/fs)

xf = dft(x3p)
plt.subplot(3, 2, 5)
plt.stem(np.abs(xf))
plt.xlabel('Hz')
plt.ylabel('Amplitude')
plt.title('Frequency plot of unfiltered signal x3(n)')

xf1 = dft(x3f)
plt.subplot(3, 2, 6)
plt.stem(freq1, np.abs(x3ft))
plt.xlabel('Hz')
plt.ylabel('Amplitude')
plt.title('Frequency plot of filtered signal')

plt.tight_layout()
plt.show()
