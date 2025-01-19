import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

def plot_sequence(x, title):
    plt.stem(x)
    plt.title(title)
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_spectrum(X, title):
    plt.stem(np.abs(X))
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel('|X[k]|')
    plt.grid(True)
    plt.show()

def plot_intermediate_result(X, stage):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.stem(X.real)
    plt.title(f'Real part of X at Stage {stage}')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.stem(X.imag)
    plt.title(f'Imaginary part of X at stage {stage}')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def fft(x):
    N = len(x)
    if N <= 4:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        X = np.concatenate([X_even + factor[:N//2] * X_odd,
                            X_even + factor[N//2:] * X_odd], axis=0)  # Ensure correct shape
        return X

t = np.linspace(0, 1, 16)
t1 = np.linspace(0,5)
x = 5 * np.sin(2 * np.pi * 10 * t+ (np.pi / 2))
x1= 5 * np.sin(2 * np.pi * 10 * t1+ (np.pi / 2))
plot_sequence(x1, 'Input Sequence')
X = fft(x)
N = len(x)
stages = int(np.log2(N))
for i in range(stages + 1):
    plot_intermediate_result(fft(x * np.exp(-2j * np.pi * i / N * np.arange(N))), i)

plot_spectrum(X, 'Magnitude Spectrum of 16-point DFT')

print("Magnitude of the sequence:", np.abs(X))

print("DFT of the sequence:", X)

plt.figure()

plt.subplot(1, 2, 1)
plt.stem(np.abs(X))
plt.title('Magnitude of 16-point DFT')
plt.xlabel('Index')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(np.angle(X))
plt.title('Phase of 16-point DFT')
plt.xlabel('Index')
plt.ylabel('Phase')
plt.grid(True)

plt.tight_layout()
plt.show()
