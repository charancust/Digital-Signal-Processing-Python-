import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

def fft(x):
    N = len(x)
    if N <= 4:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        X = np.concatenate((X_even + factor[:N//2] * X_odd, 
                             X_even + factor[N//2:] * X_odd))
        return X

fs = 10000  
fc = 3500  
M = 15     
nfc = 2 * np.pi * fc / fs

lpf = np.sinc(2 * fc * (np.arange(M+1) - M/2) / fs)

hrect = lpf * np.ones(M+1)  
hhamming = lpf * np.hamming(M+1)  
hhanning = lpf * np.hanning(M+1)  

w = np.linspace(0, np.pi, num=8000)
Hrect = np.fft.fft(hrect, 8000)
Hhamming = np.fft.fft(hhamming, 8000)
Hhanning = np.fft.fft(hhanning, 8000)

t = np.linspace(0, 0.01, num=fs//100, endpoint=False)  
x1 = 5 * np.sin(2 * np.pi * 3000 * t + np.pi/2)
x2 = 10 * np.cos(2 * np.pi * 4000 * t + np.pi/4)
x3 = x1 + x2

s = 1024
n = np.arange(s)
x1p = 5 * np.sin(2 * np.pi * 1000 * n / s + np.deg2rad(90))  
x2p = 10 * np.cos(2 * np.pi * 2000 * n / s + np.deg2rad(45))  
x3p = x1p + x2p
f = 1000  
bw = 20  
ha = np.ones(s)
for k in range(s):
    if abs(k - f) <= bw or abs(k + f) <= bw:
        ha[k] = 0

x3f = np.fft.ifft(fft(x3p) * ha)

s1 = 4096
n1 = np.arange(s1)
x1p1 = 5 * np.sin(2 * np.pi * 1000 * n1 / s1 + np.deg2rad(90))  
x2p1 = 10 * np.cos(2 * np.pi * 2000 * n1 / s1 + np.deg2rad(45))  
x3p1 = x1p1 + x2p1
f1 = 4000  
bw1 = 20  
ha1 = np.ones(s1)
for k1 in range(s1):
    if abs(k1 - f1) <= bw1 or abs(k1 + f1) <= bw1:
        ha1[k1] = 0

x3f1 = np.fft.ifft(fft(x3p1) * ha1)

yrect = np.convolve(x3, hrect, mode='same')
yhamming = np.convolve(x3, hhamming, mode='same')
yhanning = np.convolve(x3, hhanning, mode='same')

plt.figure(figsize=(14, 24))

plt.subplot(6, 2, 1)
plt.plot(n, x3p)
plt.title('Input Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

xf11 = dft(x3f1)
plt.subplot(6, 2, 2)
plt.stem(np.abs(xf11))
plt.title('Output signal')
plt.xlabel("time")
plt.ylabel("Amplitude")

plt.subplot(6, 2, 4)
plt.plot(t, yrect, label='Rectangular Window')
plt.title('Output Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(6, 2, 6)
plt.plot(t, yhamming, label='Hamming Window')
plt.title('Output Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(6, 2, 8)
plt.plot(t, yhanning, label='Hanning Window')
plt.title('Output Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(6, 2, 3)
plt.magnitude_spectrum(x3, Fs=fs, scale='dB')
plt.title('Input Signal Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

plt.subplot(6, 2, 5)
plt.magnitude_spectrum(yrect, Fs=fs, scale='dB')
plt.title('Output Signal Spectrum (Rectangular Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

plt.subplot(6, 2, 7)
plt.magnitude_spectrum(yhamming, Fs=fs, scale='dB')
plt.title('Output Signal Spectrum (HammingWindow)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

plt.subplot(6, 2, 9)
plt.magnitude_spectrum(yhanning, Fs=fs, scale='dB')
plt.title('Output Signal Spectrum (Hanning Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

plt.subplot(6, 2, 10)
plt.plot(w, np.abs(Hrect))
plt.title('Magnitude Response of Rectangular Window')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(6, 2, 11)
plt.plot(w, np.abs(Hhamming))
plt.title('Magnitude Response of Hamming Window')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(6, 2, 12)
plt.plot(w, np.abs(Hhanning))
plt.title('Magnitude Response of Hanning Window')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.figure(figsize=(14, 12))

plt.subplot(3, 2, 1)
plt.plot(w, np.angle(Hrect))
plt.title('Phase Response of Rectangular Window')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Phase (radians)')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(w, np.angle(Hhamming))
plt.title('Phase Response of Hamming Window')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Phase (radians)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(w, np.angle(Hhanning))
plt.title('Phase Response of Hanning Window')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Phase (radians)')
plt.grid(True)

plt.tight_layout()
plt.show()