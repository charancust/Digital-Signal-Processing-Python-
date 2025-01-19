import numpy as np
from scipy.signal import butter, freqs, TransferFunction, bilinear, residue, freqz
import matplotlib.pyplot as plt
import math

def bw (N, wc, analog = True):
    if not analog:
        raise ValueError("Only analog")
    k = np.arange(1, N+1)
    theta = (2*k-1)*np.pi/(2*N)
    poles = wc*np.exp(1j*theta)
    p = np.poly(poles.real)
    k = np.prod(-poles)
    b = [k.real]
    a = np.real(p)
    return b,a

wp = 200
ws = 600
Ap = 5
As = 40
aplin = 10**(0.1*Ap)
aslin = 10**(0.1*As)
wpc = wp/(2*np.pi)
wsc = ws/(2*np.pi)

N1 = np.log10(((aslin-1)/(aplin-1))**0.5)
N2 = (N1/(np.log10(ws/wp)))
N = np.ceil(N2)

wc = wp/((aplin-1)**(1/(2*N)))
print("Filter order is", N)
print("Cutoff frequency is", wc)

b,a = bw(N, wc, analog = True)

print("transfer funtion H(s):")
print("H(s) = ", end = "")
for i in range (len(b)):
    if i == len(b) - 1:
        print(f"({b[i]:.5f}*s^{len(b)-1-i})", end = "")
    else:
        print(f"({b[i]:.5f}*s^{len(b)-1-i})"+"+", end = "")
    print("\n")

w, h = freqs(b, a)

plt.subplot(221)
plt.semilogx(w, 20*np.log10(abs(h)))
plt.title("Butterworth filter frequency response")
plt.xlabel("Frequency [rad/sec]")
plt.ylabel("Amplitude [dB]")
plt.grid(which = 'both', axis = 'both')
plt.axvline(wp, color = 'r', linestyle = '--', label = 'Passband edge')
plt.axvline(ws, color = 'b', linestyle = '--', label = 'Stopband edge')
plt.legend()
plt.show()

numct = [1]
denct = [1, 1]
sr = 90
fcc = 15
t = 1/sr

def bt(s):
    return (2/t)*(1-1/np.exp(s*t))/(1+1/np.exp(s*t))

numdt = [numct[0]/denct[1]]
a1, a2 = denct[1], denct[0]
b1 = a1-a2
b0 = a2
dendt = [b0, b1]
wp1 = 2*sr*np.tan(np.pi*fcc/sr)
wc1 = 2*np.pi*fcc
wcpre = 2/t*np.tan(wc*t/2)

plt.subplot(222)
plt.axvline(wcpre, color = 'r', linestyle = '--', label = 'Pre-warped cutoff frequency')
plt.xlabel('Frequency [rad/sec]')
plt.ylabel('Magnitude')
plt.title("Frequency pre-warping")
plt.legend()
plt.show()    

print("Pre-warped cutoff frequency = ", wcpre)
print("\nDigital filter numerator coefficients = ", numdt)
print("\nDigital filter denominator coefficients = ", dendt)

r,p,_ = residue(numdt, dendt)
t1 = 1
t2 = 0.1

bz1, az1 = bilinear(numdt, dendt, fs = 1/t1)
bz2, az2 = bilinear(numdt, dendt, fs = 1/t2)
w1, h1 = freqz(bz1, az1)
w2, h2 = freqz(bz2, az2)
plt.subplot(223)
plt.plot(w1, 20*np.log10(abs(h1)))
plt.title("Frequency response for t=1 second")
plt.xlabel('Frequency')
plt.ylabel("Amplitude")
plt.subplot(224)
plt.plot(w2, 20*np.log10(abs(h2)))
plt.title("Frequency response for t=0.1 second")
plt.xlabel('Frequency')
plt.ylabel("Amplitude")
plt.show()
plt.tight_layout()


