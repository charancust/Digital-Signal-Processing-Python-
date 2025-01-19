import matplotlib.pyplot as plt
import numpy as np
def dft(x):
    N=len(x)
    X=np.zeros(N,dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k]+=x[n]*np.exp(-2j*k*n*np.pi/N)
    return X

def idft(X):
    N=len(X)
    x=np.zeros(N,dtype=complex)
    for n in range(N):
        for k in range(N):
           x[n]+=X[k]*np.exp(2j*k*n*np.pi/N)
    return x

N=32
x_axis=np.arange(0,N,1)
Fs=320
freq1=(Fs/N)*np.arange(-N/2,0,1)
freq2=(Fs/N)*np.arange(0,N/2,1)
freq=np.concatenate((freq2,freq1))

s1=5*np.sin(2*np.pi*20*x_axis/Fs+(np.pi/2))
s2=10*np.sin(2*np.pi*30*x_axis/Fs+(np.pi))
s3=s1+s2

def labels():
    plt.ylabel("Amplitude")

plt.subplot(421)
plt.plot(s1)
plt.title('s1')
labels()

plt.subplot(422)
plt.plot(s2)
plt.title('s2')
labels()

plt.subplot(423)
plt.plot(s3)
plt.title("s3 = s1+s2")
labels()

plt.subplot(424)
X=dft(s3)
plt.stem(freq,abs(X)/N)
plt.title("dft using own algorithm")
labels()

plt.subplot(425)
xft=np.fft.fft(s3)
plt.stem(freq,abs(xft)/N)
plt.title("dft using in built")
labels()

plt.subplot(426)
xift=idft(X)
plt.stem(xift/N)
plt.title("idft using own algorithm")
labels()

plt.subplot(427)
x3=np.fft.ifft(X)
plt.stem(x3)
plt.title("idft using in built")
labels()

plt.tight_layout()
plt.show()
