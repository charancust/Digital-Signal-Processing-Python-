import numpy as np
import matplotlib.pyplot as plt

c = np.arange(-10,10)
d = np.arange(-10,10)

def uimp(t):
    return np.where(t==-5, 1, 0)

def ustep(t):
    return np.where(t>=7, 1, 0)

def expris(t):
    return np.exp(t)

def expdec(t):
    return np.exp(-t)

def rampris(t):
    return np.where(t>=0, t, 0)

def rampdec(t):
    return np.where(t>=0, -t, 0)

plt.subplot(7,2,1)
a = uimp(c)
plt.stem(c,a)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('unit impulse')

plt.subplot(7,2,2)
b = uimp(d)
plt.stem(d,b)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('unit impulse')

plt.subplot(7,2,3)
e = ustep(c)
plt.plot(c,e)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('unit step')

plt.subplot(7,2,4)
f = ustep(d)
plt.stem(d,f)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('unit step')

plt.subplot(7,2,5)
g = expris(c)
plt.plot(c,g)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('exponential rising')

plt.subplot(7,2,6)
h = expris(d)
plt.stem(d,h)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('exponential rising')

plt.subplot(7,2,7)
i = expdec(c)
plt.plot(c,i)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('exponential decay')

plt.subplot(7,2,8)
j = expdec(d)
plt.stem(d,j)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('exponential decay')

plt.subplot(7,2,9)
k = rampris(c)
plt.plot(c,k)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('rising ramp')

plt.subplot(7,2,10)
l = rampris(d)
plt.stem(d,l)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('rising ramp')

plt.subplot(7,2,11)
m = rampdec(c)
plt.plot(c,m)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('falling ramp')

plt.subplot(7,2,12)
n = rampdec(d)
plt.stem(d,n)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('falling ramp')

s = np.arange(0,5,0.1)
plt.subplot(7,2,13)
o = 5*np.sin((2*np.pi*s)+(np.pi/6))
plt.plot(s,o)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('sine')

plt.subplot(7,2,14)
o = 5*np.sin((2*np.pi*s)+(np.pi/6))
plt.stem(s,o)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('sine')


 

