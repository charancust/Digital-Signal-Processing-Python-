import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack as ft

def titles():
    plt.xlabel('jw')
    plt.ylabel('Amplitude')

t = np.linspace(0,5)
n = (1/510)
s1 = 10*np.sin(2*np.pi*50*t)
s2 = 20*np.sin(2*np.pi*100*t + np.pi/4)
s3 = 30*np.sin(2*np.pi*150*t + np.pi/2)
plt.subplot(10,2,1)
plt.stem(t,s1)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('S1[n]')

plt.subplot(10,2,2)
ft1 = ft.fft(s1)
freq1 = np.fft.fftfreq(len(t), n)
mag1 = np.abs(ft1)/n
plt.plot(freq1,mag1)
titles()
plt.title('S1[jW]')

plt.subplot(10,2,3)
plt.stem(t,s2)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('S2[n]')

plt.subplot(10,2,4)
ft2 = ft.fft(s2)
freq2 = np.fft.fftfreq(len(t), n)
mag2 = np.abs(ft2)/n
plt.plot(freq2, mag2)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('S2[jW]')

plt.subplot(10,2,5)
plt.stem(t,s3)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('S3[n]')

plt.subplot(10,2,6)
ft3 = ft.fft(s3)
freq3 = np.fft.fftfreq(len(t), n)
mag3 = np.abs(ft3)/n
plt.plot(freq3, mag3)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('S3[jW]')

plt.subplot(10,2,7)
s4 = s1+s2
plt.stem(t,s4)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('S4[n]')

plt.subplot(10,2,8)
ft4 = ft.fft(s4)
freq4 = np.fft.fftfreq(len(t), n)
mag4 = np.abs(ft4)/n
plt.plot(freq4, mag4)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('S4[jW]')

plt.subplot(10,2,9)
s5 = s4-s3
plt.stem(t,s5)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('S5[n]')

plt.subplot(10,2,10)
ft5 = ft.fft(s5)
freq5 = np.fft.fftfreq(len(t), n)
mag5 = np.abs(ft5)/n
plt.plot(freq5, mag5)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('S5[jW]')

plt.subplot(10,2,11)
s6 = s1*s3
plt.stem(t,s6)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('S6[n]')

plt.subplot(10,2,12)
ft6 = ft.fft(s6)
freq6 = np.fft.fftfreq(len(t), n)
mag6 = np.abs(ft6)/n
plt.plot(freq6, mag6)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('S6[jW]')

plt.subplot(10,2,13)
s7 = s1+10
plt.stem(t,s7)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('Scaling')

plt.subplot(10,2,14)
ft7 = ft.fft(s7)
freq7 = np.fft.fftfreq(len(t), n)
mag7 = np.abs(ft7)/n
plt.plot(freq7, mag7)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('Scaling')

plt.subplot(10,2,15)
s8 = 30*np.sin(2*np.pi*150*t + (np.pi/2)*2)
plt.stem(t,s8)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('90 phase shift of S3')

plt.subplot(10,2,16)
ft8 = ft.fft(s8)
freq8 = np.fft.fftfreq(len(t), n)
mag8 = np.abs(ft8)/n
plt.plot(freq8, mag8)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('90 phase shift of S3')

plt.subplot(10,2,17)
s9 = 30*np.sin(2*np.pi*150*t + (np.pi/2)*3)
plt.stem(t,s9)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('180 phase shift of S3')

plt.subplot(10,2,18)
ft9 = ft.fft(s9)
freq9 = np.fft.fftfreq(len(t), n)
mag9 = np.abs(ft9)/n
plt.plot(freq9, mag9)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('180 phase shift of S3')

plt.subplot(10,2,19)
s10 = np.flip(s2)
plt.stem(t,s10)
plt.xlabel('n index')
plt.ylabel('Amplitude')
plt.title('S2[-n]')

plt.subplot(10,2,20)
ft10 = ft.fft(s10)
freq10 = np.fft.fftfreq(len(t), n)
mag10 = np.abs(ft10)/n
plt.plot(freq10, mag10)
plt.xlabel('jW')
plt.ylabel('Amplitude')
plt.title('S2[-jW]')