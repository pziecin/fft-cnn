import cv2
import numpy as np
from matplotlib import pyplot as py
import radialProfile as rp

img = cv2.imread('imageMP.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
print("FFT", f)
print("FFT shift", fshift)
# print(len(fshift), len(fshift[0]))
filtr = [0] * 100
#
# print(2000*np.log(np.abs(fshift)))
# print(filtr)

# fshift = fshift * filtr

magnitude_spectrum2 = np.abs(fshift)**2
magnitude_spectrum = 20*np.log(np.abs(fshift))

psd1D = rp.azimuthalAverage(magnitude_spectrum)

print(psd1D)

py.figure(1)
py.clf()
py.imshow(img)
py.figure(2)
py.clf()
py.imshow(magnitude_spectrum)
py.figure(3)
py.clf()
py.semilogy( psd1D )
py.xlabel("Spatial Frequency")
py.ylabel("Power Spectrum")
py.show()
# #
# plt.subplot(121),plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()