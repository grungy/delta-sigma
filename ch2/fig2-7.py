import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
# f = fs / 8 --> 8 = fs/f
N = 600

# sample spacing
f_s = 10000.0  # sample / time
T = 1.0 / f_s  # time / sample --> N*T --> total_time
t = np.linspace(0.0, N*T, N)
time_step = T

#  sine wave stuff
frequency = 50
A = 16
y = A * np.sin(2 * np.pi * frequency * t)

# sub sampling
subf = 8  # every x samples
ys = y[::subf]
ts = t[::subf]

#  plotting parameters
sp = int(1.0 / (frequency / f_s))  # samples_in_one_period
ssp = int(sp/subf)  # samples in one period of sub sample

# quantizer parameters
delta = 2  # smallest step size -- LSB
M = 16  # number of steps / bins
lvls = M + 1 # number of levels
full_scale = 2 * (M / 2.0) * delta
input_range = ( -1 * ( (lvls) * delta)/2.0, (lvls)*delta/2.0)

# quantize sine wave
bins = np.arange(input_range[0], input_range[1] + 1, delta)  # add 1 because arange does not include right bound
inds = np.digitize(ys, bins, right=True)

y_quant = bins[inds] - 1  # shift bins by 1 because digitize function doesn't include left most part of bin

#  calculate error from quantization
error = ys - y_quant

max = np.amax(y_quant)
print("max: ", max)


# Plotting section

# plot the spectrum
yfsq = scipy.fftpack.fft(y_quant)
yf = scipy.fftpack.fft(y)
yfe = scipy.fftpack.fft(error)

time_step = ts[1] - ts[0]

NSQ = y_quant.size
NSQ2 = NSQ // 2

xfsq = scipy.fftpack.fftfreq(y_quant.size, d=T*subf)
xf = scipy.fftpack.fftfreq(y.size, d=T)
xfe = scipy.fftpack.fftfreq(error.size, d=T*subf)

yfsq_mag = 1.0/NSQ * np.abs(yfsq[:NSQ2])
dbfsq = 20.0 * np.log10(yfsq_mag / np.amax(yfsq_mag))
print(xfsq)

fig, ax = plt.subplots()
ax.stem(xfsq[:NSQ2], 20 * np.log(yfsq_mag), '.-')
ax.set_title("Quantized Signal Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

fig4, ax4 = plt.subplots()
ax4.stem(xf[:N//2], 1.0/N * np.abs(yf[:N//2]), '.-')
ax4.set_title("Original Signal Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

fig2, ax2 = plt.subplots(2, 1)

ax2[0].stem(ts[:ssp], y_quant[:ssp], '-.', label="quantized", use_line_collection=False)
ax2[0].plot(t[:sp], y[:sp], label="original")
ax2[1].stem(ts[:ssp], error[:ssp], '-.', label="error", use_line_collection=True)
fig2.legend()
plt.xlabel("time (s)")
ax2[0].set_ylabel("Amplitude")
ax2[0].set_title("Original Signal and Sub-sampled Quantized Signal")
ax2[1].set_title("Quantizer Error")

fig3, ax3 = plt.subplots()
ax3.plot(ts[:sp], y_quant[:sp], '.')
ax3.set_title("Quantized Signal")
plt.xlabel("time (s)")
plt.ylabel("Amplitude")
plt.show()