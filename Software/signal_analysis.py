import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def load_data(filename):
    # 读入数据文件
    data = np.loadtxt(filename)
    return data


def plot_signal_waveform(data, fs):
    # 绘制信号波形
    duration = len(data) / fs # 持续时间，单位为秒
    time = np.linspace(0, duration, len(data))
    plt.subplot(3,1,1)
    plt.plot(time, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original Signal")


def plot_stft_spectrogram(data, fs, window, nperseg, noverlap):
    # 进行STFT
    f, t, Zxx = signal.stft(data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # 绘制时频图
    plt.subplot(3,1,2)
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='YlOrBr')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')


def plot_fft_magnitude(data, fs):
    # 进行FFT
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_data), 1/fs)

    # 绘制FFT图
    plt.subplot(3,1,3)
    plt.plot(freqs, np.abs(fft_data))
    plt.title('FFT Magnitude')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [Hz]')


if __name__ == '__main__':
    filename = 'Software\data\\1.csv'
    data = load_data(filename)
    fs = 1000
    window = signal.windows.hann(128) # 窗函数
    nperseg = 128 # STFT段长
    noverlap = nperseg//2 # STFT重叠长度

    plot_signal_waveform(data, fs)
    plot_stft_spectrogram(data, fs, window, nperseg, noverlap)
    plot_fft_magnitude(data, fs)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()
