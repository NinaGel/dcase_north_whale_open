import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch


class Act:
    """
    ACT类用于计算信号的短时傅里叶变换（STFT）和逆变换（ISTFT），
    以及频谱图和Chirp率的计算。
    """

    def __init__(self, windowSize=1024, hopSize=256, windowType='hann', fftSize=0):
        """
        初始化ACT类的实例。

        参数：
        windowSize -- 窗口大小（样本数）
        hopSize -- 步长（重叠量）
        windowType -- 窗口类型（默认为汉宁窗）
        fftSize -- FFT大小（默认为大于等于windowSize的下一个2的整数次幂）
        """
        if fftSize != 0 and not (fftSize & (fftSize - 1) == 0):
            raise ValueError("FFT size must be a power of 2!")
        
        if fftSize >= windowSize:
            self._fftSize = fftSize
        else:
            self._fftSize = self.next_power_of_two(windowSize)
        self._hopSize = hopSize
        self._windowSize = windowSize
        self._windowType = windowType
        # 获取窗口函数的样本值
        self._windowSamples = np.hanning(self._windowSize)
        # 计算逆STFT的归一化增益
        self._gain = 1 / (self._fftSize * np.sum(self._windowSamples ** 2) / self._hopSize)

    @staticmethod
    def next_power_of_two(n):
        """
        计算大于等于n的下一个2的整数次幂。

        参数：
        n -- 输入整数

        返回值：
        大于等于n的下一个2的整数次幂
        """
        return 1 << (n - 1).bit_length()

    def Direct(self, samples):
        """
        计算信号的直接STFT（短时傅里叶变换）。

        参数：
        samples -- 输入信号的样本数组

        返回值：
        STFT的列表，每个元素是一个包含实部数组和虚部数组的元组
        """
        scct = []
        # 遍历信号，按步长进行窗函数分帧
        for pos in range(0, len(samples) - self._windowSize, self._hopSize):
            # 初始化实部数组
            re = np.zeros(self._fftSize)
            # 复制当前帧的信号到实部数组中
            re[:self._windowSize] = samples[pos:pos + self._windowSize]
            # 如果使用的不是矩形窗，则应用窗口函数
            if self._windowType != 'boxcar':
                re[:self._windowSize] *= self._windowSamples
            # 计算FFT
            fft_result = np.fft.fft(re, n=self._fftSize)
            # 将实部和虚部存储起来
            scct.append((fft_result.real, fft_result.imag))
        return scct

    def Inverse(self, stft):
        """ISTFT实现可以优化"""
        spectraCount = len(stft)
        output = np.zeros(spectraCount * self._hopSize + self._windowSize)
        pos = 0
        for i in range(spectraCount):
            # 可以简化复数处理
            complex_spectrum = stft[i][0] + 1j * stft[i][1]
            ifft_result = np.fft.ifft(complex_spectrum).real[:self._windowSize]
            
            # 应用窗口并重叠相加
            output[pos:pos + self._windowSize] += ifft_result * self._windowSamples
            pos += self._hopSize
            
            # 实时归一化处理，参考C#版本
            if pos >= self._hopSize:
                output[pos-self._hopSize:pos] *= self._gain
                
        # 处理最后一个窗口
        output[pos:pos + self._windowSize] *= self._gain
        return output

    def Spectrogram(self, samples):
        """计算频谱图，温和增强信号特征"""
        spectrogram = []
        
        # 非常温和的预加重，仅轻微增强高频
        pre_emphasis = 0.97
        emphasized_samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])
        
        for pos in range(0, len(emphasized_samples) - self._windowSize, self._hopSize):
            frame = emphasized_samples[pos:pos + self._windowSize].copy()
            
            if self._windowType != 'boxcar':
                # 使用汉宁窗，提供更好的频率分辨率
                frame *= self._windowSamples
                
            # 进行FFT并计算功率谱
            fft_result = np.fft.fft(frame, n=self._fftSize)
            
            # 只取到2kHz的频率范围
            max_freq_bin = int(self._fftSize * 2000 / 8000)  # 2kHz对应的bin数
            power_spectrum = np.abs(fft_result[:max_freq_bin]) ** 2
            
            # 非常温和的频率加权，主要增强中频段
            n_freqs = len(power_spectrum)
            freq_weights = np.concatenate([
                np.ones(n_freqs//3),                    # 低频段保持不变 (0-667Hz)
                np.linspace(1, 1.1, n_freqs//3),        # 中频段轻微增强 (667-1333Hz)
                np.linspace(1.1, 1, n_freqs - 2*(n_freqs//3))  # 高频段渐变回正常 (1333-2000Hz)
            ])
            power_spectrum *= freq_weights
            
            # 使用更温和的对数压缩
            power_spectrum = np.log(1 + power_spectrum)
            spectrogram.append(power_spectrum)
        
        # 转换为numpy数组
        spectrogram = np.array(spectrogram).T
        
        # 使用更保守的动态范围压缩
        p_low, p_high = np.percentile(spectrogram, [1, 99])
        spectrogram = np.clip(spectrogram, p_low, p_high)
        
        # 归一化到[0,1]区间
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        
        return spectrogram

    def ChirpRate(self, siglen, fs, winlen):
        """
        计算Chirp率，并调整窗口函数的样本值。

        参数：
        siglen -- 信号长度（样本数）
        fs -- 采样率
        winlen -- 窗口长度
        """
        t = siglen / fs
        # 创建大小为winlen的数组，所有元素值为π/4
        slope = np.full(winlen, np.pi / 4)
        # 初始化chirp rate
        # 初始化chirplet window（未使用）
        for j in range(winlen):
            chirprate = np.tan(slope[j]) * (fs / 2) / t
            # 计算复指数
            exponent = -1j * np.pi * chirprate
            # 调整窗口样本值
            self._windowSamples[j] *= np.exp(exponent).real
            # 上面取实部，因为原代码中将结果转换为float
            # 如果需要保留复数部分，可以修改代码

    @property
    def size(self):
        """添加属性访问器，与C#版本保持一致"""
        return self._fftSize


def process_audio_to_features(filepath_wav, n_fft=1024, hop_length=256, window_type='hann'):
    """改进的特征处理函数"""
    # 读取音频文件
    samples, sr = librosa.load(filepath_wav, sr=None)
    
    # 创建ACT实例
    act = Act(windowSize=n_fft, hopSize=hop_length, windowType=window_type, fftSize=n_fft)
    
    # 计算频谱图
    spectrogram = act.Spectrogram(samples)
    
    # 标准化处理
    scaler = StandardScaler()
    scaled_spectrogram = scaler.fit_transform(spectrogram)
    
    # 转换为PyTorch张量
    input_tensor = torch.tensor(scaled_spectrogram, dtype=torch.float32)
    
    return input_tensor, spectrogram, sr


def plot_spectrogram(spectrogram, sr, hop_length, n_fft):
    """改进的频谱图绘制函数"""
    time_axis = np.linspace(0, len(spectrogram[0]) * hop_length / sr, len(spectrogram[0]))
    freq_axis = np.linspace(0, 2, spectrogram.shape[0])  # 频率以kHz为单位，最大2kHz
    
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower', 
              cmap='viridis',
              interpolation='nearest',
              extent=[0, time_axis[-1], 0, freq_axis[-1]])
    
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.title('ACT Time-Frequency Spectrogram')
    
    # 更温和的颜色范围调整
    plt.clim(0.05, 0.95)
    
    plt.show()


# 主程序
if __name__ == "__main__":

    filepath_wav = 'scaper_audio/snr_scaper_audio/test/snr_high/wav/soundscape_whale_test_146.wav'  # 替换为您的音频文件路径
    # filepath_wav = 'scaper_audio/scaper_k_fold/fold_1/train/wav/overlap_731_40_87.wav'

    # 处理音频文件并提取特征，返回处理后的特征张量、频谱图和采样率
    features, processed_spectrogram, sr = process_audio_to_features(filepath_wav, n_fft=1024, hop_length=256,
                                                                    window_type='hann')

    # 打印特征张量的形状
    print("Processed features tensor shape:", features.shape)

    # 绘制频谱图
    plot_spectrogram(processed_spectrogram, sr, hop_length=256, n_fft=1024)
