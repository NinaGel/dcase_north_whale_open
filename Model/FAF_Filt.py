"""
FAF-Filt: 频率感知傅里叶滤波器 / Frequency-aware Fourier Filter for Sound Event Detection
基于 / Based on: ICASSP 2025论文 - Sun et al., ByteDance

核心组件 / Key Components:
1. FrequencyAwareFourierFilter: 傅里叶域可学习滤波器 / Learnable Fourier domain filter
2. FrequencyAdaptiveConv: 频率自适应卷积 / Frequency-adaptive convolution
3. FAF_Filt_Model: 完整CRNN模型 / Complete CRNN model

架构 / Architecture:
Conv2d → BN → CG → AvgPool → [FAF-Filt → FA-Conv → BN → CG → AvgPool] × 4 → BiGRU → Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# try:
#     import config_dcase as cfg
# except ImportError:
#     cfg = None  # Will be set when imported from training script

from Model.FeatureProjection import FrequencyProjection


class FrequencyAwareFourierFilter(nn.Module):
    """频率感知傅里叶滤波器 / Frequency-aware Fourier Filter (FAF-Filt)

    在傅里叶域进行可学习滤波
    Performs learnable filtering in the Fourier domain

    流程 / Pipeline: LayerNorm → 2D FFT → 频率自适应滤波 → 2D IFFT → MLP

    Args:
        num_channels: 通道数 / number of channels
        num_freq_bins: 频率bin数 / number of frequency bins
        reduction_ratio: SE模块压缩比 / SE module reduction ratio
    """
    def __init__(self, num_channels, num_freq_bins, reduction_ratio=4):
        super(FrequencyAwareFourierFilter, self).__init__()

        self.num_channels = num_channels
        self.num_freq_bins = num_freq_bins
        self.freq_bins_fft = num_freq_bins // 2 + 1  # F/2+1 for rfft2

        # Layer normalization as in paper (Equation 1)
        # Normalize over channel dimension
        self.ln1 = nn.LayerNorm(num_channels)
        self.ln2 = nn.LayerNorm(num_channels)

        # Frequency-aware coefficient network (SE-like module for computing e)
        # Paper: avgpool(x) along C and T → Linear(F→F/r) → ReLU → Linear(F/r→F/2+1) → Sigmoid
        # Output: e ∈ R^(1×1×(F/2+1))
        self.freq_coeff_net = nn.Sequential(
            nn.Linear(num_freq_bins, num_freq_bins // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(num_freq_bins // reduction_ratio, self.freq_bins_fft),
            nn.Sigmoid()
        )

        # Learnable complex-valued filter F ∈ C^(C×1×(F/2+1)) (Equation 2)
        # Paper doesn't specify initialization, using small random values
        self.complex_filter_real = nn.Parameter(torch.randn(num_channels, 1, self.freq_bins_fft) * 0.02)
        self.complex_filter_imag = nn.Parameter(torch.randn(num_channels, 1, self.freq_bins_fft) * 0.02)

        # MLP after IFFT
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 2),
            nn.GELU(),
            nn.Linear(num_channels * 2, num_channels)
        )

    def forward(self, x):
        """
        Args:
            x: Input feature [batch, C, T, F] or [batch, C, F, T]
        Returns:
            out: Filtered feature [batch, C, T, F]
        """
        # Standardize input format to [batch, C, T, F]
        if x.dim() == 4:
            batch, C, F, T = x.shape
            # Permute to [batch, C, T, F] if needed
            if F == self.num_freq_bins:
                x = x.permute(0, 1, 3, 2)  # [batch, C, T, F]

        batch, C, T, F = x.shape
        assert C == self.num_channels and F == self.num_freq_bins, \
            f"Expected channels={self.num_channels}, freq={self.num_freq_bins}, got {C}, {F}"

        residual = x

        # 1. Layer normalization (Paper Equation 1: X = F(LN(x)))
        # LayerNorm expects input: [B, T, F, C], normalize over C
        x_perm = x.permute(0, 2, 3, 1)  # [batch, T, F, C]
        x_norm = self.ln1(x_perm)  # [batch, T, F, C]
        x_norm = x_norm.permute(0, 3, 1, 2)  # [batch, C, T, F]

        # 2. 2D FFT along time and frequency dimensions
        # rfft2 performs FFT along last 2 dimensions (T, F)
        # Note: cuFFT requires FP32 for non-power-of-2 dimensions
        dtype_before = x_norm.dtype
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.float()

        X_fft = torch.fft.rfft2(x_norm, dim=(-2, -1))  # [batch, C, T_fft, F_fft]
        # T_fft = T, F_fft = F/2+1

        # 3. Calculate frequency-adaptive coefficients e ∈ [1, 1, F/2+1]
        # Avgpool along channel and time: [batch, C, T, F] → [batch, 1, 1, F]
        x_pooled = x.mean(dim=(1, 2), keepdim=False)  # [batch, F]
        freq_coeffs = self.freq_coeff_net(x_pooled)  # [batch, F/2+1]
        freq_coeffs = freq_coeffs.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, F/2+1]

        # 4. Construct learnable complex filter F
        complex_filter = torch.complex(self.complex_filter_real, self.complex_filter_imag)  # [C, 1, F/2+1]
        complex_filter = complex_filter.unsqueeze(0)  # [1, C, 1, F/2+1]

        # 5. Frequency-aware filter H = e ⊙ F
        # freq_coeffs: [batch, 1, 1, F/2+1], complex_filter: [1, C, 1, F/2+1]
        H = freq_coeffs * complex_filter  # [batch, C, 1, F/2+1]

        # 6. Apply filter: X̂ = H ⊙ X
        X_filtered = H * X_fft  # [batch, C, T_fft, F_fft]

        # 7. 2D IFFT back to time-frequency domain
        x_ifft = torch.fft.irfft2(X_filtered, s=(T, F), dim=(-2, -1))  # [batch, C, T, F]

        # Convert back to original dtype if needed
        if dtype_before == torch.float16:
            x_ifft = x_ifft.half()

        # 8. Layer normalization + MLP (Paper: after IFFT)
        # Permute to [batch, T, F, C] for LayerNorm and MLP
        x_ifft = x_ifft.permute(0, 2, 3, 1)  # [batch, T, F, C]
        x_ifft = self.ln2(x_ifft)  # [batch, T, F, C]

        # MLP operates on channel dimension
        x_mlp = self.mlp(x_ifft)  # [batch, T, F, C]
        x_mlp = x_mlp.permute(0, 3, 1, 2)  # [batch, C, T, F]

        # Residual connection
        out = residual + x_mlp

        return out


class FrequencyAdaptiveConv(nn.Module):
    """
    Frequency-Adaptive Convolution (FA-Conv)

    Applies frequency-dependent attention to inputs and outputs of 2D convolution.

    Pipeline:
    xˆ = Aout ⊙ (W(Ain ⊙ x))

    Where:
    - Ain: Input frequency attention [Cin, 1, F]
    - Aout: Output frequency attention [Cout, 1, F]
    - W: Standard 2D convolution weights

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Convolution kernel size
        num_freq_bins (int): Number of frequency bins
        stride (int or tuple): Convolution stride (default: 1)
        padding (int or tuple): Convolution padding (default: 0)
        reduction_ratio (int): Channel reduction ratio for attention (default: 4)
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_freq_bins,
                 stride=1, padding=0, reduction_ratio=4):
        super(FrequencyAdaptiveConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_freq_bins = num_freq_bins

        # Calculate output frequency dimension after convolution
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        # For freq dimension (assuming kernel operates on freq axis)
        self.out_freq_bins = (num_freq_bins + 2 * padding[0] - kernel_size[0]) // stride[0] + 1

        # Input frequency attention (SE-like module)
        # Paper: Avgpool over time → Conv1d → BN → ReLU → Conv1d → Sigmoid
        # Output: Ain ∈ R^(Cin×1×F)
        self.input_freq_squeeze = nn.Conv1d(in_channels, max(1, in_channels // reduction_ratio), 1)
        self.input_freq_bn = nn.BatchNorm1d(max(1, in_channels // reduction_ratio))
        self.input_freq_excite = nn.Conv1d(max(1, in_channels // reduction_ratio), in_channels, 1)

        # Standard 2D convolution (W in paper)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)

        # Output frequency attention (SE-like module)
        # Paper: Avgpool over time → Conv1d → BN → ReLU → Conv1d → Sigmoid
        # Output: Aout ∈ R^(Cout×1×F')
        self.output_freq_squeeze = nn.Conv1d(out_channels, max(1, out_channels // reduction_ratio), 1)
        self.output_freq_bn = nn.BatchNorm1d(max(1, out_channels // reduction_ratio))
        self.output_freq_excite = nn.Conv1d(max(1, out_channels // reduction_ratio), out_channels, 1)

    def forward(self, x):
        """
        Paper Equation 5: x̂ = Aout ⊙ (W(Ain ⊙ x))

        Args:
            x: Input feature [batch, Cin, F, T] or [batch, Cin, T, F]
        Returns:
            out: Output feature after FA-Conv [batch, Cout, F', T']
        """
        batch, C, dim1, dim2 = x.shape

        # Standardize input to [B, C, F, T] (frequency first)
        if dim1 == self.num_freq_bins:
            is_freq_first = True
        else:
            is_freq_first = False
            # Permute to [B, C, F, T]
            x = x.permute(0, 1, 3, 2)

        # 1. Calculate input frequency attention Ain ∈ R^(Cin×1×F)
        # Avgpool over time dimension: [batch, Cin, F, T] → [batch, Cin, F, 1]
        x_pool_in = nn.functional.adaptive_avg_pool2d(x, (self.num_freq_bins, 1))
        x_pool_in = x_pool_in.squeeze(-1)  # [batch, Cin, F]

        # SE-like module: Conv1d → BN → ReLU → Conv1d → Sigmoid
        Ain = self.input_freq_squeeze(x_pool_in)  # [batch, Cin/r, F]
        Ain = self.input_freq_bn(Ain)
        Ain = F.relu(Ain, inplace=True)
        Ain = self.input_freq_excite(Ain)  # [batch, Cin, F]
        Ain = torch.sigmoid(Ain)
        Ain = Ain.unsqueeze(-1)  # [batch, Cin, F, 1]

        # 2. Apply input attention: Ain ⊙ x (element-wise multiplication)
        x_weighted = Ain * x  # [batch, Cin, F, T]

        # 3. Standard 2D convolution: W(Ain ⊙ x)
        x_conv = self.conv(x_weighted)  # [batch, Cout, F', T']

        # 4. Calculate output frequency attention Aout ∈ R^(Cout×1×F')
        _, _, freq_out, time_out = x_conv.shape
        x_pool_out = nn.functional.adaptive_avg_pool2d(x_conv, (freq_out, 1))
        x_pool_out = x_pool_out.squeeze(-1)  # [batch, Cout, F']

        # SE-like module: Conv1d → BN → ReLU → Conv1d → Sigmoid
        Aout = self.output_freq_squeeze(x_pool_out)  # [batch, Cout/r, F']
        Aout = self.output_freq_bn(Aout)
        Aout = F.relu(Aout, inplace=True)
        Aout = self.output_freq_excite(Aout)  # [batch, Cout, F']
        Aout = torch.sigmoid(Aout)
        Aout = Aout.unsqueeze(-1)  # [batch, Cout, F', 1]

        # 5. Apply output attention: Aout ⊙ (W(Ain ⊙ x))
        out = Aout * x_conv  # [batch, Cout, F', T']

        # Permute back to original format if needed
        if not is_freq_first:
            out = out.permute(0, 1, 3, 2)  # [batch, Cout, T', F']

        return out


class ContextGating(nn.Module):
    """Context Gating module from Mean Teacher DCASE 2018"""
    def __init__(self, num_channels):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        """
        Args:
            x: [batch, channels, freq, time]
        Returns:
            gated_x: [batch, channels, freq, time]
        """
        # Global average pooling
        batch, C, F, T = x.shape
        x_pooled = x.mean(dim=(-2, -1))  # [batch, C]

        # Gating weights
        gates = torch.sigmoid(self.fc(x_pooled))  # [batch, C]
        gates = gates.unsqueeze(-1).unsqueeze(-1)  # [batch, C, 1, 1]

        return x * gates


class FAF_Filt_Model(nn.Module):
    """
    Complete FAF-Filt Model for Sound Event Detection

    Architecture (based on Figure 1 in paper):
    Input MelSpectrogram → [Optional: Projection 512->128] → Conv2d → BN → CG → AvgPool
    → [FAF-Filt → FA-Conv → BN → CG → AvgPool] × 4
    → BiGRU → Linear → Prediction

    Args:
        num_classes (int): Number of output classes (default: 10 for DCASE)
        input_freq_bins (int): Number of input frequency bins (default: 128 for LogMel)
        conv_channels (list): List of conv channel numbers for each block
        gru_hidden (int): GRU hidden size
        gru_layers (int): Number of GRU layers
        reduction_ratio (int): Reduction ratio for FAF-Filt and FA-Conv
        use_projection (bool): Whether to use frequency projection (512->128)
        projection_method (str): Projection method ('adaptive', 'conv1d', 'linear')
    """
    def __init__(self, num_classes=10, input_freq_bins=128,
                 conv_channels=[32, 64, 128, 128, 128],
                 gru_hidden=128, gru_layers=2, reduction_ratio=4,
                 use_projection=False, projection_method='conv1d',
                 projection_target=128):
        super(FAF_Filt_Model, self).__init__()

        self.num_classes = num_classes
        self.original_freq_bins = input_freq_bins
        self.use_projection = use_projection

        # 频率降维（可选）
        if use_projection and input_freq_bins > projection_target:
            print(f"  [投影] {input_freq_bins} -> {projection_target} 频率维度 (方法: {projection_method})")
            self.freq_projection = FrequencyProjection(
                input_freq=input_freq_bins,
                output_freq=projection_target,
                method=projection_method
            )
            self.input_freq_bins = projection_target
        else:
            self.freq_projection = None
            self.input_freq_bins = input_freq_bins

        # First convolution block (no FAF-Filt/FA-Conv)
        self.conv1 = nn.Conv2d(1, conv_channels[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.cg1 = ContextGating(conv_channels[0])
        self.pool1 = nn.AvgPool2d((2, 1))  # Pool frequency by 2

        # Subsequent blocks with FAF-Filt and FA-Conv
        self.blocks = nn.ModuleList()
        freq_bins = self.input_freq_bins // 2  # After first pooling (use projected freq if enabled)

        for i in range(len(conv_channels) - 1):
            in_ch = conv_channels[i]
            out_ch = conv_channels[i + 1]

            # FAF-Filt
            faf_filt = FrequencyAwareFourierFilter(in_ch, freq_bins, reduction_ratio)

            # FA-Conv
            fa_conv = FrequencyAdaptiveConv(in_ch, out_ch, kernel_size=3,
                                           num_freq_bins=freq_bins,
                                           stride=1, padding=1,
                                           reduction_ratio=reduction_ratio)

            # BN, CG, Pool
            bn = nn.BatchNorm2d(out_ch)
            cg = ContextGating(out_ch)
            pool = nn.AvgPool2d((2, 1))  # Pool frequency by 2

            self.blocks.append(nn.ModuleDict({
                'faf_filt': faf_filt,
                'fa_conv': fa_conv,
                'bn': bn,
                'cg': cg,
                'pool': pool
            }))

            freq_bins = freq_bins // 2  # Update freq bins after pooling

        # Final feature dimension after all pooling
        # self.input_freq_bins → ... → final_freq_bins (after all pooling)
        self.final_freq_bins = self.input_freq_bins // (2 ** len(conv_channels))
        self.final_channels = conv_channels[-1]
        self.gru_input_size = self.final_freq_bins * self.final_channels

        # BiGRU
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if gru_layers > 1 else 0
        )

        # Final classifier
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)  # *2 for bidirectional

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input LogMel spectrogram [batch, freq, time] or [batch, 1, freq, time]
        Returns:
            output: Frame-level predictions [batch, time, num_classes]
        """
        # Handle input format: standardize to [batch, 1, freq, time]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, freq, time]

        batch, _, freq, time = x.shape

        # Frequency projection (optional)
        if self.freq_projection is not None:
            # Current: [batch, 1, freq_original, time]
            assert freq == self.original_freq_bins, \
                f"Expected freq={self.original_freq_bins}, got {freq}"
            x = self.freq_projection(x)  # [batch, 1, freq_projected, time]
            freq = self.input_freq_bins
        else:
            assert freq == self.input_freq_bins, \
                f"Expected freq={self.input_freq_bins}, got {freq}"

        # First conv block (no FAF-Filt/FA-Conv)
        # Paper Figure 1: Input → Conv2d → BN → CG → AvgPool
        x = self.conv1(x)  # [batch, C1, F, T]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.cg1(x)
        x = self.pool1(x)  # [batch, C1, F/2, T]

        # Subsequent blocks with FAF-Filt and FA-Conv
        # Paper Figure 1: [FAF-Filt → FA-Conv → BN → CG → AvgPool] × 4
        for block in self.blocks:
            batch, C, F_cur, T_cur = x.shape

            # FAF-Filt expects [batch, C, T, F]
            # Current format: [batch, C, F, T]
            x_perm = x.permute(0, 1, 3, 2)  # [batch, C, T, F]
            x_perm = block['faf_filt'](x_perm)  # [batch, C, T, F]
            x = x_perm.permute(0, 1, 3, 2)  # [batch, C, F, T]

            # FA-Conv expects [batch, C, F, T] or [batch, C, T, F]
            # We pass [batch, C, F, T], it will handle the format
            x = block['fa_conv'](x)  # [batch, C', F', T]
            x = block['bn'](x)
            x = F.relu(x)
            x = block['cg'](x)
            x = block['pool'](x)  # [batch, C', F'/2, T]

        # Reshape for GRU: [batch, C, F, T] → [batch, T, C*F]
        batch, C, F_final, T_final = x.shape
        x = x.permute(0, 3, 1, 2)  # [batch, T, C, F]
        x = x.reshape(batch, T_final, -1)  # [batch, T, C*F]

        # BiGRU
        x, _ = self.gru(x)  # [batch, T, gru_hidden*2]

        # Classifier
        output = self.classifier(x)  # [batch, T, num_classes]

        return output


if __name__ == "__main__":
    """Test FAF-Filt model"""
    print("Testing FAF-Filt Model...")

    # Test FrequencyAwareFourierFilter
    print("\n1. Testing FrequencyAwareFourierFilter...")
    faf_filt = FrequencyAwareFourierFilter(num_channels=32, num_freq_bins=64)
    x_faf = torch.randn(4, 32, 309, 64)  # [batch, C, T, F]
    out_faf = faf_filt(x_faf)
    print(f"   Input shape: {x_faf.shape}")
    print(f"   Output shape: {out_faf.shape}")
    assert out_faf.shape == x_faf.shape, "FAF-Filt shape mismatch!"
    print("   [PASS] FrequencyAwareFourierFilter test passed!")

    # Test FrequencyAdaptiveConv
    print("\n2. Testing FrequencyAdaptiveConv...")
    fa_conv = FrequencyAdaptiveConv(32, 64, kernel_size=3, num_freq_bins=64, padding=1)
    x_conv = torch.randn(4, 32, 64, 309)  # [batch, C, F, T]
    out_conv = fa_conv(x_conv)
    print(f"   Input shape: {x_conv.shape}")
    print(f"   Output shape: {out_conv.shape}")
    assert out_conv.shape[1] == 64, "FA-Conv channel mismatch!"
    print("   [PASS] FrequencyAdaptiveConv test passed!")

    # Test complete model
    print("\n3. Testing complete FAF_Filt_Model...")
    model = FAF_Filt_Model(num_classes=10, input_freq_bins=128)
    x_model = torch.randn(4, 128, 309)  # [batch, freq, time]
    out_model = model(x_model)
    print(f"   Input shape: {x_model.shape}")
    print(f"   Output shape: {out_model.shape}")
    assert out_model.shape == (4, 309, 10), f"Model output shape mismatch! Expected (4, 309, 10), got {out_model.shape}"
    print("   [PASS] Complete model test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n4. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    print("\n[SUCCESS] All tests passed successfully!")
