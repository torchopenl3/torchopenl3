import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor as T
import numpy as np
from torch.nn.functional import conv1d

import torchopenl3.core


class CustomSTFT(nn.Module):
    """
    STFT implemented like kapre 0.1.4.
    Attributes
    ----------
      n_dft: int
        The window size for the STFT
      n_hop: int
        The hop (or stride) size
      power_spectrogram: float
        2.0 to get power spectrogram, 1.0 to get amplitude spectrogram.
      return_decibel_spectrogram: bool
        Whether to return in decibel or not, i.e. returns
        log10(amplitude spectrogram) if True

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.

    Examples
    --------
    >>> stftlayer = CustomSTFT(n_dft = 512, n_hop = 242,
        power_spectrogram = 2.0,
        return_decibel_spectrogram=False)
    >>> stftlayer = speclayer(x)
    """

    def __init__(
        self,
        n_dft=512,
        n_hop=None,
        power_spectrogram=2.0,
        return_decibel_spectrogram=False,
    ):

        super().__init__()
        if n_hop is None:
            n_hop = n_dft // 2
        self.n_dft = n_dft
        self.n_hop = n_hop
        self.power_spectrogram = float(power_spectrogram)
        self.return_decibel_spectrogram = return_decibel_spectrogram

        dft_real_kernels, dft_imag_kernels = self.get_stft_kernels(self.n_dft)
        self.register_buffer(
            "dft_real_kernels",
            T(dft_real_kernels, requires_grad=False, dtype=torch.float32)
            .squeeze(1)
            .swapaxes(0, 2),
        )
        self.register_buffer(
            "dft_imag_kernels",
            T(dft_imag_kernels, requires_grad=False, dtype=torch.float32)
            .squeeze(1)
            .swapaxes(0, 2),
        )

    def forward(self, x):
        """
        Convert a batch of waveforms to STFT forms.

        Parameters
        ----------
        x : torch tensor
        """
        if x.is_cuda and not self.dft_real_kernels.is_cuda:
            self.dft_real_kernels = self.dft_real_kernels.cuda()
            self.dft_imag_kernels = self.dft_imag_kernels.cuda()

        output_real = conv1d(
            x, self.dft_real_kernels, stride=self.n_hop, padding=0
        ).unsqueeze(3)
        output_imag = conv1d(
            x, self.dft_imag_kernels, stride=self.n_hop, padding=0
        ).unsqueeze(3)
        output = output_real.pow(2) + output_imag.pow(2)

        if self.power_spectrogram != 2.0:
            output = torch.pow(torch.sqrt(output), self.power_spectrogram)
        if self.return_decibel_spectrogram:
            output = self.amplitude_to_decibel(output)
        return output

    def get_stft_kernels(self, n_dft):
        """
        Get the STFT kernels.
        Implemented similar to kapre=0.1.4
        """

        nb_filter = int(n_dft // 2 + 1)

        # prepare DFT filters
        timesteps = np.array(range(n_dft))
        w_ks = np.arange(nb_filter) * 2 * np.pi / float(n_dft)
        dft_real_kernels = np.cos(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))
        dft_imag_kernels = -np.sin(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))

        # windowing DFT filters
        dft_window = librosa.filters.get_window(
            "hann", n_dft, fftbins=True
        )  # _hann(n_dft, sym=False)
        dft_window = dft_window.astype(np.float32)
        dft_window = dft_window.reshape((1, -1))
        dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
        dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

        dft_real_kernels = dft_real_kernels.transpose()
        dft_imag_kernels = dft_imag_kernels.transpose()
        dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
        dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

        return (
            dft_real_kernels.astype(np.float32),
            dft_imag_kernels.astype(np.float32),
        )

    def amplitude_to_decibel(self, x, amin=1e-10, dynamic_range=80.0):
        """
        Convert (linear) amplitude to decibel (log10(x)).
        Implemented similar to kapre=0.1.4
        """

        log_spec = (
            10 * torch.log(torch.clamp(x, min=amin)) / np.log(10).astype(np.float32)
        )
        if x.ndim > 1:
            axis = tuple(range(x.ndim)[1:])
        else:
            axis = None

        log_spec = log_spec - torch.amax(log_spec, dim=axis, keepdims=True)
        log_spec = torch.clamp(log_spec, min=-1 * dynamic_range)
        return log_spec


class CustomMelSTFT(CustomSTFT):
    """
    MelSTFT implemented like kapre 0.1.4.
    Attributes
    ----------
      sr: int

      n_dft: int
        The window size for the STFT
      n_hop: int
        The hop (or stride) size
      power_spectrogram: float
        2.0 to get power spectrogram, 1.0 to get amplitude spectrogram.
      return_decibel_spectrogram: bool
        Whether to return in decibel or not, i.e. returns
        log10(amplitude spectrogram) if True

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.

    Examples
    --------
    >>> stftlayer = CustomSTFT(n_dft = 512, n_hop = 242,
        power_spectrogram = 2.0,
        return_decibel_spectrogram=False)
    >>> stftlayer = speclayer(x)
    """

    def __init__(
        self,
        sr,
        n_dft=512,
        n_hop=None,
        n_mels=128,
        htk=True,
        power_melgram=1.0,
        return_decibel_melgram=False,
        padding="same",
    ):
        super().__init__(
            n_dft=n_dft,
            n_hop=n_hop,
            power_spectrogram=2.0,
            return_decibel_spectrogram=False,
        )
        self.padding = padding
        self.sr = sr
        self.power_melgram = power_melgram
        self.return_decibel_melgram = return_decibel_melgram

        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=n_dft,
            n_mels=n_mels,
            fmin=0,
            fmax=sr // 2,
            htk=htk,
            norm=1,
        )
        self.register_buffer("mel_basis", T(mel_basis, requires_grad=False))

    def forward(self, x):

        if x.is_cuda and not self.dft_real_kernels.is_cuda:
            self.dft_real_kernels = self.dft_real_kernels.cuda()
            self.dft_imag_kernels = self.dft_imag_kernels.cuda()
            self.mel_basis = self.mel_basis.cuda()

        if self.padding == "same":
            x = self.custom_pad(x)

        output = super().forward(x)
        output = torch.matmul(self.mel_basis, output.squeeze(-1)).unsqueeze(-1)

        if self.power_melgram != 2.0:
            output = torch.pow(torch.sqrt(output), self.power_melgram)
        if self.return_decibel_melgram:
            output = self.amplitude_to_decibel(output)
        return output

    def custom_pad(self, x):
        """
        Pad sequence.
        Implemented similar to keras version used in kapre=0.1.4
        """

        filter_width = self.n_dft
        strides = self.n_hop
        in_width = self.sr

        if in_width % strides == 0:
            pad_along_width = max(filter_width - strides, 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides), 0)

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        x = torch.nn.ZeroPad2d((pad_left, pad_right, 0, 0))(x)
        return x


class PytorchOpenl3(nn.Module):
    def __init__(self, content_type, input_repr, embedding_size):
        # Note: content_type is unused
        super(PytorchOpenl3, self).__init__()
        self.AUDIO_POOLING_SIZES = {
            "linear": {512: (32, 24), 6144: (8, 8)},
            "mel128": {512: (16, 24), 6144: (4, 8)},
            "mel256": {512: (32, 24), 6144: (8, 8)},
        }

        if input_repr == "linear":
            self.speclayer = CustomSTFT(
                n_dft=512,
                n_hop=242,
                power_spectrogram=1.0,
                return_decibel_spectrogram=True,
            )

        elif input_repr == "mel128":
            self.speclayer = CustomMelSTFT(
                sr=48000,
                n_dft=2048,
                n_hop=242,
                n_mels=128,
                htk=True,
                power_melgram=1.0,
                return_decibel_melgram=True,
                padding="same",
            )

        elif input_repr == "mel256":
            self.speclayer = CustomMelSTFT(
                sr=48000,
                n_dft=2048,
                n_hop=242,
                n_mels=256,
                htk=True,
                power_melgram=1.0,
                return_decibel_melgram=True,
                padding="same",
            )

        self.input_repr = input_repr
        self.embedding_size = embedding_size
        self.batch_normalization_1 = self.__batch_normalization(
            2,
            "batch_normalization_1",
            num_features=1,
            eps=0.001,
            momentum=0.99,
        )
        self.conv2d_1 = self.__conv(
            2,
            name="conv2d_1",
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_2 = self.__batch_normalization(
            2,
            "batch_normalization_2",
            num_features=64,
            eps=0.001,
            momentum=0.99,
        )
        self.conv2d_2 = self.__conv(
            2,
            name="conv2d_2",
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_3 = self.__batch_normalization(
            2,
            "batch_normalization_3",
            num_features=64,
            eps=0.001,
            momentum=0.99,
        )
        self.conv2d_3 = self.__conv(
            2,
            name="conv2d_3",
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_4 = self.__batch_normalization(
            2,
            "batch_normalization_4",
            num_features=128,
            eps=0.001,
            momentum=0.99,
        )
        self.conv2d_4 = self.__conv(
            2,
            name="conv2d_4",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_5 = self.__batch_normalization(
            2,
            "batch_normalization_5",
            num_features=128,
            eps=0.001,
            momentum=0.99,
        )
        self.conv2d_5 = self.__conv(
            2,
            name="conv2d_5",
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_6 = self.__batch_normalization(
            2,
            "batch_normalization_6",
            num_features=256,
            eps=0.001,
            momentum=0.99,
        )
        self.conv2d_6 = self.__conv(
            2,
            name="conv2d_6",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_7 = self.__batch_normalization(
            2,
            "batch_normalization_7",
            num_features=256,
            eps=0.001,
            momentum=0.99,
        )
        self.conv2d_7 = self.__conv(
            2,
            name="conv2d_7",
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_8 = self.__batch_normalization(
            2,
            "batch_normalization_8",
            num_features=512,
            eps=0.001,
            momentum=0.99,
        )
        self.audio_embedding_layer = self.__conv(
            2,
            name="audio_embedding_layer",
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )

    def forward(self, x, keep_all_outputs=False):
        if keep_all_outputs:
            all_outputs = []
        x = self.speclayer(x)
        x = x.squeeze(-1).unsqueeze(1)
        if keep_all_outputs:
            all_outputs.append(x)
        batch_normalization_1 = self.batch_normalization_1(x)
        if keep_all_outputs:
            all_outputs.append(batch_normalization_1)
        conv2d_1_pad = F.pad(batch_normalization_1, (1, 1, 1, 1))
        conv2d_1 = self.conv2d_1(conv2d_1_pad)
        if keep_all_outputs:
            all_outputs.append(conv2d_1)
        batch_normalization_2 = self.batch_normalization_2(conv2d_1)
        if keep_all_outputs:
            all_outputs.append(batch_normalization_2)
        activation_1 = F.relu(batch_normalization_2)
        if keep_all_outputs:
            all_outputs.append(activation_1)
        conv2d_2_pad = F.pad(activation_1, (1, 1, 1, 1))
        conv2d_2 = self.conv2d_2(conv2d_2_pad)
        if keep_all_outputs:
            all_outputs.append(conv2d_2)
        batch_normalization_3 = self.batch_normalization_3(conv2d_2)
        if keep_all_outputs:
            all_outputs.append(batch_normalization_3)
        activation_2 = F.relu(batch_normalization_3)
        if keep_all_outputs:
            all_outputs.append(activation_2)
        max_pooling2d_1 = F.max_pool2d(
            activation_2,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=0,
            ceil_mode=False,
        )
        if keep_all_outputs:
            all_outputs.append(max_pooling2d_1)
        conv2d_3_pad = F.pad(max_pooling2d_1, (1, 1, 1, 1))
        conv2d_3 = self.conv2d_3(conv2d_3_pad)
        if keep_all_outputs:
            all_outputs.append(conv2d_3)
        batch_normalization_4 = self.batch_normalization_4(conv2d_3)
        if keep_all_outputs:
            all_outputs.append(batch_normalization_4)
        activation_3 = F.relu(batch_normalization_4)
        if keep_all_outputs:
            all_outputs.append(activation_3)
        conv2d_4_pad = F.pad(activation_3, (1, 1, 1, 1))
        conv2d_4 = self.conv2d_4(conv2d_4_pad)
        if keep_all_outputs:
            all_outputs.append(conv2d_4)
        batch_normalization_5 = self.batch_normalization_5(conv2d_4)
        if keep_all_outputs:
            all_outputs.append(batch_normalization_5)
        activation_4 = F.relu(batch_normalization_5)
        if keep_all_outputs:
            all_outputs.append(activation_4)
        max_pooling2d_2 = F.max_pool2d(
            activation_4,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=0,
            ceil_mode=False,
        )
        if keep_all_outputs:
            all_outputs.append(max_pooling2d_2)
        conv2d_5_pad = F.pad(max_pooling2d_2, (1, 1, 1, 1))
        conv2d_5 = self.conv2d_5(conv2d_5_pad)
        if keep_all_outputs:
            all_outputs.append(conv2d_5)
        batch_normalization_6 = self.batch_normalization_6(conv2d_5)
        if keep_all_outputs:
            all_outputs.append(batch_normalization_6)
        activation_5 = F.relu(batch_normalization_6)
        if keep_all_outputs:
            all_outputs.append(activation_5)
        conv2d_6_pad = F.pad(activation_5, (1, 1, 1, 1))
        conv2d_6 = self.conv2d_6(conv2d_6_pad)
        if keep_all_outputs:
            all_outputs.append(conv2d_6)
        batch_normalization_7 = self.batch_normalization_7(conv2d_6)
        if keep_all_outputs:
            all_outputs.append(batch_normalization_7)
        activation_6 = F.relu(batch_normalization_7)
        if keep_all_outputs:
            all_outputs.append(activation_6)
        max_pooling2d_3 = F.max_pool2d(
            activation_6,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=0,
            ceil_mode=False,
        )
        if keep_all_outputs:
            all_outputs.append(max_pooling2d_3)
        conv2d_7_pad = F.pad(max_pooling2d_3, (1, 1, 1, 1))
        conv2d_7 = self.conv2d_7(conv2d_7_pad)
        if keep_all_outputs:
            all_outputs.append(conv2d_7)
        batch_normalization_8 = self.batch_normalization_8(conv2d_7)
        if keep_all_outputs:
            all_outputs.append(batch_normalization_8)
        activation_7 = F.relu(batch_normalization_8)
        if keep_all_outputs:
            all_outputs.append(activation_7)
        audio_embedding_layer_pad = F.pad(activation_7, (1, 1, 1, 1))
        audio_embedding_layer = self.audio_embedding_layer(audio_embedding_layer_pad)
        if keep_all_outputs:
            all_outputs.append(audio_embedding_layer)
        max_pooling2d_4 = F.max_pool2d(
            audio_embedding_layer,
            kernel_size=self.AUDIO_POOLING_SIZES[self.input_repr][self.embedding_size],
            stride=self.AUDIO_POOLING_SIZES[self.input_repr][self.embedding_size],
            padding=0,
            ceil_mode=False,
        )

        if keep_all_outputs:
            all_outputs.append(max_pooling2d_4)
        # Might just use view ?
        squeeze = (
            max_pooling2d_4.swapaxes(1, 2)
            .swapaxes(2, 3)
            .reshape((max_pooling2d_4.shape[0], -1))
        )
        if keep_all_outputs:
            all_outputs.append(squeeze)
            return all_outputs
        else:
            return squeeze

    def __batch_normalization(self, dim, name, **kwargs):
        if dim == 0 or dim == 1:
            layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:
            layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)
        else:
            raise NotImplementedError()
        return layer

    def __conv(self, dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        return layer


def load_audio_embedding_model(input_repr, content_type, embedding_size):
    return torchopenl3.core.load_audio_embedding_model(
        input_repr, content_type, embedding_size
    )
