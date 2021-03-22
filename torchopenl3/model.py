import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor as T
from nnAudio import Spectrogram
import librosa


class CustomSpectrogram(nn.Module):
    """
    Custom Spectrogram implemented to mimic---but unfortunately not
    completely replicate---behavior of kapre 0.1.4, which is required
    by openl3 0.3.1
    Attributes
    ----------
    type: str
      the type of the spectrogram. one option from "lin", "mel128" or "mel256"
    n_fft: int
      The window size for the STFT
    n_hop: int
      The hop (or stride) size
    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.
    Examples
    --------
    >>> speclayer = CustomSpectrogram("mel256", n_fft = 512, n_hop = 242, asr = 48000)
    >>> specs = speclayer(x)
    """

    def __init__(self, type, n_fft, n_hop, asr):
        super().__init__()
        self.type = type
        self.stft = Spectrogram.STFT(
            n_fft=n_fft,
            win_length=None,
            freq_bins=None,
            hop_length=n_hop,
            window="hann",
            freq_scale="no",
            center=(type != "linear"),
            pad_mode=(None if type == "linear" else "constant"),
            iSTFT=False,
            fmin=0,
            fmax=asr // 2,
            sr=asr,
            trainable=False,
            output_format="Magnitude",
            verbose=False,
        )
        if self.type == "mel128":
            self.mel_basis = librosa.filters.mel(
                sr=asr, n_fft=n_fft, n_mels=128, fmin=0, fmax=asr // 2, htk=True, norm=1
            )
            self.mel_basis = T(self.mel_basis, dtype=torch.float32)
        elif self.type == "mel256":
            self.mel_basis = librosa.filters.mel(
                sr=asr, n_fft=n_fft, n_mels=256, fmin=0, fmax=asr // 2, htk=True, norm=1
            )
            self.mel_basis = T(self.mel_basis, dtype=torch.float32)

    def forward(self, x):
        """
        Convert a batch of waveforms to Mel spectrograms or spectrogram
        depening on the type.
        Parameters
        ----------
        x : torch tensor
        """

        x_stft = self.stft(x)
        if self.type == "linear":
            x_stft = x_stft
        elif self.type == "mel128" or self.type == "mel256":
            x_stft = torch.pow(x_stft, 2)
            x_stft = torch.sqrt(torch.matmul(self.mel_basis, x_stft))
        else:
            raise ValueError("The type should either be linear or mel128 or mel256")

        return self.amplitude_to_decibel(x_stft)

    def amplitude_to_decibel(self, x, amin=1e-10, dynamic_range=80.0):
        """
        Convert (linear) amplitude to decibel (log10(x)).
        Implemented similar to kapre-0.1.4
        """

        log_spec = (
            T(10.0, dtype=torch.float32)
            * torch.log(torch.maximum(x, T(amin)))
            / torch.log(T(10.0, dtype=torch.float32))
        )
        if x.ndim > 1:
            axis = tuple(range(x.ndim)[1:])
        else:
            axis = None

        log_spec = log_spec - torch.amax(log_spec, dim=axis, keepdims=True)
        log_spec = torch.maximum(log_spec, T(-1 * dynamic_range))
        return log_spec


class PytorchOpenl3(nn.Module):
    def __init__(self, input_repr, embedding_size):
        super(PytorchOpenl3, self).__init__()
        self.AUDIO_POOLING_SIZES = {
            "linear": {512: (32, 24), 6144: (8, 8)},
            "mel128": {512: (16, 24), 6144: (4, 8)},
            "mel256": {512: (32, 24), 6144: (8, 8)},
        }
        self.speclayer = CustomSpectrogram(input_repr, n_fft=512, n_hop=242, asr=48000)

        # Old approach, commenting it out if we need it
        """
        elif input_repr == "mel128":
            self.speclayer = Spectrogram.MelSpectrogram(
                sr=48000, n_fft=2048, n_mels=128, hop_length=242, power=1.0, htk=True
            )
        elif input_repr == "mel256":
            self.speclayer = Spectrogram.MelSpectrogram(
                sr=48000, n_fft=2048, n_mels=256, hop_length=242, power=1.0, htk=True
            )
        """

        self.input_repr = input_repr
        self.embedding_size = embedding_size
        self.batch_normalization_1 = self.__batch_normalization(
            2, "batch_normalization_1", num_features=1, eps=0.001, momentum=0.99
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
            2, "batch_normalization_2", num_features=64, eps=0.001, momentum=0.99
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
            2, "batch_normalization_3", num_features=64, eps=0.001, momentum=0.99
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
            2, "batch_normalization_4", num_features=128, eps=0.001, momentum=0.99
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
            2, "batch_normalization_5", num_features=128, eps=0.001, momentum=0.99
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
            2, "batch_normalization_6", num_features=256, eps=0.001, momentum=0.99
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
            2, "batch_normalization_7", num_features=256, eps=0.001, momentum=0.99
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
            2, "batch_normalization_8", num_features=512, eps=0.001, momentum=0.99
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

    def forward(self, x):
        all_outputs = []
        x = self.speclayer(x)
        x = x.unsqueeze(1)
        all_outputs.append(x)
        batch_normalization_1 = self.batch_normalization_1(x)
        all_outputs.append(batch_normalization_1)
        conv2d_1_pad = F.pad(batch_normalization_1, (1, 1, 1, 1))
        conv2d_1 = self.conv2d_1(conv2d_1_pad)
        all_outputs.append(conv2d_1)
        batch_normalization_2 = self.batch_normalization_2(conv2d_1)
        all_outputs.append(batch_normalization_2)
        activation_1 = F.relu(batch_normalization_2)
        all_outputs.append(activation_1)
        conv2d_2_pad = F.pad(activation_1, (1, 1, 1, 1))
        conv2d_2 = self.conv2d_2(conv2d_2_pad)
        all_outputs.append(conv2d_2)
        batch_normalization_3 = self.batch_normalization_3(conv2d_2)
        all_outputs.append(batch_normalization_2)
        activation_2 = F.relu(batch_normalization_3)
        all_outputs.append(activation_2)
        max_pooling2d_1 = F.max_pool2d(
            activation_2, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False
        )
        conv2d_3_pad = F.pad(max_pooling2d_1, (1, 1, 1, 1))
        conv2d_3 = self.conv2d_3(conv2d_3_pad)
        all_outputs.append(conv2d_3)
        batch_normalization_4 = self.batch_normalization_4(conv2d_3)
        all_outputs.append(batch_normalization_4)
        activation_3 = F.relu(batch_normalization_4)
        all_outputs.append(activation_3)
        conv2d_4_pad = F.pad(activation_3, (1, 1, 1, 1))
        conv2d_4 = self.conv2d_4(conv2d_4_pad)
        all_outputs.append(conv2d_4)
        batch_normalization_5 = self.batch_normalization_5(conv2d_4)
        all_outputs.append(batch_normalization_5)
        activation_4 = F.relu(batch_normalization_5)
        all_outputs.append(activation_4)
        max_pooling2d_2 = F.max_pool2d(
            activation_4, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False
        )
        conv2d_5_pad = F.pad(max_pooling2d_2, (1, 1, 1, 1))
        conv2d_5 = self.conv2d_5(conv2d_5_pad)
        all_outputs.append(conv2d_5)
        batch_normalization_6 = self.batch_normalization_6(conv2d_5)
        all_outputs.append(batch_normalization_6)
        activation_5 = F.relu(batch_normalization_6)
        all_outputs.append(activation_5)
        conv2d_6_pad = F.pad(activation_5, (1, 1, 1, 1))
        conv2d_6 = self.conv2d_6(conv2d_6_pad)
        all_outputs.append(conv2d_6)
        batch_normalization_7 = self.batch_normalization_7(conv2d_6)
        all_outputs.append(batch_normalization_7)
        activation_6 = F.relu(batch_normalization_7)
        all_outputs.append(activation_6)
        max_pooling2d_3 = F.max_pool2d(
            activation_6, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False
        )
        conv2d_7_pad = F.pad(max_pooling2d_3, (1, 1, 1, 1))
        conv2d_7 = self.conv2d_7(conv2d_7_pad)
        all_outputs.append(conv2d_7)
        batch_normalization_8 = self.batch_normalization_8(conv2d_7)
        all_outputs.append(batch_normalization_8)
        activation_7 = F.relu(batch_normalization_8)
        all_outputs.append(activation_7)
        audio_embedding_layer_pad = F.pad(activation_7, (1, 1, 1, 1))
        audio_embedding_layer = self.audio_embedding_layer(audio_embedding_layer_pad)
        all_outputs.append(audio_embedding_layer)
        max_pooling2d_4 = F.max_pool2d(
            audio_embedding_layer,
            kernel_size=self.AUDIO_POOLING_SIZES[self.input_repr][self.embedding_size],
            stride=self.AUDIO_POOLING_SIZES[self.input_repr][self.embedding_size],
            padding=0,
            ceil_mode=False,
        )
        all_outputs.append(max_pooling2d_4)
        return all_outputs

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
