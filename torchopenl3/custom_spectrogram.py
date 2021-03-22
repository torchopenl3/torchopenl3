import torch
from torch import tensor as T
from torch import nn
import librosa
import nnAudio.Spectrogram


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
        self.stft = nnAudio.Spectrogram.STFT(
            n_fft=n_fft,
            win_length=None,
            freq_bins=None,
            hop_length=n_hop,
            window="hann",
            freq_scale="no",
            center=False,  # or True?
            pad_mode=None,  # or "constant"? etc.
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
