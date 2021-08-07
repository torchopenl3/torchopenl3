import itertools
from collections import OrderedDict

import numpy as np
import soundfile as sf
import torch
from torch import tensor as T

import torchopenl3

PARAMS = OrderedDict(
    {
        "center": [True, False],
        "hop_size": [0.1, 0.2718, 0.5],
    }
)

paramlist = [
    dict(zip(PARAMS.keys(), p)) for p in itertools.product(*list(PARAMS.values()))
]


class TestEmbeddingTimestamps:
    """
    Test timestamps of output
    """

    def test_timestamps_no_channels(self):

        # Could do this for all models
        sounds = [torch.rand(48000) for _ in range(3)]
        # Add silence to change the embeddings
        sounds[0][0:16000] = torch.zeros(16000)
        sounds[1][16000:32000] = torch.zeros(16000)
        sounds[2][32000:48000] = torch.zeros(16000)

        for params in paramlist:
            emb0, ts0 = torchopenl3.get_audio_embedding(
                sounds, 48000, batch_size=32, sampler="resampy", **params
            )
            ts0 = np.vstack(ts0)
            emb1, ts1 = torchopenl3.get_audio_embedding(
                torch.stack(sounds), 48000, sampler="resampy", batch_size=32, **params
            )
            assert torch.mean(torch.abs(ts1 - ts0)) <= 1e-6

    def test_timestamps_stereo(self):
        # Could do this for all models
        sounds = [torch.rand(48000, 2) for _ in range(3)]
        # Add silence to change the embeddings
        sounds[0][0:16000, :] = torch.zeros(16000, 2)
        sounds[1][16000:32000, :] = torch.zeros(16000, 2)
        sounds[2][32000:48000, :] = torch.zeros(16000, 2)

        for params in paramlist:
            emb0, ts0 = torchopenl3.get_audio_embedding(
                sounds, 48000, batch_size=32, sampler="resampy", **params
            )
            ts0 = np.vstack(ts0)
            emb1, ts1 = torchopenl3.get_audio_embedding(
                torch.stack(sounds), 48000, batch_size=32, sampler="resampy", **params
            )
            assert torch.mean(torch.abs(ts1 - ts0)) <= 1e-6
