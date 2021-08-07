import numpy as np
import soundfile as sf
import torch
import torchopenl3


class TestEmbeddingShapes:
    """
    Test shapes of output
    """

    def test_shapes_no_channels(self):
        # Could do this for all models
        sounds = [torch.rand(48000) for _ in range(3)]
        # Add silence to change the embeddings
        sounds[0][0:16000] = torch.zeros(16000)
        sounds[1][16000:32000] = torch.zeros(16000)
        sounds[2][32000:48000] = torch.zeros(16000)

        emb0, ts0 = torchopenl3.get_audio_embedding(
            sounds, 48000, batch_size=32, sampler="resampy"
        )
        emb0 = torch.stack(emb0)
        ts0 = np.vstack(ts0)
        assert emb0.shape == (3, 6, 6144)
        # assert ts0.shape == (3, 6, 1)

        emb1, ts1 = torchopenl3.get_audio_embedding(
            torch.stack(sounds), 48000, batch_size=32, sampler="resampy"
        )
        assert emb1.shape == (3, 6, 6144)
        # assert ts1.shape == (3, 6, 1)

        assert torch.mean(torch.abs(emb1 - emb0)) <= 1e-6
        # assert torch.mean(torch.abs(ts1 - ts0)) <= 1e-6

    def test_shapes_stereo(self):
        # Could do this for all models
        sounds = [torch.rand(48000, 2) for _ in range(3)]
        # Add silence to change the embeddings
        sounds[0][0:16000, :] = torch.zeros(16000, 2)
        sounds[1][16000:32000, :] = torch.zeros(16000, 2)
        sounds[2][32000:48000, :] = torch.zeros(16000, 2)
        emb0, ts0 = torchopenl3.get_audio_embedding(
            sounds, 48000, batch_size=32, sampler="resampy"
        )

        emb0 = torch.stack(emb0)
        ts0 = np.vstack(ts0)
        assert emb0.shape == (3, 6, 6144)
        # assert ts0.shape == (3, 6, 1)

        emb1, ts1 = torchopenl3.get_audio_embedding(
            torch.stack(sounds), 48000, batch_size=32, sampler="resampy"
        )
        assert emb1.shape == (3, 6, 6144)
        # assert ts1.shape == (3, 6, 1)

        assert torch.mean(torch.abs(emb1 - emb0)) <= 1e-6
        # assert torch.mean(torch.abs(ts1 - ts0)) <= 1e-6
