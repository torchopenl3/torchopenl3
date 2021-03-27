import torch
import torchopenl3


def test_preprocess():
    audio = torch.rand(1024, 48000)
    assert torchopenl3.utils.preprocess_audio_batch(audio, sr=48000).shape == (
        6144,
        1,
        48000,
    )

    audio = torch.rand(1024, 48000, 1)
    assert torchopenl3.utils.preprocess_audio_batch(audio, sr=48000).shape == (
        6144,
        1,
        48000,
    )

    audio = torch.rand(1024, 48000, 2)
    assert torchopenl3.utils.preprocess_audio_batch(audio, sr=48000).shape == (
        6144,
        1,
        48000,
    )


if __name__ == "__main__":
    test_preprocess()
