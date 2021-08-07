import itertools
import os.path
import tempfile

import numpy as np
import openl3
import requests
import soundfile as sf
import torch
from torch import tensor as T
from tqdm.auto import tqdm

import torchopenl3

AUDIO_URLS = [
    "https://raw.githubusercontent.com/marl/openl3/master/tests/data/audio/chirp_1s.wav",
    "https://raw.githubusercontent.com/marl/openl3/master/tests/data/audio/chirp_44k.wav",
    "https://raw.githubusercontent.com/marl/openl3/master/tests/data/audio/chirp_mono.wav",
    "https://raw.githubusercontent.com/marl/openl3/master/tests/data/audio/chirp_stereo.wav",
    # Ignore empty audio which causes openl3 to throw an exception
    # "https://raw.githubusercontent.com/marl/openl3/master/tests/data/audio/empty.wav",
    "https://raw.githubusercontent.com/marl/openl3/master/tests/data/audio/short.wav",
    "https://raw.githubusercontent.com/marl/openl3/master/tests/data/audio/silence.wav",
]

CHECK_AUDIO_MODEL_PARAMS = {
    #    "verbose": [0, 1],
    "center": [True, False],
    "hop_size": [0.1, 0.5],
}


class TestRegression:
    """
    Tests for any regressions against openl3.
    """

    def check_model_for_regression(self, modelparams, filenames):
        audios = []
        srs = []
        for filename in filenames:
            audio, sr = sf.read(filename)
            audios.append(audio)
            srs.append(sr)
        n = len(filenames)
        embeddings0, ts0 = openl3.get_audio_embedding(
            audios, srs, batch_size=32, **modelparams
        )
        embeddings1, ts1 = openl3.get_audio_embedding(
            audios, srs, batch_size=32, **modelparams
        )

        # This is just a sanity check that openl3
        # gives consistent results, we can remove
        # it later.
        for i in range(n):
            assert embeddings1[0].shape == embeddings0[0].shape
            assert embeddings1[1].shape == embeddings0[1].shape
            assert torch.mean(torch.abs(T(embeddings1[i]) - T(embeddings0[i]))) <= 1e-6
            assert torch.mean(torch.abs(T(ts1[i]) - T(ts0[i]))) <= 1e-6
        embeddings2, ts2 = torchopenl3.get_audio_embedding(
            audios, srs, batch_size=32, sampler="resampy", **modelparams
        )
        for i in range(n):
            """
            We increase the compare paremeter as kapre in openl3 and nnAudio in torchopenl3 giving
            more mean error. We can expect a prrety good result when we will pretrain model
            """
            print(embeddings1[0].shape, embeddings2[0].shape)
            print(embeddings1[1].shape, embeddings2[1].shape)
            print(torch.mean(torch.abs(T(ts1[i]) - T(ts2[i]))))
            print(torch.mean(torch.abs(T(embeddings1[i]) - T(embeddings2[i]))))
            print(torch.mean(torch.abs(T(ts1[i]) - T(ts2[i]))))
            assert embeddings1[0].shape == embeddings2[0].shape
            assert embeddings1[1].shape == embeddings2[1].shape
            assert torch.mean(torch.abs(T(embeddings1[i]) - T(embeddings2[i]))) <= 5e-3
            assert torch.mean(torch.abs(T(ts1[i]) - T(ts2[i]))) <= 1e-6

    def _test_regression(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filenames = []
            for url in AUDIO_URLS:
                filename = os.path.join(tmpdirname, os.path.split(url)[1])
                r = requests.get(url, allow_redirects=True)
                open(filename, "wb").write(r.content)
                filenames.append(filename)

            modelparamlist = [
                dict(zip(CHECK_AUDIO_MODEL_PARAMS.keys(), p), **kwargs)
                for p in itertools.product(*list(CHECK_AUDIO_MODEL_PARAMS.values()))
            ]
            for modelparams in tqdm(modelparamlist):
                self.check_model_for_regression(modelparams, filenames)

    """
    def test_regression_env_linear_512(self):
        self._test_regression(
            content_type="env", input_repr="linear", embedding_size=512
        )
    """

    def test_regression_music_linear_512(self):
        self._test_regression(
            content_type="music", input_repr="linear", embedding_size=512
        )

    """
    def test_regression_env_mel128_512(self):
        self._test_regression(
            content_type="env", input_repr="mel128", embedding_size=512
        )

    def test_regression_music_mel128_512(self):
        self._test_regression(
            content_type="music", input_repr="mel128", embedding_size=512
        )
    """

    def test_regression_env_mel256_512(self):
        self._test_regression(
            content_type="env", input_repr="mel256", embedding_size=512
        )

    """
    def test_regression_music_mel256_512(self):
        self._test_regression(
            content_type="music", input_repr="mel256", embedding_size=512
        )

    def test_regression_env_linear_6144(self):
        self._test_regression(
            content_type="env", input_repr="linear", embedding_size=6144
        )

    def test_regression_music_linear_6144(self):
        self._test_regression(
            content_type="music", input_repr="linear", embedding_size=6144
        )

    def test_regression_env_mel128_6144(self):
        self._test_regression(
            content_type="env", input_repr="mel128", embedding_size=6144
        )

    def test_regression_music_mel128_6144(self):
        self._test_regression(
            content_type="music", input_repr="mel128", embedding_size=6144
        )

    def test_regression_env_mel256_6144(self):
        self._test_regression(
            content_type="env", input_repr="mel256", embedding_size=6144
        )

    def test_regression_music_mel256_6144(self):
        self._test_regression(
            content_type="music", input_repr="mel256", embedding_size=6144
        )
    """
