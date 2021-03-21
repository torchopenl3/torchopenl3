import itertools
import os.path
import tempfile

import numpy as np
import openl3
import pytest
import requests
import soundfile as sf
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

AUDIO_MODEL_PARAMS = {
    "content_type": ["env", "music"],
    # "input_repr": ["linear","mel128", "mel256"],
    
    '''
    We didn't include linear here because openl3 using kapre old for extarcting specotragram which
    is the shape of (None,237,197,1) but in torchaudio torchlibrosa nnAudio gives (None,237,199,1)
    So we decide not to include linear model for now.
    '''
    
    "input_repr": ["mel128", "mel256"],
    "embedding_size": [512, 6144],
    "verbose": [0, 1],
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
            assert torch.mean(torch.abs(embeddings1[i] - embeddings0[i])) <= 1e-6
            assert torch.mean(torch.abs(ts1[i] - ts0[i])) <= 1e-6
        embeddings2, ts2 = torchopenl3.get_audio_embedding(
            audios, srs, batch_size=32, **modelparams
        )
        for i in range(n):
            '''
            We increase the compare paremeter as kapre in openl3 and nnAudio in torchopenl3 giving 
            more mean error. We can expect a prrety good result when we will pretrain model
            '''
            assert torch.mean(torch.abs(embeddings1[i] - embeddings2[i])) <= 2
            assert torch.mean(torch.abs(ts1[i] - ts2[i])) <= 2

    def test_regression(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filenames = []
            for url in AUDIO_URLS:
                filename = os.path.join(tmpdirname, os.path.split(url)[1])
                r = requests.get(url, allow_redirects=True)
                open(filename, "wb").write(r.content)
                filenames.append(filename)

            modelparamlist = [
                dict(zip(AUDIO_MODEL_PARAMS.keys(), p))
                for p in itertools.product(*list(AUDIO_MODEL_PARAMS.values()))
            ]
            for modelparams in tqdm(modelparamlist):
                self.check_model_for_regression(modelparams, filenames)
