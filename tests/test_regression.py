import itertools
import os.path
import tempfile
import torch.tensor as T
import torch
import numpy as np
import openl3
import pytest
import requests
import soundfile as sf
from tqdm.auto import tqdm
import resampy
import torchopenl3
from keras import Model, Input

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
    """
    We didn't include linear here because openl3 using kapre old for extarcting specotragram which
    is the shape of (None,237,197,1) but in torchaudio torchlibrosa nnAudio gives (None,237,199,1)
    So we decide not to include linear model for now.
    """
    "input_repr": ["linear", "mel128", "mel256"],
    "embedding_size": [512, 6144],
    "verbose": [0, 1],
    "center": [True, False],
    "hop_size": [0.1, 0.5],
}

TARGET_SR = 48000


def center_audio(audio, frame_len):
    return np.pad(audio, (int(frame_len / 2.0), 0), mode="constant", constant_values=0)


def pad_audio(audio, frame_len, hop_len):
    audio_len = audio.size
    if audio_len < frame_len:
        pad_length = frame_len - audio_len
    else:
        pad_length = int(
            np.ceil((audio_len - frame_len) / float(hop_len))
        ) * hop_len - (audio_len - frame_len)

    if pad_length > 0:
        audio = np.pad(audio, (0, pad_length),
                       mode="constant", constant_values=0)

    return audio


def get_num_windows(audio_len, frame_len, hop_len, center):
    if center:
        audio_len += int(frame_len / 2.0)

    if audio_len <= frame_len:
        return 1
    else:
        return 1 + int(np.ceil((audio_len - frame_len) / float(hop_len)))


def preprocess_audio_batch(audio, sr, center=True, hop_size=0.1):
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = resampy.resample(
            audio, sr_orig=sr, sr_new=TARGET_SR, filter="kaiser_best"
        )

    frame_len = TARGET_SR
    hop_len = int(hop_size * TARGET_SR)

    if center:
        audio = center_audio(audio, frame_len)

    audio = pad_audio(audio, frame_len, hop_len)

    n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
    x = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_len, n_frames),
        strides=(audio.itemsize, hop_len * audio.itemsize),
    ).T

    x = x.reshape((x.shape[0], 1, x.shape[-1]))

    return x

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
            
        batch = []
        for audio, sr in zip(audios, srs):
            x = preprocess_audio_batch(
                audio, sr)
            batch.append(x)
            
        batch = np.vstack(batch)
                
        #Openl3 Model
        model = openl3.models.load_audio_embedding_model(**modelparams)
        inp = model.get_input_at(0)
        oups = [
            model.layers[1].output,
            model.layers[2].output,
            model.layers[3].output,
            model.layers[4].output,
            model.layers[5].output,
            model.layers[6].output,
            model.layers[7].output,
            model.layers[8].output,
            model.layers[10].output,
            model.layers[11].output,
            model.layers[12].output,
            model.layers[13].output,
            model.layers[14].output,
            model.layers[15].output,
            model.layers[17].output,
            model.layers[18].output,
            model.layers[19].output,
            model.layers[20].output,
            model.layers[21].output,
            model.layers[22].output,
            model.layers[24].output,
            model.layers[25].output,
            model.layers[26].output,
            model.layers[27].output,
            model.layers[28].output,
        ]
        openl3_model = Model(inputs=[inp], outputs=oups)
        
        #Torchopenl3 Model
        torchopenl3_model = torchopenl3.model.PytorchOpenl3(
            modelparams[0], modelparams[2])
        torchopenl3_model.load_state_dict(torch.load(
            torchopenl3.core.get_model_path(**modelparams)))
        torchopenl3_model = torchopenl3_model.eval()
        
        #Openl3 Model All layers output
        openl3_output = openl3_model.predict(batch)
        
        #TorchOpenl3 Model All layers output
        torchopenl3_output = torchopenl3_model(torch.tensor(batch))
        
        for i in range(25):
            assert np.mean(np.abs(
                openl3_output[i] - torchopenl3_output[i].swapaxes(1, 2).swapaxes(2, 3).detach().numpy())) <= 2

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
