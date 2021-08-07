import itertools
import os.path
import random
import tempfile

import numpy as np
import openl3
import requests
import resampy
import soundfile as sf
import torch
from torch import tensor as T
import torchopenl3
from keras import Model
from tqdm.auto import tqdm

random.seed(0)

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
    "input_repr": ["linear", "mel128", "mel256"],
    "embedding_size": [512, 6144],
    # "verbose": [0, 1],
    # "center": [True, False],
    # "hop_size": [0.1, 0.5],
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
        audio = np.pad(audio, (0, pad_length), mode="constant", constant_values=0)

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


class LayerByLayer:
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
            x = preprocess_audio_batch(audio, sr)
            batch.append(x)

        # Downsample for faster testing
        batch = np.vstack([b for i, b in enumerate(batch) if i % 7 == 0])
        print("Batch shape:", batch.shape)

        # Openl3 Model
        model = openl3.models.load_audio_embedding_model(**modelparams)
        inp = model.get_input_at(0)
        oups = [model.layers[l].output for l in range(1, len(model.layers))]
        # oups = [model.layers[l].output for l in range(29)]
        openl3_model = Model(inputs=[inp], outputs=oups)

        # Torchopenl3 Model
        torchopenl3_model = torchopenl3.core.load_audio_embedding_model(**modelparams)
        #        torchopenl3_model = torchopenl3.model.PytorchOpenl3(**modelparams)
        #        torchopenl3_model.load_state_dict(
        #            torch.load(torchopenl3.core.get_model_path(**modelparams))
        #        )
        #        torchopenl3_model = torchopenl3_model.eval()

        # Openl3 Model All layers output
        openl3_output = openl3_model.predict(batch)

        # TorchOpenl3 Model All layers output
        torchopenl3_output = torchopenl3_model(
            torch.tensor(batch, dtype=torch.float32), keep_all_outputs=True
        )

        print("Open L3 layers:     ", len(model.layers))
        print("Open L3 output:     ", len(openl3_output))
        print("Torchopen L3 output:", len(torchopenl3_output))
        for i in range(len(openl3_output)):
            if i < len(torchopenl3_output):
                if i != len(torchopenl3_output) - 1:
                    print(
                        f"{i} {openl3_output[i].shape} {torchopenl3_output[i].swapaxes(1, 2).swapaxes(2, 3).shape}"
                    )
                else:
                    print(f"{i} {openl3_output[i].shape} {torchopenl3_output[i].shape}")
            else:
                print(f"{i} {openl3_output[i].shape}")
        for i in range(len(torchopenl3_output)):
            if i != len(torchopenl3_output) - 1:
                torcho = torchopenl3_output[i].swapaxes(1, 2).swapaxes(2, 3).clone()
            else:
                torcho = torchopenl3_output[i]
            torcho = torcho.detach().numpy()
            err = np.mean(np.abs(openl3_output[i] - torcho))
            # pearson = scipy.stats.pearsonr(
            #    np.sort(openl3_output[i].flatten()), np.sort(torcho.flatten())
            # )[0]
            print(err, i, modelparams)

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
            random.shuffle(modelparamlist)
            for modelparams in tqdm(modelparamlist):
                self.check_model_for_regression(modelparams, filenames)


if __name__ == "__main__":
    lbl = LayerByLayer()
    lbl.test_regression()
