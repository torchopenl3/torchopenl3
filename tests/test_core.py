import numpy as np
import os
import soundfile as sf
import torchopenl3
import torchopenl3.models

TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, "data", "audio")

CHIRP_MONO_PATH = os.path.join(TEST_AUDIO_DIR, "chirp_mono.wav")
CHIRP_STEREO_PATH = os.path.join(TEST_AUDIO_DIR, "chirp_stereo.wav")
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, "chirp_44k.wav")
CHIRP_1S_PATH = os.path.join(TEST_AUDIO_DIR, "chirp_1s.wav")
EMPTY_PATH = os.path.join(TEST_AUDIO_DIR, "empty.wav")
SHORT_PATH = os.path.join(TEST_AUDIO_DIR, "short.wav")
SILENCE_PATH = os.path.join(TEST_AUDIO_DIR, "silence.wav")


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    else:
        try:
            a = a.numpy()
        except Exception:
            a = a.detach().numpy()
        return a


def test_get_audio_embedding():
    hop_size = 0.1
    tol = 1e-5

    # Make sure all embedding types work fine
    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel256",
        content_type="music",
        embedding_size=512,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )

    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel256",
        content_type="music",
        embedding_size=6144,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )

    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel128",
        content_type="music",
        embedding_size=512,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel128",
        content_type="music",
        embedding_size=6144,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="linear",
        content_type="music",
        embedding_size=512,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="linear",
        content_type="music",
        embedding_size=6144,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel256",
        content_type="env",
        embedding_size=512,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel256",
        content_type="env",
        embedding_size=6144,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel128",
        content_type="env",
        embedding_size=512,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel128",
        content_type="env",
        embedding_size=6144,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="linear",
        content_type="env",
        embedding_size=6144,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = torchopenl3.get_audio_embedding(
        audio,
        sr,
        input_repr="linear",
        content_type="env",
        embedding_size=512,
        center=True,
        hop_size=hop_size,
        verbose=True,
    )
    emb1, ts1 = to_numpy(emb1), to_numpy(ts1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[-1] == 512
    assert not np.any(np.isnan(emb1))

    # Make sure we can load a model and pass it in
    model = torchopenl3.models.load_audio_embedding_model("linear", "env", 512)
    emb1load, ts1load = torchopenl3.get_audio_embedding(
        audio, sr, model=model, center=True, hop_size=hop_size, verbose=True
    )
    emb1load, ts1load = to_numpy(emb1load), to_numpy(ts1load)
    assert np.all(np.abs(emb1load - emb1) < tol)
    assert np.all(np.abs(ts1load - ts1) < tol)

    # Make sure that the embeddings are approximately the same with mono and stereo
    audio, sr = sf.read(CHIRP_STEREO_PATH)
    emb2, ts2 = torchopenl3.get_audio_embedding(
        audio, sr, model=model, center=True, hop_size=0.1, verbose=True
    )
    emb2, ts2 = to_numpy(emb2), to_numpy(ts2)
    # assert np.all(np.abs(emb1 - emb2) < tol)
    # assert np.all(np.abs(ts1 - ts2) < tol)
    assert not np.any(np.isnan(emb2))

    # Make sure that the embeddings are approximately the same if we resample the audio
    audio, sr = sf.read(CHIRP_44K_PATH)
    emb3, ts3 = torchopenl3.get_audio_embedding(
        audio, sr, model=model, center=True, hop_size=0.1, verbose=True
    )
    emb3, ts3 = to_numpy(emb3), to_numpy(ts3)
    # assert np.all(np.abs(emb1 - emb3) < tol)
    # assert np.all(np.abs(ts1 - ts3) < tol)
    assert not np.any(np.isnan(emb3))

    # Check for centering
    audio, sr = sf.read(CHIRP_1S_PATH)
    emb6, _ = torchopenl3.get_audio_embedding(
        audio, sr, model=model, center=True, hop_size=hop_size, verbose=True
    )
    n_frames = 1 + int((audio.shape[0] + sr // 2 - sr) / float(int(hop_size * sr)))
    assert emb6.shape[1] == n_frames

    emb7, _ = torchopenl3.get_audio_embedding(
        audio,
        sr,
        model=model,
        center=False,
        hop_size=hop_size,
        verbose=True,
    )
    n_frames = 1 + int((audio.shape[0] - sr) / float(int(hop_size * sr)))
    assert emb7.shape[1] == n_frames

    # Check for hop size
    hop_size = 0.2
    emb8, _ = torchopenl3.get_audio_embedding(
        audio,
        sr,
        model=model,
        center=False,
        hop_size=hop_size,
        verbose=True,
    )
    n_frames = 1 + int((audio.shape[0] - sr) / float(int(hop_size * sr)))
    assert emb8.shape[1] == n_frames

    # Check batch processing with multiple files with a single sample rate
    audio, sr = sf.read(CHIRP_MONO_PATH)
    hop_size = 0.1
    emb_list, ts_list = torchopenl3.get_audio_embedding(
        [audio, audio],
        sr,
        model=model,
        center=True,
        hop_size=hop_size,
        batch_size=4,
    )
    n_frames = 1 + int((audio.shape[0] + sr // 2 - sr) / float(int(hop_size * sr)))
    assert len(emb_list) == 2
    assert len(ts_list) == 2
    assert emb_list[0].shape[0] == n_frames
    assert np.allclose(to_numpy(emb_list[0]), to_numpy(emb_list[1]))
    assert np.allclose(ts_list[0], ts_list[1])

    # Check batch processing with multiple files with individually given sample rates
    emb_list, ts_list = torchopenl3.get_audio_embedding(
        [audio, audio],
        [sr, sr],
        model=model,
        center=True,
        hop_size=hop_size,
        batch_size=4,
    )
    n_frames = 1 + int((audio.shape[0] + sr // 2 - sr) / float(int(hop_size * sr)))
    assert type(emb_list) == list
    assert type(ts_list) == list
    assert len(emb_list) == 2
    assert len(ts_list) == 2
    assert emb_list[0].shape[0] == n_frames
    assert np.allclose(to_numpy(emb_list[0]), to_numpy(emb_list[1]))
    assert np.allclose(ts_list[0], ts_list[1])

    # Check batch processing with multiple files with different sample rates
    emb_list, ts_list = torchopenl3.get_audio_embedding(
        [audio, audio],
        [sr, sr / 2],
        model=model,
        center=True,
        hop_size=hop_size,
        batch_size=4,
    )
    n_frames = 1 + int((audio.shape[0] + sr // 2 - sr) / float(int(hop_size * sr)))
    n_frames_2 = 1 + int(
        (audio.shape[0] + sr // 4 - sr / 2) / float(int(hop_size * sr / 2))
    )
    assert type(emb_list) == list
    assert type(ts_list) == list
    assert len(emb_list) == 2
    assert len(ts_list) == 2
    assert emb_list[0].shape[0] == n_frames
    assert emb_list[1].shape[0] == n_frames_2
