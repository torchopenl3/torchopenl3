import os

from numbers import Real
from math import ceil

import six
import traceback

import soundfile as sf
import numpy as np

import torch
from torch import tensor as T

from .models import PytorchOpenl3
from .utils import preprocess_audio_batch
from .torchopenl3_exceptions import TorchOpenL3Error

TARGET_SR = 48000


def get_model_path(input_repr, content_type, embedding_size):
    base_url = "https://github.com/torchopenl3/torchopenl3-models/raw/master/"
    return f"{base_url}torchopenl3_{input_repr}_{content_type}_{embedding_size}.pth.tar"


def load_audio_embedding_model(
    input_repr, content_type, embedding_size,
):
    model = PytorchOpenl3(
        input_repr=input_repr, embedding_size=embedding_size, content_type=content_type,
    )

    weight_path = get_model_path(input_repr, content_type, embedding_size)
    model.load_state_dict(torch.hub.load_state_dict_from_url(weight_path))
    model = model.eval()
    return model


def get_audio_embedding(
    audio,
    sr,
    model=None,
    input_repr="mel256",
    content_type="music",
    embedding_size=6144,
    center=True,
    hop_size=0.1,
    batch_size=32,
    verbose=True,
    weight_path="",
    sampler="resampy",
):
    if model is None:
        model = load_audio_embedding_model(input_repr, content_type, embedding_size)

    if torch.cuda.is_available():
        # Won't work with multigpu
        device = "cuda"
    else:
        device = "cpu"

    if isinstance(audio, np.ndarray):
        audio = T(audio, device=device, dtype=torch.float64)
    if isinstance(audio, torch.Tensor):
        # nsounds x nsamples (x nchannels)
        if audio.ndim == 1:
            audio = audio.view(1, -1)
        elif audio.ndim == 2 and audio.shape[1] == 2:
            audio = audio.view(1, audio.shape[0], audio.shape[1])
        assert audio.ndim == 2 or audio.ndim == 3
        nsounds = audio.shape[0]
        if audio.is_cuda:
            model = model.cuda()
        audio = preprocess_audio_batch(audio, sr, center, hop_size, sampler=sampler).to(
            torch.float32
        )
        total_size = audio.size()[0]
        audio_embedding = []
        with torch.set_grad_enabled(False):
            for i in range((total_size // batch_size) + 1):
                small_batch = audio[i * batch_size : (i + 1) * batch_size]
                if small_batch.shape[0] > 0:
                    # print("small_batch.shape", small_batch.shape)
                    audio_embedding.append(model(small_batch))
        audio_embedding = torch.vstack(audio_embedding)
        # This is broken, doesn't use hop-size or center
        ts_list = torch.arange(audio_embedding.size()[0] // nsounds) * hop_size
        ts_list = ts_list.expand(nsounds, audio_embedding.size()[0] // nsounds)

        if audio.is_cuda:
            ts_list = ts_list.cuda()

        assert audio_embedding.shape[0] % nsounds == 0
        assert ts_list.shape[0] % nsounds == 0
        # return nsounds x nframes x ndim
        return (
            audio_embedding.view(nsounds, audio_embedding.shape[0] // nsounds, -1),
            ts_list,
        )
    elif isinstance(audio, list):
        if isinstance(audio[0], np.ndarray):
            for i in range(len(audio)):
                audio[i] = T(audio[i], device=device, dtype=torch.float64)
        if audio[0].is_cuda:
            model = model.cuda()
        audio_list = audio
        if isinstance(sr, Real):
            sr_list = [sr] * len(audio_list)
        elif isinstance(sr, list):
            sr_list = sr

        embedding_list = []
        ts_list = []

        batch = []
        file_batch_size_list = []
        for audio, sr in zip(audio_list, sr_list):
            # nsamples (x nchannels)
            assert audio.ndim == 1 or audio.ndim == 2
            if audio.ndim == 1:
                audio = audio.view(1, -1)
            elif audio.ndim == 2:
                audio = audio.view(1, audio.shape[0], audio.shape[1])
            else:
                assert False
            x = preprocess_audio_batch(
                audio, sr, hop_size=hop_size, center=center, sampler=sampler
            ).to(torch.float32)
            batch.append(x)
            file_batch_size_list.append(x.size()[0])

        batch = torch.vstack(batch)
        total_size = batch.size()[0]
        audio_embeddings = []
        with torch.set_grad_enabled(False):
            for i in range((total_size // batch_size) + 1):
                small_batch = batch[i * batch_size : (i + 1) * batch_size]
                if small_batch.shape[0] > 0:
                    audio_embeddings.append(model(small_batch))

        audio_embeddings = torch.vstack(audio_embeddings)
        # with torch.set_grad_enabled(False):
        #   audio_embeddings = model(batch)
        audio_embeddings = audio_embeddings.view(total_size, -1)
        start_idx = 0
        for file_batch_size in file_batch_size_list:
            end_idx = start_idx + file_batch_size
            embedding = audio_embeddings[start_idx:end_idx, ...]
            # This is broken, doesn't use center
            ts = np.arange(embedding.shape[0]) * hop_size

            embedding_list.append(embedding)
            ts_list.append(ts)
            start_idx = end_idx

        return embedding_list, ts_list
    else:
        assert False


def _get_num_windows(audio_len, frame_len, hop_len, center):
    if center:
        audio_len += int(frame_len / 2.0)

    if audio_len <= frame_len:
        return 1
    else:
        return 1 + int(np.ceil((audio_len - frame_len) / float(hop_len)))


def process_audio_file(
    filepath,
    output_dir=None,
    suffix=None,
    model=None,
    input_repr="mel256",
    content_type="music",
    embedding_size=6144,
    center=True,
    hop_size=0.1,
    batch_size=32,
    overwrite=False,
    verbose=True,
):
    """
    Computes and saves L3 embedding for a given audio file
    Parameters
    ----------
    filepath : str or list[str]
        Path or list of paths to WAV file(s) to be processed.
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    suffix : str or None
        String to be appended to the output filename, i.e. <base filename>_<suffix>.npz.
        If None, then no suffix will be added, i.e. <base filename>.npz.
    model : keras.models.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used as model input. Ignored if `model` is
        a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train the embedding model. Ignored if `model` is
        a valid Keras model.
    embedding_size : 6144 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.
    hop_size : float
        Hop size in seconds.
    batch_size : int
        Batch size used for input to embedding model
    overwrite : bool
        If True, overwrites existing output files
    verbose : bool
        If True, prints verbose messages.
    Returns
    -------
    """
    if isinstance(filepath, six.string_types):
        filepath_list = [filepath]
    elif isinstance(filepath, list):
        filepath_list = filepath
    else:
        err_msg = "filepath should be type str or list[str], but got {}."
        raise TorchOpenL3Error(err_msg.format(filepath))

    if not suffix:
        suffix = ""

    # Load model
    if not model:
        model = load_audio_embedding_model(input_repr, content_type, embedding_size)

    audio_list = []
    sr_list = []
    batch_filepath_list = []

    total_batch_size = 0

    num_files = len(filepath_list)
    for file_idx, filepath in enumerate(filepath_list):
        if not os.path.exists(filepath):
            raise TorchOpenL3Error('File "{}" could not be found.'.format(filepath))

        if verbose:
            print(
                "torchopenl3: Processing {} ({}/{})".format(
                    filepath, file_idx + 1, num_files
                )
            )

        # Skip if overwriting isn't enabled and output file exists
        output_path = get_output_path(filepath, suffix + ".npz", output_dir=output_dir)
        if os.path.exists(output_path) and not overwrite:
            err_msg = "torchopenl3: {} exists and overwriting not enabled, skipping."
            print(err_msg.format(output_path))
            continue

        try:
            audio, sr = sf.read(filepath)
        except Exception:
            err_msg = 'Could not open file "{}":\n{}'
            raise TorchOpenL3Error(err_msg.format(filepath, traceback.format_exc()))

        audio_list.append(audio)
        sr_list.append(sr)
        batch_filepath_list.append(filepath)

        audio_length = ceil(audio.shape[0] / float(TARGET_SR / sr))
        frame_length = TARGET_SR
        hop_length = int(hop_size * TARGET_SR)
        num_windows = _get_num_windows(audio_length, frame_length, hop_length, center)
        total_batch_size += num_windows

        if total_batch_size >= batch_size or file_idx == (num_files - 1):
            embedding_list, ts_list = get_audio_embedding(
                audio_list,
                sr_list,
                model=model,
                input_repr=input_repr,
                content_type=content_type,
                embedding_size=embedding_size,
                center=center,
                hop_size=hop_size,
                batch_size=batch_size,
                verbose=verbose,
            )
            for fpath, embedding, ts in zip(
                batch_filepath_list, embedding_list, ts_list
            ):
                embedding, ts = to_numpy(embedding), to_numpy(ts)
                output_path = get_output_path(
                    fpath, suffix + ".npz", output_dir=output_dir
                )

                np.savez(output_path, embedding=embedding, timestamps=ts)
                assert os.path.exists(output_path)

                if verbose:
                    print("torchopenl3: Saved {}".format(output_path))

            audio_list = []
            sr_list = []
            batch_filepath_list = []
            total_batch_size = 0


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    else:
        try:
            a = a.numpy()
        except Exception:
            a = a.detach().numpy()
        return a


def get_output_path(filepath, suffix, output_dir=None):
    """
    Returns path to output file corresponding to the given input file.
    Parameters
    ----------
    filepath : str
        Path to audio file to be processed
    suffix : str
        String to append to filename (including extension)
    output_dir : str or None
        Path to directory where file will be saved. If None,
        will use directory of given filepath.
    Returns
    -------
    output_path : str
        Path to output file
    """
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    if not output_dir:
        output_dir = os.path.dirname(filepath)

    if suffix[0] != ".":
        output_filename = "{}_{}".format(base_filename, suffix)
    else:
        output_filename = base_filename + suffix

    return os.path.join(output_dir, output_filename)
