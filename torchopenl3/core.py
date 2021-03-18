import numpy as np
from .model import PytorchOpenl3
from .utils import preprocess_audio_batch
from numbers import Real
import torch
import requests
import os


def get_audio_embedding(audio, sr, model=None, input_repr="mel256",
                        content_type="music", embedding_size=6144,
                        center=True, hop_size=0.1, batch_size=32,
                        verbose=True, weight_path=''):
    if isinstance(audio, np.ndarray):
        audio_list = [audio]
        list_input = False
    elif isinstance(audio, list):
        audio_list = audio
        list_input = True

    if isinstance(sr, Real):
        sr_list = [sr] * len(audio_list)
    elif isinstance(sr, list):
        sr_list = sr

    model = PytorchOpenl3(input_repr, embedding_size, weight_path).eval()

    embedding_list = []
    ts_list = []

    batch = []
    file_batch_size_list = []
    for audio, sr in zip(audio_list, sr_list):
        x = preprocess_audio_batch(
            audio, sr, hop_size=hop_size, center=center, input_repr=input_repr, content_type=content_type, embedding_size=embedding_size)
        batch.append(x)
        file_batch_size_list.append(x.shape[0])

    batch = np.vstack(batch)
    total_size = batch.shape[0]
    batch_embedding = []
    with torch.set_grad_enabled(False):
        for i in range((total_size//batch_size) + 1):
            small_batch = batch[i*batch_size:(i+1)*batch_size]
            small_batch = torch.tensor(small_batch).float()
            batch_embedding.append(
                model(small_batch).detach().numpy())
    batch_embedding = np.vstack(batch_embedding)
    start_idx = 0
    for file_batch_size in file_batch_size_list:
        end_idx = start_idx + file_batch_size
        embedding = batch_embedding[start_idx:end_idx, ...]
        ts = np.arange(embedding.shape[0]) * hop_size

        embedding_list.append(embedding)
        ts_list.append(ts)
        start_idx = end_idx

    if not list_input:
        return embedding_list[0], ts_list[0]
    else:
        return embedding_list, ts_list
