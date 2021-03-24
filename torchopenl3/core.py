# import requests
import os
from numbers import Real

import numpy as np
import torch
import torch.tensor as T

from .models import PytorchOpenl3
from .utils import preprocess_audio_batch


def get_model_path(input_repr, content_type, embedding_size):
    return os.path.join(
        os.path.dirname(__file__),
        "torchopenl3_{}_{}_{}.pth".format(input_repr, content_type, embedding_size),
    )


def load_np_weights(weight_file):
    weights_dict = np.load(weight_file, allow_pickle=True).item()
    # try:
    #    weights_dict = np.load(weight_file, allow_pickle=True).item()
    # except:
    #    weights_dict = np.load(weight_file, encoding="bytes", allow_pickle=True).item()
    return weights_dict


def load_audio_embedding_model(
    input_repr,
    content_type,
    embedding_size,
):
    model = PytorchOpenl3(
        input_repr=input_repr, embedding_size=embedding_size, content_type=content_type
    )
    try:
        weight_path = get_model_path(input_repr, content_type, embedding_size)
        model.load_state_dict(torch.load(weight_path))
    except FileNotFoundError:
        wd = os.path.split(os.path.normcase(__file__))[0]
        # 6144 and 512 weights are the same
        npweights = load_np_weights(
            os.path.join(
                wd,
                f"openl3_{input_repr}_{content_type}_layer_weights.pkl",
            )
        )

        def update_batch_norm(layer, name):
            layer.state_dict()["weight"].copy_(
                torch.from_numpy(npweights[name]["scale"])
            )
            layer.state_dict()["bias"].copy_(torch.from_numpy(npweights[name]["bias"]))
            layer.state_dict()["running_mean"].copy_(
                torch.from_numpy(npweights[name]["mean"])
            )
            layer.state_dict()["running_var"].copy_(
                torch.from_numpy(npweights[name]["var"])
            )

        update_batch_norm(model.batch_normalization_1, "batch_normalization_1")
        update_batch_norm(model.batch_normalization_2, "batch_normalization_2")
        update_batch_norm(model.batch_normalization_3, "batch_normalization_3")
        update_batch_norm(model.batch_normalization_4, "batch_normalization_4")
        update_batch_norm(model.batch_normalization_5, "batch_normalization_5")
        update_batch_norm(model.batch_normalization_6, "batch_normalization_6")
        update_batch_norm(model.batch_normalization_7, "batch_normalization_7")
        update_batch_norm(model.batch_normalization_8, "batch_normalization_8")

        def update_conv(layer, name):
            layer.state_dict()["weight"].copy_(
                torch.from_numpy(npweights[name]["weights"])
            )
            layer.state_dict()["bias"].copy_(torch.from_numpy(npweights[name]["bias"]))

        update_conv(model.conv2d_1, "conv2d_1")
        update_conv(model.conv2d_2, "conv2d_2")
        update_conv(model.conv2d_3, "conv2d_3")
        update_conv(model.conv2d_4, "conv2d_4")
        update_conv(model.conv2d_5, "conv2d_5")
        update_conv(model.conv2d_6, "conv2d_6")
        update_conv(model.conv2d_7, "conv2d_7")
        update_conv(model.audio_embedding_layer, "audio_embedding_layer")

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
):
    if model is None:
        model = load_audio_embedding_model(input_repr, content_type, embedding_size)

    if isinstance(audio, np.ndarray):
        audio = torch.Tensor(audio)
        if torch.cuda.is_available():
            audio = audio.cuda()
    if isinstance(audio, torch.Tensor):
        if audio.is_cuda:
            model = model.cuda()
        audio = preprocess_audio_batch(audio, sr, center, hop_size)
        total_size = audio.size()[0]
        audio_embedding = []
        with torch.set_grad_enabled(False):
            for i in range((total_size // batch_size) + 1):
                small_batch = audio[i * batch_size : (i + 1) * batch_size]
                audio_embedding.append(model(small_batch))
        audio_embedding = torch.vstack(audio_embedding)
        if audio.is_cuda:
            ts_list = torch.arange(audio_embedding.size()[0]).cuda()
        else:
            ts_list = torch.arange(audio_embedding.size()[0])
        return audio_embedding, ts_list

    if isinstance(audio, list):
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
            x = preprocess_audio_batch(
                T(audio, device="cuda"), sr, hop_size=hop_size, center=center
            )
            batch.append(x)
            file_batch_size_list.append(x.size()[0])

        batch = torch.vstack(batch)
        batch_embedding = batch
        #        total_size = batch.size()[0]
        #        batch_embedding = []
        #        with torch.set_grad_enabled(False):
        #            for i in range((total_size // batch_size) + 1):
        #                small_batch = batch[i * batch_size : (i + 1) * batch_size]
        #                batch_embedding.append(model(small_batch).detach().numpy())
        #        batch_embedding = np.vstack(batch_embedding)
        batch_embedding = batch_embedding.swapaxes(1, 2).swapaxes(2, 3)
        batch_embedding = batch_embedding.reshape(total_size, -1)
        start_idx = 0
        for file_batch_size in file_batch_size_list:
            end_idx = start_idx + file_batch_size
            embedding = batch_embedding[start_idx:end_idx, ...]
            ts = np.arange(embedding.shape[0]) * hop_size

            embedding_list.append(embedding)
            ts_list.append(ts)
            start_idx = end_idx

        return embedding_list, ts_list
