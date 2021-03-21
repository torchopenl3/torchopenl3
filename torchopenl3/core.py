import numpy as np
from .model import PytorchOpenl3
from .utils import preprocess_audio_batch
from numbers import Real
import torch
import requests
import os


def get_model_path(input_repr,content_type, embedding_size):
    return os.path.join(os.path.dirname(__file__), "torchopenl3_{}_{}_{}.pth".format(input_repr, content_type,embedding_size))

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
    if model==None:
        weight_path = get_model_path(input_repr, content_type,embedding_size)
        model = PytorchOpenl3(input_repr, embedding_size)
        model.load_state_dict(torch.load(weight_path))
        model = model.eval()
        
    if isinstance(audio, np.ndarray):
        audio = torch.Tensor(audio)
        if torch.cuda.is_available():
            audio = audio.cuda()
    if isinstance(audio,torch.Tensor):
        if audio.is_cuda:
            model = model.cuda()
        audio = preprocess_audio_batch(audio,sr,center,hop_size)
        total_size = audio.size()[0]
        audio_embedding = []
        with torch.set_grad_enabled(False):
            for i in range((total_size // batch_size) + 1):
                small_batch = audio[i * batch_size: (i + 1) * batch_size]
                audio_embedding.append(
                    model(small_batch))
        audio_embedding = torch.vstack(audio_embedding)
        if audio.is_cuda:
            ts_list = torch.arange(audio_embedding.size()[0]).cuda()
        else:
            ts_list = torch.arange(audio_embedding.size()[0])
        return audio_embedding,ts_list
            
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
                torch.Tensor(audio),
                sr,
                hop_size=hop_size,
                center=center
            )
            batch.append(x)
            file_batch_size_list.append(x.size()[0])

        batch = torch.vstack(batch)
        total_size = batch.size()[0]
        batch_embedding = []
        with torch.set_grad_enabled(False):
            for i in range((total_size // batch_size) + 1):
                small_batch = batch[i * batch_size : (i + 1) * batch_size]
                batch_embedding.append(model(small_batch).detach().numpy())
        batch_embedding = np.vstack(batch_embedding)
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
