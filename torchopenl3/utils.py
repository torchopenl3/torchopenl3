import numpy as np
import resampy

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

    audio_len = audio.size
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