import julius
import resampy
import torch
import logging

TARGET_SR = 48000


def center_audio(audio, frame_len):
    return torch.nn.functional.pad(
        audio, (int(frame_len / 2.0), 0), mode="constant", value=0
    )


def pad_audio(audio, frame_len, hop_len):
    audio_len = audio.size()[1]
    if audio_len < frame_len:
        pad_length = frame_len - audio_len
    else:
        pad_length = torch.ceil(
            torch.tensor((audio_len - frame_len) / float(hop_len))
        ).int() * hop_len - (audio_len - frame_len)

    if pad_length > 0:
        audio = torch.nn.functional.pad(
            audio, (0, pad_length), mode="constant", value=0
        )

    return audio


def preprocess_audio_batch(audio, sr, center=True, hop_size=0.1, sampler="julian"):
    if audio.ndim == 3:
        audio = torch.mean(audio, axis=2)

    if sr != TARGET_SR:
        if sampler == "julian":
            audio = julius.resample_frac(audio, sr, TARGET_SR)

        elif sampler == "resampy":
            logging.warning(
                "To get accurate results we have to move "
                "the data the CPU to use resampy"
            )
            audio = torch.tensor(
                resampy.resample(
                    audio.detach().cpu().numpy(),
                    sr_orig=sr,
                    sr_new=TARGET_SR,
                    filter="kaiser_best",
                ),
                dtype=audio.dtype,
                device=audio.device,
            )

        else:
            raise ValueError("Only julian and resampy works!")

    frame_len = TARGET_SR
    hop_len = int(hop_size * TARGET_SR)
    if center:
        audio = center_audio(audio, frame_len)

    audio = pad_audio(audio, frame_len, hop_len)
    n_frames = 1 + int((audio.size()[1] - frame_len) / float(hop_len))
    x = []
    xframes_shape = None
    for i in range(audio.shape[0]):
        xframes = (
            torch.as_strided(
                audio[i],
                size=(frame_len, n_frames),
                stride=(1, hop_len),
            )
            .transpose(0, 1)
            .unsqueeze(1)
        )
        if xframes_shape is None:
            xframes_shape = xframes.shape
        assert xframes.shape == xframes_shape
        x.append(xframes)
    x = torch.vstack(x)
    return x
