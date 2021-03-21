import torch.nn as nn
import torch.nn.functional as F

# from nnAudio import Spectrogram

from torchopenl3.custom_spectrogram import CustomSpectrogram


class PytorchOpenl3(nn.Module):
    def __init__(self, input_repr, embedding_size):
        super(PytorchOpenl3, self).__init__()
        self.AUDIO_POOLING_SIZES = {
            "linear": {512: (32, 24), 6144: (8, 8)},
            "mel128": {512: (16, 24), 6144: (4, 8)},
            "mel256": {512: (32, 24), 6144: (8, 8)},
        }
        self.speclayer = CustomSpectrogram(input_repr, n_fft=512, n_hop=242, asr=48000)
        self.input_repr = input_repr
        self.embedding_size = embedding_size
        self.batch_normalization_1 = self.__batch_normalization(
            2, "batch_normalization_1", num_features=1, eps=0.001, momentum=0.99
        )
        self.conv2d_1 = self.__conv(
            2,
            name="conv2d_1",
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_2 = self.__batch_normalization(
            2, "batch_normalization_2", num_features=64, eps=0.001, momentum=0.99
        )
        self.conv2d_2 = self.__conv(
            2,
            name="conv2d_2",
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_3 = self.__batch_normalization(
            2, "batch_normalization_3", num_features=64, eps=0.001, momentum=0.99
        )
        self.conv2d_3 = self.__conv(
            2,
            name="conv2d_3",
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_4 = self.__batch_normalization(
            2, "batch_normalization_4", num_features=128, eps=0.001, momentum=0.99
        )
        self.conv2d_4 = self.__conv(
            2,
            name="conv2d_4",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_5 = self.__batch_normalization(
            2, "batch_normalization_5", num_features=128, eps=0.001, momentum=0.99
        )
        self.conv2d_5 = self.__conv(
            2,
            name="conv2d_5",
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_6 = self.__batch_normalization(
            2, "batch_normalization_6", num_features=256, eps=0.001, momentum=0.99
        )
        self.conv2d_6 = self.__conv(
            2,
            name="conv2d_6",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_7 = self.__batch_normalization(
            2, "batch_normalization_7", num_features=256, eps=0.001, momentum=0.99
        )
        self.conv2d_7 = self.__conv(
            2,
            name="conv2d_7",
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.batch_normalization_8 = self.__batch_normalization(
            2, "batch_normalization_8", num_features=512, eps=0.001, momentum=0.99
        )
        self.audio_embedding_layer = self.__conv(
            2,
            name="audio_embedding_layer",
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )

    def forward(self, x):
        x = self.speclayer(x)
        x = x.unsqueeze(1)
        batch_normalization_1 = self.batch_normalization_1(x)
        conv2d_1_pad = F.pad(batch_normalization_1, (1, 1, 1, 1))
        conv2d_1 = self.conv2d_1(conv2d_1_pad)
        batch_normalization_2 = self.batch_normalization_2(conv2d_1)
        activation_1 = F.relu(batch_normalization_2)
        conv2d_2_pad = F.pad(activation_1, (1, 1, 1, 1))
        conv2d_2 = self.conv2d_2(conv2d_2_pad)
        batch_normalization_3 = self.batch_normalization_3(conv2d_2)
        activation_2 = F.relu(batch_normalization_3)
        max_pooling2d_1 = F.max_pool2d(
            activation_2, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False
        )
        conv2d_3_pad = F.pad(max_pooling2d_1, (1, 1, 1, 1))
        conv2d_3 = self.conv2d_3(conv2d_3_pad)
        batch_normalization_4 = self.batch_normalization_4(conv2d_3)
        activation_3 = F.relu(batch_normalization_4)
        conv2d_4_pad = F.pad(activation_3, (1, 1, 1, 1))
        conv2d_4 = self.conv2d_4(conv2d_4_pad)
        batch_normalization_5 = self.batch_normalization_5(conv2d_4)
        activation_4 = F.relu(batch_normalization_5)
        max_pooling2d_2 = F.max_pool2d(
            activation_4, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False
        )
        conv2d_5_pad = F.pad(max_pooling2d_2, (1, 1, 1, 1))
        conv2d_5 = self.conv2d_5(conv2d_5_pad)
        batch_normalization_6 = self.batch_normalization_6(conv2d_5)
        activation_5 = F.relu(batch_normalization_6)
        conv2d_6_pad = F.pad(activation_5, (1, 1, 1, 1))
        conv2d_6 = self.conv2d_6(conv2d_6_pad)
        batch_normalization_7 = self.batch_normalization_7(conv2d_6)
        activation_6 = F.relu(batch_normalization_7)
        max_pooling2d_3 = F.max_pool2d(
            activation_6, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False
        )
        conv2d_7_pad = F.pad(max_pooling2d_3, (1, 1, 1, 1))
        conv2d_7 = self.conv2d_7(conv2d_7_pad)
        batch_normalization_8 = self.batch_normalization_8(conv2d_7)
        activation_7 = F.relu(batch_normalization_8)
        audio_embedding_layer_pad = F.pad(activation_7, (1, 1, 1, 1))
        audio_embedding_layer = self.audio_embedding_layer(audio_embedding_layer_pad)
        max_pooling2d_4 = F.max_pool2d(
            audio_embedding_layer,
            kernel_size=self.AUDIO_POOLING_SIZES[self.input_repr][self.embedding_size],
            stride=self.AUDIO_POOLING_SIZES[self.input_repr][self.embedding_size],
            padding=0,
            ceil_mode=False,
        )
        return max_pooling2d_4

    def __batch_normalization(self, dim, name, **kwargs):
        if dim == 0 or dim == 1:
            layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:
            layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)
        else:
            raise NotImplementedError()
        return layer

    def __conv(self, dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()
        return layer
