from torchopenl3.models import load_audio_embedding_model
import torch
import torchopenl3


class TestModels:
    def _test_model(self, input_repr, content_type, embedding_size):
        m = load_audio_embedding_model(input_repr, content_type, embedding_size)
        assert m.embedding_size == embedding_size
        assert m.input_repr == input_repr
        if input_repr == "linear":
            for i, child in enumerate(m.children()):
                if i == 0:
                    assert isinstance(child, torchopenl3.models.CustomSTFT)
                elif i % 2 != 0:
                    assert isinstance(child, torch.nn.modules.batchnorm.BatchNorm2d)
                else:
                    assert isinstance(child, torch.nn.modules.conv.Conv2d)
        else:
            for i, child in enumerate(m.children()):
                if i == 0:
                    assert isinstance(child, torchopenl3.models.CustomMelSTFT)
                elif i % 2 != 0:
                    assert isinstance(child, torch.nn.modules.batchnorm.BatchNorm2d)
                else:
                    assert isinstance(child, torch.nn.modules.conv.Conv2d)

    def test_model_env_linear_512(self):
        self._test_model(content_type="env", input_repr="linear", embedding_size=512)

    def test_model_music_linear_512(self):
        self._test_model(content_type="music", input_repr="linear", embedding_size=512)

    def test_model_env_mel128_512(self):
        self._test_model(content_type="env", input_repr="mel128", embedding_size=512)

    def test_model_music_mel128_512(self):
        self._test_model(content_type="music", input_repr="mel128", embedding_size=512)

    def test_model_env_mel256_512(self):
        self._test_model(content_type="env", input_repr="mel256", embedding_size=512)

    def test_model_music_mel256_512(self):
        self._test_model(content_type="music", input_repr="mel256", embedding_size=512)

    def test_model_env_linear_6144(self):
        self._test_model(content_type="env", input_repr="linear", embedding_size=6144)

    def test_model_music_linear_6144(self):
        self._test_model(content_type="music", input_repr="linear", embedding_size=6144)

    def test_model_env_mel128_6144(self):
        self._test_model(content_type="env", input_repr="mel128", embedding_size=6144)

    def test_model_music_mel128_6144(self):
        self._test_model(content_type="music", input_repr="mel128", embedding_size=6144)

    def test_model_env_mel256_6144(self):
        self._test_model(content_type="env", input_repr="mel256", embedding_size=6144)

    def test_model_music_mel256_6144(self):
        self._test_model(content_type="music", input_repr="mel256", embedding_size=6144)
