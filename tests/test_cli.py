import pytest
import os
from torchopenl3.cli import (
    positive_int,
    positive_float,
    get_file_list,
    parse_args,
    main,
)
from argparse import ArgumentTypeError
from torchopenl3.torchopenl3_exceptions import TorchOpenL3Error
import tempfile
import numpy as np
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, "data", "audio")

CHIRP_MONO_PATH = os.path.join(TEST_AUDIO_DIR, "chirp_mono.wav")
CHIRP_STEREO_PATH = os.path.join(TEST_AUDIO_DIR, "chirp_stereo.wav")
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, "chirp_44k.wav")
CHIRP_1S_PATH = os.path.join(TEST_AUDIO_DIR, "chirp_1s.wav")
EMPTY_PATH = os.path.join(TEST_AUDIO_DIR, "empty.wav")
SHORT_PATH = os.path.join(TEST_AUDIO_DIR, "short.wav")
SILENCE_PATH = os.path.join(TEST_AUDIO_DIR, "silence.wav")


def test_positive_float():

    # test that returned value is float
    f = positive_float(5)
    assert f == 5.0
    assert type(f) is float

    # test it works for valid strings
    f = positive_float("1.3")
    assert f == 1.3
    assert type(f) is float

    # make sure error raised for all invalid values:
    invalid = [-5, -1.0, None, "hello"]
    for i in invalid:
        pytest.raises(ArgumentTypeError, positive_float, i)


def test_positive_int():
    # test that returned value is int
    i = positive_int(5)
    assert i == 5
    assert type(i) is int

    i = positive_int(5.0)
    assert i == 5
    assert type(i) is int

    # test it works for valid strings
    i = positive_int("1")
    assert i == 1
    assert type(i) is int

    # make sure error raised for all invalid values:
    invalid = [-5, -1.0, None, "hello"]
    for i in invalid:
        pytest.raises(ArgumentTypeError, positive_int, i)


def test_get_file_list():

    # test for invalid input (must be iterable, e.g. list)
    pytest.raises(ArgumentTypeError, get_file_list, CHIRP_44K_PATH)

    # test for valid list of file paths
    flist = get_file_list([CHIRP_44K_PATH, CHIRP_1S_PATH])
    assert len(flist) == 2
    assert flist[0] == CHIRP_44K_PATH and flist[1] == CHIRP_1S_PATH

    # test for valid folder
    flist = get_file_list([TEST_AUDIO_DIR])
    assert len(flist) == 7

    flist = sorted(flist)
    assert flist[0] == CHIRP_1S_PATH
    assert flist[1] == CHIRP_44K_PATH
    assert flist[2] == CHIRP_MONO_PATH
    assert flist[3] == CHIRP_STEREO_PATH
    assert flist[4] == EMPTY_PATH
    assert flist[5] == SHORT_PATH
    assert flist[6] == SILENCE_PATH

    # combine list of files and folders
    flist = get_file_list([TEST_AUDIO_DIR, CHIRP_44K_PATH])
    assert len(flist) == 8

    # nonexistent path
    pytest.raises(TorchOpenL3Error, get_file_list, ["/fake/path/to/file"])


def test_parse_args():

    # test for all the defaults
    args = [CHIRP_44K_PATH]
    args = parse_args(args)
    assert args.inputs == [CHIRP_44K_PATH]
    assert args.output_dir is None
    assert args.suffix is None
    assert args.input_repr == "mel256"
    assert args.content_type == "music"
    assert args.audio_embedding_size == 6144
    assert args.no_audio_centering is False
    assert args.audio_hop_size == 0.1
    assert args.quiet is False


def test_main():

    tempdir = tempfile.mkdtemp()
    with patch(
        "sys.argv",
        ["torchopenl3", CHIRP_44K_PATH, "--output-dir", tempdir],
    ):
        main()

    # check output file created
    outfile = os.path.join(tempdir, "chirp_44k.npz")
    assert os.path.isfile(outfile)

    # This need to generate a regression save path after that we can run this assertion
    # regression test
    # data_reg = np.load(REG_CHIRP_44K_PATH)
    # data_out = np.load(outfile)

    # assert sorted(data_out.files) == sorted(data_out.files) == sorted(
    #     ['embedding', 'timestamps'])
    # assert np.allclose(data_out['timestamps'], data_reg['timestamps'],
    #                    rtol=1e-05, atol=1e-05, equal_nan=False)
    # assert np.allclose(data_out['embedding'], data_reg['embedding'],
    #                    rtol=1e-05, atol=1e-05, equal_nan=False)


def test_script_main():

    # Duplicate audio regression test from test_run just to hit coverage
    tempdir = tempfile.mkdtemp()
    with patch(
        "sys.argv",
        ["torchopenl3", CHIRP_44K_PATH, "--output-dir", tempdir],
    ):
        import torchopenl3.__main__

    # check output file created
    outfile = os.path.join(tempdir, "chirp_44k.npz")
    assert os.path.isfile(outfile)

    # This need to generate a regression save path after that we can run this assertion
    # regression test
    # data_reg = np.load(REG_CHIRP_44K_PATH)
    # data_out = np.load(outfile)

    # assert sorted(data_out.files) == sorted(data_out.files) == sorted(
    #     ['embedding', 'timestamps'])
    # assert np.allclose(data_out['timestamps'], data_reg['timestamps'],
    #                    rtol=1e-05, atol=1e-05, equal_nan=False)
    # assert np.allclose(data_out['embedding'], data_reg['embedding'],
    #                    rtol=1e-05, atol=1e-05, equal_nan=False)
