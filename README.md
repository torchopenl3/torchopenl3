# torchopenl3

WARNING: Because of weird kapre 0.1.3 STFT implementation, only 'linear' models have low mean-absolute-error:

```
0.00027512445 27 {'content_type': 'env', 'input_repr': 'linear', 'embedding_size': 512}
0.00035646933 27 {'content_type': 'music', 'input_repr': 'linear', 'embedding_size': 512}
0.0004159477 27 {'content_type': 'env', 'input_repr': 'linear', 'embedding_size': 6144}
0.00050214445 27 {'content_type': 'music', 'input_repr': 'linear', 'embedding_size': 6144}
0.019344347 27 {'content_type': 'env', 'input_repr': 'mel128', 'embedding_size': 6144}
0.021740034 27 {'content_type': 'env', 'input_repr': 'mel128', 'embedding_size': 512}
0.025367895 27 {'content_type': 'music', 'input_repr': 'mel128', 'embedding_size': 6144}
0.03770243 27 {'content_type': 'music', 'input_repr': 'mel128', 'embedding_size': 512}
0.039656915 27 {'content_type': 'env', 'input_repr': 'mel256', 'embedding_size': 6144}
0.05372108 27 {'content_type': 'music', 'input_repr': 'mel256', 'embedding_size': 6144}
0.07465961 27 {'content_type': 'env', 'input_repr': 'mel256', 'embedding_size': 512}
0.11598873 27 {'content_type': 'music', 'input_repr': 'mel256', 'embedding_size': 512}
```

## Development$

We want `pytest` to pass, specifically we want `tests/test_regression.py`
to run to demonstrate that torchopenl3 API matches the original
openl3 API.

Here is what you need to do to get `pytest` running.

Make sure you have pre-commit hooks installed:
```
pre-commit install
```
This helps us avoid checking dirty jupyter notebook cells into the
repo.

Install the package with all dev libraries (i.e. tensorflow openl3)
```
pip3 install -e ".[dev]"
```

If that doesn't work, you might need to first run:
```
pip3 install "Cython>=0.23.4"
```

If it works and you can run:
```
pytest
```
Then you can do regression testing of torchopenl3 API vs openl3..

If that doesn't work (and it might not because openl3 has tricky
requirements), install Docker and work within the Docker environment.
Unfortunately this Docker image is quite big (about 4 GB) because
of pytorch AND tensorflow dependencies AND openl3 models, but you
only need to download it once:

```
docker pull turian/torchopenl3
# Or, build the docker yourself
#docker build -t turian/torchopenl3 .
docker run --mount source=`pwd`,target=/home/openl3/,type=bind -it turian/torchopenl3 bash
```

Inside docker, run:
```
pip3 install -e ".[dev]"
pytest 2>&1 | less
```

If it says `killed`, increase Docker memory.

If none of this works, ask for help.
