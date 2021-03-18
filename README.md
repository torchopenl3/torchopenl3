# torchopenl3

[forthcoming]

For developers:
```
pip3 install -e ".[dev]"
```

If that doesn't work, you might need to first run:
```
pip3 install "Cython>=0.23.4"
```

Make sure you have pre-commit hooks installed:
```
pre-commit install
```
This helps us avoid checking dirty jupyter notebook cells into the
repo.

If you need an environment to play around with openl3 and are having
trouble installing its strange brew of requirements, use this docker:

```
docker pull turian/torchopenl3
# Or, build the docker yourself
#docker build -t turian/torchopenl3 .
docker run --mount source=`pwd`,target=/home/openl3/,type=bind -it turian/torchopenl3 bash
```
