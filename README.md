# torchopenl3

[forthcoming]

## Development

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
requirements), install Docker and work within the Docker environment:

```
docker pull turian/torchopenl3
# Or, build the docker yourself
#docker build -t turian/torchopenl3 .
docker run --mount source=`pwd`,target=/home/openl3/,type=bind -it turian/torchopenl3 bash
```

Inside docker, run:
```
pip3 install -e ".[dev]"
pytest
```

If it says `killed`, increase Docker memory.

If none of this works, ask for help.
