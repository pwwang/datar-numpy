# datar-numpy

The numpy backend for [datar][1].

Note that only `base` APIs are implemented.

## Installation

```bash
pip install -U datar-numpy
# or
pip install -U datar[numpy]
```

## Usage

```python
from datar.base import ceiling

# without it
ceiling(1.2)  # NotImplementedByCurrentBackendError

# with it
ceiling(1.2)  # 2
```

[1]: https://github.com/pwwang/datar
