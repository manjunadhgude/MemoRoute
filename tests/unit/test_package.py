
from pathlib import Path

import memoroute


def test_memoroute_package_is_importable():
    assert memoroute.__file__ is not None
    assert Path(memoroute.__file__).exists()
