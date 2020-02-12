# -*- coding: utf-8 -*-

import pytest
from rfm_deployment.skeleton import fib

__author__ = "mmontero"
__copyright__ = "mmontero"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
