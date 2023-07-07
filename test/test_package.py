def test_package():
    try:
        import argparse
        import logging
        import os
        import pickle
        import tarfile

        import black
        import flake8
        import isort
        import numpy
        import pandas
        import scipy
        import six
        import sklearn

        import modules

    except ModuleNotFoundError:
        assert False
