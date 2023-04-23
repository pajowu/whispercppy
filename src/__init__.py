"""
whispercpp: A Pybind11-based Python wrapper for whisper.cpp, a C++ implementation of OpenAI Whisper.

Install with pip:

.. code-block:: bash

    pip install whispercpp

The binding provides a ``Whisper`` class:

.. code-block:: python

    from whispercpp import Whisper

    w = Whisper.from_pretrained("tiny.en")
    print(w.transcribe_from_file("test.wav"))
"""

from __future__ import annotations

import os as _os
import typing as t
from dataclasses import dataclass

from . import utils

if t.TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from . import api
else:
    api = utils.LazyLoader("api", globals(), "whispercppy.api_cpp2py_export")


try:  # pragma: no cover
    with open(_os.path.join(_os.path.dirname(__file__), "__about__.py")) as f:
        exec(f.read())
except FileNotFoundError:
    # Fallback for when __about__.py is not available.
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0, "dev0")


@dataclass
class Whisper:
    """
    A wrapper class for Whisper C++ API.

    This class should only be instantiated using ``from_pretrained()``.
    ``__init__()`` will raise a ``RuntimeError``.
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        """Empty init method. This will raise a ``RuntimeError``."""
        raise RuntimeError(
            "Using '__init__()' is not allowed. Use 'from_pretrained()' instead."
        )

    if t.TYPE_CHECKING:
        # The following will be populated by from_pretrained.
        _ref: Whisper
        context: api.Context
        params: api.Params
        no_state: bool
        basedir: str | None
        _transcript: list[str]

    _context_initialized: bool = False

    @staticmethod
    def from_pretrained(
        model_name: str, basedir: str | None = None, no_state: bool = False
    ):
        """Load a preconverted model from a given model name.

        Currently it doesn't support custom preconverted models yet. PRs are welcome.

        Args:
            model_name (str): Name of the preconverted model.
            basedir (str, optional): Base directory to store the model. Defaults to None.
                                     Default will be "$XDG_DATA_HOME/whispercpp" for directory.
            no_state (bool, optional): Whether to initialize the model state. Defaults to False.

        Returns:
            A ``Whisper`` object.

        Raises:
            RuntimeError: If the given model name is not a valid preconverted model.
        """
        if model_name not in utils.MODELS_URL and not _os.path.isfile(model_name):
            raise RuntimeError(
                f"'{model_name}' is not a valid preconverted model or a file path. \
                    Choose one of {list(utils.MODELS_URL)}"
            )
        _ref = object.__new__(Whisper)

        if model_name in utils.MODELS_URL:
            context = api.Context.from_file(
                utils.download_model(model_name, basedir=basedir), no_state=no_state
            )
        else:
            context = api.Context.from_file(model_name, no_state=no_state)

        params = (
            api.Params.from_enum(api.SAMPLING_GREEDY)
            .with_print_progress(False)
            .with_print_realtime(False)
            .build()
        )
        context.reset_timings()
        _context_initialized = not no_state
        _transcript = []
        _ref.__dict__.update(locals())
        return _ref

    @staticmethod
    def from_params(
        model_name: str,
        params: api.Params,
        basedir: str | None = None,
        no_state: bool = False,
    ):
        """Load a preconverted model from a given model name and params.

        Currently it doesn't support custom preconverted models yet. PRs are welcome.

        Args:
            model_name (str): Name of the preconverted model.
            params (api.Params): Params to be passed to Whisper.
            basedir (str, optional): Base directory to store the model. Defaults to None.
                                     Default will be "$XDG_DATA_HOME/whispercpp" for directory.
            no_state (bool, optional): Whether to initialize the model state. Defaults to False.

        Returns:
            A ``Whisper`` object.

        Raises:
            RuntimeError: If the given model name is not a valid preconverted model.
        """
        if model_name not in utils.MODELS_URL and not _os.path.isfile(model_name):
            raise RuntimeError(
                f"'{model_name}' is not a valid preconverted model or a file path. \
                    Choose one of {list(utils.MODELS_URL)}"
            )
        _ref = object.__new__(Whisper)

        if model_name in utils.MODELS_URL:
            context = api.Context.from_file(
                utils.download_model(model_name, basedir=basedir), no_state=no_state
            )
        else:
            context = api.Context.from_file(model_name, no_state=no_state)

        context.reset_timings()
        _context_initialized = not no_state
        _transcript = []
        _ref.__dict__.update(locals())
        return _ref

    def transcribe(
        self, data: NDArray[np.float32], num_proc: int = 1, strict: bool = False
    ):
        """Transcribe audio from a given numpy array.

        Args:
            data (np.ndarray): Audio data as a numpy array.
            num_proc (int, optional): Number of processes to use for transcription. Defaults to 1.
                                      Note that if num_proc > 1, transcription accuracy may decrease.
            strict (bool, optional): If False, then ``context.init_state()`` will be called if no_state=True.
                                     Default to False.

        Returns:
            Transcribed text.
        """
        if strict:
            assert (
                self.context.is_initialized
            ), "strict=True and context is not initialized. Make sure to call 'context.init_state()' before."
        else:
            if not self.context.is_initialized and not self._context_initialized:
                self.context.init_state()
                # NOTE: Make sure context should only be initialized once.
                self._context_initialized = True

        self.context.full_parallel(self.params, data, num_proc)

        return "".join(
            [
                self.context.full_get_segment_text(i)
                for i in range(self.context.full_n_segments())
            ]
        )

    def transcribe_from_file(
        self, filename: str, num_proc: int = 1, strict: bool = False
    ):
        """Transcribe audio from a given file. This function uses a simple C++ implementation for loading audio file.

        Currently only WAV files are supported. PRs are welcome for other format supports.

        See ``Whisper.transcribe()`` for more details.

        Args:
            filename (str): Path to the audio file.
            num_proc (int, optional): Number of processes to use for transcription. Defaults to 1.
                                      Note that if num_proc > 1, transcription accuracy may decrease.
            strict (bool, optional): If False, then ``context.init_state()`` will be called if no_state=True.
                                     Default to False.

        Returns:
            Transcribed text.
        """
        return self.transcribe(
            api.load_wav_file(filename).mono, num_proc=num_proc, strict=strict
        )


__all__ = ["Whisper", "api", "utils"]
