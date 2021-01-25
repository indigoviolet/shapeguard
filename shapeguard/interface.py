from __future__ import annotations

import builtins
import inspect
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from types import FrameType
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

from lark import LarkError

from .exception import ShapeGuardError
from .guard import ShapeGuard


class InterfaceMeta(type):
    """Metaclass for Interface.

    - We implement __call__ on this metaclass, so that `Interface()`
      is not a constructor but invokes this method instead.

    - We also track the current ShapeGuard and forks as instance
      attributes ie. on the Instance class. -> mainly because
      `__call__` needs to refer to `get()`
    """

    _current: Optional[ShapeGuard] = None
    _all: Dict[Optional[HashableDict], ShapeGuard] = {}
    _noop = False

    @contextmanager
    def noop(self):
        self._noop = True
        yield
        self._noop = False

    def reset(self):
        self._current = None
        self._all = {}

    def get(self) -> ShapeGuard:
        if self._current is None:
            self._current = self.get_base()
        return self._current

    def get_base(self) -> ShapeGuard:
        return self._get(None)

    def get_throwaway(self) -> ShapeGuard:
        return self._get({})

    def _get(self, params: Optional[Dict[str, Any]]) -> ShapeGuard:
        if params is not None and len(params) == 0:
            # throwaway
            return ShapeGuard()
        elif params is None:
            # base
            if None not in self._all:
                self._all[None] = ShapeGuard()
            return self._all[None]
        else:
            # fork
            params = HashableDict(params)
            if params not in self._all:
                self._all[params] = ShapeGuard(params=params)
            return self._all[params]

    def __call__(self, arg, template: Union[str, List[str], Set[str], Tuple[str, ...]]):  # type: ignore[override]
        if self._noop:
            return arg

        try:
            if _is_listy(template):
                assert _is_listy(
                    arg
                ), f"Found sequence template {template}, but non-sequence tensor {type(arg)}"

                assert (
                    len(arg) >= 1
                ), f"Found sequence template {template}, but empty sequence tensor"

                if len(template) == 1:
                    template = list(template) * len(arg)

                assert len(template) == len(
                    arg
                ), f"Found {len(template)} templates, but {len(arg)} args"

                for t, m in zip(arg, template):
                    sg(t, m)
            else:
                assert isinstance(template, str)
                self.get().guard(arg, template)
        except (ShapeGuardError, LarkError) as e:
            _annotate_and_raise(e, inspect.currentframe())

        return arg


def _is_listy(a) -> bool:
    return isinstance(a, Sequence) and not isinstance(a, (str, bytes, bytearray))


class Interface(metaclass=InterfaceMeta):
    def __init__(self, arg, template: Union[str, List[str], Set[str], Tuple[str, ...]]):
        # This only exists to persuade mypy that Interface() has this
        # type -- since its implementation is actually in
        # InterfaceMeta.__call__
        assert False, "Should not be called"

    @classmethod
    @contextmanager
    def fork(cls, **kwargs) -> Generator[ShapeGuard, None, None]:
        prev = cls._current
        fork_sg = cls._get(params=kwargs)
        cls._switch_fork(fork_sg)
        yield fork_sg
        cls._switch_fork(prev)

    @classmethod
    def _switch_fork(cls, nxt: Optional[ShapeGuard]):
        if cls._current:
            cls._checkin_fork(cls._current)
        if nxt:
            cls._checkout_fork(nxt)
        cls._current = nxt

    @classmethod
    def _checkout_fork(cls, fork_sg: ShapeGuard) -> None:
        if fork_sg is not cls.get_base():
            fork_sg.dims.update(cls.get_base().dims)

    @classmethod
    def _checkin_fork(cls, fork_sg: ShapeGuard):
        if fork_sg is not cls.get_base():
            transferred_dims = {k: v for k, v in fork_sg.dims.items() if k[0].isupper()}
            cls.get_base().dims.update(transferred_dims)

    @classmethod
    def install(cls, sg="sg"):
        setattr(builtins, sg, cls)

    @classmethod
    def uninstall(cls, sg="sg"):
        delattr(builtins, sg)


# https://stackoverflow.com/a/16162138/14044156
class HashableDict(dict):
    def __hash__(self):
        return hash((frozenset(self), frozenset(self.values())))


def _annotate_and_raise(
    e: Union[ShapeGuardError, LarkError], current_frame: Optional[FrameType]
):
    """Adds information about the template to the exception"""
    assert current_frame is not None
    offending_frame = current_frame.f_back
    if offending_frame is None or offending_frame.f_code is current_frame.f_code:
        # if we didn't find a frame (weird!) or for a recursive
        # call (list templates), we simply pass it up for the
        # "terminal" sg() call to handle
        raise
    else:
        code_context = inspect.getframeinfo(offending_frame).code_context
        if code_context is None:
            # Probably from a repl
            raise
        offending_line = code_context[0].strip()
        if offending_line is not None:
            # We can't necessarily create other exception types
            # with a single str arg, so we cast them to
            # ShapeGuardError and chain them
            error_type = type(e) if isinstance(e, ShapeGuardError) else ShapeGuardError
            chained_error = None if isinstance(e, ShapeGuardError) else e
            raise error_type(f"\n\t>>> {offending_line}\n {str(e)}").with_traceback(
                sys.exc_info()[2]
            ) from chained_error


sg = Interface
