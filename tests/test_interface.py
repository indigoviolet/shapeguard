import pytest
from shapeguard import sg


@pytest.fixture(autouse=True)
def reset_singletons():
    sg.reset()


def test_base():
    sg([1, 2], "A,B")
    assert sg.get().dims == {"A": 1, "B": 2}
    assert sg.get().params == {}


def test_fork():
    with sg.fork(foo=1):
        sg([1, 2], "a,b")
        assert sg.get().dims == {"a": 1, "b": 2}
        assert sg.get().params == {"foo": 1}
    with sg.fork(foo=2):
        sg([3, 4], "a,b")
        assert sg.get().dims == {"a": 3, "b": 4}
        assert sg.get().params == {"foo": 2}


def test_base_propagates_to_fork():
    sg([1, 2], "A,B")
    with sg.fork(foo=1):
        sg([3, 4, 1, 2], "a,b,A,B")
        assert sg.get().dims == {"a": 3, "b": 4, "A": 1, "B": 2}


def test_fork_propagates_to_base():
    sg([1, 2], "A,B")
    with sg.fork(foo=1):
        sg([3, 4, 1, 2], "C,d,A,B")

    assert sg.get().dims == {"C": 3, "A": 1, "B": 2}


def test_noop():
    with sg.noop():
        sg([1, 2], "A,B")
        assert sg.get().dims == {}
