import inspect
import sys
from typing import Optional, NamedTuple


def get_frame_above(fn: str) -> Optional[inspect.FrameInfo]:
    """ Returns the code for the first frame above fn """

    frames = inspect.getouterframes(sys._getframe(1), context=1)
    found = False
    for f in frames:
        if found:
            return f
        if f.function == fn:
            found = True

    return None
