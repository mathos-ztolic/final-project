from typing import Optional, cast
from collections.abc import Generator

def prev_curr_next_iter[T](
    lst: list[T]
) -> Generator[tuple[Optional[T], T, Optional[T]], None, None]:
    if not lst:
        return
    if len(lst) == 1:
        yield (None, lst[0], None)
        return
    yield (None, lst[0], lst[1])
    for i in range(0, len(lst)-2):
        yield cast(tuple[T, T, T], tuple(lst[i:i+3]))
    yield (lst[-2], lst[-1], None)