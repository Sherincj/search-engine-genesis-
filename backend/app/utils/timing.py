"""
Timing utilities for measuring latency.

Provides both a context manager and a decorator so any code path
can be instrumented with minimal effort.
"""

import time
import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable

logger = logging.getLogger(__name__)


@contextmanager
def timer(label: str = "operation"):
    """
    Context manager that logs elapsed wall-clock time.

    Usage::

        with timer("search"):
            results = engine.search(query)
    """
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s completed in %.2f ms", label, elapsed_ms)


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that logs execution time of a function.

    Usage::

        @timed
        def build_index(documents):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s completed in %.2f ms", func.__qualname__, elapsed_ms)
        return result

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s completed in %.2f ms", func.__qualname__, elapsed_ms)
        return result

    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper
