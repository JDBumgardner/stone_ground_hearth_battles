import asyncio
import functools
import logging
from typing import Awaitable, Optional, TypeVar


def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as e:
        asyncio.set_event_loop(asyncio.new_event_loop())
        return asyncio.get_event_loop()

# From https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/
T = TypeVar('T')

def create_task(
    coroutine: Awaitable[T],
    *,
    logger: logging.Logger,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> 'asyncio.Task[T]':  # This type annotation has to be quoted for Python < 3.9, see https://www.python.org/dev/peps/pep-0585/
    '''
    This helper function wraps a ``loop.create_task(coroutine())`` call and ensures there is
    an exception handler added to the resulting task. If the task raises an exception it is logged
    using the provided ``logger``.
    '''
    if loop is None:
        loop = asyncio.get_running_loop()
    task = loop.create_task(coroutine)
    task.add_done_callback(
        functools.partial(_handle_task_result, logger=logger)
    )
    return task


def _handle_task_result(
    task: asyncio.Task,
    *,
    logger: logging.Logger,
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    # Ad the pylint ignore: we want to handle all exceptions here so that the result of the task
    # is properly logged. There is no point re-raising the exception in this callback.
    except Exception:  # pylint: disable=broad-except
        logger.exception("Exception in asyncio task:")
