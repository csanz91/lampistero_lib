import functools
import logging
import time
from typing import Callable


logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 5,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for delay after each retry
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = Exception("Function failed after all attempts")
            delay = initial_delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:  # Don't sleep after the last attempt
                        sleep_time = delay

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: "
                            f"{str(e)}. Retrying in {sleep_time:.2f} seconds..."
                        )
                        time.sleep(sleep_time)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {str(e)}"
                        )

            # If we get here, all attempts failed
            raise last_exception

        return wrapper

    return decorator