import logging
import time
import asyncio
import functools

default_logger = logging.getLogger

def log_execution_time(func):
    logger = logging.getLogger(func.__module__)
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"'{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"'{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper