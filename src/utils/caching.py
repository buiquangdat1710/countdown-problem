import contextlib
import hashlib
import logging
import pickle
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CacheManager:
    """Manages file-based caching for function results."""

    def __init__(self, cache_dir: str | Path = "cache", max_age: int | None = None):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_age: Maximum age of cache files in seconds (None for no expiration)
        """
        self.cache_dir = Path(cache_dir)
        self.max_age = max_age
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key for function call.

        Args:
            func: The function being cached
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Unique cache key string
        """
        # Create a unique key based on function name, args, and kwargs
        key_data = {
            "func_name": func.__name__,
            "module": func.__module__,
            "args": args,
            "kwargs": kwargs,
        }

        # Serialize to bytes for consistent hashing
        key_bytes = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)

        # Generate hash
        return hashlib.sha256(key_bytes).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key.

        Args:
            cache_key: The cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is valid and not expired.

        Args:
            cache_path: Path to cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False

        if self.max_age is None:
            return True

        # Check if file is within max_age
        file_age = time.time() - cache_path.stat().st_mtime
        return file_age <= self.max_age

    def _save_to_cache(self, cache_path: Path, data: Any) -> None:
        """Save data to cache file using pickle serialization.

        Args:
            cache_path: Path to save cache file
            data: Data to cache
        """
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.debug(f"Cached result to {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")

    def _load_from_cache(self, cache_path: Path) -> Any:
        """Load data from cache file using pickle deserialization.

        Args:
            cache_path: Path to cache file

        Returns:
            Cached data
        """
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            logger.debug(f"Loaded cached result from {cache_path}")
            return data

        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            # Remove corrupted cache file
            with contextlib.suppress(Exception):
                cache_path.unlink()
            raise

    def get_cached_result(
        self, func: Callable, args: tuple, kwargs: dict
    ) -> tuple[bool, Any]:
        """Get cached result if available and valid.

        Args:
            func: The function being cached
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Tuple of (cache_hit: bool, result: Any)
        """
        cache_key = self._get_cache_key(func, args, kwargs)
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            try:
                result = self._load_from_cache(cache_path)
                return True, result
            except Exception:
                # Cache loading failed, will recompute
                pass

        return False, None

    def save_result(
        self, func: Callable, args: tuple, kwargs: dict, result: Any
    ) -> None:
        """Save function result to cache using pickle serialization.

        Args:
            func: The function being cached
            args: Positional arguments
            kwargs: Keyword arguments
            result: Result to cache
        """
        cache_key = self._get_cache_key(func, args, kwargs)
        cache_path = self._get_cache_path(cache_key)
        self._save_to_cache(cache_path, result)

    def clear_cache(self, func: Callable | None = None) -> None:
        """Clear cache files.

        Args:
            func: If provided, only clear cache for this function. Otherwise clear all.
        """
        if func is None:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    try:
                        cache_file.unlink()
                        logger.debug(f"Removed cache file {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        else:
            # Clear cache for specific function (this is approximate since we don't have args/kwargs)
            func_pattern = f"*{func.__name__}*"
            for cache_file in self.cache_dir.glob(func_pattern):
                if cache_file.is_file():
                    try:
                        cache_file.unlink()
                        logger.debug(f"Removed cache file {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {cache_file}: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*"))
        total_files = len([f for f in cache_files if f.is_file()])
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


def cached(
    cache_dir: str | Path = "cache",
    max_age: int | None = None,
    ignore_errors: bool = True,
) -> Callable[[F], F]:
    """Decorator to cache function results to disk using pickle serialization.

    Args:
        cache_dir: Directory to store cache files
        max_age: Maximum age of cache files in seconds (None for no expiration)
        ignore_errors: If True, ignore cache errors and proceed with function execution

    Returns:
        Decorated function with caching capability
    """
    cache_manager = CacheManager(cache_dir, max_age)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Try to get cached result
                cache_hit, cached_result = cache_manager.get_cached_result(
                    func, args, kwargs
                )

                if cache_hit:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result

            except Exception as e:
                if not ignore_errors:
                    raise
                logger.warning(f"Cache read error for {func.__name__}: {e}")

            # Execute function
            logger.debug(f"Cache miss for {func.__name__}, executing function")
            result = func(*args, **kwargs)

            try:
                # Save result to cache
                cache_manager.save_result(func, args, kwargs, result)

            except Exception as e:
                if not ignore_errors:
                    raise
                logger.warning(f"Cache write error for {func.__name__}: {e}")

            return result

        # Add cache management methods to the wrapped function
        wrapper.clear_cache = lambda: cache_manager.clear_cache(func)
        wrapper.clear_all_cache = lambda: cache_manager.clear_cache()
        wrapper.get_cache_stats = cache_manager.get_cache_stats

        return wrapper

    return decorator


# Convenience decorators with common configurations
def cached_temporary(max_age: int = 3600, cache_dir: str | Path = "temp_cache"):
    """Convenience decorator for temporary caching (1 hour default)."""
    return cached(cache_dir=cache_dir, max_age=max_age)
