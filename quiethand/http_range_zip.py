"""Fail-closed, seekable HTTP Range reader for selective ZIP access."""

from __future__ import annotations

import io
from http.cookiejar import CookieJar
import re
import time
from typing import Callable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_ALLOWED_SOURCE_HOSTS = {"www.dropbox.com", "dropbox.com"}
_ALLOWED_RESPONSE_SUFFIXES = (".dropboxusercontent.com",)


class HTTPRangeError(RuntimeError):
    """A remote object did not satisfy the pinned byte-range contract."""


class HTTPRangeReader(io.RawIOBase):
    """Expose one immutable HTTPS object as a seekable read-only stream."""

    def __init__(
        self,
        url: str,
        size: int,
        *,
        max_request_bytes: int = 128 * 1024**2,
        prefetch_bytes: int = 32 * 1024**2,
        timeout_seconds: int = 120,
        opener: Callable[..., object] | None = None,
    ) -> None:
        super().__init__()
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_SOURCE_HOSTS:
            raise HTTPRangeError("range source must be a pinned Dropbox HTTPS URL")
        if size <= 0:
            raise HTTPRangeError("range source size must be positive")
        if max_request_bytes <= 0:
            raise HTTPRangeError("range request cap must be positive")
        if prefetch_bytes <= 0 or prefetch_bytes > max_request_bytes:
            raise HTTPRangeError("range prefetch must fit inside the request cap")
        self._url = url
        self._size = size
        self._position = 0
        self._max_request_bytes = max_request_bytes
        self._prefetch_bytes = prefetch_bytes
        self._timeout_seconds = timeout_seconds
        self._opener = opener or build_opener(HTTPCookieProcessor(CookieJar())).open
        self._cache_start = 0
        self._cache = b""
        self.bytes_fetched = 0
        self.requests_made = 0

    def _open_range(self, request: Request):
        last_error: URLError | None = None
        for attempt in range(3):
            try:
                return self._opener(request, timeout=self._timeout_seconds)
            except URLError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(attempt + 1)
        assert last_error is not None
        raise HTTPRangeError("Dropbox range connection failed after three attempts") from last_error

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError("invalid seek mode")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed range reader")
        if self._position >= self._size:
            return b""
        if size is None or size < 0:
            requested = self._size - self._position
        else:
            requested = min(size, self._size - self._position)
        if requested == 0:
            return b""
        if requested > self._max_request_bytes:
            raise HTTPRangeError("single HTTP range request exceeds the hard cap")

        start = self._position
        cache_end = self._cache_start + len(self._cache)
        if self._cache_start <= start and start + requested <= cache_end:
            offset = start - self._cache_start
            payload = self._cache[offset : offset + requested]
            self._position += requested
            return payload

        fetched = min(
            self._size - start,
            max(requested, self._prefetch_bytes),
        )
        if fetched > self._max_request_bytes:
            fetched = requested
        end = start + fetched - 1
        request = Request(
            self._url,
            method="GET",
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "User-Agent": "QuietHand-M3/1.0",
            },
        )
        with self._open_range(request) as response:
            status = getattr(response, "status", None)
            if status != 206:
                raise HTTPRangeError("server did not honor the byte range")
            final = urlparse(response.geturl())
            if final.scheme != "https" or not (
                final.hostname in _ALLOWED_SOURCE_HOSTS
                or any(
                    final.hostname and final.hostname.endswith(suffix)
                    for suffix in _ALLOWED_RESPONSE_SUFFIXES
                )
            ):
                raise HTTPRangeError("Dropbox range request redirected off the allowlist")
            content_range = response.headers.get("Content-Range", "")
            match = _CONTENT_RANGE.fullmatch(content_range)
            if match is None:
                raise HTTPRangeError("missing or malformed Content-Range")
            observed = tuple(int(value) for value in match.groups())
            if observed != (start, end, self._size):
                raise HTTPRangeError("Content-Range differs from the pinned object")
            declared = response.headers.get("Content-Length")
            if declared is None or not declared.isdigit() or int(declared) != fetched:
                raise HTTPRangeError("range response length declaration is invalid")
            fetched_payload = response.read(fetched + 1)
            if len(fetched_payload) != fetched:
                raise HTTPRangeError("range response ended at the wrong byte")

        self._cache_start = start
        self._cache = fetched_payload
        payload = fetched_payload[:requested]
        self._position += requested
        self.bytes_fetched += fetched
        self.requests_made += 1
        return payload
