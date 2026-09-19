#!/usr/bin/env python3
"""Serve local review artifacts with HTTP byte-range support for video."""

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or "Range" not in self.headers:
            return super().send_head()

        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(source.fileno()).st_size
        try:
            unit, requested = self.headers["Range"].split("=", 1)
            if unit != "bytes" or "," in requested:
                raise ValueError
            start_text, end_text = requested.split("-", 1)
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            else:
                suffix = int(end_text)
                start = max(0, size - suffix)
                end = size - 1
            if start < 0 or start >= size or end < start:
                raise IndexError
            end = min(end, size - 1)
        except (ValueError, IndexError):
            source.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.fstat(source.fileno()).st_mtime))
        self.end_headers()
        self._requested_range = (start, end)
        return source

    def copyfile(self, source, outputfile):
        requested = getattr(self, "_requested_range", None)
        if requested is None:
            return super().copyfile(source, outputfile)
        start, end = requested
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)
        del self._requested_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--directory", default=".")
    args = parser.parse_args()
    handler = lambda *handler_args, **kwargs: RangeRequestHandler(
        *handler_args, directory=args.directory, **kwargs
    )
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
