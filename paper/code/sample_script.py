def normpath(path: str, options: Options) -> str:
    """Convert path to absolute; but to relative in bazel mode.

    (Bazel's distributed cache doesn't like filesystem metadata to
    end up in output files.)
    """
    # TODO: Could we always use relpath?  (A worry in non-bazel
    # mode would be that a moved file may change its full module
    # name without changing its size, mtime or hash.)
    if options.bazel:
        return os.path.relpath(path)
    else:
        return os.path.abspath(path)