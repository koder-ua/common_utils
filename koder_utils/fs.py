import atexit
import os
import stat
from pathlib import Path
from typing import List, Union, BinaryIO, TextIO, Optional, cast

from .subprocess import run


def make_secure(*files: Path):
    for fl in files:
        if fl.exists():
            fl.unlink()
        os.close(os.open(str(fl), os.O_WRONLY | os.O_CREAT, 0o600))


async def make_cert_and_key(key_file: Path, cert_file: Path, subj: str):
    await run(f"openssl genrsa 1024 2>/dev/null > {key_file}")
    cmd = f'openssl req -new -x509 -nodes -sha1 -days 365 -key "{key_file}" -subj "{subj}" > {cert_file} 2>/dev/null'
    await run(cmd)


def read_inventory(path: str) -> List[str]:
    names = [name_or_ip.strip() for name_or_ip in open(path)]
    return [name_or_ip for name_or_ip in names if name_or_ip and not name_or_ip.startswith("#")]


def open_to_append(fname: str, is_bin: bool = False) -> Union[BinaryIO, TextIO]:
    if os.path.exists(fname):
        fd = open(fname, "rb+" if is_bin else "r+")
        fd.seek(0, os.SEEK_END)
    else:
        fd = open(fname, "wb" if is_bin else "w")
        os.chmod(fname, stat.S_IRGRP | stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH)
    return fd


def open_for_append_or_create(fname: str) -> TextIO:
    if not os.path.exists(fname):
        return cast(TextIO, open(fname, "w"))

    fd = open(fname, 'r+')
    fd.seek(0, os.SEEK_END)
    return cast(TextIO, fd)


def which(program: str) -> Optional[str]:
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        exe_file = os.path.join(path, program)
        if is_exe(exe_file):
            return exe_file

    return None


FILES_TO_REMOVE = []  # type: List[str]


def tmpnam(remove_after: bool = True, **kwargs) -> str:
    fd, name = tempfile.mkstemp(**kwargs)
    os.close(fd)
    if remove_after:
        FILES_TO_REMOVE.append(name)
    return name


def clean_tmp_files() -> None:
    for fname in FILES_TO_REMOVE:
        try:
            os.unlink(fname)
        except IOError:
            pass
    FILES_TO_REMOVE[:] = []


atexit.register(clean_tmp_files)
