import os
import stat
import time
import atexit
import asyncio
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Any, Callable, TypeVar, Coroutine, Tuple, List, Union, BinaryIO, \
    TextIO, Optional, cast

from .cli import run


AnyPath = Union[str, Path]


class Timeout(Iterable[float]):
    def __init__(self, timeout: int, message: str = None, min_tick: int = 1, no_exc: bool = False) -> None:
        self.end_time = time.time() + timeout
        self.message = message
        self.min_tick = min_tick
        self.prev_tick_at = time.time()
        self.no_exc = no_exc

    def tick(self) -> bool:
        current_time = time.time()

        if current_time > self.end_time:
            if self.message:
                msg = "Timeout: {}".format(self.message)
            else:
                msg = "Timeout"

            if self.no_exc:
                return False

            raise TimeoutError(msg)

        sleep_time = self.min_tick - (current_time - self.prev_tick_at)
        if sleep_time > 0:
            time.sleep(sleep_time)
            self.prev_tick_at = time.time()
        else:
            self.prev_tick_at = current_time

        return True

    def __iter__(self) -> Iterator[float]:
        return cast(Iterator[float], self)

    def __next__(self) -> float:
        if not self.tick():
            raise StopIteration()
        return self.end_time - time.time()


class AttredDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def flatten(data: Iterable[Any]) -> List[Any]:
    res = []
    for i in data:
        if isinstance(i, (list, tuple, set)):
            res.extend(flatten(i))
        else:
            res.append(i)
    return res


Tp = TypeVar('Tp')


def find(lst: List[Tp], check: Callable[[Tp], bool], default: Tp = None) -> Tp:
    for obj in lst:
        if check(obj):
            return obj
    return default


FM_FUNC_INPUT = TypeVar("FM_FUNC_INPUT")
FM_FUNC_RES = TypeVar("FM_FUNC_RES")


def flatmap(func: Callable[[FM_FUNC_INPUT], Iterable[FM_FUNC_RES]],
            inp_iter: Iterable[FM_FUNC_INPUT]) -> Iterator[FM_FUNC_RES]:
    for val in inp_iter:
        for res in func(val):
            yield res


T = TypeVar("T")
R = TypeVar("R")


async def async_map(func: Callable[[T], Coroutine[Any, Any, R]],
                    values: Iterable[T],
                    max_workers: int = 0) -> Iterable[R]:

    semaphore = asyncio.Semaphore(max_workers) if max_workers else None

    async def worker(val: T) -> R:
        if semaphore:
            await semaphore.acquire()
        return await func(val)

    return asyncio.gather(*map(worker, values))


class IgnoreAll:
    def __enter__(self) -> 'IgnoreAll':
        return self

    def __exit__(self, x, y, z) -> bool:
        return True


ignore_all = IgnoreAll()


async def async_run(*coros: Coroutine[Any, Any, R]) -> List[R]:
    async def mcoro(idx: int, coro: Coroutine[Any, Any, R]) -> Tuple[int, R]:
        return idx, (await coro)

    coros_with_idx = [mcoro(idx, coro) for idx, coro in enumerate(coros)]

    return [val for _, val in (await asyncio.gather(*coros_with_idx))]


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
