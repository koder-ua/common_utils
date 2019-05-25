from __future__ import annotations

import bisect
import os
import random
import re
import math
import stat
import time
import socket
import atexit
import asyncio
import tempfile
import ipaddress
import contextlib
import subprocess
from pathlib import Path
from collections import Counter
from typing import (Iterable, Iterator, Any, Callable, TypeVar, Coroutine, Tuple, List, Union, BinaryIO,
                    TextIO, Optional, cast, Dict, Mapping, Sequence, AsyncIterator, Generic)

from dataclasses import dataclass, field

from . import run


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


class RAttredDict:
    def __init__(self, val: Mapping[str, Any]):
        self.__val = val

    def __getattr__(self, name: str) -> Any:
        try:
            vl = self.__val[name]
            is_dict_like = hasattr(vl, '__getitem__') and not isinstance(vl, (str, bytes))
            return self.__class__(vl) if is_dict_like else vl
        except KeyError:
            raise AttributeError(f"No {name} key exists, only {list(self.__val.keys())}") from None


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


def read_inventory(path: Path) -> List[str]:
    names = [name_or_ip.strip() for name_or_ip in path.open()]
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


FILES_TO_REMOVE: List[str] = []


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


def is_ip(data: str) -> bool:
    try:
        ipaddress.ip_address(data)
        return True
    except ValueError:
        return False


def parse_creds(creds: str) -> Tuple[str, str, str]:
    """Parse simple credentials format user[:passwd]@host"""
    user, passwd_host = creds.split(":", 1)

    if '@' not in passwd_host:
        passwd, host = passwd_host, None
    else:
        passwd, host = passwd_host.rsplit('@', 1)

    return user, passwd, host


def get_ip_for_target(target_ip: str) -> str:
    if not is_ip(target_ip):
        target_ip = socket.gethostbyname(target_ip)

    first_dig = map(int, target_ip.split("."))
    if first_dig == 127:
        return '127.0.0.1'

    data = subprocess.check_output(['ip', 'route', 'get', 'to', target_ip]).decode()
    data_line = data.split("\n")[0].strip()

    rr1 = r'{0} via [.0-9]+ dev (?P<dev>.*?) src (?P<ip>[.0-9]+)$'
    rr1 = rr1.replace(" ", r'\s+')
    rr1 = rr1.format(target_ip.replace('.', r'\.'))

    rr2 = r'{0} dev (?P<dev>.*?) src (?P<ip>[.0-9]+)$'
    rr2 = rr2.replace(" ", r'\s+')
    rr2 = rr2.format(target_ip.replace('.', r'\.'))

    res1 = re.match(rr1, data_line)
    res2 = re.match(rr2, data_line)

    if res1 is not None:
        return res1.group('ip')

    if res2 is not None:
        return res2.group('ip')

    raise OSError("Can't define interface for {0}".format(target_ip))


@contextlib.contextmanager
def empty_ctx(val: Any = None) -> Iterator[Any]:
    yield val


def to_ip(host_or_ip: str) -> str:
    # translate hostname to address
    try:
        ipaddress.ip_address(host_or_ip)
        return host_or_ip
    except ValueError:
        ip_addr = socket.gethostbyname(host_or_ip)
        return ip_addr


def shape2str(shape: Iterable[int]) -> str:
    return "*".join(map(str, shape))


def str2shape(shape: str) -> Tuple[int, ...]:
    return tuple(map(int, shape.split('*')))


def dict2str_helper(dct: Dict[str, Any], prefix: str) -> List[str]:
    res = []
    for k, v in dct.items():
        assert isinstance(k, str)
        if isinstance(v, dict):
            res.extend(dict2str_helper(v, prefix + k + "::"))
        else:
            res.append("{}{} {}".format(prefix, k, v))
    return res


def dict2str(dct: Dict[str, Any]) -> str:
    return "\n".join(dict2str_helper(dct, ""))


def auto_group_by(items: Sequence[T], border: float = 0.3) -> Iterable[Iterable[T]]:
    all_attrs = set(items[0].__dict__)
    assert all(set(item.__dict__) == all_attrs for item in items)
    entropy = {}
    for attr in all_attrs:
        vals: Dict[str, int] = Counter()
        for item in items:
            vals[getattr(item, attr)] += 1

        entropy[attr] = -sum(vals[val] / len(items) * math.log2(vals[val] / len(items)) for val in vals)

    mutable_keys = [attr for attr, val in entropy.items() if val > border]
    for group in group_by((item.__dict__ for item in items), mutable_keys):
        yield [items[idx] for idx in group]


def group_by(items: Iterable[Dict[str, Any]], mutable_keys=Union[str, Tuple[str, ...]]) -> Iterable[List[int]]:
    if isinstance(mutable_keys, str):
        mutable_keys = (mutable_keys,)

    mutable_keys_s = set(mutable_keys)

    grouped: Dict[Tuple, List[int]] = {}
    for idx, dct in enumerate(items):
        group_idx = tuple(dct[key] for key in dct if key not in mutable_keys_s)
        grouped.setdefault(group_idx, []).append(idx)

    return grouped.values()


async def async_wait_cycle(timeout: float) -> AsyncIterator[float]:
    ptime = time.time()
    while True:
        yield ptime
        wtime = ptime + timeout - time.time()
        if wtime >= 0:
            await asyncio.sleep(wtime)
        ptime = time.time()


def get_discretizer(output_max_val: int, step_coef: float) -> Tuple[Callable[[float], int], Callable[[int], int]]:
    val = 0
    table: List[int] = [val]
    for _ in range(output_max_val):
        val = max(round(step_coef * val), val + 1)
        table.append(val)

    def discretize(vl: float) -> int:
        return min(output_max_val, bisect.bisect_left(table, round(vl)))

    return discretize, table.__getitem__


@dataclass
class SamplingList(Generic[T]):
    count: int
    total_processed_count: int = field(init=False, default=0)
    skip: int = field(init=False, default=0)
    curr_counter: int = field(init=False, default=0)
    samples: List[T] = field(init=False, default_factory=list)
    pretenders: List[T] = field(init=False, default_factory=list)

    def add(self, obj: T) -> None:
        self.total_processed_count += 1
        if self.curr_counter == self.skip:
            self.pretenders.append(obj)
            self.curr_counter = 0
            if len(self.pretenders) == self.count:
                self.samples = random.sample(self.samples + self.pretenders, self.count)
        else:
            self.curr_counter += 1


def partition(items: Iterable[T], size: int) -> Iterable[List[T]]:
    curr = []
    for idx, val in enumerate(items):
        curr.append(val)
        if (idx + 1) % size == 0:
            yield curr
            curr = []

    if curr:
        yield curr


def partition_by_len(items: Iterable[Union[T, Tuple[T, int]]],
                     chars_per_line: int, delimiter_len: int) -> Iterable[List[T]]:
    curr: List[T] = []
    curr_len = 0
    for el_r in items:
        if isinstance(el_r, tuple):
            el, el_len = el_r
        else:
            el = el_r
            el_len = len(str(el))
        if curr_len + delimiter_len + el_len <= chars_per_line:
            curr.append(el)
            curr_len += delimiter_len + el_len
        else:
            yield curr
            curr = [el]
            curr_len = el_len
    if curr:
        yield curr