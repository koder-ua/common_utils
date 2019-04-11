import abc
import asyncio
import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Union, BinaryIO, Dict, AsyncIterator, List, Iterable, Generic, TypeVar, Callable, Coroutine, \
    AsyncIterable, Tuple

from . import AnyPath, CmdType, CMDResult, run


class ISyncNode(metaclass=abc.ABCMeta):
    """Remote node interface"""
    conn_addr: str
    conn: Any

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def run(self, cmd: CmdType, input_data: Union[bytes, None, BinaryIO] = None,
            merge_err: bool = True, timeout: float = 60, output_to_devnull: bool = False,
            term_timeout: float = 1, env: Dict[str, str] = None) -> CMDResult:
        pass

    @abc.abstractmethod
    def copy(self, local_path: AnyPath, remote_path: AnyPath, compress: bool = False) -> None:
        pass

    @abc.abstractmethod
    def read(self, path: AnyPath, compress: bool = False) -> bytes:
        pass

    @abc.abstractmethod
    def write(self, path: AnyPath, content: Union[BinaryIO, bytes], compress: bool = False) -> None:
        pass

    @abc.abstractmethod
    def stat(self, path: AnyPath) -> os.stat_result:
        pass

    @abc.abstractmethod
    def stat_many(self, path: List[AnyPath]) -> List[os.stat_result]:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    def __enter__(self) -> 'ISyncNode':
        return self

    def __exit__(self, x, y, z) -> bool:
        self.disconnect()
        return False


class ISimpleAsyncNode(metaclass=abc.ABCMeta):
    """Remote node interface"""
    conn_addr: str
    conn: Any

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    async def run(self, cmd: CmdType, input_data: Union[bytes, None, BinaryIO] = None,
                  merge_err: bool = True, timeout: float = 60, output_to_devnull: bool = False,
                  term_timeout: float = 1, env: Dict[str, str] = None,
                  compress: bool = True) -> CMDResult:
        pass

    async def run_bytes(self, cmd: CmdType, input_data: Union[bytes, None, BinaryIO] = None,
                         merge_err: bool = True, timeout: float = 60,
                         term_timeout: float = 1, env: Dict[str, str] = None,
                         compress: bool = True) -> bytes:
        return (await self.run(cmd, input_data=input_data, merge_err=merge_err, timeout=timeout,
                               term_timeout=term_timeout, env=env, compress=compress)).stdout_b

    async def run_str(self, *args, **kwargs) -> str:
        return (await self.run_bytes(*args, **kwargs)).decode()

    async def run_json(self, *args, **kwargs) -> Dict[str, Any]:
        return json.loads(await self.run_str(*args, **kwargs))

    @abc.abstractmethod
    async def copy(self, local_path: AnyPath, remote_path: AnyPath, compress: bool = False):
        pass


class IAsyncNode(ISimpleAsyncNode):
    @abc.abstractmethod
    async def read(self, path: AnyPath, compress: bool = False) -> bytes:
        pass

    async def read_str(self, path: AnyPath, compress: bool = True) -> str:
        return (await self.read(path, compress)).decode()

    @abc.abstractmethod
    async def iter_file(self, path: AnyPath, compress: bool = False) -> AsyncIterator[bytes]:
        pass

    @abc.abstractmethod
    async def write_tmp(self, content: Union[BinaryIO, bytes], compress: bool = False) -> Path:
        pass

    @abc.abstractmethod
    async def stat(self, path: AnyPath) -> os.stat_result:
        pass

    @abc.abstractmethod
    async def write(self, path: AnyPath, content: Union[BinaryIO, bytes], compress: bool = False):
        pass

    @abc.abstractmethod
    async def iterdir(self, path: AnyPath) -> Iterable[Path]:
        pass

    @abc.abstractmethod
    async def disconnect(self) -> None:
        pass

    @abc.abstractmethod
    async def __aenter__(self) -> 'IAsyncNode':
        pass

    async def __aexit__(self, x, y, z) -> bool:
        await self.disconnect()
        return False

    async def copy(self, local_path: AnyPath, remote_path: AnyPath, compress: bool = False):
        await self.write(remote_path, open(local_path, 'rb'), compress=compress)

    async def exists(self, fname: AnyPath) -> bool:
        try:
            await self.stat(fname)
            return True
        except OSError:
            return False

    async def copy_to_tmp(self, local_path: AnyPath, compress: bool = False) -> Path:
        return await self.write_tmp(Path(local_path).open('rb'), compress=compress)


class LocalHost(IAsyncNode):
    conn_addr = "<localhost>"
    conn = None

    def __str__(self) -> str:
        return "<Local>"

    async def write(self, path: AnyPath, content: Union[BinaryIO, bytes], compress: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        with path.open("wb") as fd:
            if isinstance(content, bytes):
                fd.write(content)
            else:
                shutil.copyfileobj(content, fd)

    async def write_tmp(self, content: Union[BinaryIO, bytes], compress: bool = False) -> Path:
        fd, path = tempfile.mkstemp(text=False)
        if isinstance(content, bytes):
            fd.write(content)
        else:
            shutil.copyfileobj(content, fd)
        os.close(fd)
        return Path(path)

    async def run(self, cmd: CmdType, input_data: Union[bytes, None, BinaryIO] = None,
                  merge_err: bool = True, timeout: float = 60, output_to_devnull: bool = False,
                  term_timeout: float = 1, env: Dict[str, str] = None, compress: bool = True) -> CMDResult:

        return await run(cmd, input_data, merge_err=merge_err, timeout=timeout,
                         output_to_devnull=output_to_devnull, term_timeout=term_timeout, env=env)

    async def read(self, path: AnyPath, compress: bool = False) -> bytes:
        return open(path, "rb").read()

    async def iter_file(self, path: AnyPath, compress: bool = False) -> AsyncIterator[bytes]:
        with open(path, "rb") as fd:
            while True:
                data = fd.read(16 * 1024)
                if not data:
                    break
                yield data

    async def stat(self, path: AnyPath) -> os.stat_result:
        return Path(path).stat()

    async def exists(self, fname: AnyPath) -> bool:
        return Path(fname).exists()

    async def iterdir(self, path: AnyPath) -> Iterable[Path]:
        return Path(path).iterdir()

    async def disconnect(self) -> None:
        pass

    async def __aenter__(self) -> 'IAsyncNode':
        return self

    async def __aexit__(self, x, y, z) -> bool:
        return False


T = TypeVar('T')


class BaseConnectionPool(Generic[T], metaclass=abc.ABCMeta):
    def __init__(self, max_conn_per_node: int = None) -> None:
        assert max_conn_per_node >= 1, f"max_conn_per_node(={max_conn_per_node}) must be >= 1"
        self.free_conn: Dict[str, List[T]] = {}
        self.conn_per_node: Dict[str, int] = {}
        self.conn_freed: Dict[str, asyncio.Condition] = {}
        self.max_conn_per_node = max_conn_per_node
        self.opened = False

    async def get_conn(self, conn_addr: str) -> T:
        assert self.opened, "Pool is not opened"
        if conn_addr not in self.conn_freed:
            self.conn_freed[conn_addr] = asyncio.Condition()

        while True:
            free_cons = self.free_conn.setdefault(conn_addr, [])
            if free_cons:
                return free_cons.pop()

            if self.conn_per_node.setdefault(conn_addr, 0) < self.max_conn_per_node:
                self.conn_per_node[conn_addr] += 1

                try:
                    return await self._rpc_connect(conn_addr)
                except Exception:
                    self.conn_per_node[conn_addr] -= 1
                    raise

            async with self.conn_freed[conn_addr]:
                await self.conn_freed[conn_addr].wait()

    async def release_conn(self, conn_addr: str, conn: T) -> None:
        assert self.opened, "Pool is not opened"
        self.free_conn[conn_addr].append(conn)

        async with self.conn_freed[conn_addr]:
            self.conn_freed[conn_addr].notify()

    @abc.abstractmethod
    async def _rpc_connect(self, conn_addr: str) -> T:
        pass

    @abc.abstractmethod
    async def _rpc_disconnect(self, conn: T) -> None:
        pass

    async def __aenter__(self) -> 'BaseConnectionPool[T]':
        assert not self.opened, "Pool already opened"
        self.opened = True
        return self

    async def __aexit__(self, x, y, z) -> None:
        assert self.opened, "Pool is not opened"
        for addr, conns in self.free_conn.items():
            assert len(conns) == self.conn_per_node[addr]
            for conn in conns:
                await self._rpc_disconnect(conn)
            self.conn_per_node[addr] = 0
        self.free_conn = {}
        self.opened = False

    @asynccontextmanager
    async def connection(self, conn_addr: str) -> AsyncIterator[T]:
        conn = await self.get_conn(conn_addr)
        try:
            yield conn
        finally:
            await self.release_conn(conn_addr, conn)


R = TypeVar('R')
V = TypeVar('V')


async def rpc_map(pool: BaseConnectionPool[R],
                  func: Callable[..., Coroutine[Any, Any, V]],
                  hostnames: List[str],
                  **kwargs) -> AsyncIterable[Tuple[str, Union[Exception, V]]]:

    conns: Dict[str, R] = {}
    coros: List[Coroutine[Any, Any, V]] = []

    try:
        for hostname in hostnames:
            conn = await pool.get_conn(hostname)
            conns[hostname] = conn
            coros.append(func(conn, hostname, **kwargs))

        for hostname, res in zip(hostnames, await asyncio.gather(*coros, return_exceptions=True)):
            yield hostname, res
    finally:
        for hostname, conn in conns.items():
            await pool.release_conn(hostname, conn)
