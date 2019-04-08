import abc
import errno
import json
import os
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Union, BinaryIO, Dict, AsyncIterator, List, Iterable, Optional, Iterator

from .utils import AnyPath
from .cli import CmdType, CMDResult, run


@dataclass
class OSRelease:
    distro: str
    release: str
    arch: str


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


async def get_hostname(node: IAsyncNode) -> str:
    return (await node.run_str("hostname")).strip()


async def get_all_ips(node: IAsyncNode) -> List[str]:
    return (await node.run_str("hostname -I")).split()


async def get_os(node: IAsyncNode) -> OSRelease:
    """return os type, release and architecture for node.
    """
    arch = await node.run_str("arch")
    dist_type = (await node.run_str("lsb_release -i -s")).lower().strip()
    codename = (await node.run_str("lsb_release -c -s")).lower().strip()
    return OSRelease(dist_type, codename, arch)

