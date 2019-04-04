import abc
import os
import re
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Union, BinaryIO, Dict, AsyncIterator, Tuple, List, Optional, Set

from .utils import AnyPath
from .subprocess import CmdType, CMDResult, run
from .converters import b2ssize


@dataclass
class OSRelease:
    distro: str
    release: str
    arch: str


class INode(metaclass=abc.ABCMeta):
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
    def copy_file(self, local_path: AnyPath, remote_path: AnyPath, expanduser: bool = False, compress: bool = False):
        pass

    @abc.abstractmethod
    def get_file_content(self, path: AnyPath, expanduser: bool = False, compress: bool = False) -> bytes:
        pass

    @abc.abstractmethod
    def put_to_file(self, path: AnyPath, content: Union[BinaryIO, bytes], expanduser: bool = False,
                    compress: bool = False):
        pass

    @abc.abstractmethod
    def stat_file(self, path: AnyPath, expanduser: bool = False) -> os.stat_result:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    def __enter__(self) -> 'INode':
        return self

    def __exit__(self, x, y, z) -> bool:
        self.disconnect()
        return False


class IAsyncNode(metaclass=abc.ABCMeta):
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

    async def run_stdout(self, cmd: CmdType, input_data: Union[bytes, None, BinaryIO] = None,
                         merge_err: bool = True, timeout: float = 60,
                         term_timeout: float = 1, env: Dict[str, str] = None,
                         compress: bool = True) -> bytes:
        return (await self.run(cmd, input_data=input_data, merge_err=merge_err, timeout=timeout,
                               term_timeout=term_timeout, env=env, compress=compress)).stdout_b

    async def run_stdout_str(self, *args, **kwargs) -> str:
        return (await self.run_stdout(*args, **kwargs)).decode("utf8")

    @abc.abstractmethod
    async def copy_file(self, local_path: AnyPath, remote_path: AnyPath, expanduser: bool = False,
                        compress: bool = False):
        pass

    @abc.abstractmethod
    async def copy_to_tmp_file(self, local_path: AnyPath, compress: bool = False):
        pass

    @abc.abstractmethod
    async def read_file(self, path: AnyPath, expanduser: bool = False, compress: bool = False) -> bytes:
        pass

    @abc.abstractmethod
    async def iter_file(self, path: AnyPath, expanduser: bool = False,
                        compress: bool = False) -> AsyncIterator[bytes]:
        pass

    @abc.abstractmethod
    async def write_file(self, path: AnyPath, content: Union[BinaryIO, bytes], expanduser: bool = False,
                         compress: bool = False):
        pass

    @abc.abstractmethod
    async def write_tmp_file(self, content: Union[BinaryIO, bytes], compress: bool = False) -> Path:
        pass

    @abc.abstractmethod
    async def stat_file(self, path: AnyPath, expanduser: bool = False) -> os.stat_result:
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


class LocalHost(IAsyncNode):
    conn_addr = "<localhost>"
    conn = None

    def __str__(self) -> str:
        return "<Local>"

    async def write_file(self, path: AnyPath, content: Union[BinaryIO, bytes], expanduser: bool = False,
                         compress: bool = False) -> None:

        path = Path(path)

        if expanduser:
            path = path.expanduser()

        path.parent.mkdir(exist_ok=True)
        with path.open("wb") as fd:
            if isinstance(content, bytes):
                fd.write(content)
            else:
                shutil.copyfileobj(content, fd)

    async def put_to_temp_file(self, content: Union[BinaryIO, bytes], compress: bool = False) -> Path:
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

    async def copy_file(self, local_path: AnyPath, remote_path: AnyPath, expanduser: bool = False,
                        compress: bool = False):
        remote_path = Path(remote_path)
        if expanduser:
            remote_path = remote_path.expanduser()

        shutil.copyfile(local_path, remote_path)

    async def copy_to_tmp_file(self, local_path: AnyPath, compress: bool = False) -> Path:
        fd, path = tempfile.mkstemp(text=False)
        os.close(fd)
        shutil.copyfile(local_path, path)
        return Path(path)

    async def read_file(self, path: AnyPath, expanduser: bool = False, compress: bool = False) -> bytes:
        if expanduser:
            path = path.expanduser()
        return path.open("rb").read()

    async def iter_file(self, path: AnyPath, expanduser: bool = False,
                        compress: bool = False) -> AsyncIterator[bytes]:
        if expanduser:
            path = path.expanduser()

        with path.open("rb") as fd:
            while True:
                data = fd.read(16 * 1024)
                if not data:
                    break
                yield data

    async def stat_file(self, path: AnyPath, expanduser: bool = False) -> os.stat_result:
        if expanduser:
            path = path.expanduser()

        return path.stat()

    async def disconnect(self) -> None:
        pass

    async def __aenter__(self) -> 'IAsyncNode':
        return self

    async def __aexit__(self, x, y, z) -> bool:
        return False


async def get_hostname(node: IAsyncNode) -> str:
    return await node.run_stdout_str("hostname")


async def get_os(node: IAsyncNode) -> OSRelease:
    """return os type, release and architecture for node.
    """
    arch = await node.run_stdout_str("arch")
    dist_type = (await node.run_stdout_str("lsb_release -i -s")).lower().strip()
    codename = (await node.run_stdout_str("lsb_release -c -s")).lower().strip()
    return OSRelease(dist_type, codename, arch)


