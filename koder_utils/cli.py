from __future__ import annotations

import os
import json
import signal
import asyncio
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Tuple, Sequence, Optional, BinaryIO, Callable, Any


logger = logging.getLogger("utils.cli")


CmdType = Union[str, Sequence[Union[str, bytes, Path]]]


@dataclass
class CMDResult:
    args: CmdType
    stdout_b: bytes
    stderr_b: Optional[bytes]
    returncode: int

    @property
    def stdout(self) -> str:
        return self.stdout_b.decode()

    def check_returncode(self):
        if self.returncode != 0:
            raise subprocess.CalledProcessError(self.returncode, self.args, self.stdout_b, self.stderr_b)


async def start_proc(cmd: CmdType,
                     input_data: Union[bytes, None, BinaryIO] = None,
                     merge_err: bool = True,
                     output_to_devnull: bool = False,
                     **kwargs) -> Tuple[asyncio.subprocess.Process, Optional[bytes]]:

    assert not isinstance(input_data, str), "String not allowed as input data, encode them"

    if isinstance(input_data, (str, bytes)):
        stdin: Any = asyncio.subprocess.PIPE
    elif input_data is None:
        stdin = None
    else:
        assert hasattr(input_data, 'read'), f"Unknown type {type(input_data)} as input data"
        stdin = input_data
        input_data = None

    if output_to_devnull:
        stderr = asyncio.subprocess.DEVNULL
        stdout = asyncio.subprocess.DEVNULL
    else:
        stderr = asyncio.subprocess.STDOUT if merge_err else asyncio.subprocess.PIPE
        stdout = asyncio.subprocess.PIPE

    if isinstance(cmd, str):
        func: Callable = asyncio.create_subprocess_shell
        cmd = [cmd]
    else:
        func = asyncio.create_subprocess_exec
        cmd = [str(arg) for arg in cmd]

    return (await func(*cmd, stdout=stdout, stderr=stderr, stdin=stdin, **kwargs)), input_data


async def terminate_proc(proc: asyncio.subprocess.Process, proc_task, term_timeout: float, cmd: CmdType,
                         term_group: Optional[int]) -> Tuple[Optional[bytes], Optional[bytes]]:
    try:
        proc.terminate()
    except ProcessLookupError:
        pass

    done, _ = await asyncio.wait({proc_task}, timeout=term_timeout)

    if not done:
        try:
            if term_group is not None:
                os.killpg(term_group, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass

        done, _ = await asyncio.wait({proc_task}, timeout=term_timeout)

        if not done:
            raise RuntimeError(f"Can't kill process '{cmd}' with pid {proc.pid}")

    return await proc_task


async def run_proc_timeout(cmd: CmdType,
                           proc: asyncio.subprocess.Process,
                           timeout: float,
                           input_data: Optional[bytes],
                           term_timeout: float,
                           term_group: int = None) -> CMDResult:
    assert timeout > 2 * term_timeout
    proc_task = asyncio.create_task(proc.communicate(input=input_data))
    done, _ = await asyncio.wait({proc_task}, timeout=timeout - 2 * term_timeout)

    if not done:
        out, err = await terminate_proc(proc, proc_task, term_timeout, cmd, term_group)
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output=out, stderr=err)

    out, err = await proc_task

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(returncode=proc.returncode, cmd=cmd, output=out, stderr=err)

    return CMDResult(cmd, stdout_b=out, stderr_b=err, returncode=proc.returncode)


async def run(cmd: CmdType,
              input_data: Union[bytes, None, BinaryIO] = None,
              merge_err: bool = True,
              timeout: float = 60,
              output_to_devnull: bool = False,
              term_timeout: float = 1,
              **kwargs) -> CMDResult:

    proc, input_data = await start_proc(cmd, input_data, merge_err, output_to_devnull=output_to_devnull, **kwargs)
    res = await run_proc_timeout(cmd, proc, timeout=timeout, input_data=input_data, term_timeout=term_timeout)

    if merge_err:
        assert res.stderr_b is None

    return res


async def run_stdout(*args, **kwargs) -> str:
    return (await run(*args, **kwargs)).stdout


async def run_json(*args, **kwargs) -> Any:
    return json.loads((await run(*args, **kwargs)).stdout)
