import asyncio
import logging
from pathlib import Path
import subprocess
from dataclasses import dataclass
from typing import Union, Tuple, Sequence, Optional, BinaryIO, Callable, Any, Dict

logger = logging.getLogger("common.cli")


CmdType = Union[str, Sequence[Union[str, bytes, Path]]]


@dataclass
class CMDResult:
    args: CmdType
    stdout_b: bytes
    stderr_b: Optional[bytes]
    returncode: int

    @property
    def stdout(self) -> str:
        return self.stdout_b.decode("utf8")

    def check_returncode(self):
        if self.returncode != 0:
            raise subprocess.CalledProcessError(self.returncode, self.args, self.stdout_b, self.stderr_b)


async def run_proc_timeout(cmd: CmdType,
                           proc: asyncio.subprocess.Process,
                           timeout: float,
                           input_data: Optional[bytes],
                           term_timeout: float) -> CMDResult:

    done, not_done = await asyncio.wait({proc.communicate(input=input_data)}, timeout=timeout - 2 * term_timeout)
    if not_done:
        proc.terminate()
        done, not_done = await asyncio.wait({proc.communicate()}, timeout=term_timeout)
        if not_done:
            proc.kill()
            proc_fut = proc.communicate()
            done, not_done = await asyncio.wait(proc_fut, timeout=term_timeout)

            if not_done:
                raise RuntimeError(f"Can't kill process {proc.pid} of {cmd}")

        proc_fut2, = done
        out2, err2 = await proc_fut2
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output=out2, stderr=err2)

    proc_fut3, = done
    out3, err3 = await proc_fut3

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(returncode=proc.returncode,
                                            cmd=cmd, output=out3, stderr=err3)

    return CMDResult(cmd, stdout_b=out3, stderr_b=err3, returncode=proc.returncode)


async def start_proc(cmd: CmdType,
                     input_data: Union[bytes, None, BinaryIO] = None,
                     merge_err: bool = True,
                     output_to_devnull: bool = False,
                     env: Dict[str, str] = None) -> Tuple[asyncio.subprocess.Process, Optional[bytes]]:

    if isinstance(input_data, bytes):
        stdin: Any = asyncio.subprocess.PIPE
    elif input_data is None:
        stdin = None
    else:
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

    return (await func(*cmd, stdout=stdout, stderr=stderr, stdin=stdin, env=env)), input_data


async def run(cmd: CmdType,
              input_data: Union[bytes, None, BinaryIO] = None,
              merge_err: bool = True,
              timeout: float = 60,
              output_to_devnull: bool = False,
              term_timeout: float = 1,
              env: Dict[str, str] = None) -> CMDResult:

    proc, input_data = await start_proc(cmd, input_data, merge_err, output_to_devnull=output_to_devnull, env=env)
    res = await run_proc_timeout(cmd, proc, timeout=timeout, input_data=input_data, term_timeout=term_timeout)

    if merge_err:
        assert res.stderr_b is None

    return res


async def run_stdout(*args, **kwargs) -> str:
    return (await run(*args, **kwargs)).stdout
