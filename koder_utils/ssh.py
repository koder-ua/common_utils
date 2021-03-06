from __future__ import annotations

import socket
import asyncio
import ipaddress
import subprocess
from typing import Tuple, Iterable, List

from . import run, CMDResult, CmdType, ISimpleAsyncNode, AnyPath


DEFAULT_OPTS = ("-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectionAttempts=3",
                "-o", "ConnectTimeout=10",
                "-o", "LogLevel=ERROR")


class SSH(ISimpleAsyncNode):
    def __init__(self, node: str, ssh_user: str, ssh_opts: Tuple[str, ...] = DEFAULT_OPTS) -> None:
        self.node = node
        self.ssh_opts = list(ssh_opts)
        self.ssh_user = ssh_user
        self.cmd_prefix = ["ssh"] + self.ssh_opts + [self.ssh_user + "@" + self.node] + ['--']
        self.cmd_prefix_s = " ".join(self.cmd_prefix) + " "

    def __str__(self) -> str:
        return f"SSH({self.node})"

    async def run(self, cmd: CmdType, *args, **kwargs) -> CMDResult:
        cmd = (self.cmd_prefix if isinstance(cmd, list) else self.cmd_prefix_s) + cmd  # type: ignore
        return await run(cmd, *args, **kwargs)

    async def copy(self, local_path: AnyPath, remote_path: AnyPath, compress: bool = False, timeout: int = 60):
        cmd = ["scp", *self.ssh_opts, local_path, f"{self.ssh_user}@{self.node}:{remote_path}"]
        await run(cmd, timeout=timeout)

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


async def get_sshable_hosts(addresses: Iterable[str], user: str) -> List[str]:
    async def check_host(addr):
        try:
            try:
                ipaddress.ip_address(addr)
            except ValueError:
                addr = socket.gethostbyname(addr)
            ssh = SSH(addr, user)
            if await ssh.run('pwd'):
                return addr
        except (subprocess.CalledProcessError, socket.gaierror):
            return None

    tasks = set(map(check_host, addresses))
    return [name for name in (await asyncio.gather(*tasks)) if name]


# import re
# import time
# import errno
# import socket
# import getpass
# import logging
# import os.path
# import selectors
# from io import StringIO
# from typing import cast, Set, Optional, List, Any, Dict, NamedTuple
#
# try:
#     import paramiko
# except ImportError:
#     paramiko = None
#
# from .utils import to_ip, Timeout
# from .storage import IStorable
#
#
# logger = logging.getLogger("cephlib")
#
#
# IP = str
# IPAddr = NamedTuple("IPAddr", [("host", IP), ("port", int)])
#
#
# class ConnCreds(IStorable):
#     def __init__(self, host: str, user: str = None, passwd: str = None, port: str = '22',
#                  key_file: str = None, key: bytes = None) -> None:
#         self.user = user
#         self.passwd = passwd
#         self.addr = IPAddr(host, int(port))
#         self.key_file = key_file
#         self.key = key
#
#     def __str__(self) -> str:
#         return "{}@{}:{}".format(self.user, self.addr.host, self.addr.port)
#
#     def __repr__(self) -> str:
#         return str(self)
#
#     def raw(self) -> Dict[str, Any]:
#         return {
#             'user': self.user,
#             'host': self.addr.host,
#             'port': self.addr.port,
#             'passwd': self.passwd,
#             'key_file': self.key_file
#         }
#
#     @classmethod
#     def fromraw(cls, data: Dict[str, Any]) -> 'ConnCreds':
#         return cls(**data)
#
#
# class URIsNamespace:
#     class ReParts:
#         user_rr = "[^:]*?"
#         host_rr = "[^:@]*?"
#         port_rr = "\\d+"
#         key_file_rr = "[^:@]*"
#         passwd_rr = ".*?"
#
#     re_dct = ReParts.__dict__
#
#     for attr_name, val in re_dct.items():
#         if attr_name.endswith('_rr'):
#             new_rr = "(?P<{0}>{1})".format(attr_name[:-3], val)
#             setattr(ReParts, attr_name, new_rr)
#
#     re_dct = ReParts.__dict__
#
#     templs = [
#         "^{host_rr}$",
#         "^{host_rr}:{port_rr}$",
#         "^{host_rr}::{key_file_rr}$",
#         "^{host_rr}:{port_rr}:{key_file_rr}$",
#         "^{user_rr}@{host_rr}$",
#         "^{user_rr}@{host_rr}:{port_rr}$",
#         "^{user_rr}@{host_rr}::{key_file_rr}$",
#         "^{user_rr}@{host_rr}:{port_rr}:{key_file_rr}$",
#         "^{user_rr}:{passwd_rr}@{host_rr}$",
#         "^{user_rr}:{passwd_rr}@{host_rr}:{port_rr}$",
#     ]
#
#     uri_reg_exprs = []  # type: List[str]
#     for templ in templs:
#         uri_reg_exprs.append(templ.format(**re_dct))
#
#
# def parse_ssh_uri(uri: str) -> ConnCreds:
#     """Parse ssh connection URL from one of following form
#         [ssh://]user:passwd@host[:port]
#         [ssh://][user@]host[:port][:key_file]
#     """
#
#     if uri.startswith("ssh://"):
#         uri = uri[len("ssh://"):]
#
#     for rr in URIsNamespace.uri_reg_exprs:
#         rrm = re.match(rr, uri)
#         if rrm is not None:
#             params = {"user": getpass.getuser()}  # type: Dict[str, str]
#             params.update(rrm.groupdict())
#             params['host'] = to_ip(params['host'])
#             return ConnCreds(**params)  # type: ignore
#
#     raise ValueError("Can't parse {0!r} as ssh uri value".format(uri))
#
#
# NODE_KEYS = {}  # type: Dict[IPAddr, Any]
# SSH_KEY_PASSWD = None  # type: Optional[str]
#
#
# def set_ssh_key_passwd(passwd: str) -> None:
#     global SSH_KEY_PASSWD
#     SSH_KEY_PASSWD = passwd
#
#
# def set_key_for_node(host_port: IPAddr, key: bytes) -> None:
#     if paramiko is None:
#         raise RuntimeError("paramiko module is not available")
#
#     with StringIO(key.decode()) as sio:
#         NODE_KEYS[host_port] = paramiko.RSAKey.from_private_key(sio)  # type: ignore
#
#
# def connect(creds: ConnCreds,
#             conn_timeout: int = 60,
#             tcp_timeout: int = 15,
#             default_banner_timeout: int = 30) -> Any:
#
#     if paramiko is None:
#         raise RuntimeError("paramiko module is not available")
#
#     ssh = paramiko.SSHClient()
#     ssh.load_host_keys('/dev/null')
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh.known_hosts = None
#
#     end_time = time.time() + conn_timeout  # type: float
#
#     logger.debug("SSH connecting to %s", creds)
#
#     while True:
#         try:
#             time_left = end_time - time.time()
#             c_tcp_timeout = min(tcp_timeout, time_left)
#
#             banner_timeout_arg = {}  # type: Dict[str, int]
#             if paramiko.__version_info__ >= (1, 15, 2):
#                 banner_timeout_arg['banner_timeout'] = int(min(default_banner_timeout, time_left))
#
#             if creds.passwd is not None:
#                 ssh.connect(creds.addr.host,
#                             timeout=c_tcp_timeout,
#                             username=creds.user,
#                             password=cast(str, creds.passwd),
#                             port=creds.addr.port,
#                             allow_agent=False,
#                             look_for_keys=False,
#                             **banner_timeout_arg)
#             elif creds.key_file is not None:
#                 ssh.connect(creds.addr.host,
#                             username=creds.user,
#                             timeout=c_tcp_timeout,
#                             pkey=paramiko.RSAKey.from_private_key_file(creds.key_file, password=SSH_KEY_PASSWD),
#                             look_for_keys=False,
#                             port=creds.addr.port,
#                             **banner_timeout_arg)
#             elif creds.key is not None:
#                 with StringIO(creds.key.decode()) as sio:
#                     ssh.connect(creds.addr.host,
#                                 username=creds.user,
#                                 timeout=c_tcp_timeout,
#                                 pkey=paramiko.RSAKey.from_private_key(sio, password=SSH_KEY_PASSWD),  # type: ignore
#                                 look_for_keys=False,
#                                 port=creds.addr.port,
#                                 **banner_timeout_arg)
#             elif (creds.addr.host, creds.addr.port) in NODE_KEYS:
#                 ssh.connect(creds.addr.host,
#                             username=creds.user,
#                             timeout=c_tcp_timeout,
#                             pkey=NODE_KEYS[creds.addr],
#                             look_for_keys=False,
#                             port=creds.addr.port,
#                             **banner_timeout_arg)
#             else:
#                 key_file = os.path.expanduser('~/.ssh/id_rsa')
#                 ssh.connect(creds.addr.host,
#                             username=creds.user,
#                             timeout=c_tcp_timeout,
#                             key_filename=key_file,
#                             look_for_keys=False,
#                             port=creds.addr.port,
#                             **banner_timeout_arg)
#             return ssh
#         except (socket.gaierror, paramiko.PasswordRequiredException):
#             raise
#         except socket.error:
#             if time.time() > end_time:
#                 raise
#             time.sleep(1)
#
#
# def wait_ssh_available(addrs: List[IPAddr],
#                        timeout: int = 300,
#                        tcp_timeout: float = 1.0) -> None:
#
#     addrs_set = set(addrs)  # type: Set[IPAddr]
#
#     for _ in Timeout(timeout):
#         selector = selectors.DefaultSelector()  # type: selectors.BaseSelector
#         with selector:
#             for addr in addrs_set:
#                 sock = socket.socket()
#                 sock.setblocking(False)
#                 try:
#                     sock.connect(addr)
#                 except BlockingIOError:
#                     pass
#                 selector.register(sock, selectors.EVENT_READ, data=addr)
#
#             etime = time.time() + tcp_timeout
#             ltime = etime - time.time()
#             while ltime > 0:
#                 # convert to greater or equal integer
#                 for key, _ in selector.select(timeout=int(ltime + 0.99999)):
#                     selector.unregister(key.fileobj)
#                     try:
#                         key.fileobj.getpeername()  # type: ignore
#                         addrs_set.remove(key.data)
#                     except OSError as exc:
#                         if exc.errno == errno.ENOTCONN:
#                             pass
#                 ltime = etime - time.time()
#
#         if not addrs_set:
#             break
#

# async def run_ssh(host: str, ssh_opts: str, cmd: str, no_retry: bool = False, max_retry: int = 3, timeout: int = 20,
#                   input_data: bytes = None, merge_err: bool = False) -> CMDResult:
#     if no_retry:
#         max_retry = 0
#
#     ssh_cmd = "ssh {0} {1} {2}".format(ssh_opts, host, cmd)
#     logger.debug("SSH %s %r", host, cmd)
#     while True:
#         try:
#             return await run(ssh_cmd, input_data=input_data, timeout=timeout, merge_err=merge_err)
#         except (subprocess.CalledProcessError, TimeoutError) as lexc:
#             if max_retry == 0:
#                 raise
#             exc = lexc
#
#         err = exc.output
#         if isinstance(err, bytes):
#             err = err.decode()
#
#         if err:
#             logger.warning("SSH error for host %s. Cmd: %r. Err is %r. Will retry", host, cmd, err)
#         else:
#             logger.warning("SSH error for host %s. Cmd: %r. Most probably host " +
#                            "is unreachable via ssh. Will retry", host, cmd)
#
#         max_retry -= 1
#         time.sleep(1)


# class SSHHost(ISSHHost):
#     def __init__(self, conn: paramiko.SSHClient, info: NodeInfo) -> None:
#         self.conn = conn
#         self.info = info
#
#     def __str__(self) -> str:
#         return self.node_id
#
#     @property
#     def node_id(self) -> str:
#         return self.info.node_id
#
#     def put_to_file(self, path: Optional[str], content: bytes) -> str:
#         if path is None:
#             path = self.run("mktemp", nolog=True).strip()
#
#         logger.debug("PUT %s bytes to %s", len(content), path)
#
#         with self.conn.open_sftp() as sftp:
#             with sftp.open(path, "wb") as fd:
#                 fd.write(content)
#
#         return path
#
#     def disconnect(self):
#         self.conn.close()
#
#     def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
#         if not nolog:
#             logger.debug("SSH:{0} Exec {1!r}".format(self, cmd))
#
#         transport = self.conn.get_transport()
#         session = transport.open_session()
#
#         try:
#             session.set_combine_stderr(True)
#             stime = time.time()
#
#             session.exec_command(cmd)
#             session.settimeout(1)
#             session.shutdown_write()
#             output = ""
#
#             while True:
#                 try:
#                     ndata = session.recv(1024).decode("utf-8")
#                     if not ndata:
#                         break
#                     output += ndata
#                 except socket.timeout:
#                     pass
#
#                 if time.time() - stime > timeout:
#                     raise OSError(output + "\nExecution timeout")
#
#             code = session.recv_exit_status()
#         finally:
#             found = False
#
#             if found:
#                 session.close()
#
#         if code != 0:
#             templ = "SSH:{0} Cmd {1!r} failed with code {2}. Output: {3}"
#             raise OSError(templ.format(self, cmd, code, output))
#
#         return output
#
# def get_data(rr: str, data: str) -> str:
#     match_res = re.search("(?ims)" + rr, data)
#     return match_res.group(0)
#
#
# @dataclass
# class SWInfo:
#     os_version: Tuple[str, ...]
#     kernel_version: str
#     mtab: Dict[str, str]
#     libvirt_version: Optional[str]
#     qemu_version: Optional[str]
#
#
# async def get_sw_info(node: IAsyncNode) -> SWInfo:
#     os_version = tuple(await get_os(node))
#     kernel_version = (await node.get_file_content('/proc/version')).decode().strip()
#     mtab: Dict[str, str] = {}
#     for line in (await node.get_file_content('/etc/mtab')).decode().split("\n"):
#         line = line.strip()
#         if line.startswith('/dev/'):
#             dev, rest = line.split(" ", 1)
#             mtab[dev] = rest
#
#     try:
#         libvirt_version = (await node.run_stdout_str("virsh -v")).strip()
#     except OSError:
#         libvirt_version = None
#
#     try:
#         qemu_version = (await node.run_stdout_str("qemu-system-x86_64 --version")).strip()
#     except OSError:
#         qemu_version = None
#
#     return SWInfo(os_version, kernel_version, mtab, libvirt_version, qemu_version)
#
#
#
# @dataclass
# class NodeInfo:
#     ssh_creds: ConnCreds
#     roles: Set[str]
#     hostname: Optional[str] = None
#     params: Dict[str, Any] = field(default_factory_attr=dict)
#     hw_info: Optional[HWInfo] = None
#     sw_info: Optional[SWInfo] = None
#     os_vm_id: Optional[int] = None
#
#     @property
#     def node_id(self) -> str:
#         return "{0.host}:{0.port}".format(self.ssh_creds.addr)
#
#     def __str__(self) -> str:
#         return self.node_id
#
#     def __repr__(self) -> str:
#         return str(self)
#
#     def raw(self) -> Dict[str, Any]:
#         dct = self.__dict__.copy()
#         dct['ssh_creds'] = self.ssh_creds.raw()
#         dct['roles'] = list(self.roles)
#         dct['hw_info'] = self.hw_info.raw() if self.hw_info is not None else None
#         dct['sw_info'] = self.sw_info.raw() if self.sw_info is not None else None
#         return dct
#
#     @classmethod
#     def fromraw(cls, data: Dict[str, Any]) -> 'NodeInfo':
#         data = data.copy()
#         data['ssh_creds'] = ConnCreds.fromraw(data['ssh_creds'])
#         data['roles'] = set(data['roles'])
#         data['hw_info'] = None if data.get('hw_info') is None else HWInfo.fromraw(data['hw_info'])
#         data['sw_info'] = None if data.get('sw_info') is None else SWInfo.fromraw(data['sw_info'])
#         obj = cls.__new__(cls)  # type: ignore
#         obj.__dict__.update(data)
#         return obj
