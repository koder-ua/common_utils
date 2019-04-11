import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Optional, Tuple, Set, Dict, Any, List

from . import IAsyncNode


def ip_and_hostname(ip_or_hostname: str) -> Tuple[str, Optional[str]]:
    """returns (ip, maybe_hostname)"""
    try:
        ipaddress.ip_address(ip_or_hostname)
        return ip_or_hostname, None
    except ValueError:
        return socket.gethostbyname(ip_or_hostname), ip_or_hostname


def parse_ipa4(data: str) -> Set[str]:
    """
    parse 'ip -o -4 a' output
    """
    res: Set[str] = set()
    # 26: eth0    inet 169.254.207.170/16
    for line in data.split("\n"):
        line = line.strip()
        if line:
            _, dev, _, ip_sz, *_ = line.split()
            ip, sz = ip_sz.split('/')
            ipaddress.IPv4Address(ip)
            res.add(ip)
    return res


def parse_var_list_file(content: str, separator: str, ignore_err: bool = False,
                        comment: str = "#", multiline_char: str = '\\') -> Dict[str, str]:
    """
    Parse file with format
    var_name: value
    """
    res: Dict[str, str] = {}
    pline: str = ""

    for ln in content.split("\n"):
        ln = ln.strip()

        if not ln:
            continue

        if comment and ln[0] in comment:
            if pline != "":
                raise ValueError("Commented line breaks multiline")
            continue

        pline += ln
        if multiline_char and pline[-1] == multiline_char:
            pline = pline[:-1]
            continue

        try:
            name, val = pline.split(separator)
        except ValueError:
            if not ignore_err:
                raise
        else:
            res[name.strip()] = val.strip()
            pline = ""

    if pline != "":
        raise ValueError("Unfinished line at the end of the file")

    return res


def parse_info_file_from_proc(content: str, ignore_err: bool = False) -> Dict[str, str]:
    return parse_var_list_file(content, ignore_err=ignore_err, separator=":", comment="", multiline_char="")


def parse_devices_tree(lsblkdct: Dict[str, Any]) -> Dict[str, str]:
    def fall_down(fnode: Dict[str, Any], root: str, res_dict: Dict[str, str]):
        res_dict['/dev/' + fnode['name']] = root
        for ch_node in fnode.get('children', []):
            fall_down(ch_node, root, res_dict)

    res: Dict[str, str] = {}
    for node in lsblkdct['blockdevices']:
        fall_down(node, '/dev/' + node['name'], res)
        res['/dev/' + node['name']] = '/dev/' + node['name']
    return res


def parse_sockstat_file(fc: str) -> Optional[Dict[str, Dict[str, str]]]:
    res: Dict[str, Dict[str, str]] = {}
    for ln in fc.split("\n"):
        if ln.strip():
            if ':' not in ln:
                return None
            name, params = ln.split(":", 1)
            params_l = params.split()
            if len(params_l) % 2 != 0:
                return None
            res[name] = dict(zip(params_l[:-1:2], params_l[1::2]))
    return res


async def get_host_interfaces(rpc: IAsyncNode) -> List[Tuple[bool, str]]:
    """Return list of host interfaces, returns pair (is_physical, name)"""

    res: List[Tuple[bool, str]] = []
    sys_net_file = await rpc.run("ls -l /sys/class/net")
    sys_net_file.check_returncode()

    for line in sys_net_file.stdout.strip().split("\n")[1:]:
        if not line.startswith('l'):
            continue

        params = line.split()
        if len(params) < 11:
            continue

        res.append(('/devices/virtual/' not in params[10], params[8]))
    return res


@dataclass
class ProcInfo:
    cmdline: str
    fd_count: int
    socket_stats_v4: Optional[Dict]
    socket_stats_v6: Optional[Dict]
    status_raw: str
    status: Dict[str, str]
    th_count: int
    io: Dict[str, str]
    memmap: List[Tuple[str, str, str]]
    sched_raw: str
    sched: Dict[str, str]
    stat: str


async def collect_process_info(conn: IAsyncNode, pid: int) -> ProcInfo:
    fpath = f"/proc/{pid}/net/sockstat"
    socket_stats_v4 = parse_sockstat_file(await conn.read_str(fpath))

    fpath = f"/proc/{pid}/net/sockstat6"
    socket_stats_v6 = parse_sockstat_file(await conn.read_str(fpath))

    # memmap
    mem_map_str = await conn.read_str(f"/proc/{pid}/maps", compress=True)
    memmap: List[Tuple[str, str, str]] = []
    for ln in mem_map_str.strip().split("\n"):
        mem_range, access, offset, dev, inode, *pathname = ln.split()
        memmap.append((mem_range, access, " ".join(pathname)))

    proc_stat = await conn.read_str(f"/proc/{pid}/status")
    sched_raw = await conn.read_str(f"/proc/{pid}/sched")
    try:
        data = "\n".join(sched_raw.strip().split("\n")[2:])
        sched = parse_info_file_from_proc(data, ignore_err=True)
    except:
        sched = {}

    return ProcInfo(
        cmdline=await conn.read_str(f"/proc/{pid}/cmdline"),
        fd_count=len(list(await conn.iterdir(f"/proc/{pid}/fd"))),
        socket_stats_v4=socket_stats_v4,
        socket_stats_v6=socket_stats_v6,
        status_raw=proc_stat,
        status=parse_info_file_from_proc(proc_stat),
        th_count=int(proc_stat.split('Threads:')[1].split()[0]),
        io=parse_info_file_from_proc(await conn.read_str(f"/proc/{pid}/io")),
        memmap=memmap,
        sched_raw=sched_raw,
        sched=sched,
        stat=await conn.read_str(f"/proc/{pid}/stat")
    )


async def get_hostname(node: IAsyncNode) -> str:
    return (await node.run_str("hostname")).strip()


async def get_all_ips(node: IAsyncNode) -> List[str]:
    return (await node.run_str("hostname -I")).split()


@dataclass
class OSRelease:
    distro: str
    release: str
    arch: str


async def get_os(node: IAsyncNode) -> OSRelease:
    """return os type, release and architecture for node.
    """
    arch = await node.run_str("arch")
    dist_type = (await node.run_str("lsb_release -i -s")).lower().strip()
    codename = (await node.run_str("lsb_release -c -s")).lower().strip()
    return OSRelease(dist_type, codename, arch)


SANE_FILE_NAME_RE = re.compile(r"[a-zA-Z.0-9_-]*$")


def check_filename_is_sane(fname: str) -> bool:
    return SANE_FILE_NAME_RE.match(fname) is not None
