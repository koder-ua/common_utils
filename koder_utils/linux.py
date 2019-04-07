import ipaddress
import socket
from typing import Optional, Tuple, Set, Dict, Any, List

from .rpc_node import IAsyncNode


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
