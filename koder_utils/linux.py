import logging
import re
import socket
import weakref
from enum import IntEnum
from ipaddress import ip_address, IPv4Address, IPv4Network
from pathlib import Path
from typing import Optional, Tuple, Set, Dict, Any, List, Iterable, Union, Iterator
import xml.etree.ElementTree as ElementTree

from . import (IAsyncNode, ssize2b, BlockUsage, IPANetDevInfo, NetStats, Disk, DiskType,
               LogicBlockDev, BlockDevType, HWModel, DFInfo, LSHWDiskInfo, LSHWInfo, LSHWNetInfo, LSHWCPUInfo,
               OSRelease, ProcInfo)


logger = logging.getLogger("utils.linux")


def ip_and_hostname(ip_or_hostname: str) -> Tuple[str, Optional[str]]:
    """returns (ip, maybe_hostname)"""
    try:
        ip_address(ip_or_hostname)
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
            IPv4Address(ip)
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

    if pline != "" and not ignore_err:
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
    except ValueError:
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


netstat_fields = "recv_bytes recv_packets rerrs rdrop rfifo rframe rcompressed" + \
                 " rmulticast send_bytes send_packets serrs sdrop sfifo scolls" + \
                 " scarrier scompressed"


def parse_ipa(data: str) -> Dict[str, IPANetDevInfo]:
    """
    parse 'ip a' output
    """
    res = {}
    for line in data.split("\n"):
        if line.strip() != '' and not line[0].isspace():
            rr = re.match(r"\d+:\s+(?P<name>[^:]*):.*?mtu\s+(?P<mtu>\d+)", line)
            if rr:
                name = rr.group('name')
                res[name] = IPANetDevInfo(name, int(rr.group('mtu')))
    return res


def parse_netdev(netdev: str) -> Dict[str, NetStats]:
    info: Dict[str, NetStats] = {}
    for line in netdev.strip().split("\n")[2:]:
        adapter, data = line.split(":")
        adapter = adapter.strip()
        assert adapter not in info
        info[adapter] = NetStats(**dict(zip(netstat_fields.split(), map(int, data.split()))))
    return info


def parse_meminfo(meminfo: str) -> Dict[str, int]:
    info = {}
    for line in meminfo.split("\n"):
        line = line.strip()
        if line == '':
            continue
        name, data = line.split(":", 1)
        data = data.strip()
        if " " in data:
            data = data.replace(" ", "")
            assert data[-1] == 'B'
            val = ssize2b(data[:-1])
        else:
            val = int(data)
        info[name] = val
    return info


def is_routable(ip: IPv4Address, net: IPv4Network) -> bool:
    return not (ip.is_loopback or net.prefixlen == 32)


class DiskStatFields(IntEnum):
    #  1 - major number
    #  2 - minor mumber
    #  3 - device name
    #  4 - reads completed successfully
    #  5 - reads merged
    #  6 - sectors read
    #  7 - time spent reading (ms)
    #  8 - writes completed
    #  9 - writes merged
    # 10 - sectors written
    # 11 - time spent writing (ms)
    # 12 - I/Os currently in progress
    # 13 - time spent doing I/Os (ms)
    # 14 - weighted time spent doing I/Os (ms)

    reads_completed = 3
    sectors_read = 5
    rtime = 6
    writes_completed = 7
    sectors_written = 9
    wtime = 10
    io_queue = 11
    io_time = 12
    weighted_io_time = 13


def parse_diskstats(data: str) -> Dict[str, BlockUsage]:

    sector_size = 512

    res = {}

    for line in data.split("\n"):
        line = line.strip()
        if line:
            vals = line.split()
            name = vals[2]

            read_bytes = int(vals[DiskStatFields.sectors_read]) * sector_size
            write_bytes = int(vals[DiskStatFields.sectors_written]) * sector_size
            read_iops = int(vals[DiskStatFields.reads_completed])
            write_iops = int(vals[DiskStatFields.writes_completed])
            io_time = int(vals[DiskStatFields.io_time])
            w_io_time = int(vals[DiskStatFields.weighted_io_time])

            iops = read_iops + write_iops
            qd = w_io_time / io_time if io_time > 1000 else None
            lat = w_io_time / iops if iops > 100 else None

            res[name] = BlockUsage(read_bytes=read_bytes,
                                   write_bytes=write_bytes,
                                   total_bytes=read_bytes + write_bytes,
                                   read_iops=read_iops,
                                   write_iops=write_iops,
                                   iops=iops,
                                   io_time=io_time,
                                   w_io_time=w_io_time,
                                   lat=lat,
                                   queue_depth=qd)

    return res


def parse_lsblkjs(data: List[Dict[str, Any]], hostname: str, diskstats: Dict[str, BlockUsage]) -> Iterable[Disk]:
    for disk_js in data:
        name = disk_js['name']

        if re.match(r"sr\d+|loop\d+", name):
            continue

        if disk_js['type'] != 'disk':
            logger.warning("lsblk for node %r return device %r, which is %r not 'disk'. Ignoring it",
                           hostname, name, disk_js['type'])
            continue

        tp = disk_js["tran"]
        if tp is None:
            if disk_js['subsystems'] == 'block:scsi:pci':
                tp = 'sas'
            elif disk_js['subsystems'] == 'block:nvme:pci':
                tp = 'nvme'
            elif disk_js['subsystems'] == 'block:virtio:pci':
                tp = 'virtio'

        # raid controllers often don't report ssd drives correctly
        # only SSD has 4k phy sec size
        phy_sec = int(disk_js["phy-sec"])
        if phy_sec == 4096:
            disk_js["rota"] = '0'

        stor_tp = {
            ('sata', '1'): DiskType.sata_hdd,
            ('sata', '0'): DiskType.sata_ssd,
            ('nvme', '0'): DiskType.nvme,
            ('nvme', '1'): DiskType.unknown,
            ('sas',  '1'): DiskType.sas_hdd,
            ('sas',  '0'): DiskType.sas_ssd,
            ('virtio', '0'): DiskType.virtio,
            ('virtio', '1'): DiskType.virtio,
        }.get((tp, disk_js["rota"]))

        if stor_tp is None:
            logger.warning("Can't detect disk type for %r in node %r. tran=%r, rota=%r, subsystem=%r." +
                           " Treating it as " + DiskType.sata_hdd.name,
                           name, hostname, disk_js['tran'], disk_js['rota'], disk_js['subsystems'])
            stor_tp = DiskType.sata_hdd

        lbd = LogicBlockDev(name=name,
                            dev_path=Path('/dev') / name,
                            size=int(disk_js['size']),
                            mountpoint=disk_js['mountpoint'],
                            fs=disk_js["fstype"],
                            tp=BlockDevType.hwdisk,
                            hostname=hostname,
                            usage=diskstats[name])

        dsk = Disk(logic_dev=lbd,
                   rota=disk_js['rota'] == '1',
                   scheduler=disk_js['sched'],
                   extra=disk_js,
                   hw_model=HWModel(vendor=disk_js['vendor'], model=disk_js['model'], serial=disk_js['serial']),
                   rq_size=int(disk_js["rq-size"]),
                   phy_sec=phy_sec,
                   min_io=int(disk_js["min-io"]),
                   tp=stor_tp)

        def fill_mountable(root: Dict[str, Any], parent: Union[Disk, LogicBlockDev]):
            if 'children' in root:
                for ch_js in root['children']:
                    assert '/' not in ch_js['name']
                    part = LogicBlockDev(name=ch_js['name'],
                                         tp=BlockDevType.unknown,
                                         dev_path=Path('/dev') / ch_js['name'],
                                         size=int(ch_js['size']),
                                         mountpoint=ch_js['mountpoint'],
                                         fs=ch_js["fstype"],
                                         parent=weakref.ref(dsk),
                                         label=ch_js.get("partlabel"),
                                         hostname=hostname,
                                         usage=diskstats[name])
                    parent.children[part.name] = part
                    fill_mountable(ch_js, part)

        fill_mountable(disk_js, dsk)

        yield dsk


def parse_df(data: str) -> Iterator[DFInfo]:
    lines = data.strip().split("\n")
    assert lines[0].split() == ["Filesystem", "1K-blocks", "Used", "Available", "Use%", "Mounted", "on"]
    for ln in lines[1:]:
        name, size, _, free, _, mountpoint = ln.split()
        yield DFInfo(path=Path(name),
                     name=name.split("/")[-1],
                     size=int(size),
                     free=int(free),
                     mountpoint=mountpoint)


def get_data(rr: str, data: str) -> str:
    return re.search("(?ims)" + rr, data).group(0)  # type: ignore


def parse_lshw_info(lshw_out: str) -> Optional[LSHWInfo]:
    lshw_et = ElementTree.fromstring(lshw_out)

    try:
        hostname = lshw_et.find("node").attrib['id']  # type: ignore
    except (AttributeError, KeyError):
        hostname = None

    try:
        sys_name = lshw_et.find("node/vendor").text + " " + lshw_et.find("node/product").text  # type: ignore
        sys_name = sys_name.lower().replace("(to be filled by o.e.m.)", "")
    except AttributeError:
        sys_name = None

    core = lshw_et.find("node/node[@id='core']")
    if core is None:
        return None

    try:
        mb: Optional[str] = " ".join(core.find(node).text for node in ['vendor', 'product', 'version'])  # type: ignore
    except AttributeError:
        mb = None

    cpu_info = []
    for cpu in core.findall("node[@class='processor']"):
        try:
            model = cpu.find('product').text  # type: ignore
            threads_node = cpu.find("configuration/setting[@id='threads']")
            cores = 1 if threads_node is None else int(threads_node.attrib['value'])  # type: ignore
        except (AttributeError, KeyError):
            pass
        else:
            assert isinstance(model, str)
            cpu_info.append(LSHWCPUInfo(model, cores))

    ram_size = 0
    for mem_node in core.findall(".//node[@class='memory']"):
        descr = mem_node.find('description')
        try:
            if descr is not None and descr.text == 'System Memory':
                mem_sz = mem_node.find('size')
                if mem_sz is None:
                    for slot_node in mem_node.find("node[@class='memory']"):  # type: ignore
                        slot_sz = slot_node.find('size')
                        if slot_sz is not None:
                            assert slot_sz.attrib['units'] == 'bytes'
                            ram_size += int(slot_sz.text)  # type: ignore
                else:
                    assert mem_sz.attrib['units'] == 'bytes'
                    ram_size += int(mem_sz.text)  # type: ignore
        except (AttributeError, KeyError, ValueError):
            pass

    net_info = {}
    for net in core.findall(".//node[@class='network']"):
        try:
            link = net.find("configuration/setting[@id='link']")
            if link.attrib['value'] == 'yes':  # type: ignore

                speed_node = net.find("configuration/setting[@id='speed']")
                speed = None if speed_node is None else speed_node.attrib['value']

                dup_node = net.find("configuration/setting[@id='duplex']")
                dup = False if dup_node is None else dup_node.attrib['value'] == "full"

                name = net.find("logicalname").text  # type: ignore
                assert isinstance(name, str)
                net_info[name] = LSHWNetInfo(name=name, speed=speed, duplex=dup)
        except (AttributeError, KeyError):
            pass

    storage_controllers = []
    for controller in core.findall(".//node[@class='storage']"):
        try:
            description = getattr(controller.find("description"), 'text', "")
            product = getattr(controller.find("product"), 'text', "")
            vendor = getattr(controller.find("vendor"), 'text', "")
            dev = getattr(controller.find("logicalname"), 'text', "")
            storage_controllers.append((f"{dev}: " if dev else "") + f"{description} {vendor} {product}")
        except AttributeError:
            pass

    disks_raw_info: Dict[str, str] = {}
    disks_info = {}
    for disk in core.findall(".//node[@class='disk']"):
        try:
            lname_node = disk.find('logicalname')
            if lname_node is not None:

                dev = lname_node.text.split('/')[-1]  # type: ignore
                if dev == "" or dev[-1].isdigit():
                    continue

                sz_node = disk.find('size')
                assert sz_node.attrib['units'] == 'bytes'  # type: ignore
                sz = int(sz_node.text)  # type: ignore
                disks_info[dev] = LSHWDiskInfo(size=sz, mount_point=None, device=None)
            else:
                description = disk.find('description').text  # type: ignore
                product = disk.find('product').text  # type: ignore
                vendor = disk.find('vendor').text  # type: ignore
                version = disk.find('version').text  # type: ignore
                serial = disk.find('serial').text  # type: ignore
                businfo = disk.find('businfo').text  # type: ignore
                assert isinstance(businfo, str)
                disks_raw_info[businfo] = f"{description} {product} {vendor} {version} {serial}"
        except (AttributeError, KeyError):
            pass

    return LSHWInfo(raw=lshw_out,
                    disks_raw_info=disks_raw_info,
                    disks_info=disks_info,
                    sys_name=sys_name,
                    hostname=hostname,
                    mb=mb,
                    cpu_info=cpu_info,
                    storage_controllers=storage_controllers,
                    net_info=net_info,
                    ram_size=ram_size)


def get_dev_file_name(path_or_name: str) -> str:
    res = path_or_name[len("/dev/"):] if path_or_name.startswith('/dev/') else path_or_name
    assert '/' not in res
    return res
