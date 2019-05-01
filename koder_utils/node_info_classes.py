from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv4Network
from typing import Optional, List, Set, Dict, Any, Callable, Tuple, Iterable, Type, TypeVar

from . import b2ssize, Array, IntArithmeticMixin, FloatArithmeticMixin


# ---------------  LSHW ------------------------------------------------------------------------------------------------


@dataclass
class LSHWNetInfo:
    name: str
    speed: Optional[str]
    duplex: bool


@dataclass
class LSHWDiskInfo:
    size: int
    mount_point: Optional[str]
    device: Optional[str]


@dataclass
class LSHWCPUInfo:
    model: str
    cores: int


@dataclass
class LSHWInfo:
    hostname: Optional[str]
    cpu_info: List[LSHWCPUInfo]
    disks_info: Dict[str, LSHWDiskInfo]
    ram_size: int
    sys_name: Optional[str]
    mb: Optional[str]
    raw: str
    disks_raw_info: Dict[str, str]
    net_info: Dict[str, LSHWNetInfo]
    storage_controllers: List[str]
    summary: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.summary = {'cores': sum(info.cores for info in self.cpu_info),
                        'ram': self.ram_size,
                        'storage': sum(info.size for info in self.disks_info.values()),
                        'disk_count': len(self.disks_info)}

    def __str__(self):
        res = ["Simmary: {cores} cores, {ram}B RAM, {storage}B storage".format(**self.summary), str(self.sys_name)]
        if self.mb is not None:
            res.append("Motherboard: " + self.mb)

        if self.ram_size == 0:
            res.append("RAM: Failed to get RAM size")
        else:
            res.append(f"RAM {b2ssize(self.ram_size)}B")

        if self.cpu_info:
            res.append("CPU cores: Failed to get CPU info")
        else:
            res.append("CPU cores:")
            for info in self.cpu_info:
                res.append(f"    {info.cores} * {info.model}" if info.cores > 1 else f"    {info.model}")

        if self.storage_controllers:
            res.append("Disk controllers:")
            for descr in self.storage_controllers:
                res.append(f"    {descr}")

        if self.disks_info != {}:
            res.append("Storage devices:")
            for dev, info in sorted(self.disks_info.items()):
                res.append(f"    {dev} {b2ssize(info.size)}B {info.model}")
        else:
            res.append("Storage devices's: Failed to get info")

        if self.disks_raw_info != {}:
            res.append("Disks devices:")
            for dev, descr in sorted(self.disks_raw_info.items()):
                res.append(f"    {dev} {descr}")
        else:
            res.append("Disks devices's: Failed to get info")

        if self.net_info != {}:
            res.append("Net adapters:")
            for name, adapter in self.net_info.items():
                res.append(f"    {name} duplex={adapter.speed}")
        else:
            res.append("Net adapters: Failed to get net info")

        return str(self.hostname) + ":\n" + "\n".join("    " + i for i in res)


@dataclass
class HWModel:
    model: str
    serial: str
    vendor: str


@dataclass
class NetAdapterAddr:
    ip: IPv4Address
    net: IPv4Network
    is_routable: bool


T = TypeVar('T')


@dataclass
class NetStats(IntArithmeticMixin):
    recv_bytes: int
    recv_packets: int
    rerrs: int
    rdrop: int
    rfifo: int
    rframe: int
    rcompressed: int
    rmulticast: int
    send_bytes: int
    send_packets: int
    serrs: int
    sdrop: int
    sfifo: int
    scolls: int
    scarrier: int
    scompressed: int

    @classmethod
    def empty(cls: Type[T]) -> T:
        return cls(*([0] * 16))

    @property
    def total_err(self) -> int:
        return self.rerrs + self.rdrop + self.serrs + self.sdrop


@dataclass
class NetworkAdapter:
    dev: str
    is_phy: bool
    usage: NetStats
    duplex: Optional[bool] = None
    mtu: Optional[int] = None
    speed: Optional[int] = None
    ips: List[NetAdapterAddr] = field(default_factory=list)
    speed_s: Optional[str] = None
    d_usage: Optional[NetStats] = None


@dataclass
class ClusterNetData:
    mask: str
    usage: NetStats
    d_usage: Optional[NetStats] = None
    roles: Set[str] = field(default_factory=set)
    mtus: Set[int] = field(default_factory=set)
    speeds: Set[int] = field(default_factory=set)


@dataclass
class NetworkBond:
    name: str
    sources: List[str]


@dataclass
class IPANetDevInfo:
    name: str
    mtu: int


class BlockDevType(Enum):
    hwdisk = 0
    partition = 1
    lvm_lv = 2
    unknown = 3


class DiskType(Enum):
    sata_hdd = 0
    sas_hdd = 1
    nvme = 2
    sata_ssd = 3
    sas_ssd = 4
    virtio = 5
    unknown = 6

    @property
    def short_name(self) -> str:
        return {DiskType.nvme: 'nvme',
                DiskType.sata_ssd: 'ssd',
                DiskType.sas_ssd: 'ssd',
                DiskType.sata_hdd: 'hdd',
                DiskType.sas_hdd: 'hdd',
                DiskType.virtio: 'hdd',
                DiskType.unknown: 'hdd'}[DiskType(self.value)]


@dataclass
class DFInfo:
    path: Path
    name: str
    size: int
    free: int
    mountpoint: str


@dataclass
class BlockUsage(FloatArithmeticMixin):
    read_bytes: float
    write_bytes: float
    total_bytes: float
    read_iops: float
    write_iops: float
    io_time: float
    w_io_time: float
    iops: float
    queue_depth: Optional[float]
    lat: Optional[float]


@dataclass
class LogicBlockDev:
    tp: BlockDevType
    name: str
    hostname: str
    dev_path: Path
    size: int
    usage: BlockUsage
    mountpoint: Optional[str] = None
    fs: Optional[str] = None
    free_space: Optional[int] = None
    parent: Optional[Callable[[], Optional[Disk]]] = None
    children: Dict[str, LogicBlockDev] = field(default_factory=dict)
    label: Optional[str] = None
    d_usage: Optional[BlockUsage] = None
    partition_num: int = 0

    def __post_init__(self):
        assert '/dev/' + self.name == str(self.dev_path), f"name={self.name}, dev_path={self.dev_path}"
        assert '/' not in self.name, f"name={self.name}"
        for key in self.children:
            assert '/' not in key, f"children={self.children}"

        for pattern in ["sd[a-z]+(\d+)", "hd[a-z]+(\d+)", "vd[a-z]+(\d+)", "nvme\d+n\d+p(\d+)"]:
            rr = re.match(pattern, self.name)
            if rr:
                self.partition_num = int(rr.group(1))


@dataclass
class Disk:
    tp: DiskType
    logic_dev: LogicBlockDev
    extra: Dict[str, Any]
    scheduler: Optional[str]
    hw_model: HWModel
    rota: bool
    rq_size: int
    phy_sec: int
    min_io: int
    parent: Optional[Callable[[], Optional[Disk]]] = None

    # have to simulate LogicBlockDev inheritance, as mypy/pycharm works poorly with dataclass inheritance
    @property
    def name(self) -> str:
        return self.logic_dev.name

    @property
    def dev_path(self) -> Path:
        return self.logic_dev.dev_path

    @property
    def children(self) -> Dict[str, LogicBlockDev]:
        return self.logic_dev.children

    @property
    def size(self) -> int:
        return self.logic_dev.size

    @property
    def mountpoint(self) -> Optional[str]:
        return self.logic_dev.mountpoint

    @property
    def partition_num(self) -> int:
        return self.logic_dev.partition_num

    def __getattr__(self, name):
        return getattr(self.logic_dev, name)


@dataclass
class AggNetStat:
    """For /proc/net/stat & /proc/net/softnet_stat & Co"""
    raw: Array[int]

    @property
    def processed_v(self) -> Array[int]:
        return self.raw[:, 0]  # type: ignore

    @property
    def dropped_no_space_in_q_v(self) -> Array[int]:
        return self.raw[:, 1]  # type: ignore

    @property
    def no_budget_v(self) -> Array[int]:
        return self.raw[:, 2]  # type: ignore

    @property
    def processed(self) -> int:
        return self.processed_v.sum()  # type: ignore

    @property
    def dropped_no_space_in_q(self) -> int:
        return self.dropped_no_space_in_q_v.sum()  # type: ignore

    @property
    def no_budget(self) -> int:
        return self.no_budget_v.sum()  # type: ignore

    def __add__(self, other: AggNetStat) -> AggNetStat:
        return self.__class__(self.raw + other.raw)

    def __iadd__(self, other: AggNetStat) -> AggNetStat:
        self.raw += other.raw  # type: ignore
        return self

    def __sub__(self, other: AggNetStat) -> AggNetStat:
        return self.__class__(self.raw - other.raw)

    def __isub__(self, other: AggNetStat) -> AggNetStat:
        self.raw -= other.raw  # type: ignore
        return self


# ----------------   CLUSTER  ------------------------------------------------------------------------------------------


@dataclass
class Host:
    name: str
    stor_id: str
    net_adapters: Dict[str, NetworkAdapter]
    disks: Dict[str, Disk]
    logic_block_devs: Dict[str, LogicBlockDev]
    hw_info: Optional[LSHWInfo]
    bonds: Dict[str, NetworkBond]
    netstat: Optional[AggNetStat]
    d_netstat: Optional[AggNetStat]

    uptime: float
    open_tcp_sock: int
    open_udp_sock: int
    mem_total: int
    mem_free: int
    swap_total: int
    swap_free: int
    load_5m: Optional[float]

    def find_interface(self, net: IPv4Network) -> Optional[NetworkAdapter]:
        for adapter in self.net_adapters.values():
            for ip in adapter.ips:
                if ip.ip in net:
                    return adapter
        return None

    def iter_net_addrs(self) -> Iterable[Tuple[NetAdapterAddr, List[NetworkAdapter]]]:
        """For each ada"""
        for adapter in self.net_adapters.values():
            adapter_name = adapter.dev.split(".")[0] if '.' in adapter.dev else adapter.dev
            sources = self.bonds[adapter_name].sources if adapter_name in self.bonds else [adapter_name]
            for addr in adapter.ips:
                yield addr, [self.net_adapters[scr] for scr in sources]

    @property
    def cpu_cores(self) -> Optional[int]:
        if self.hw_info:
            return sum(inf.cores for inf in self.hw_info.cpu_info)
        else:
            return None

@dataclass
class OSRelease:
    distro: str
    release: str
    arch: str


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
