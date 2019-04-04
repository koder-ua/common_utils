"""
This module contains interfaces for storage classes
"""

import os
import re
import json
import shutil
import collections
from typing import Any, Type, IO, Tuple, cast, List, Dict, Iterable, Iterator, NamedTuple, Optional

from .types import NumVector, get_arr_info
from .istorage_nnp import IStorable, ISimpleStorage, ISerializer, _Raise, ObjClass, IStorageNNP

try:
    import yaml

    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

except ImportError:
    yaml = None

try:
    import pyaml
except ImportError:
    pyaml = None


try:
    import numpy
    from .istorage import IStorage
    IStorageBase = IStorage
except ImportError:
    numpy = None
    IStorageBase = IStorageNNP

ArrayData = NamedTuple("ArrayData",
                       [('header', List[str]), ('histo_bins', Optional[NumVector]), ('data', Optional[NumVector])])


class FSStorage(ISimpleStorage):
    """Store all data in files on FS"""

    def __init__(self, root_path: str, existing: bool) -> None:
        self.root_path = root_path
        self.existing = existing
        self.ignored = {'.', '..'}

    def j(self, path: str) -> str:
        return os.path.join(self.root_path, path)

    def isdir(self, path):
        return os.path.isdir(self.j(path))

    def put(self, value: bytes, path: str) -> None:
        jpath = self.j(path)
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        with open(jpath, "wb") as fd:
            fd.write(value)

    def get(self, path: str) -> bytes:
        try:
            with open(self.j(path), "rb") as fd:
                return fd.read()
        except FileNotFoundError as exc:
            raise KeyError(path) from exc

    def rm(self, path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.unlink(path)

    def __contains__(self, path: str) -> bool:
        return os.path.exists(self.j(path))

    def get_fname(self, path: str) -> str:
        return self.j(path)

    def get_fd(self, path: str, mode: str = "rb+") -> IO[bytes]:
        jpath = self.j(path)

        if mode in ('cb', 'ct'):
            create_on_fail = True
            mode = "rb+" if mode == 'cb' else 'rt+'
            dpath = os.path.dirname(jpath)
            if not os.path.exists(dpath):
                os.makedirs(dpath)
        else:
            create_on_fail = False

        try:
            fd = open(jpath, mode)
        except IOError:
            if not create_on_fail:
                raise
            fd = open(jpath, "wt" if 't' in mode else "wb")

        return cast(IO[bytes], fd)

    def sub_storage(self, path: str) -> 'FSStorage':
        return self.__class__(self.j(path), self.existing)

    def sync(self) -> None:
        pass

    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        path = self.j(path)

        if os.path.exists(path):
            if not os.path.isdir(path):
                raise OSError("{!r} is not a directory".format(path))
            return ((not fobj.is_dir(), fobj.name)
                    for fobj in os.scandir(path)
                    if fobj.path not in self.ignored)


class RawSerializer(ISerializer):
    """Placeholder for no serialization"""
    def pack(self, value: IStorable) -> bytes:
        if not isinstance(value, bytes):
            raise ValueError("Can't serialize object {!r}".format(type(value)))
        return value

    def unpack(self, data: bytes) -> bytes:
        return data


class TxtSerializer(ISerializer):
    """Placeholder for no serialization"""
    encoding = 'utf8'

    def pack(self, value: IStorable) -> bytes:
        if not isinstance(value, str):
            raise ValueError("Can't serialize object {!r}".format(type(value)))
        return value.encode(self.encoding)

    def unpack(self, data: bytes) -> str:
        return data.decode(self.encoding)


class JsonSerializer(ISerializer):
    """Serialize data to json"""
    def pack(self, value: IStorable) -> bytes:
        try:
            return json.dumps(value).encode('utf8')
        except Exception as exc:
            raise ValueError("Can't pickle object {!r} to json.") from exc

    def unpack(self, data: bytes) -> Any:
        # TODO: dirty hack for issue with 10.2.x ceph version osd df output
        return json.loads(data.decode("utf8").replace('"utilization":-nan,"var":-nan', '"utilization":0,"var":0'))


if yaml:
    class YAMLSerializer(ISerializer):
        """Serialize data to yaml"""
        def pack(self, value: IStorable) -> bytes:
            try:
                return yaml.dump(value, Dumper=Dumper, encoding="utf8", width=160)
            except Exception as exc:
                raise ValueError("Can't pickle to yaml: {!r}".format(type(value))) from exc

        def unpack(self, data: bytes) -> Any:
            return yaml.load(data.decode("utf8"), Loader=Loader)


    class SAFEYAMLSerializer(ISerializer):
        """Serialize data to yaml"""
        def pack(self, value: IStorable) -> bytes:
            try:
                return yaml.safe_dump(value, encoding="utf8", width=160)
            except Exception as exc:
                raise ValueError("Can't pickle to yaml: {!r}".format(type(value))) from exc

        def unpack(self, data: bytes) -> Any:
            return yaml.safe_load(data.decode('utf8'))

    if pyaml:
        class PYAMLSerializer(ISerializer):
            """Serialize data to yaml"""
            def pack(self, value: IStorable) -> bytes:
                try:
                    if isinstance(value, IStorable) and hasattr(value.__class__, 'raw'):
                        return pyaml.dumps(value.raw(), width=160)
                    return pyaml.dumps(value, width=160)
                except Exception as exc:
                    raise ValueError("Can't pickle to yaml: {!r}".format(type(value))) from exc

            def unpack(self, data: bytes) -> Any:
                return yaml.safe_load(data.decode('utf8'))
    else:
        PYAMLSerializer = None
else:
    PYAMLSerializer = YAMLSerializer = SAFEYAMLSerializer = None


class Storage(IStorageBase):
    """interface for storage"""
    csv_file_encoding = 'utf8'

    def __init__(self, sstorage: ISimpleStorage, serializer: ISerializer) -> None:
        self.sstorage = sstorage
        self.serializer = serializer
        self.cache = {}
        self.other_caches = collections.defaultdict(dict)

    def flush(self):
        self.cache = {}
        self.other_caches = collections.defaultdict(dict)

    def sub_storage(self, *path: str) -> 'Storage':
        fpath = "/".join(path)
        return self.__class__(self.sstorage.sub_storage(fpath), self.serializer)

    def put(self, value: Any, *path: str) -> None:
        dct_value = cast(IStorable, value).raw() if isinstance(value, IStorable) else value
        serialized = self.serializer.pack(dct_value)  # type: ignore
        path_s = "/".join(path)
        self.sstorage.put(serialized, path_s)
        self.cache.pop(path_s, None)

    def get(self, path: str, default: Any = _Raise) -> Any:
        try:
            vl = self.sstorage.get(path)
        except:
            if default is _Raise:
                raise
            return default
        return self.serializer.unpack(vl)

    def rm(self, *path: str) -> None:
        path_s = "/".join(path)
        self.sstorage.rm("/".join(path))
        self.cache.pop(path_s, None)

    def load(self, obj_class: Type[ObjClass], *path: str) -> ObjClass:
        path_s = "/".join(path)
        if path_s not in self.cache:
            self.cache[path_s] = cast(ObjClass, obj_class.fromraw(self.get(path_s)))
        return self.cache[path_s]

    # ---------------  List of values ----------------------------------------------------------------------------------

    def put_list(self, value: Iterable[IStorable], *path: str) -> None:
        serialized = self.serializer.pack([obj.raw() for obj in value])  # type: ignore
        path_s = "/".join(path)
        self.sstorage.put(serialized, "/".join(path))
        self.cache.pop(path_s, None)

    def load_list(self, obj_class: Type[ObjClass], *path: str) -> List[ObjClass]:
        path_s = "/".join(path)
        if path_s not in self.cache:
            raw_val = cast(List[Dict[str, Any]], self.get(path_s))
            assert isinstance(raw_val, list)
            self.cache[path_s] = [cast(ObjClass, obj_class.fromraw(val)) for val in raw_val]
        return self.cache[path_s]

    def __contains__(self, path: str) -> bool:
        return path in self.cache or path in self.sstorage

    def put_raw(self, val: bytes, *path: str) -> str:
        path_s = "/".join(path)
        self.sstorage.put(val, path_s)
        self.cache.pop(path_s, None)
        return self.get_fname(path_s)

    def get_fname(self, fpath: str) -> str:
        return cast(FSStorage, self.sstorage).j(fpath)

    def get_raw(self, *path: str) -> bytes:
        return self.sstorage.get("/".join(path))

    def append_raw(self, value: bytes, *path: str) -> None:
        path_s = "/".join(path)
        with self.sstorage.get_fd(path_s, "rb+") as fd:
            fd.seek(0, os.SEEK_END)
            fd.write(value)
        self.cache.pop(path_s, None)

    def get_fd(self, path: str, mode: str = "r") -> IO:
        self.cache.pop(path, None)
        return self.sstorage.get_fd(path, mode)

    def sync(self) -> None:
        self.sstorage.sync()

    def __enter__(self) -> 'Storage':
        return self

    def __exit__(self, x: Any, y: Any, z: Any) -> None:
        self.sync()

    def list(self, *path: str) -> Iterator[Tuple[bool, str]]:
        return self.sstorage.list("/".join(path))

    def iter_paths(self, root: str, path_parts: List[str],
                   already_found_groups: Dict[str, str]) -> Iterator[Tuple[bool, str, Dict[str, str]]]:

        curr = path_parts[0]
        rest = path_parts[1:]

        for is_file, name in self.list(root):
            if rest and is_file:
                continue

            rr = re.match(pattern=curr + "$", string=name)
            if rr:
                if root:
                    path = root + "/" + name
                else:
                    path = name

                new_groups = rr.groupdict().copy()
                new_groups.update(already_found_groups)

                if rest:
                    yield from self.iter_paths(path, rest, new_groups)
                else:
                    yield is_file, path, new_groups

    # --------------  Arrays -------------------------------------------------------------------------------------------

    def read_headers(self, fd) -> Tuple[str, List[str], List[str], Optional[NumVector]]:
        assert numpy is not None
        header = fd.readline().decode(self.csv_file_encoding).rstrip().split(",")
        dtype, has_header2, header2_dtype, *ext_header = header

        if has_header2 == 'true':
            ln = fd.readline().decode(self.csv_file_encoding).strip()
            header2 = numpy.fromstring(ln, sep=',', dtype=header2_dtype)
        else:
            assert has_header2 == 'false', \
                "In file {} has_header2 is not true/false, but {!r}".format(fd.name, has_header2)
            header2 = None
        return dtype, ext_header, header, header2

    def get_array(self, path: str, nc: bool = False) -> ArrayData:
        assert numpy is not None

        with self.sstorage.get_fd(path, "rb") as fd:
            fd.seek(0, os.SEEK_SET)

            if not nc:
                stats = os.fstat(fd.fileno())
                if path in self.cache:
                    size, atime, arr_info = self.cache[path]
                    if size == stats.st_size and atime == stats.st_atime_ns:
                        return arr_info

            data_dtype, header, _, header2 = self.read_headers(fd)
            dt = fd.read().decode(self.csv_file_encoding).strip()

        if len(dt) != 0:
            arr = numpy.fromstring(dt.replace("\n", ','), sep=',', dtype=data_dtype)
            lines = dt.count("\n") + 1
            assert len(set(ln.count(',') for ln in dt.split("\n"))) == 1, \
                "Data lines in {!r} have different element count".format(path)
            arr.shape = [lines] if lines == arr.size else [lines, -1]
        else:
            arr = None

        arr_data = ArrayData(header, header2, arr)
        if not nc:
            self.cache[path] = (stats.st_size, stats.st_atime_ns, arr_data)

        return arr_data

    def put_array(self, path: str,
                  data: NumVector,
                  header: List[str],
                  header2: NumVector = None,
                  append_on_exists: bool = False) -> None:
        self.cache.pop(path, None)
        dtype, shape = get_arr_info(data)
        dtype2 = None if header2 is None else get_arr_info(header2)[0]
        header = [dtype] + (['false', ''] if header2 is None else ['true', dtype2]) + header
        exists = append_on_exists and path in self.sstorage

        vw = data.view().reshape((data.shape[0], 1)) \
             if (numpy and isinstance(data, numpy.ndarray) and len(shape) == 1) \
             else data

        mode = "cb" if not exists else "rb+"

        with self.sstorage.get_fd(path, mode) as fd:
            if exists:
                data_dtype, _, full_header, curr_header2 = self.read_headers(fd)

                assert data_dtype == dtype, \
                    "Path {!r}. Passed data type ({!r}) and current data type ({!r}) doesn't match"\
                        .format(path, dtype, data_dtype)

                assert header == full_header, \
                    "Path {!r}. Passed header ({!r}) and current header ({!r}) doesn't match"\
                        .format(path, header, full_header)

                assert header2 == curr_header2, \
                    "Path {!r}. Passed header2 != current header2: {!r}\n{!r}".format(path, header2, curr_header2)

                fd.seek(0, os.SEEK_END)
                self.cache.pop(path, None)
            else:
                self.cache.pop(path, None)
                fd.write((",".join(header) + "\n").encode(self.csv_file_encoding))
                if header2 is not None:
                    fd.write((",".join(map(str, header2)) + "\n").encode(self.csv_file_encoding))

            if numpy and isinstance(data, numpy.ndarray):
                numpy.savetxt(fd, vw, delimiter=',', newline="\n", fmt="%lu")
            else:
                assert len(shape) == 1
                fc = ("{}\n" * len(vw)).format(*vw)
                fd.write(fc.encode(self.csv_file_encoding))

            if not exists:
                fd.truncate()


class _Def:
    pass


class AttredStorage:
    def __init__(self, storage: FSStorage, serializer: ISerializer, ext: str) -> None:
        self.__dict__.update({
            "_AttredStorage__storage": storage,
            "_AttredStorage__serializer" : serializer,
            "_AttredStorage__ext": ext,
            "_AttredStorage__r": storage.root_path
        })

    def __load(self, spath: str, ext: str, dir_allowed: bool = True) -> Tuple[bool, Any]:
        path = spath.split("/")
        curr = self

        if ext is not None:
            last = path[-1]
            path = path[:-1]
        else:
            last = None

        for step in path:
            if not curr.__storage.isdir(step):
                raise KeyError("Path {0!r} expected to be a dir, but it's a file at {1!r}".format(step, curr.__r))
            curr = curr.__class__(curr.__storage.sub_storage(step), curr.__serializer, curr.__ext)

        if not last:
            return True, curr

        if curr.__storage.isdir(last):
            if dir_allowed:
                return True, curr.__class__(curr.__storage.sub_storage(last), curr.__serializer, curr.__ext)
            else:
                raise KeyError("Path {0!r} expected to be a dir, but it's a file at {1!r}".format(last, curr.__r))

        return False, curr.__storage.get(last + ("." + ext if ext != '' else ''))

    def __getitem__(self, path: str) -> Any:
        isdir, val = self.__load(path, self.__ext)
        return val if isdir else self.__serializer.unpack(val)

    def __setitem__(self, path: str, val: IStorable) -> None:
        self.__storage.put(self.__serializer.pack(val), path + '.' + self.__ext)

    def __setattr__(self, name: str, val: IStorable) -> None:
        self.__storage.put(self.__serializer.pack(val), name + '.' + self.__ext)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError("Can't found file '{0}.{1}' or dir '{0}' at {2!r}. {3!s}"
                                 .format(name, self.__ext, self.__r, exc))

    def __str__(self) -> str:
        return "{0.__class__.__name__}({0.__r})".format(self)

    def __iter__(self) -> Iterator[Tuple[bool, str]]:
        return iter(self.__storage.list("."))

    def __contains__(self, value) -> bool:
        return value in self.__storage or (value + "." + self.__ext) in self.__storage

    def get(self, path: str, default: Any = None, ext: str = _Def) -> Any:
        try:
            return self.__load(path, self.__ext if ext is _Def else ext, dir_allowed=False)[1]
        except (AttributeError, KeyError):
            return default

    def put(self, path: str, val: IStorable, ext=_Def):
        if ext is _Def:
            ext = self.__ext
        return self.__storage.put(val, path + ("." + ext if ext != '' else ''))

    def __len__(self) -> int:
        return len(self.__storage.list("."))


serializer_map = {
    'safe': SAFEYAMLSerializer,
    'pretty': PYAMLSerializer,
    'yaml': YAMLSerializer,
    'json': JsonSerializer,
    'js': JsonSerializer,
    'raw': RawSerializer,
    'xml': RawSerializer,
    'txt': TxtSerializer,
}


def make_attr_storage(storage, ext, serializer=None):
    if serializer is None:
        serializer = serializer_map[ext]()
    return AttredStorage(storage, serializer, ext)


def make_storage(url, existing=False, serializer='safe'):
    fstor = FSStorage(url, existing)
    return Storage(fstor, serializer_map[serializer]())


class TypedStorage:
    def __init__(self, storage):
        self.txt = make_attr_storage(storage.sstorage, 'txt')
        self.json = make_attr_storage(storage.sstorage, 'json')
        self.xml = make_attr_storage(storage.sstorage, 'xml')
        self.raw = storage

    def substorage(self, path):
        return self.__class__(self.raw.substorage(path))

