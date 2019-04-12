#!/usr/bin/env python3
import configparser
import sys
import stat
import shutil
import hashlib
import tempfile
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Optional
from distutils.spawn import find_executable

from .. import b2ssize


EXT_TO_REMOVE_DEFAULT = "*.c *.pyi *.h *.cpp *.hpp *.dist-info *.pyx *.pxd"
MODULES_TO_REMOVE_DEFAULT = "ensurepip lib2to3 venv tkinter"
TRUE_VALS = {'True', 'true', 'yes', 'y', 't'}


unpacker_script = """#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

readonly ARCHNAME="${0}"
readonly CMD="${1:-}"

function help() {
    echo "${ARCHNAME} --list|--help|-h|--install [INSTALLATION_PATH(%DEFPATH%)]"
}

if [[ "${CMD}" == "--help" ]] || [[ "${CMD}" == "-h" ]] ; then
    help
    exit 0
fi

readonly ARCH_CONTENT_POS=$(awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0; }' "${ARCHNAME}")

if [[ "${CMD}" == "--list" ]] ; then
    tail "-n+${ARCH_CONTENT_POS}" "${ARCHNAME}" | tar --gzip --list
    exit 0
fi

if [[ "${CMD}" == "--install" ]] ; then
    readonly INSTALL_PATH="${2:-%DEFPATH%}"

    if [[ "${INSTALL_PATH}" == "" ]] ; then
        help
        exit 1
    fi

    mkdir --parents "${INSTALL_PATH}"
    tail "-n+${ARCH_CONTENT_POS}" "${ARCHNAME}" | tar -zx -C "${INSTALL_PATH}"
    tail "-n+${ARCH_CONTENT_POS}" "${ARCHNAME}" > "${INSTALL_PATH}/distribution.tar.gz"
    exit 0
fi

help
exit 1

__ARCHIVE_BELOW__
"""


@dataclass
class ArchConfig:
    sources: List[str]
    remove_ext: List[str]
    requirements: str
    version: str
    remove_modules: List[str]
    remove_pycache: bool
    remove_config_folder: bool
    path: Optional[Path]
    postinstall_help: Optional[str]


def load_config(path: Path) -> ArchConfig:
    cfg = configparser.ConfigParser()
    cfg.read_file(path.open())

    pkg = cfg['package']
    deploy = cfg['deploy']

    return ArchConfig(
        sources=pkg['sources'].split(),
        remove_ext=pkg.get('remove_ext', EXT_TO_REMOVE_DEFAULT).split(),
        requirements=pkg.get('requirements', 'requirements.txt'),
        version=pkg.get('version', f"{sys.version_info.major}.{sys.version_info.minor}"),
        remove_modules=pkg.get('remove_modules', MODULES_TO_REMOVE_DEFAULT).split(),
        remove_pycache=pkg.get('', 'True') in TRUE_VALS,
        remove_config_folder=pkg.get('remove_config_folder', 'True') in TRUE_VALS,
        path=deploy.get('path', None),
        postinstall_help=pkg.get('postinstall_help', None)
    )


def install_deps(target: Path, py_name: str, requirements: Path) -> None:
    tempo_libs = Path(tempfile.mkdtemp())
    opts = "--no-warn-script-location --no-compile --ignore-installed"
    cmd = f"{py_name} -m pip install {opts} --prefix {tempo_libs} -r {requirements}"
    subprocess.check_call(cmd, shell=True)

    site_packages = tempo_libs / 'lib' / py_name / 'site-packages'
    if not site_packages.exists():
        print("No libs installed by pip. If project has no requirements - remove requirements = XXX line from config")
        exit(1)

    site_packages.rename(target)
    shutil.rmtree(tempo_libs, ignore_errors=True)


def remove_files(root_dir: Path, ext_to_remove: List[str]) -> None:
    for fname in list(root_dir.rglob("*")):
        if fname.suffix in ext_to_remove:
            try:
                if fname.is_dir():
                    shutil.rmtree(fname, ignore_errors=True)
                else:
                    fname.unlink()
            except FileNotFoundError:
                pass


def copy_files(root_dir: Path, target: Path, patterns: List[str]) -> None:
    copy_anything = False
    for pattern in patterns:
        names = list(root_dir.glob(pattern))
        for name in names:
            copy_anything = True
            target_fl = target / name.relative_to(root_dir)
            target_fl.parent.mkdir(parents=True, exist_ok=True)
            if name.is_file():
                shutil.copyfile(name, target_fl)
            elif name.is_dir():
                shutil.copytree(name, target_fl, symlinks=False)

        if '*' not in pattern and '?' not in pattern and names == []:
            print(f"Failed: can't find name {pattern}")
            exit(1)

        print(f"Adding {root_dir}/{pattern} => {len(names)} files/dirs")

    if not copy_anything:
        print("Failed: Find nothing to copy to archive")
        exit(1)


def copy_py_binary(py_name: str, bin_target: Path):
    py_path = find_executable(py_name)
    assert py_path is not None
    shutil.copyfile(py_path, bin_target)
    bin_target.chmod(stat.S_IXUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def copy_py_lib(py_name: str, lib_target: Path):
    cmd = [py_name, "-c", "import sys; print('\\n'.join(sys.path))"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout

    for line in result.stdout.decode().split("\n"):
        line = line.strip()
        if not line:
            continue
        lib_path = Path(line)
        if (lib_path / 'encodings').is_dir():
            break
    else:
        raise RuntimeError(f"Can't find std library path for {py_name}")

    shutil.copytree(src=str(lib_path), dst=str(lib_target))


def clear_lib(lib_target: Path, cfg: ArchConfig):
    for name in cfg.remove_modules:
        tgt = lib_target / name
        if tgt.is_dir():
            shutil.rmtree(tgt)

    if cfg.remove_config_folder:
        for itemname in lib_target.iterdir():
            if itemname.name.startswith("config-") and itemname.is_dir():
                shutil.rmtree(itemname)

    if cfg.remove_pycache:
        for pycache_name in lib_target.rglob("__pycache__"):
            if pycache_name.is_dir():
                shutil.rmtree(pycache_name)


def get_hash(path: Path) -> str:
    hashobj = hashlib.md5()
    with path.open("rb") as target_fd:
        while True:
            data = target_fd.read(2 ** 20)
            if not data:
                break
            hashobj.update(data)

    return hashobj.hexdigest()


def parse_arge(argv: List[str]) -> Any:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--standalone", action="store_true",
                        help="Make standalone archive with python interpreter and std library")
    parser.add_argument("--config", default=None, help="Archive config file")
    parser.add_argument("base_folder", help="Path to package directory (it parent must contains requirements.txt)")
    parser.add_argument("arch_path", help="Path to store self-unpacking archive to")
    return parser.parse_args(argv[1:])


def main(argv: List[str]) -> int:
    opts = parse_arge(argv)
    root_dir = Path(opts.base_folder).absolute()
    config = opts.config if opts.config else root_dir / 'arch_config.cfg'
    cfg = load_config(Path(config))
    py_name = f"python{cfg.version}"

    requirements = root_dir / cfg.requirements if cfg.requirements else None

    if requirements and not requirements.exists():
        print(f"Can't find requirements at {requirements}")
        return 1

    target = Path(tempfile.mkdtemp())
    arch_target = Path(opts.arch_path).absolute()
    temp_arch_target = arch_target.parent / (arch_target.name + ".tmp")

    copy_files(root_dir, target, cfg.sources)

    if opts.standalone:
        standalone_root = target / 'python'
        standalone_root.mkdir(parents=True, exist_ok=True)
        standalone_stdlib = target / 'python' / 'lib' / py_name
        standalone_stdlib.parent.mkdir(parents=True, exist_ok=True)
        copy_py_binary(py_name, standalone_root / py_name)
        copy_py_lib(py_name, standalone_stdlib)
        clear_lib(standalone_stdlib, cfg)

        if requirements:
            install_deps(target / 'python' / 'lib' / py_name / 'dist-packages', py_name, requirements)
    elif requirements:
        install_deps(target / 'libs' , py_name, requirements)

    if cfg.remove_pycache:
        for pycache_name in target.rglob("__pycache__"):
            if pycache_name.is_dir():
                shutil.rmtree(pycache_name)

    tar_cmd = ["tar", "--create", "--gzip", "--directory=" + str(target), "--file", str(temp_arch_target),
               *(item.name for item in target.iterdir())]

    if opts.standalone:
        tar_cmd.append("python")

    subprocess.run(tar_cmd).check_returncode()

    with arch_target.open("wb") as target_fd:
        with temp_arch_target.open("rb") as source_arch:
            def_path = cfg.path if cfg.path else ""
            target_fd.write(unpacker_script.replace("%DEFPATH%", def_path).encode('ascii'))
            shutil.copyfileobj(source_arch, target_fd, length=2 ** 20)

    temp_arch_target.unlink()
    shutil.rmtree(str(target))

    ssize = b2ssize(arch_target.stat().st_size)
    print(f"Results stored into {arch_target}. Size = {ssize}B. MD5 = {get_hash(arch_target)}")

    return 0


if __name__ == "__main__":
    exit(main(sys.argv))
