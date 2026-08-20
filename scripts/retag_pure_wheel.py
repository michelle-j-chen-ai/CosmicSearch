"""Retag a pure-Python wheel that carries a CPython-specific tag.

`lilypad_py` is published as ``cp310-cp310-linux_x86_64`` but ships no compiled
extensions -- every entry lives under ``.data/purelib`` -- so it imports fine on
any Python 3. Installers reject it on the tag alone, which is what blocks the
image build on a 3.13 base.

This rewrites the wheel under a ``py3-none-any`` name, copying every entry byte
for byte and changing only the ``Tag:`` line of ``.dist-info/WHEEL``. It refuses
to touch a wheel that actually contains an extension module, so it cannot
silently mislabel something that really is version-specific.

    python scripts/retag_pure_wheel.py vendor/lilypad_py-*.whl
"""

from __future__ import annotations

import glob
import sys
import zipfile

_WHEEL_METADATA = (
    b"Wheel-Version: 1.0\n"
    b"Generator: retag_pure_wheel\n"
    b"Root-Is-Purelib: false\n"
    b"Tag: py3-none-any\n"
)
_NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")


def retag(src: str, dst: str) -> str:
    with zipfile.ZipFile(src) as zin:
        native = [n for n in zin.namelist() if n.endswith(_NATIVE_SUFFIXES)]
        if native:
            raise SystemExit(
                f"{src} contains {len(native)} compiled extension(s) "
                f"(e.g. {native[0]}); it is NOT pure Python and must not be retagged"
            )
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename.endswith(".dist-info/WHEEL"):
                    data = _WHEEL_METADATA
                zout.writestr(info, data)
    return dst


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    matches = sorted(glob.glob(argv[1]))
    if not matches:
        raise SystemExit(f"no wheel matched {argv[1]!r}")
    src = matches[0]
    name = src.rsplit("/", 1)[-1]
    dist, version = name.split("-")[0], name.split("-")[1]
    dst = src[: -len(name)] + f"{dist}-{version}-py3-none-any.whl"
    print(retag(src, dst))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
