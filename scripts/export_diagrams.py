"""Export editable Wiki diagrams as SVGs with native vector text.

Run from any directory with draw.io Desktop installed. Explicit newlines in
the source control label wrapping; html=0 and whiteSpace=nowrap let draw.io
emit SVG text instead of HTML labels with raster fallback images.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = (ROOT / "docs/assets/concepts/diagrams", ROOT / "docs/assets/workflows")
SVG = "{http://www.w3.org/2000/svg}"


def validate_source(source):
    for cell in ET.parse(source).findall(".//mxCell"):
        if not cell.get("value"):
            continue
        style = dict(part.split("=", 1) for part in cell.get("style", "").split(";") if "=" in part)
        if style.get("html") != "0" or style.get("whiteSpace") != "nowrap":
            raise ValueError(f"{source.name}: label {cell.get('id')} needs html=0;whiteSpace=nowrap;")


def validate_export(path):
    root = ET.parse(path).getroot()
    forbidden = {SVG + tag for tag in ("image", "foreignObject", "script")}
    if any(node.tag in forbidden for node in root.iter()):
        raise ValueError(f"{path.name}: raster images, HTML labels, or scripts found")
    if "content" in root.attrib or not root.findall(f".//{SVG}text"):
        raise ValueError(f"{path.name}: expected standalone SVG text without embedded diagram source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio", default="drawio", help="draw.io Desktop executable (draw.io.exe on Windows)")
    args = parser.parse_args()
    executable = shutil.which(args.drawio)
    if executable is None:
        raise SystemExit(f"draw.io executable not found: {args.drawio}")
    startup = None
    if hasattr(subprocess, "STARTUPINFO"):
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
    sources = sorted(p for folder in FOLDERS for p in folder.glob("*.drawio"))
    for source in sources:
        validate_source(source)
    for source in sources:
        output = source.with_suffix(".svg")
        subprocess.run([executable, "--export", "--format", "svg", "--theme", "light",
                        "--border", "0", "--output", str(output), str(source)],
                       check=True, startupinfo=startup)
        validate_export(output)
    print(f"Exported and validated {len(sources)} SVGs ({sum(p.with_suffix('.svg').stat().st_size for p in sources):,} bytes).")


if __name__ == "__main__":
    main()
