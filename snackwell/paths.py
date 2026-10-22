"""Repository paths shared by the command-line tools and simulator."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rtl_sources() -> list[Path]:
    """Return the package first, followed by every RTL implementation file."""
    package = ROOT / "rtl/snackwell_pkg.sv"
    sources = sorted((ROOT / "rtl").rglob("*.sv")) + sorted((ROOT / "rtl").rglob("*.v"))
    return [package, *(path for path in sources if path != package)]
