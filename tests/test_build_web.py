"""Tests for the browser bundle assembled by scripts/build_web.py."""

import ast
import importlib.util
import json
import re
import struct
from collections import deque
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "game_of_life_text"


def _load_build_script() -> ModuleType:
    """Import scripts/build_web.py, which lives outside the installed package."""

    path = REPO_ROOT / "scripts" / "build_web.py"
    spec = importlib.util.spec_from_file_location("build_web", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_web = _load_build_script()


def _package_imports(module_name: str) -> set[str]:
    """Return the sibling modules ``module_name`` imports, however it does it."""

    tree = ast.parse((PACKAGE_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
    package = build_web.PACKAGE_NAME
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:
                found.update(alias.name for alias in node.names)
            elif node.level and node.module:
                found.add(node.module.split(".")[0])
            elif node.module and node.module.startswith(f"{package}."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{package}."):
                    found.add(alias.name.split(".")[1])
    return found


def _reachable_modules() -> set[str]:
    """Walk imports out from browser.py, the worker's only entry point."""

    seen = {"browser"}
    queue = deque(seen)
    while queue:
        for name in _package_imports(queue.popleft()):
            if name not in seen and (PACKAGE_ROOT / f"{name}.py").exists():
                seen.add(name)
                queue.append(name)
    return seen


def test_bundled_modules_match_what_the_worker_imports() -> None:
    """The copy list must be the import closure of browser.py, plus __init__."""

    bundled = {name.removesuffix(".py") for name in build_web.WEB_MODULES}

    assert bundled == _reachable_modules() | {"__init__"}


def test_bundle_contains_the_site_and_the_python_payload(tmp_path: Path) -> None:
    """The static half of a build should be servable from its own root."""

    out = tmp_path / "dist"
    written = build_web.build(out)

    assert (out / "index.html").is_file()
    assert (out / "styles.css").is_file()
    assert all(path.is_file() for path in written)

    manifest = json.loads((out / "pysrc" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["package"] == build_web.PACKAGE_NAME
    for name in manifest["modules"]:
        assert (out / "pysrc" / manifest["package"] / name).is_file()


def test_bundle_leaves_out_the_desktop_gui(tmp_path: Path) -> None:
    """PySide6 has no Pyodide wheel, so the GUI must never reach the browser."""

    out = tmp_path / "dist"
    build_web.build(out)
    payload = out / "pysrc" / build_web.PACKAGE_NAME

    assert not (payload / "gui.py").exists()
    assert not any(path.name.startswith("profile_generator") for path in payload.iterdir())


def test_rebuilding_replaces_stale_files(tmp_path: Path) -> None:
    """The output directory is wiped, so a renamed file cannot linger."""

    out = tmp_path / "dist"
    build_web.build(out)
    stale = out / "stale.js"
    stale.write_text("// removed", encoding="utf-8")

    build_web.build(out)

    assert not stale.exists()


def test_worker_reads_the_module_list_from_the_manifest() -> None:
    """The worker must not carry its own copy of the module list."""

    worker = (REPO_ROOT / "web" / "engine-worker.ts").read_text(encoding="utf-8")

    assert "manifest.json" in worker
    for name in build_web.WEB_MODULES:
        assert name not in worker


@pytest.mark.parametrize("name", build_web.SITE_FILES)
def test_declared_site_files_exist(name: str) -> None:
    """Catch a renamed asset before the build fails in CI."""

    assert (build_web.WEB_ROOT / name).is_file()


def _page() -> str:
    return (build_web.WEB_ROOT / "index.html").read_text(encoding="utf-8")


def test_the_site_agrees_with_itself_about_where_it_lives() -> None:
    """One domain, spelled the same way everywhere a crawler will read it.

    A stale absolute URL is invisible in the browser and expensive in search:
    it points canonicalisation, the sitemap and every link preview somewhere
    the site is not.
    """

    canonical = re.search(r'<link rel="canonical" href="([^"]+)"', _page())
    assert canonical is not None
    site = canonical.group(1)
    assert site.startswith("https://")
    assert site.endswith("/")

    page = _page()
    assert f'<meta property="og:url" content="{site}"' in page
    assert f'<meta property="og:image" content="{site}og.png"' in page
    assert f'<meta name="twitter:image" content="{site}og.png"' in page

    sitemap = (build_web.WEB_ROOT / "sitemap.xml").read_text(encoding="utf-8")
    assert f"<loc>{site}</loc>" in sitemap

    robots = (build_web.WEB_ROOT / "robots.txt").read_text(encoding="utf-8")
    assert f"Sitemap: {site}sitemap.xml" in robots

    for node in json.loads(_structured_data())["@graph"]:
        assert node["@id"].startswith(site)


def _structured_data() -> str:
    match = re.search(
        r'<script type="application/ld\+json">(.*?)</script>', _page(), flags=re.DOTALL
    )
    assert match is not None
    return match.group(1)


def test_structured_data_is_valid_json_and_claims_nothing_it_cannot_back() -> None:
    """Fabricated ratings or prices are a manual-action risk, so assert we ship none."""

    graph = json.loads(_structured_data())["@graph"]
    types = {node["@type"] for node in graph}

    assert {"WebSite", "WebApplication"} <= types
    assert not any("aggregateRating" in node or "review" in node for node in graph)

    app = next(node for node in graph if node["@type"] == "WebApplication")
    assert app["offers"]["price"] == "0"
    assert app["isAccessibleForFree"] is True


def test_the_social_card_is_a_png_at_the_size_the_page_advertises() -> None:
    """Crawlers trust og:image:width/height; a mismatch crops the preview."""

    data = (build_web.WEB_ROOT / "og.png").read_bytes()

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    page = _page()
    assert f'<meta property="og:image:width" content="{width}"' in page
    assert f'<meta property="og:image:height" content="{height}"' in page
