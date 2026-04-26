"""Conway's Game of Life GUI simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gui import main as main


def __getattr__(name: str) -> object:
    """Load the GUI entry point only when it is requested."""

    if name == "main":
        from .gui import main

        return main
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["main"]
