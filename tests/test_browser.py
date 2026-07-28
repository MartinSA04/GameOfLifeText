"""Tests for the JSON bridge the browser worker calls."""

import json

import pytest

from game_of_life_text.browser import BOARD_PADDING, generate_text, supported_characters
from game_of_life_text.simulator import Board, Pattern, SimulationConfig


def test_generated_payload_settles_into_the_target() -> None:
    """Running the payload's initial cells should reproduce its target cells."""

    payload = json.loads(generate_text("Hi"))
    config = SimulationConfig(width=payload["width"], height=payload["height"], wrap=False)
    board = Board.from_points(config, [(x, y) for x, y in payload["initial"]])

    settled = board.step_n(payload["generations"])

    assert settled.live_cells == Pattern.from_points((x, y) for x, y in payload["target"])


def test_payload_is_json_with_room_around_the_construction() -> None:
    """The board should be padded so nothing starts on an edge."""

    payload = json.loads(generate_text("I"))
    xs = [x for x, _ in payload["initial"]]
    ys = [y for _, y in payload["initial"]]

    assert min(xs) >= BOARD_PADDING
    assert min(ys) >= BOARD_PADDING
    assert max(xs) <= payload["width"] - BOARD_PADDING
    assert max(ys) <= payload["height"] - BOARD_PADDING


def test_progress_runs_from_zero_to_a_stable_total() -> None:
    """The browser's plan bar needs a monotonic count against a fixed total."""

    reports: list[tuple[int, int]] = []

    generate_text("Hi", lambda done, total: reports.append((done, total)))

    done_values = [done for done, _ in reports]
    totals = {total for _, total in reports}
    assert len(totals) == 1
    assert done_values[0] == 0
    assert done_values == sorted(done_values)
    assert done_values[-1] == totals.pop()


def test_progress_does_not_change_the_payload() -> None:
    """Reporting is observation only: the plan must come out identical."""

    assert generate_text("Hi", lambda done, total: None) == generate_text("Hi")


def test_trailing_newlines_do_not_add_an_empty_line() -> None:
    """A trailing newline from a textarea should not change the plan."""

    assert generate_text("Hi\n") == generate_text("Hi")


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_blank_text_is_rejected(text: str) -> None:
    """Whitespace-only input should raise instead of planning an empty board."""

    with pytest.raises(ValueError, match="visible character"):
        generate_text(text)


def test_supported_characters_covers_the_visible_ascii_range() -> None:
    """The font list is what the browser validates typing against."""

    supported = supported_characters()

    assert set("abcXYZ0189") <= set(supported)
    assert " " in supported
    assert len(supported) == len(set(supported))
