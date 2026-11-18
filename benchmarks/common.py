from collections.abc import Callable

from torch.utils.benchmark import Timer


def median_seconds(function: Callable[[], object], min_run_time: float = 0.15) -> float:
    for _ in range(3):
        function()
    measurement = Timer(stmt="fn()", globals={"fn": function}).blocked_autorange(
        min_run_time=min_run_time
    )
    return measurement.median


def milliseconds(seconds: float) -> str:
    return f"{seconds * 1e3:.3f}"
