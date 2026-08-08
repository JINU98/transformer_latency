from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SCENARIOS = "1,10,20%,50%,100%"


@dataclass(frozen=True)
class TokenScenario:
    """One 'how many tokens did we generate' point on the sweep.

    label      -- human/CSV friendly tag, e.g. "1tok", "10tok", "20pct"
    display    -- human readable string for chart axes, e.g. "1 token", "20% of L"
    is_percent -- True if this scenario was specified as a percentage of L
    raw_value  -- the number as written by the user (1, 10, 20.0, ...)
    """

    label: str
    display: str
    is_percent: bool
    raw_value: float

    def resolve(self, seq_len: int) -> int:
        """Return the number of decode steps (generated tokens) for a given L."""
        if self.is_percent:
            tokens = round(seq_len * self.raw_value / 100.0)
        else:
            tokens = round(self.raw_value)
        return max(1, int(tokens))


def parse_token_scenarios(value: str | None) -> list[TokenScenario]:
    """Parse a comma separated scenario spec.

    Examples of accepted tokens:
      "1"      -> exactly 1 generated token
      "10"     -> exactly 10 generated tokens
      "20%"    -> 20% of the sequence length L, rounded, min 1
      "1.5x"   -> same as percent but written as a multiplier of L (150%)
    """
    raw = value if value else DEFAULT_SCENARIOS
    scenarios: list[TokenScenario] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if item.endswith("%"):
            pct = float(item[:-1])
            label = f"{_fmt_num(pct)}pct"
            display = f"{_fmt_num(pct)}% of L"
            scenarios.append(TokenScenario(label, display, True, pct))
        elif item.lower().endswith("x"):
            mult = float(item[:-1])
            pct = mult * 100.0
            label = f"{_fmt_num(mult)}x"
            display = f"{_fmt_num(mult)}x L"
            scenarios.append(TokenScenario(label, display, True, pct))
        else:
            n = float(item)
            label = f"{_fmt_num(n)}tok"
            display = f"{_fmt_num(n)} token{'s' if n != 1 else ''}"
            scenarios.append(TokenScenario(label, display, False, n))
    return scenarios


def _fmt_num(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)
