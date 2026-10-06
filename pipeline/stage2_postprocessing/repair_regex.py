"""Repair common formatting slips in Stage 2 output.

Target line shapes:
    - [DATE] [EXACT|APPROX] <event>      DATE is YYYY, YYYY-MM, YYYY-MM-DD or "START to END"
    - [PRE ADM] <event>
    - [INDETERMINATE] <event>
([NO DATE PROVIDED] from the older v1 prompt is also accepted.)

Each repair is a small regex applied per line in a fixed order. Lines that are
still malformed after repair are kept as they are, never dropped.
"""

import re

_DATE = r"\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?"
_DATE_OR_RANGE = rf"{_DATE}(?:\s+to\s+{_DATE})?"

_CANONICAL_DATELESS = re.compile(r"^- \[(?:PRE ADM|INDETERMINATE|NO DATE PROVIDED)\] \S.*$")
_CANONICAL_DATED = re.compile(
    r"^- \[\d{4}(?:-\d{2}(?:-\d{2})?)?(?: to \d{4}(?:-\d{2}(?:-\d{2})?)?)?\] "
    r"\[(?:EXACT|APPROX)\] \S.*$"
)

# Bullets: '*', '•', '·' -> '- '.
_BULLET = re.compile(r"^(\s*)[*•·]\s+")
# Missing bullet on a tag-led line, e.g. '[PRE ADM] ...' or '[-2110-07-05] [EXACT] ...'.
_MISSING_BULLET = re.compile(r"^(\s*)\[-?(?=\d{4}|EXACT|APPROX|PRE ADM|INDETERMINATE|NO DATE PROVIDED)")
# Markdown **bold** and __underline__. MIMIC de-identification tokens [**...**] are not bold.
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_DEID = re.compile(r"\[\*\*[^\]]*?\*\*\]")
_DEID_MASK = re.compile("\x00(\\d+)\x00")
_UNDERLINE = re.compile(r"__([^_]+)__")
# Tag spelling variants.
_ALIAS_EXACT = re.compile(r"\[(?:EXACT_DATE|EXACT DATE)\]")
_ALIAS_APPROX = re.compile(r"\[(?:APPROXIMATE|APPROX_DATE|APPROX DATE)\]")
_ALIAS_NO_DATE = re.compile(r"\[(?:UNKNOWN|UNDATED|NO DATE|NODATE|NO_DATE|NO_DATE_PROVIDED|NODATEPROVIDED)\]")
# Date and tag in one bracket: '[2137-1-14, EXACT]' or '[EXACT 2137-01-14]'.
_COMBINED_COMMA = re.compile(rf"^- \[({_DATE_OR_RANGE})[,;]\s*(EXACT|APPROX)\]")
_COMBINED_TYPE_FIRST = re.compile(rf"^- \[(EXACT|APPROX)\s+({_DATE_OR_RANGE})\]")
# Tag written at the end of the line instead of the front.
_TRAILING_TAG = re.compile(rf"^- (?!\[)(.+?)\s+\[({_DATE_OR_RANGE})\]\s+\[(EXACT|APPROX)\]\s*$")
_TRAILING_NO_DATE = re.compile(r"^- (?!\[)(.+?)\s+\[NO DATE PROVIDED\]\s*$")
_TYPE_AFTER_NO_DATE = re.compile(r"^- \[NO DATE PROVIDED\] \[(?:EXACT|APPROX)\] ")
# Leading date bracket only, so dates inside the event text are never touched.
_LEADING_DATE = re.compile(rf"^- \[({_DATE_OR_RANGE})\](?= )")
_DOUBLE_SPACE = re.compile(r" {2,}")
_TRAILING_SPACE = re.compile(r"\s+$")

def is_canonical(line):
    return bool(_CANONICAL_DATELESS.match(line) or _CANONICAL_DATED.match(line))


def _sub(pattern, repl, s):
    """pattern.sub, counting only the matches that actually changed the text."""
    changed = 0

    def replace(m):
        nonlocal changed
        new = m.expand(repl) if isinstance(repl, str) else repl(m)
        changed += new != m.group(0)
        return new

    return pattern.sub(replace, s), changed


def _strip_bold(s):
    """Remove **bold** but leave de-identification tokens like [**Hospital1 18**] intact."""
    if "**" not in s:
        return s, 0
    tokens = []

    def mask(m):
        tokens.append(m.group(0))
        return f"\x00{len(tokens) - 1}\x00"

    s, n = _sub(_BOLD, r"\1", _DEID.sub(mask, s))
    if tokens:
        s = _DEID_MASK.sub(lambda m: tokens[int(m.group(1))], s)
    return s, n


def _pad(date):
    """'2175-3-1' -> '2175-03-01'; keeps the original precision."""
    parts = date.split("-")
    return "-".join([parts[0]] + [f"{int(p):02d}" for p in parts[1:3]])


def _normalize_date(m):
    sides = re.split(r"\s+to\s+", m.group(1))
    return f"- [{' to '.join(_pad(s.strip()) for s in sides)}]"


# (name, pattern, replacement), applied in this order. Bold uses _strip_bold.
_STEPS = [
    # Trailing whitespace first so end-of-line patterns match.
    ("strip_trailing_ws", _TRAILING_SPACE, ""),
    ("bullet_normalize", _BULLET, r"\1- "),
    ("bullet_restore", _MISSING_BULLET, r"\1- ["),
    ("strip_markdown_bold", None, None),
    ("strip_markdown_underline", _UNDERLINE, r"\1"),
    ("type_alias_exact", _ALIAS_EXACT, "[EXACT]"),
    ("type_alias_approx", _ALIAS_APPROX, "[APPROX]"),
    ("type_alias_no_date", _ALIAS_NO_DATE, "[NO DATE PROVIDED]"),
    ("combined_tag_comma", _COMBINED_COMMA, r"- [\1] [\2]"),
    ("combined_tag_typefirst", _COMBINED_TYPE_FIRST, r"- [\2] [\1]"),
    ("trailing_tag_swap", _TRAILING_TAG, r"- [\2] [\3] \1"),
    ("trailing_no_date", _TRAILING_NO_DATE, r"- [NO DATE PROVIDED] \1"),
    ("strip_type_after_no_date", _TYPE_AFTER_NO_DATE, "- [NO DATE PROVIDED] "),
    # Dates last, once every path has moved the date bracket to the front.
    ("date_normalize", _LEADING_DATE, _normalize_date),
    ("collapse_double_space", _DOUBLE_SPACE, " "),
]
REPAIR_NAMES = [name for name, _, _ in _STEPS]


def repair(line):
    """Repair one line. Returns (line, {repair_name: times_applied})."""
    counts = {}
    for name, pattern, repl in _STEPS:
        if pattern is None:
            line, counts[name] = _strip_bold(line)
        else:
            line, counts[name] = _sub(pattern, repl, line)
    return line, counts


def repair_text(text):
    """Repair every line; drop blank and non-bullet lines. Returns (text, meta)."""
    lines = (text or "").splitlines()
    totals = dict.fromkeys(REPAIR_NAMES, 0)
    out, blank, non_bullet = [], 0, 0
    for raw in lines:
        if not raw.strip():
            blank += 1
            continue
        line, counts = repair(raw)
        if not line.startswith("- "):
            non_bullet += 1
            continue
        for name, n in counts.items():
            totals[name] += n
        out.append(line)
    return "\n".join(out), {
        "lines_in": len(lines),
        "bullet_lines_in": len(out),
        "non_bullet_lines_dropped": non_bullet,
        "blank_lines_dropped": blank,
        "repairs_applied": totals,
    }
