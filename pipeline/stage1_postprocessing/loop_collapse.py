"""Loop collapse: undo a model repeating the same block of bullets at the end.

When a model runs into its token limit it often starts cycling through the same
lines until it is cut off. Only that repeating tail is collapsed to one copy;
repeats anywhere else are real and are kept.

The same file is used by stage1_postprocessing and stage2_postprocessing.
"""

import re

BULLETS = ("- ", "* ", "• ", "· ")
_SPACES = re.compile(r"\s+")


def bullet_key(line):
    """Bullet text used to compare lines, or None if the line is not a bullet."""
    s = line.strip()
    for bullet in BULLETS:
        if s.startswith(bullet):
            return _SPACES.sub(" ", s[len(bullet):])
    return None


def lines_to_keep(keys):
    """How many leading lines to keep once the repeating tail is collapsed.

    For each period k, walk back from the end while keys[j] == keys[j + k]. A run
    covering at least two full periods is a loop. Collapse the loop that removes
    the most lines (ties: the shortest period), keeping one copy of the block.
    """
    n = len(keys)
    best = None  # (lines_removed, -k, loop_start)
    for k in range(1, n // 2 + 1):
        s = n - k
        while s > 0 and keys[s - 1] == keys[s - 1 + k]:
            s -= 1
        if n - s >= 2 * k:
            candidate = (n - s - k, -k, s)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return n
    _, neg_k, s = best
    return s - neg_k


def collapse_loops(text):
    """Drop blank and non-bullet lines, then collapse a looping tail.

    Returns (clean_text, meta).
    """
    lines = (text or "").splitlines()
    bullets, keys = [], []
    blank = non_bullet = 0
    for line in lines:
        if not line.strip():
            blank += 1
            continue
        key = bullet_key(line)
        if key is None:
            non_bullet += 1
            continue
        bullets.append(line.rstrip())
        keys.append(key)

    keep = lines_to_keep(keys)
    return "\n".join(bullets[:keep]), {
        "lines_in": len(lines),
        "bullet_lines_in": len(bullets),
        "lines_out": keep,
        "loop_lines_removed": len(bullets) - keep,
        "non_bullet_lines_dropped": non_bullet,
        "blank_lines_dropped": blank,
    }


def count_bullets(text):
    return sum(line.lstrip().startswith(BULLETS) for line in (text or "").splitlines())
