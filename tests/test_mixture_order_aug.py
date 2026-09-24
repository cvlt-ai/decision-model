"""Order-augmentation (s1-v2): the mixture must teach P(key | option-SET),
not P(key | option-LIST-ORDER). Each choice row emits a shuffled twin with the
SAME gold key but a DIFFERENT option order. This is the direct countermeasure to
the measured permutation gap (mmlu_pro 0.475 flip in s1-v1).

Kept cheap: builders run with a small cap; we assert structural invariants,
not dataset content.
"""

from s1 import mixture


def _option_block(prompt: str) -> list[str]:
    """Return the ordered option lines ([i] key ...) of a rendered prompt."""
    lines = prompt.splitlines()
    i = lines.index("Options:")
    out = []
    for ln in lines[i + 1:]:
        if ln.startswith("["):
            out.append(ln)
        elif ln == "" or ln == "Answer:":
            break
    return out


def _option_contents(lines: list[str]) -> list[str]:
    """Strip the '[i] ' index prefix -> the option text the model sees.

    The index is positional (changes on reorder); the content is what must be
    invariant under a permutation.
    """
    return [ln.split("] ", 1)[1] if "] " in ln else ln for ln in lines]


def test_banking77_order_aug_doubles_rows_and_keeps_gold():
    base = mixture.build_banking77(cap=40)
    aug = mixture.build_banking77(cap=40, order_aug=True)
    # canonical + one shuffled twin per source row
    assert len(aug) == 2 * len(base)
    by_uid = {r["uid"]: r for r in aug}
    for b in base:
        twin = by_uid[b["uid"] + "/s"]
        assert twin["family"] == "banking77"
        # SAME gold key, regardless of option order
        assert twin["target"] == b["target"]


def test_banking77_shuffled_twin_is_a_permutation_not_the_identity():
    aug = mixture.build_banking77(cap=20, order_aug=True)
    by_uid = {r["uid"]: r for r in aug}
    saw_difference = 0
    checked = 0
    for uid, base in by_uid.items():
        if not uid.endswith("/s"):
            continue
        checked += 1
        canon = by_uid[uid[: -len("/s")]]
        cb, tb = _option_block(canon["prompt"]), _option_block(base["prompt"])
        assert len(cb) == len(tb) == 77
        # same SET of option texts (keys travel with order), different index order
        assert sorted(_option_contents(cb)) == sorted(_option_contents(tb))
        if cb != tb:  # genuinely reordered
            saw_difference += 1
    assert checked == 20
    # 77! possible orders across 20 seeded rows: at least one must actually differ
    assert saw_difference > 0


def test_shuffled_options_deterministic_by_uid():
    o1 = mixture._shuffled_options("banking77/5", [(k, k) for k in "abcdefgh"])
    o2 = mixture._shuffled_options("banking77/5", [(k, k) for k in "abcdefgh"])
    o3 = mixture._shuffled_options("banking77/6", [(k, k) for k in "abcdefgh"])
    assert [k for k, _ in o1] == [k for k, _ in o2]  # same uid -> same order
    # each is a permutation of the input keys
    assert sorted(k for k, _ in o1) == sorted(k for k, _ in o3) == list("abcdefgh")


def test_noul_families_not_order_augmented():
    # injection is noul (2 options): order_aug must NOT double it
    base = mixture.build_injection(cap=30)
    aug = mixture.build_injection(cap=30, order_aug=True)
    assert len(aug) == len(base)


def test_severity_order_aug_preserves_gold_and_permutes():
    base = mixture.build_severity(cap=30)
    aug = mixture.build_severity(cap=30, order_aug=True)
    by_uid = {r["uid"]: r for r in aug}
    for b in base:
        twin = by_uid[b["uid"] + "/s"]
        assert twin["target"] == b["target"]
        cb, tb = _option_block(b["prompt"]), _option_block(twin["prompt"])
        assert len(cb) == len(tb) == 4
        assert sorted(_option_contents(cb)) == sorted(_option_contents(tb))  # same 4 levels