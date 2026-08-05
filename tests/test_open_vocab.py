from limo4si.open_vocab import Candidate, parse_referent, resolve_candidate


def candidate(index: int, x: float, y: float) -> Candidate:
    return Candidate(index, "cup", 0.9, [x - 5, y - 5, x + 5, y + 5], [x, y])


def test_chinese_shelf_reference():
    query = parse_referent("第四层架子上的右数第五个杯子")
    assert query.target == "cup"
    assert query.support == "shelf"
    assert query.level_from_top == 4
    assert query.order == "right_to_left"
    assert query.ordinal == 5


def test_resolve_fourth_row_fifth_from_right():
    rows = []
    index = 0
    for y in (20, 60, 100, 140):
        for x in (10, 30, 50, 70, 90, 110):
            rows.append(candidate(index, x, y))
            index += 1
    selected, detail = resolve_candidate(
        rows, parse_referent("第四层架子上的右数第五个杯子")
    )
    assert detail["status"] == "resolved"
    assert selected is not None
    assert selected.center_xy == [30, 140]


def test_ambiguous_reference_is_not_guessed():
    selected, detail = resolve_candidate(
        [candidate(0, 10, 20), candidate(1, 30, 20)],
        parse_referent("杯子"),
    )
    assert selected is None
    assert detail["status"] == "needs_confirmation"


def test_longer_chinese_noun_wins():
    assert parse_referent("蚝油瓶").target == "oyster sauce bottle"
    assert parse_referent("油瓶").target == "cooking oil bottle"
