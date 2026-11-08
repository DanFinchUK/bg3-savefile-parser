"""Ability attribution from synthetic LSMF rows, without player save data."""

import struct
from pathlib import Path

import pytest

from bg3parser.lsmf import parse_lsmf_ability_scores_by_template

STATS = 1024
TEMPLATES = 1664
STRINGS = 2304
STATS_OWNERS = 3328
TEMPLATE_OWNERS = 3584


def make_blob(count=14):
    blob = bytearray(4608)
    names = ['game.stats.v3.StatsComponent', 'game.templates.v0.TemplateComponent']
    struct.pack_into('<QQQ', blob, 8, 0, 128 - 48, 512)
    struct.pack_into('<IH', blob, 32, 256, 2)
    name_offset = 0
    for ci, (name, size, offset, owner_start) in enumerate(
        zip(names, (36, 24), (STATS, TEMPLATES), (STATS_OWNERS, TEMPLATE_OWNERS), strict=True)
    ):
        encoded = name.encode('ascii')
        blob[128 + name_offset : 128 + name_offset + len(encoded)] = encoded
        struct.pack_into(
            '<QQQIIQQ',
            blob,
            384 + ci * 48,
            name_offset,
            len(encoded),
            0,
            size,
            0,
            count,
            offset - 48,
        )
        struct.pack_into(
            '<QQQQ',
            blob,
            4096 + ci * 32,
            owner_start - 48,
            owner_start - 48 + count * 4,
            ci,
            count,
        )
        # These preceding words must never be mistaken for owners.
        struct.pack_into('<12I', blob, owner_start - 48, *range(1000, 1012))
        name_offset += len(encoded)

    expected = {}
    for i in range(count):
        template = f'{i + 1:08x}-0000-0000-0000-000000000000'
        scores = (8 + i, 11, 12, 13, 14, 15)
        expected[template] = scores
        struct.pack_into('<Q7i', blob, STATS + i * 36, 0x1234, *scores, 2)
        struct.pack_into('<I', blob, STATS_OWNERS + i * 4, 100 + i)
        # Template rows are in the reverse order; join by owner, not row index.
        k = count - 1 - i
        struct.pack_into('<QI', blob, TEMPLATES + k * 24, STRINGS + i * 36 - 48, 36)
        struct.pack_into('<I', blob, TEMPLATE_OWNERS + k * 4, 100 + i)
        blob[STRINGS + i * 36 : STRINGS + (i + 1) * 36] = template.encode('ascii')
    return blob, expected


@pytest.mark.parametrize('count', [3, 14])
def test_ability_scores_use_relative_pointer_base_and_exact_owners(count):
    blob, expected = make_blob(count)
    assert parse_lsmf_ability_scores_by_template(bytes(blob)) == expected


def test_conflicting_shared_template_is_not_assigned():
    blob, expected = make_blob(3)
    first, second, third = expected
    blob[STRINGS + 36 : STRINGS + 72] = first.encode('ascii')
    result = parse_lsmf_ability_scores_by_template(bytes(blob))
    assert result == {third: expected[third]}
    assert first not in result and second not in result


def test_invalid_ability_row_does_not_shift_other_characters():
    blob, expected = make_blob(3)
    struct.pack_into('<i', blob, STATS + 36 + 8, -1)
    del expected[list(expected)[1]]
    assert parse_lsmf_ability_scores_by_template(bytes(blob)) == expected


def test_invalid_template_pointer_does_not_shift_other_characters():
    blob, expected = make_blob(3)
    struct.pack_into('<Q', blob, TEMPLATES, len(blob) + 100)
    del expected[list(expected)[-1]]
    assert parse_lsmf_ability_scores_by_template(bytes(blob)) == expected


def test_missing_template_component():
    blob, _ = make_blob(3)
    struct.pack_into('<H', blob, 36, 1)
    assert parse_lsmf_ability_scores_by_template(bytes(blob)) == {}


def test_truncated_blob():
    blob, _ = make_blob(3)
    assert parse_lsmf_ability_scores_by_template(bytes(blob[:128])) == {}


def test_report_reads_abilities_without_a_spellbook(monkeypatch):
    from bg3parser import model

    monkeypatch.setattr(model, 'parse_lsmf_spellbooks', lambda _blob: {})
    fixture = Path(__file__).parent / 'fixtures' / 'autosave_shadowheart_tutorial.lsv'
    report = model.gather_report(str(fixture))
    character = next(c for c in report.characters if c.name.startswith('Shadowheart'))
    assert character.abilities == {'str': 13, 'dex': 13, 'con': 14, 'int': 10, 'wis': 17, 'cha': 8}
