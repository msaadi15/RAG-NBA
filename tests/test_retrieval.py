"""Regression checks for wrong season and MVP category selection; no downloads."""
import numpy as np
from nba_rag.engine import NBAEngine
from nba_rag.ingest import chunk_document


def fixture_engine():
    e = NBAEngine()
    docs = [
        ('Regular-season MVP winners', '2022-23 — Joel Embiid\n2023-24 — Nikola Jokić'),
        ('NBA Finals MVP winners', '2022-23 — Nikola Jokic\n2023-24 — Jaylen Brown'),
        ('NBA champions', '2022-23 — Denver Nuggets\n2023-24 — Boston Celtics')
    ]
    e.chunks = [c for title, text in docs for c in chunk_document(dict(title=title, text=text, url='https://www.nba.com/'+title, fetched_at='today'))]
    e.vectors = np.ones((len(e.chunks), 1))
    e.embed = lambda texts: np.ones((len(texts), 1))
    return e


def test_regular_mvp_matches_exact_season():
    result = fixture_engine().retrieve('Who won regular-season MVP in 2023-24?')
    assert len(result) == 1
    assert 'Nikola Jokić' in result[0]['text']
    assert result[0]['title'] == 'Regular-season MVP winners'


def test_calendar_year_means_season_end_and_preserves_both_categories():
    result = fixture_engine().retrieve('Who won the 2023 championship and Finals MVP?')
    assert {p['title'] for p in result} == {'NBA champions', 'NBA Finals MVP winners'}
    assert all('2022-23' in p['text'] for p in result)


def test_unknown_season_has_no_evidence():
    assert fixture_engine().retrieve('Who won MVP in 2099?') == []


def test_slash_season_is_unambiguous():
    result = fixture_engine().retrieve('Who won MVP in 2023/24?')
    assert len(result) == 1 and 'Nikola Jokić' in result[0]['text']


def test_player_query_excludes_other_winners():
    e = fixture_engine()
    result = e.retrieve('Which seasons did Nikola Jokić win regular-season MVP?')
    assert len(result) == 1
    assert 'Nikola Jokić' in result[0]['text']
    assert 'Joel Embiid' not in result[0]['text']


def test_incomplete_generated_lookup_uses_original_records():
    from nba_rag.engine import verified_record_answer
    records = fixture_engine().retrieve('Who won the 2023 championship and Finals MVP?')
    answer, mode = verified_record_answer('Nikola Jokic won the championship.', records)
    assert mode == 'verified_records'
    assert 'Denver Nuggets' in answer and 'Nikola Jokic' in answer
    assert '[1]' in answer and '[2]' in answer


def test_complete_cited_answer_is_preserved():
    from nba_rag.engine import verified_record_answer
    records = fixture_engine().retrieve('Who won regular-season MVP in 2023-24?')
    original = 'Nikola Jokić won regular-season MVP in 2023-24 [1].'
    assert verified_record_answer(original, records) == (original, 'generated')
