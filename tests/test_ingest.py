import pytest
from nba_rag.ingest import chunk_document, extract_article


def test_extraction_preserves_rows_and_excludes_navigation():
    rows = ''.join(f'<p>20{i:02d}-{i+1:02d}: Player {i}, Team {i}</p>' for i in range(20))
    html = f'<nav>WRONG WINNER</nav><article><h1>MVP winners</h1>{rows}<script>BAD</script></article>'
    result = extract_article(html)
    assert '2000-01: Player 0' in result
    assert 'WRONG WINNER' not in result
    assert 'BAD' not in result


def test_missing_body_rejected():
    with pytest.raises(ValueError):
        extract_article('<html><nav>Access denied</nav></html>')


def test_chunks_keep_provenance_and_overlap():
    doc = dict(title='MVP', url='https://www.nba.com/test', fetched_at='2026-01-01',
               text='\n'.join(f'Year {i}: winner player {i}' for i in range(20)))
    chunks = chunk_document(doc, max_chars=100)
    assert len(chunks) > 1
    assert all(c['url'] == doc['url'] and c['text'].startswith('MVP\n') for c in chunks)
    assert len({c['id'] for c in chunks}) == len(chunks)
    assert chunks[0]['text'].splitlines()[-1] in chunks[1]['text']
    for line in doc['text'].splitlines():
        assert any(line in c['text'] for c in chunks)


def test_failed_refresh_preserves_existing_corpus(tmp_path, monkeypatch):
    from nba_rag import ingest as module
    import requests
    destination = tmp_path / 'documents.json'
    destination.write_text('[{"existing": true}]')
    monkeypatch.setattr(module, 'DATA', tmp_path)
    monkeypatch.setattr(module, 'SOURCES', [('MVP', 'https://www.nba.com/test')])
    def fail(*args, **kwargs):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(module.requests, 'get', fail)
    with pytest.raises(RuntimeError, match='existing documents kept'):
        module.ingest()
    assert destination.read_text() == '[{"existing": true}]'
