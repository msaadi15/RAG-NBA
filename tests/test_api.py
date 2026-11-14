from fastapi.testclient import TestClient
from nba_rag.app import app, engine

# No lifespan: unit checks must never download models.
client = TestClient(app)


def test_home_and_status():
    assert client.get('/').status_code == 200
    state = client.get('/api/status').json()
    assert 'model' in state and 'ready' in state


def test_question_limits():
    for question in ['', '  ', 'x' * 601]:
        assert client.post('/api/chat', json={'question': question}).status_code == 422


def test_not_ready_has_actionable_status(monkeypatch):
    monkeypatch.setattr(engine, 'ready', False)
    assert client.post('/api/chat', json={'question': 'Who won MVP?'}).status_code == 503


def test_chat_returns_engine_evidence(monkeypatch):
    monkeypatch.setattr(engine, 'ready', True)
    monkeypatch.setattr(engine, 'ask', lambda q: {'answer': 'Nikola Jokić [1]', 'sources': [{'citation': 1}], 'seconds': 0.1})
    response = client.post('/api/chat', json={'question': 'Who won MVP in 2024?'})
    assert response.status_code == 200
    assert response.json()['sources'][0]['citation'] == 1
