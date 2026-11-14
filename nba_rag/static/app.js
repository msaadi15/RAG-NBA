const $ = s => document.querySelector(s);
let ready = false, busy = false, poll;
const initial = $('#conversation').innerHTML;
function element(tag, className, text) {
  const node = document.createElement(tag); node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function updateSend() { $('#send').disabled = !ready || busy; }
async function status() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Server unavailable');
    const state = await response.json(); ready = state.ready;
    $('#status').textContent = state.error || (ready ? `● ${state.model.split('/')[1]} · ${state.chunks} passages · running locally` : state.phase + '…');
    $('#doc-count').textContent = state.documents.length;
    $('#documents').replaceChildren(...state.documents.map(doc => {
      const link = element('a', 'document', '↗  ' + doc.title);
      link.href = doc.url; link.target = '_blank'; link.rel = 'noreferrer';
      link.append(element('small', '', 'NBA.COM · fetched ' + new Date(doc.fetched_at).toLocaleDateString()));
      return link;
    }));
    if (!state.documents.length) $('#documents').textContent = 'Run setup to download the official NBA archive.';
    if (ready || state.error) clearInterval(poll);
  } catch (err) { ready = false; $('#status').textContent = 'Cannot reach the local server. Check that it is running.'; }
  updateSend();
}
async function ask(question) {
  question = question.trim();
  if (!question || busy) return;
  if (!ready) { $('#status').textContent = 'Models are not ready yet. Run setup and wait for loading to finish.'; return; }
  if (question.length < 3) { $('#status').textContent = 'Please enter at least 3 characters.'; return; }
  busy = true; updateSend(); $('#question').value = '';
  if ($('.suggestions')) $('#conversation').replaceChildren();
  const turn = element('div', 'turn');
  const user = element('div', 'user-message'); user.append(element('span', '', question));
  const label = element('div', 'answer-label', '◉ CourtSide');
  const answer = element('div', 'answer', 'Searching the archive and reading the evidence…');
  turn.append(user, label, answer); $('#conversation').append(turn);
  turn.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  try {
    const response = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question})});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The request could not be processed.');
    answer.textContent = result.answer;
    const mode = result.answer_mode === 'verified_records' ? 'Verified archive records' : 'Local model answer';
    turn.append(element('div', 'answer-meta', `${mode} · ${result.sources.length} retrieved passages · ${result.seconds}s · ${result.model.split('/')[1]}`));
    for (const source of result.sources) {
      const detail = element('details', 'evidence');
      detail.append(element('summary', '', `[${source.citation}] ${source.title} · similarity ${source.similarity}`));
      detail.append(element('p', '', source.text));
      const link = element('a', '', 'Open official NBA source ↗'); link.href = source.url; link.target = '_blank'; link.rel = 'noreferrer';
      detail.append(link, element('p', '', 'Snapshot downloaded ' + new Date(source.fetched_at).toLocaleString()));
      turn.append(detail);
    }
  } catch (err) { answer.textContent = err.message; answer.classList.add('error'); }
  finally { busy = false; updateSend(); $('#question').focus(); }
}
$('#chat-form').addEventListener('submit', e => { e.preventDefault(); ask($('#question').value); });
$('#question').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(e.target.value); } });
document.addEventListener('click', e => { const button = e.target.closest('[data-question]'); if (button) { $('#question').value = button.dataset.question; ask(button.dataset.question); } });
$('#new-chat').addEventListener('click', () => { if (!busy) { $('#conversation').innerHTML = initial; $('#question').value = ''; $('#question').focus(); } });
poll = setInterval(status, 3000); status();
