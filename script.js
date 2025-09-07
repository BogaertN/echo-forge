const ws = new WebSocket('ws://127.0.0.1:8000/ws/session1');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  document.getElementById('output').innerText += JSON.stringify(data) + '\n';
};

async function runFlow() {
  const input = document.getElementById('input').value;
  const enableTools = document.getElementById('enableTools').checked;
  const response = await fetch('http://127.0.0.1:8000/run-flow', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_input: input, enable_tools: enableTools})
  });
  const result = await response.json();
  document.getElementById('output').innerText += result.synthesis + '\n';
}
