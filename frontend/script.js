const ws = new WebSocket('ws://127.0.0.1:8000/ws/session1');

// WebSocket message handler for real-time updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  const output = document.getElementById('output');
  output.innerHTML += getFormattedOutput(data);  // Use advanced formatting function
  output.scrollTop = output.scrollHeight;  // Auto-scroll to bottom
};

// Advanced formatting for WS messages
function getFormattedOutput(data) {
  let html = '';
  switch (data.type) {
    case 'clarified':
      html = `<h3>Clarified Question:</h3><p>${data.content}</p>`;
      break;
    case 'pro_arg':
      html = `<h3>Round ${data.round} - Proponent:</h3><p>${data.content}</p>`;
      break;
    case 'opp_arg':
      html = `<h3>Round ${data.round} - Opponent:</h3><p>${data.content}</p>`;
      break;
    case 'synthesis':
      html = `<h3>Synthesis:</h3><p>${data.content}</p>`;
      break;
    case 'tools_result':
      html = `<h3>Additional Info from Tools:</h3><p>${data.content}</p>`;
      break;
    case 'journal_entry':
      html = `<h3>Journal Entry Created (ID: ${data.id}):</h3><p>${data.content}</p>`;
      break;
    case 'voice_transcription':
      html = `<h3>Voice Transcription:</h3><p>${data.content}</p>`;
      // Auto-set input to transcription for convenience
      document.getElementById('input').value = data.content;
      break;
    case 'resonance_map_updated':
      loadResonanceMap();  // Refresh map on update
      html = `<p>Resonance Map Updated</p>`;
      break;
    case 'flow_complete':
      html = `<p><strong>Flow Complete</strong></p>`;
      updateGamification();  // Refresh gamification on completion
      break;
    case 'error':
      html = `<p><strong>Error:</strong> ${data.content}</p>`;
      break;
    default:
      html = `<p>${JSON.stringify(data)}</p>`;
  }
  return html + '<hr>';  // Add separator for readability
}

// Run full flow with user input
async function runFlow() {
  const input = document.getElementById('input').value;
  const enableTools = document.getElementById('enableTools').checked;
  const rounds = parseInt(document.getElementById('rounds').value);
  try {
    const response = await fetch('http://127.0.0.1:8000/run-flow', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_input: input, enable_tools: enableTools, rounds: rounds, session_id: 'session1'})
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    // Updates come via WS; no need to handle here
  } catch (error) {
    document.getElementById('output').innerHTML += `<p><strong>Error:</strong> ${error.message}</p>`;
  }
}

// Upload and transcribe voice
async function uploadVoice() {
  const file = document.getElementById('voiceFile').files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', 'session1');
  try {
    const response = await fetch('http://127.0.0.1:8000/upload-voice/', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    // Transcription sent via WS
  } catch (error) {
    alert('Voice upload failed: ' + error.message);
  }
}

// Search journal entries
async function searchJournal() {
  const query = document.getElementById('journalQuery').value;
  try {
    const response = await fetch(`http://127.0.0.1:8000/journal/search?query=${encodeURIComponent(query)}`);
    const entries = await response.json();
    const results = document.getElementById('journalResults');
    results.innerHTML = entries.map(entry => `<p>ID: ${entry[0]} - ${entry[2]}</p>`).join('');
  } catch (error) {
    console.error('Journal search failed:', error);
  }
}

// Load full journal (advanced: paginated or filtered)
async function loadJournal() {
  searchJournal();  // Reuse search with empty query for all
}

// Update gamification display (fetch weekly report)
async function updateGamification() {
  // Assume /gamification endpoint added; placeholder fetch
  // const response = await fetch('http://127.0.0.1:8000/gamification/report');
  // const report = await response.text();
  // document.getElementById('gamification').innerText = report;
  document.getElementById('gamification').innerText = 'Streak: 5 days | Weekly Entries: 3';  // Mock
}

// Load and visualize resonance map with D3.js
async function loadResonanceMap() {
  try {
    const response = await fetch('/resonance_map.json');  // Assume exported JSON endpoint or file
    const data = await response.json();
    const width = 800, height = 600;
    const svg = d3.select('#map-container').html('').append('svg')
      .attr('width', width)
      .attr('height', height);

    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id))
      .force('charge', d3.forceManyBody().strength(-50))
      .force('center', d3.forceCenter(width / 2, height / 2));

    const link = svg.append('g').selectAll('line')
      .data(data.links)
      .enter().append('line')
      .attr('stroke-width', d => Math.sqrt(d.value || 1))
      .attr('stroke', '#999');

    const node = svg.append('g').selectAll('circle')
      .data(data.nodes)
      .enter().append('circle')
      .attr('r', 5)
      .attr('fill', d => d.group === 1 ? '#ff0000' : '#00ff00')  // Color by type
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    node.append('title').text(d => d.id);

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  } catch (error) {
    console.error('Failed to load resonance map:', error);
  }
}

// Initial loads
window.onload = () => {
  updateGamification();
  loadResonanceMap();
};
