// EchoForge Main JS: Core UI Logic
// Handles onboarding, core flows, event listeners, API interactions,
// "Show Reasoning" toggle, help/documentation, accessibility.

// Global variables
let sessionId = null;
let clarificationHistory = [];
let debateId = null;
let ws = null;  // WebSocket instance

// Utility Functions
function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {'Content-Type': 'application/json'}
    };
    if (data) options.body = JSON.stringify(data);
    
    return fetch(`/api${endpoint}`, options)
        .then(res => {
            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            return res.json();
        })
        .catch(err => {
            console.error(err);
            alert('API error: ' + err.message);
        });
}

function updateUI(elementId, content, append = false) {
    const el = document.getElementById(elementId);
    if (append) {
        el.innerHTML += content;
    } else {
        el.innerHTML = content;
    }
    // Accessibility: Announce live updates
    el.setAttribute('aria-live', 'polite');
}

// Onboarding and Help
function showHelp() {
    alert(`EchoForge Help:
- Clarify: Refine your question via Socratic dialogue.
- Debate: Configure and run multi-agent debate.
- Journal: Save insights, search entries.
- Resonance: View knowledge graph.
- Dashboard: Track progress.
For details, see manual.`);
}

// Core Flows
function startClarification(e) {
    e.preventDefault();
    const question = document.getElementById('initial-question').value;
    console.log('Starting clarification with question:', question);  // Debug: Confirm button click

    apiCall('/session/start', 'POST', {initial_question: question})
        .then(res => {
            console.log('Session start response:', res);  // Debug: Check session response
            sessionId = res.session_id;
            return apiCall('/clarification/start', 'POST', {initial_question: question});
        })
        .then(res => {
            console.log('Clarification start response:', res);  // Debug: Check clarification response
            updateUI('clarification-dialogue', `<p>Clarifier: ${res.clarifier_question}</p>`, true);
            clarificationHistory.push({question: res.clarifier_question});
        })
        .catch(err => {
            console.error('Clarification error:', err);  // Debug: Catch and log errors
            alert('Error starting clarification: ' + err.message);
        });
}

function continueClarification(response) {
    apiCall('/clarification/continue', 'POST', {
        session_id: sessionId,
        user_response: response,
        conversation_history: clarificationHistory
    })
        .then(res => {
            if (res.status === 'complete') {
                updateUI('clarified-prompt', `Clarified: ${res.clarified_prompt}`);
                document.getElementById('debate').hidden = false;
                document.getElementById('complete-clarification').disabled = true;
            } else {
                updateUI('clarification-dialogue', `<p>Clarifier: ${res.next_question}</p>`, true);
                clarificationHistory.push({question: res.next_question});
            }
        });
}

function startDebate(e) {
    e.preventDefault();
    const config = {
        rounds: parseInt(document.getElementById('rounds').value),
        tone: document.getElementById('tone').value,
        specialists: Array.from(document.querySelectorAll('#debate-config input[type="checkbox"]:checked')).map(cb => cb.value),
        enable_tools: document.getElementById('enable-tools').checked
    };
    
    apiCall('/debate/start', 'POST', {session_id: sessionId, config})
        .then(res => {
            debateId = res.debate_id;
            runDebateRounds(config.rounds);
        });
}

async function runDebateRounds(rounds) {
    for (let round = 1; round <= rounds; round++) {
        const result = await apiCall(`/debate/round/${debateId}?round_num=${round}`, 'POST');
        const argsHtml = Object.entries(result.responses).map(([role, resp]) => 
            `<div class="argument"><strong>${role}:</strong> ${resp.content}</div>`
        ).join('');
        updateUI('debate-transcript', `<h3>Round ${round}</h3>${argsHtml}`, true);
    }
    document.getElementById('synthesize').disabled = false;
}

function synthesizeDebate() {
    const tone = document.getElementById('tone').value;
    apiCall('/synthesis/generate', 'POST', {session_id: sessionId, debate_id: debateId, tone})
        .then(res => {
            updateUI('synthesis', res.synthesis);
            document.getElementById('journal-synthesis').disabled = false;
        });
}

function journalSynthesis() {
    const content = document.getElementById('synthesis').innerText;
    apiCall('/journal/create', 'POST', {session_id: sessionId, content})
        .then(res => alert('Journal entry created: ' + res.entry_id));
}

// Journal Search and Display
function searchJournal() {
    const query = document.getElementById('journal-search').value;
    apiCall(`/journal/search?query=${encodeURIComponent(query)}`)
        .then(res => {
            const list = res.results.map(entry => 
                `<li><strong>${entry.title}</strong><p>${entry.summary}</p><tags>${entry.tags.join(', ')}</tags></li>`
            ).join('');
            updateUI('entry-list', list);
        });
}

// Resonance Map (Placeholder for graph lib, e.g., Cytoscape.js)
function loadResonanceMap() {
    // Fetch map data
    apiCall('/resonance/map')  // Assume endpoint
        .then(graph => {
            // Render with canvas or lib
            const ctx = document.getElementById('map-canvas').getContext('2d');
            // Simple render placeholder
            ctx.fillText('Resonance Map (Nodes: ' + Object.keys(graph.nodes).length + ')', 10, 50);
        });
}

// Dashboard
function loadDashboard() {
    apiCall('/gamification/stats')
        .then(stats => {
            const html = `
                <div class="stat-card">Streak: ${stats.streak_count}</div>
                <div class="stat-card">Badges: ${stats.badges.join(', ')}</div>
                <div class="stat-card">Avg Clarity: ${stats.clarity_metrics.average}</div>
            `;
            updateUI('gamification-stats', html);
        });
}

// WebSocket Setup (from websocket.js)
function initWebSocket() {
    ws = new WebSocket(`ws://localhost:8000/ws/${sessionId || 'default'}`);
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleWebSocketMessage(msg);
    };
}

function handleWebSocketMessage(msg) {
    switch (msg.type) {
        case 'clarification_continued':
            // Update dialogue
            break;
        case 'debate_round_complete':
            // Append to transcript
            break;
        case 'journal_entry_created':
            alert('New journal entry!');
            break;
        // Handle other real-time updates
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    setupLogging('DEBUG');  // If needed
    initWebSocket();
    
    document.getElementById('clarification-form').addEventListener('submit', startClarification);
    document.getElementById('debate-config').addEventListener('submit', startDebate);
    document.getElementById('synthesize').addEventListener('click', synthesizeDebate);
    document.getElementById('journal-synthesis').addEventListener('click', journalSynthesis);
    document.getElementById('journal-search').addEventListener('input', searchJournal);
    document.getElementById('journal-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const content = document.getElementById('journal-content').value;
        apiCall('/journal/create', 'POST', {session_id: sessionId, content});
    });
    
    // Toggle reasoning
    document.getElementById('show-reasoning').addEventListener('change', (e) => {
        document.querySelectorAll('.reasoning').forEach(el => {
            el.style.display = e.target.checked ? 'block' : 'none';
        });
    });
    
    // Load initial data
    loadDashboard();
    loadResonanceMap();
});

// Help link
document.querySelector('a[href="#help"]').addEventListener('click', showHelp);
