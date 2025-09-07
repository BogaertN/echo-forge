// EchoForge Debate JS: Handles debate UI logic, including initialization,
// round execution, replay, counterfactual injection, output formatting,
// failure/timeout handling, specialist integration, tool use notifications.

let currentRound = 0;
let transcript = [];
let config = null;

// Start Debate
function initDebate(debateConfig) {
    config = debateConfig;
    document.getElementById('debate-transcript').innerHTML = '';
    transcript = [];
    currentRound = 0;
    runNextRound();
}

// Run Rounds Sequentially
async function runNextRound() {
    if (currentRound >= config.rounds) {
        document.getElementById('synthesize').disabled = false;
        return;
    }
    
    currentRound++;
    try {
        const result = await apiCall(`/debate/round/${debateId}?round_num=${currentRound}`, 'POST');
        transcript.push(result);
        
        // Format and append to UI
        const roundHtml = formatDebateOutput(result);
        updateUI('debate-transcript', roundHtml, true);
        
        // Handle specialists and tools
        displaySpecialists(result);
        
        runNextRound();
    } catch (err) {
        handleDebateError(err);
    }
}

function formatDebateOutput(roundData) {
    let html = `<h3>Round ${roundData.round}</h3>`;
    for (const [role, arg] of Object.entries(roundData.arguments)) {
        html += `
            <div class="argument ${role}">
                <strong>${role.charAt(0).toUpperCase() + role.slice(1)}:</strong>
                <p>${arg.content}</p>
            </div>
        `;
    }
    return html;
}

function displaySpecialists(roundData) {
    const specialistsHtml = Object.entries(roundData.arguments)
        .filter(([role]) => role !== 'proponent' && role !== 'opponent')
        .map(([role, arg]) => `<div class="specialist ${role}">${role}: ${arg.content}</div>`)
        .join('');
    
    // Append to specific div if needed
    updateUI('specialists-section', specialistsHtml);  // Assume element
}

// Replay Debate
function replayDebate() {
    document.getElementById('debate-transcript').innerHTML = '';
    transcript.forEach(round => {
        updateUI('debate-transcript', formatDebateOutput(round), true);
    });
}

// Counterfactual Injection
function injectCounterfactual() {
    const injection = prompt('Enter counterfactual assumption:');
    if (injection) {
        apiCall('/debate/inject', 'POST', {debate_id: debateId, injection})  // Assume endpoint
            .then(res => {
                alert('Counterfactual injected. Rerunning debate.');
                initDebate(config);  // Rerun
            });
    }
}

// Handle Errors/Timeouts
function handleDebateError(err) {
    alert('Debate error: ' + err.message);
    // Retry logic
    if (confirm('Retry round?')) {
        runNextRound();
    }
}

// Synthesize
function synthesize() {
    const tone = document.getElementById('tone').value;
    apiCall('/synthesis/generate', 'POST', {session_id: sessionId, debate_id: debateId, tone})
        .then(res => {
            updateUI('synthesis', `<h3>Synthesis (${tone}):</h3><p>${res.synthesis}</p>`);
            document.getElementById('journal-synthesis').disabled = false;
        });
}

// Tool Use Notifications (from WS)
function handleToolNotification(msg) {
    if (msg.type === 'tool_notification') {
        const transcript = document.getElementById('debate-transcript');
        transcript.innerHTML += `<p class="tool-note">${msg.message}</p>`;
    }
}

// Lock/Replay Controls
document.getElementById('lock-debate').addEventListener('click', () => {
    // Send lock request
    apiCall('/debate/lock', 'POST', {debate_id: debateId});
});

document.getElementById('replay-debate').addEventListener('click', replayDebate);

document.getElementById('inject-counterfactual').addEventListener('click', injectCounterfactual);

// Integrate with main.js for WS handling
// Assume handleWebSocketMessage calls handleToolNotification if relevant
