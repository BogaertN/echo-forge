// EchoForge WebSocket JS: Handles real-time connections, session-scoped updates,
// debate round streaming, journal notifications, error handling, ping/pong for keep-alive.

let ws = null;
let reconnectInterval = 1000 * 5;  // 5 seconds
let sessionId = localStorage.getItem('sessionId') || 'default';  // Persist session

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        sendPing();  // Start keep-alive
    };
    
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(initWebSocket, reconnectInterval);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function sendMessage(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    } else {
        console.warn('WebSocket not open');
    }
}

function sendPing() {
    sendMessage({type: 'ping'});
    setTimeout(sendPing, 30000);  // Every 30s
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'session_started':
            sessionId = msg.session_id;
            localStorage.setItem('sessionId', sessionId);
            break;
            
        case 'clarification_started':
        case 'clarification_continued':
            // Update clarification dialogue
            const dialogue = document.getElementById('clarification-dialogue');
            dialogue.innerHTML += `<p>Clarifier: ${msg.clarifier_question || msg.data.next_question}</p>`;
            if (msg.status === 'complete') {
                document.getElementById('clarified-prompt').textContent = msg.data.clarified_prompt;
                document.getElementById('debate').hidden = false;
            }
            break;
            
        case 'debate_started':
            debateId = msg.debate_id;
            // Show progress bar or something
            break;
            
        case 'agent_thinking':
            // Show loading for agent
            const transcript = document.getElementById('debate-transcript');
            transcript.innerHTML += `<p>${msg.agent} is thinking...</p>`;
            break;
            
        case 'debate_round_update':
            const roundHtml = `
                <h3>Round ${msg.round}</h3>
                ${Object.entries(msg.data.arguments).map(([role, arg]) => 
                    `<div><strong>${role}:</strong> ${arg.content}</div>`
                ).join('')}
            `;
            document.getElementById('debate-transcript').innerHTML += roundHtml;
            // Update progress
            const progress = (msg.progress.current_round / msg.progress.total_rounds) * 100;
            // Assume progress bar element
            break;
            
        case 'debate_completed':
            document.getElementById('synthesize').disabled = false;
            break;
            
        case 'synthesis_generated':
            document.getElementById('synthesis').textContent = msg.synthesis;
            document.getElementById('journal-synthesis').disabled = false;
            break;
            
        case 'journal_entry_created':
            alert('Journal entry created!');
            searchJournal();  // Refresh list
            break;
            
        case 'voice_transcribed':
            document.getElementById('journal-content').value = msg.transcription;
            break;
            
        case 'system_notification':
            alert(`Notification: ${msg.message}`);
            break;
            
        case 'pong':
            // Keep-alive response
            break;
            
        default:
            console.warn('Unknown message type:', msg.type);
    }
}

// Reconnect on load if needed
document.addEventListener('DOMContentLoaded', initWebSocket);
