// EchoForge Journal JS: Handles journal UI logic, entry creation, search/filter/retrieval,
// voice upload/transcription, rephrasing, metadata display, ghost loop management,
// auto-suggestions, streaks/badges updates, export options.

function initJournal() {
    loadJournalEntries();
    document.getElementById('journal-form').addEventListener('submit', createJournalEntry);
    document.getElementById('voice-upload').addEventListener('change', handleVoiceUpload);
}

// Create Entry
function createJournalEntry(e) {
    e.preventDefault();
    const content = document.getElementById('journal-content').value;
    const edits = '';  // From user input if needed
    
    apiCall('/journal/create', 'POST', {
        session_id: sessionId,
        content: content,
        user_edits: edits
    })
        .then(res => {
            alert('Entry created: ' + res.entry_id);
            loadJournalEntries();
            updateGamification();
        });
}

// Voice Transcription
function handleVoiceUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    
    fetch('/api/voice/transcribe', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(res => {
            document.getElementById('journal-content').value = res.transcription;
            // Auto-rephrase if configured
            if (confirm('Rephrase transcription?')) {
                rephraseContent(res.transcription);
            }
        })
        .catch(err => console.error(err));
}

function rephraseContent(content) {
    // Call agent via API (assume endpoint)
    apiCall('/journal/rephrase', 'POST', {content})
        .then(res => {
            document.getElementById('journal-content').value = res.rephrased;
        });
}

// Load/Search Entries
function loadJournalEntries(query = '') {
    apiCall(`/journal/search?query=${encodeURIComponent(query)}&limit=20`)
        .then(res => {
            const list = res.results.map(entry => formatEntry(entry)).join('');
            updateUI('entry-list', list);
        });
}

function formatEntry(entry) {
    const tags = entry.tags.join(', ');
    const weights = Object.entries(entry.weights).map(([k, v]) => `${k}: ${v}`).join(' | ');
    const ghost = entry.ghost_loop ? '<span class="ghost-loop">Ghost Loop</span>' : '';
    
    return `
        <li data-id="${entry.id}">
            <h4>${entry.title} ${ghost}</h4>
            <p>${entry.summary}</p>
            <div>Tags: ${tags}</div>
            <div>Weights: ${weights}</div>
            <button onclick="editEntry('${entry.id}')">Edit</button>
            <button onclick="closeGhostLoop('${entry.id}')">Close Loop</button>
        </li>
    `;
}

// Edit Entry
function editEntry(entryId) {
    apiCall(`/journal/${entryId}`)
        .then(entry => {
            document.getElementById('journal-content').value = entry.content;
            // Set form to edit mode
        });
    // On submit, call update API
}

// Close Ghost Loop
function closeGhostLoop(entryId) {
    const resolution = prompt('Resolution notes:');
    if (resolution) {
        apiCall('/journal/close-loop', 'POST', {entry_id: entryId, resolution})  // Assume endpoint
            .then(() => {
                alert('Loop closed');
                loadJournalEntries();
                updateGamification();
            });
    }
}

// Auto-Suggestions
function showAutoSuggestions(entryId) {
    apiCall(`/journal/suggestions/${entryId}`)
        .then(sugs => {
            // Display suggestions in UI
            const sugHtml = Object.entries(sugs).map(([k, v]) => `<p>${k}: ${v}</p>`).join('');
            // Append to entry li
        });
}

// Gamification Updates
function updateGamification() {
    apiCall('/gamification/stats')
        .then(stats => {
            // Update dashboard (from main.js)
            sendMessage({type: 'update_dashboard', stats});  // Via WS if needed
        });
}

// Export
function exportJournal() {
    apiCall('/journal/export', 'GET')
        .then(res => {
            const blob = new Blob([JSON.stringify(res)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'journal.json';
            a.click();
        });
}

// Event Listeners
document.getElementById('journal-search').addEventListener('input', (e) => loadJournalEntries(e.target.value));

// Integrate with WS for real-time journal updates
function handleJournalMessage(msg) {
    if (msg.type === 'journal_entry_created') {
        loadJournalEntries();
    } else if (msg.type === 'voice_transcribed') {
        document.getElementById('journal-content').value = msg.transcription;
    }
}

// Assume called from main handleWebSocketMessage
