// frontend/app.js

// ==========================================
// ☁️ CLOUD DEPLOYMENT SETTINGS
// ==========================================
// When running on your laptop, use localhost:3000
// When deploying to Vercel, change this to your Render Node.js URL!
// Example: const BACKEND_URL = 'https://upgcs-node-backend.onrender.com';

const BACKEND_URL = 'http://localhost:3000'; 

// ==========================================
// 🎨 UI ELEMENTS & LOGIC
// ==========================================
const chatForm = document.getElementById('chat-form');
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');

function addMessage(text, sender) {
    const div = document.createElement('div');
    div.classList.add('message', `${sender}-message`);
    div.innerText = text;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight; // Auto-scroll to bottom
}

function showTypingIndicator() {
    const div = document.createElement('div');
    div.classList.add('typing-indicator');
    div.id = 'typing';
    div.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingIndicator() {
    const typingElement = document.getElementById('typing');
    if (typingElement) {
        typingElement.remove();
    }
}

// ==========================================
// 🚀 CHAT SUBMIT EVENT
// ==========================================
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = userInput.value.trim();
    if (!message) return;

    // 1. Show user message in the chat
    addMessage(message, 'user');
    userInput.value = '';

    // 2. Show bouncy loading dots
    showTypingIndicator();

    try {
        // 3. Connect to the Node.js Gatekeeper (which will securely call Python)
        const response = await fetch(`${BACKEND_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Note: Your chat.js expects "message", not "question"
            body: JSON.stringify({ message: message }) 
        });

        const result = await response.json();
        
        // 4. Remove bouncy dots
        removeTypingIndicator();

        // 5. Display the AI's answer
        if (response.ok && result.success && result.data.answer) {
            addMessage(result.data.answer, 'ai');
        } else {
            // This catches errors sent cleanly from your Node.js server
            const errorMessage = result.message || "Error: Could not process that request.";
            addMessage(errorMessage, 'ai');
        }

    } catch (error) {
        removeTypingIndicator();
        console.error("Fetch Error:", error);
        addMessage("Connection lost. Is the Node.js server running? 🔌", 'ai');
    }
});