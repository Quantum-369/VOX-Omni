document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const themeToggle = document.getElementById('themeToggle');
    const clearChat = document.getElementById('clearChat');
    const typingIndicator = document.getElementById('typingIndicator');
    const suggestionChips = document.getElementById('suggestionChips');
    let isProcessing = false;

    // Theme Management
    function toggleTheme() {
        document.body.dataset.theme = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('theme', document.body.dataset.theme);
        
        // Update theme toggle icon
        const icon = themeToggle.querySelector('i');
        icon.className = document.body.dataset.theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    // Initialize theme from localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.body.dataset.theme = savedTheme;
        const icon = themeToggle.querySelector('i');
        icon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    // Auto-resize textarea
    function adjustTextareaHeight() {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
    }

    // Create message element from template
    function createMessage(content, isUser = false) {
        const template = document.getElementById('messageTemplate');
        const messageElement = template.content.cloneNode(true);
        const message = messageElement.querySelector('.message');
        
        message.classList.add(isUser ? 'user-message' : 'bot-message');
        
        const messageContent = message.querySelector('.message-content');
        // Parse markdown for bot messages, plain text for user messages
        messageContent.innerHTML = isUser ? content.replace(/\n/g, '<br>') : marked.parse(content);
        
        const timestamp = message.querySelector('.message-timestamp');
        timestamp.textContent = new Date().toLocaleTimeString();
        
        const copyBtn = message.querySelector('.copy-btn');
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(content);
            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
            }, 2000);
        });
        
        return message;
    }

    // Add message to chat
    function addMessage(content, isUser = false) {
        const message = createMessage(content, isUser);
        chatMessages.appendChild(message);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Save chat history to localStorage
        saveChatHistory();
    }

    // Save chat history to localStorage
    function saveChatHistory() {
        const messages = Array.from(chatMessages.querySelectorAll('.message')).map(msg => ({
            content: msg.querySelector('.message-content').innerHTML,
            isUser: msg.classList.contains('user-message'),
            timestamp: msg.querySelector('.message-timestamp').textContent
        }));
        localStorage.setItem('chatHistory', JSON.stringify(messages));
    }

    // Load chat history from localStorage
    function loadChatHistory() {
        const history = localStorage.getItem('chatHistory');
        if (history) {
            const messages = JSON.parse(history);
            messages.forEach(msg => {
                addMessage(msg.content.replace(/<br>/g, '\n'), msg.isUser);
            });
        } else {
            // Add initial greeting if no history
            addMessage('Hello! I am your database assistant. I can help you with:\n- Running SQL queries\n- Creating CSV files\n- Advanced data analysis\n- Real-time information\n\nWhat would you like to do?');
        }
    }

    // Set loading state
    function setLoading(loading) {
        isProcessing = loading;
        messageInput.disabled = loading;
        sendButton.disabled = loading;
        typingIndicator.classList.toggle('hidden', !loading);
    }

    // Send message to server
    async function sendMessage() {
        if (isProcessing) return;

        const message = messageInput.value.trim();
        if (!message) return;

        setLoading(true);
        addMessage(message, true);
        messageInput.value = '';
        messageInput.style.height = 'auto';

        try {
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.response) {
                // Add small delay for natural feel
                await new Promise(resolve => setTimeout(resolve, 500));
                addMessage(data.response);
            } else if (data.error) {
                addMessage(`Error: ${data.error}`);
            } else {
                addMessage('Sorry, I encountered an error processing your request.');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, I encountered a network error. Please try again.');
        } finally {
            setLoading(false);
            messageInput.focus();
        }
    }

    // Handle file drops
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        chatMessages.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        chatMessages.classList.remove('drag-over');
    }

    async function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        chatMessages.classList.remove('drag-over');

        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            const fileNames = files.map(file => file.name).join(', ');
            addMessage(`Uploading files: ${fileNames}`, true);
            
            // Here you would implement file upload logic
            // For now, we'll just acknowledge the drop
            await new Promise(resolve => setTimeout(resolve, 1000));
            addMessage('Files received. You can now ask questions about the uploaded data.');
        }
    }

    // Event Listeners
    themeToggle.addEventListener('click', toggleTheme);
    
    clearChat.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear the chat history?')) {
            while (chatMessages.firstChild) {
                chatMessages.removeChild(chatMessages.firstChild);
            }
            // Re-add message template
            const template = document.createElement('template');
            template.id = 'messageTemplate';
            template.innerHTML = document.querySelector('#messageTemplate').innerHTML;
            chatMessages.appendChild(template);
            
            // Clear localStorage
            localStorage.removeItem('chatHistory');
            
            // Add initial greeting
            addMessage('Hello! I am your database assistant. How can I help you today?');
        }
    });

    messageInput.addEventListener('input', adjustTextareaHeight);
    
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);

    // Suggestion Chips
    suggestionChips.addEventListener('click', (e) => {
        if (e.target.classList.contains('chip')) {
            messageInput.value = e.target.textContent;
            messageInput.focus();
            adjustTextareaHeight();
        }
    });

    // File drop event listeners
    chatMessages.addEventListener('dragover', handleDragOver);
    chatMessages.addEventListener('dragleave', handleDragLeave);
    chatMessages.addEventListener('drop', handleDrop);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + / to toggle theme
        if (e.key === '/' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            toggleTheme();
        }
        // Ctrl/Cmd + L to clear chat
        if (e.key === 'l' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            clearChat.click();
        }
    });

    // Initialize
    loadChatHistory();

    // Check connection status
    async function checkConnection() {
        try {
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: 'ping' })
            });
            
            if (!response.ok) {
                addMessage('Warning: Connection to server seems unstable.');
            }
        } catch (error) {
            addMessage('Error: Unable to connect to server. Please check if the server is running.');
        }
    }

    // Check connection on startup
    checkConnection();

    // Add window focus handling
    let windowFocused = true;
    window.addEventListener('focus', () => {
        windowFocused = true;
        document.title = 'Vox Chat Interface';
    });

    window.addEventListener('blur', () => {
        windowFocused = false;
    });

    // Modify addMessage to handle notifications
    const originalAddMessage = addMessage;
    addMessage = (content, isUser = false) => {
        originalAddMessage(content, isUser);
        
        // Show notification for bot messages when window is not focused
        if (!isUser && !windowFocused) {
            document.title = '(1) New Message - Vox Chat';
            // If browser supports notifications and permission is granted
            if ("Notification" in window && Notification.permission === "granted") {
                new Notification("New Message from Vox", {
                    body: content.slice(0, 100) + (content.length > 100 ? '...' : ''),
                    icon: '/favicon.ico'
                });
            }
        }
    };

    // Request notification permission
    if ("Notification" in window && Notification.permission === "default") {
        Notification.requestPermission();
    }
});