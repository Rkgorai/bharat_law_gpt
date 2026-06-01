import streamlit as st
import streamlit.components.v1 as components

def render_dictation_mic():
    """
    Renders the HTML/JS real-time Speech Dictation button widget inside the column.
    """
    st.iframe("""
    <!DOCTYPE html>
    <html>
    <head>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" />
    <style>
    body {
        margin: 0;
        padding: 0;
        background: transparent;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        width: 44px;
        height: 44px;
    }
    button {
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        outline: none !important;
    }
    .material-symbols-rounded {
        color: #7f8c8d;
        font-size: 24px;
        transition: all 0.2s ease;
    }
    button:hover .material-symbols-rounded {
        color: #4285F4;
        transform: scale(1.1);
    }
    button.listening {
        background-color: #EA4335 !important;
        animation: pulse 1.5s infinite;
    }
    button.listening .material-symbols-rounded {
        color: white !important;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(234, 67, 53, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(234, 67, 53, 0); }
        100% { box-shadow: 0 0 0 0 rgba(234, 67, 53, 0); }
    }
    </style>
    </head>
    <body>
    <button id="mic-btn">
        <span class="material-symbols-rounded">mic</span>
    </button>
    <script>
    const btn = document.getElementById('mic-btn');
    const icon = btn.querySelector('span');
    let recognition = null;
    let isListening = false;
    let finalTranscript = '';
    // Helper to programmatically update React controlled inputs securely
    const setReactInputValue = (input, val) => {
        try {
            // 1. Primary Strategy: Use browser native execCommand to simulate real typing.
            // This naturally triggers React's state and Streamlit's input sync flawlessly.
            input.focus();
            input.setSelectionRange(0, input.value.length);
            parent.document.execCommand('insertText', false, val);
        } catch (e) {
            console.warn("[STT] execCommand failed, falling back to prototype setter: ", e);
            try {
                // 2. Fallback Strategy: Override the property prototype and trigger React's _valueTracker
                const parentWindow = parent.window;
                const setter = Object.getOwnPropertyDescriptor(parentWindow.HTMLInputElement.prototype, "value").set;
                const lastVal = input.value;
                setter.call(input, val);
                const tracker = input._valueTracker;
                if (tracker) {
                    tracker.setValue(lastVal);
                }
                input.dispatchEvent(new parentWindow.Event('input', { bubbles: true }));
                input.dispatchEvent(new parentWindow.Event('change', { bubbles: true }));
            } catch (err) {
                console.error("[STT] Prototype fallback failed: ", err);
            }
        }
    };

    try {
        // Inject Custom Styles to Parent once
        if (!parent.document.getElementById('chatbar-custom-styles')) {
            const style = parent.document.createElement('style');
            style.id = 'chatbar-custom-styles';
            style.innerHTML = `
                .chatbar-block { align-items: center !important; }
                .chatbar-block > div[data-testid="column"] { display: flex !important; align-items: center !important; justify-content: center !important; }
                .chatbar-block > div[data-testid="column"]:nth-child(1) { justify-content: flex-start !important; }
                .chatbar-block > div[data-testid="column"] > div { margin-bottom: 0 !important; margin-top: 0 !important; }
                .chatbar-block > div[data-testid="column"]:nth-child(4) button { border-radius: 50% !important; width: 44px !important; height: 44px !important; background-color: #4285F4 !important; border: none !important; padding: 0 !important; display: flex !important; align-items: center !important; justify-content: center !important; transition: all 0.2s ease !important; }
                .chatbar-block > div[data-testid="column"]:nth-child(4) span.material-symbols-rounded { color: white !important; font-size: 24px !important; }
                .chatbar-block > div[data-testid="column"]:nth-child(4) { transition: all 0.2s ease !important; }
                .chatbar-block.chatbar-empty > div[data-testid="column"]:nth-child(4) { opacity: 0 !important; pointer-events: none !important; transform: scale(0.8) !important; }
            `;
            parent.document.head.appendChild(style);
        }
    } catch (e) {
        console.warn("[STT] Parent document style injection blocked by CORS: ", e);
    }

    // Fix visual parent alignment for the input pill
    const fixParentUI = () => {
        try {
            const inputs = parent.document.querySelectorAll('input');
            let chatInput = null;
            for (let i = 0; i < inputs.length; i++) {
                if (inputs[i].placeholder === "Ask a legal question...") { chatInput = inputs[i]; break; }
            }
            if (!chatInput) return;
            let block = chatInput.closest('div[data-testid="stHorizontalBlock"]');
            if (block) {
                if (!block.classList.contains('chatbar-block')) { block.classList.add('chatbar-block'); }
                const updateSendBtn = () => {
                    if (chatInput.value.trim() === '') { block.classList.add('chatbar-empty'); } else { block.classList.remove('chatbar-empty'); }
                };
                if (!chatInput.dataset.listenerAdded) {
                    chatInput.addEventListener('input', updateSendBtn);
                    chatInput.addEventListener('change', updateSendBtn);
                    chatInput.dataset.listenerAdded = 'true';
                }
                updateSendBtn();
            }

            // Find the Send button reliably inside the same horizontal block as the text input!
            const sendBtn = block.querySelector('button');
            if (sendBtn && !sendBtn.dataset.listenerAdded) {
                // Hook mousedown to force immediate text box blur and state sync before click event processes!
                sendBtn.addEventListener('mousedown', (e) => {
                    chatInput.blur();
                });
                sendBtn.dataset.listenerAdded = 'true';
            }
        } catch (e) {
            console.warn("[STT] fixParentUI blocked by CORS: ", e);
        }
    };
    fixParentUI();
    setInterval(fixParentUI, 500);

    // Use parent window to bypass sandboxed iframe microphone permission limits, with local fallback
    let SpeechRecognition = null;
    try {
        SpeechRecognition = parent.window.SpeechRecognition || parent.window.webkitSpeechRecognition || window.SpeechRecognition || window.webkitSpeechRecognition;
    } catch (e) {
        console.warn("[STT] Parent window access blocked by CORS, falling back to local iframe window: ", e);
        SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    }
    
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-IN'; // Optimized for Indian accented English

        recognition.onstart = () => {
            isListening = true;
            btn.classList.add('listening');
            icon.textContent = 'stop';
        };

        recognition.onresult = (event) => {
            let interimTranscript = '';
            let currentFinal = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    currentFinal += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            
            if (currentFinal) {
                finalTranscript += currentFinal + ' ';
            }
            
            const textToShow = (finalTranscript + interimTranscript).trim() + '\u200b';
            
            try {
                // Safely update parent input box using our custom React state synchronizer!
                const chatInput = parent.document.querySelector('input[placeholder="Ask a legal question..."]');
                if (chatInput) {
                    setReactInputValue(chatInput, textToShow);
                    
                    // Force parent UI Send button visibility update
                    fixParentUI();
                }
            } catch (e) {
                console.warn("[STT] Input value update blocked by CORS: ", e);
            }
        };

        recognition.onerror = (event) => {
            console.error("[STT] Dictation Error: ", event.error);
            stopListening();
        };

        recognition.onend = () => {
            stopListening();
        };
    } else {
        btn.style.opacity = '0.5';
        btn.title = "Speech recognition is not supported in this browser.";
    }

    function startListening() {
        if (!recognition) return;
        finalTranscript = '';
        
        try {
            // Safely clear the parent input box before dictating starts
            const chatInput = parent.document.querySelector('input[placeholder="Ask a legal question..."]');
            if (chatInput) {
                setReactInputValue(chatInput, '');
            }
        } catch (e) {
            console.warn("[STT] Input clear blocked by CORS: ", e);
        }
        try {
            recognition.start();
        } catch (e) {
            console.error("[STT] Start Dictation failed: ", e);
        }
    }

    function stopListening() {
        if (!isListening) return;
        isListening = false;
        btn.classList.remove('listening');
        icon.textContent = 'mic';
        try {
            recognition.stop();
        } catch (e) {}
        
        fixParentUI();
    }

    btn.addEventListener('click', () => {
        if (isListening) {
            stopListening();
        } else {
            startListening();
        }
    });
    </script>
    </body>
    </html>
    """, height=44)
