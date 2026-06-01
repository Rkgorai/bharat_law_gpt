import streamlit as st

def render_send_button():
    """
    Renders the HTML/JS real-time custom Send button widget inside the column.
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
        background-color: #4285F4 !important;
        border: none !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        outline: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    }
    button:hover {
        background-color: #357ae8 !important;
        transform: scale(1.05) !important;
    }
    button:active {
        transform: scale(0.95) !important;
    }
    .material-symbols-rounded {
        color: white !important;
        font-size: 24px;
    }
    /* Disabled state when text input is empty */
    button.disabled {
        background-color: rgba(128, 128, 128, 0.2) !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
        pointer-events: none !important;
    }
    button.disabled .material-symbols-rounded {
        color: rgba(128, 128, 128, 0.4) !important;
    }
    </style>
    </head>
    <body>
    <button id="send-btn" class="disabled">
        <span class="material-symbols-rounded">arrow_upward</span>
    </button>
    <script>
    const btn = document.getElementById('send-btn');
    
    // Sync button state (disabled/enabled) based on parent text input value
    const syncButtonState = () => {
        try {
            const chatInput = parent.document.querySelector('input[placeholder="Ask a legal question..."]');
            if (chatInput) {
                if (chatInput.value.trim() === '') {
                    btn.classList.add('disabled');
                } else {
                    btn.classList.remove('disabled');
                }
            }
        } catch (e) {}
    };
    
    // Run sync periodically
    syncButtonState();
    setInterval(syncButtonState, 200);
    
    // Handle click event to submit query securely
    btn.addEventListener('click', () => {
        try {
            const chatInput = parent.document.querySelector('input[placeholder="Ask a legal question..."]');
            if (chatInput && chatInput.value.trim() !== '') {
                // Force immediate text box blur and state sync
                chatInput.blur();
                
                // Wait 50ms for state to sync, then trigger Enter keypress
                setTimeout(() => {
                    const enterEvent = new parent.window.KeyboardEvent('keydown', {
                        bubbles: true,
                        cancelable: true,
                        key: 'Enter',
                        code: 'Enter',
                        keyCode: 13,
                        which: 13
                    });
                    chatInput.dispatchEvent(enterEvent);
                }, 50);
            }
        } catch (e) {
            console.error("[Send Button] Trigger failed: ", e);
        }
    });
    </script>
    </body>
    </html>
    """, height=44)
