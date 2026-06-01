import streamlit as st

def get_theme_css():
    if st.session_state.get("theme", "light") == "dark":
        return """
        :root {
            --bg-color: #10121a;
            --text-color: #e3e3e3;
            --chat-user: #1f2230;
            --chat-bot: #12141c;
            --border-color: #262936;
            --shadow: 0 4px 15px rgba(0,0,0,0.4);
            --gradient-start: #0c0d12;
            --gradient-end: #161822;
        }
        """
    else:
        return """
        :root {
            --bg-color: #ffffff;
            --text-color: #202124;
            --chat-user: #e3effb;
            --chat-bot: #ffffff;
            --border-color: #e2e6f0;
            --shadow: 0 4px 15px rgba(0,0,0,0.06);
            --gradient-start: #eff3f9;
            --gradient-end: #fafbfe;
        }
        """

def inject_custom_styles():
    theme_css = get_theme_css()
    st.markdown(f"""
<style>
{theme_css}

/* Custom dictation iframe centring in column 3 */
div[data-testid="column"]:nth-child(3) iframe {{
    width: 44px !important;
    height: 44px !important;
    border: none !important;
    overflow: hidden !important;
    display: block !important;
    margin: 0 auto !important;
}}

/* Custom send button iframe centring in column 4 */
div[data-testid="column"]:nth-child(4) iframe {{
    width: 44px !important;
    height: 44px !important;
    border: none !important;
    overflow: hidden !important;
    display: block !important;
    margin: 0 auto !important;
}}

/* Hide programmatic voice active checkbox */
div[data-testid="stCheckbox"]:has(input[aria-label="Voice Active"]) {{
    display: none !important;
}}

/* 1. Transparent Default Header */
[data-testid="stHeader"] {{ 
    background-color: transparent !important; 
}}

/* 2. Base App Styling */
.stApp, [data-testid="stAppViewContainer"] {{
    background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%) !important;
    color: var(--text-color) !important;
}}
[data-testid="stMainViewContainer"], 
[data-testid="stMain"], 
.main, 
.stAppHeader {{
    background: transparent !important;
    background-color: transparent !important;
}}
.stApp * {{
    color: var(--text-color);
}}

/* 3. Layout Width */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 150px !important;
    max-width: 1200px !important;
}}

/* 4. Chat Bubbles */
[data-testid="stChatMessage"] {{
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    background-color: var(--chat-bot);
}}
[data-testid="stChatMessage"] * {{
    color: var(--text-color) !important;
}}
[data-testid="chat-message-user"] {{
    background-color: var(--chat-user) !important;
}}

/* 5. Tool Expanders */
[data-testid="stStatusWidget"] {{
    background-color: transparent !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: 8px;
    margin-top: 10px;
}}
[data-testid="stStatusWidget"] * {{
    color: var(--text-color) !important;
}}

/* 6. GUARANTEED FIXED OVAL CHATBAR - SEPARATED SELECTORS TO PREVENT BROWSER CANCELLATIONS */
div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) {{
    position: fixed !important;
    bottom: 25px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 90% !important;
    max-width: 1000px !important;
    background-color: var(--bg-color) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 40px !important;
    padding: 5px 15px !important;
    box-shadow: var(--shadow) !important;
    z-index: 9999 !important;
    align-items: center !important;
}}
div.chatbar-block {{
    position: fixed !important;
    bottom: 25px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 90% !important;
    max-width: 1000px !important;
    background-color: var(--bg-color) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 40px !important;
    padding: 5px 15px !important;
    box-shadow: var(--shadow) !important;
    z-index: 9999 !important;
    align-items: center !important;
}}

div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]):focus-within {{
    border-color: #4285F4 !important;
    box-shadow: 0 0 0 1px #4285F4 !important;
}}
div.chatbar-block:focus-within {{
    border-color: #4285F4 !important;
    box-shadow: 0 0 0 1px #4285F4 !important;
}}

/* 7. Transparent Inner Inputs */
.stTextInput div[data-baseweb="base-input"], .stTextInput div[data-baseweb="input"] {{
    background-color: transparent !important;
    border: none !important;
}}
.stTextInput input {{
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    color: var(--text-color) !important;
    -webkit-text-fill-color: var(--text-color) !important;
    font-size: 16px !important;
    padding: 15px 5px !important; /* Taller input */
}}
.stTextInput input:focus {{
    background-color: transparent !important;
    box-shadow: none !important;
}}

/* 8. Model Selector styling */
.stSelectbox > div > div {{
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    color: #4285F4 !important;
    font-weight: 500;
    border-radius: 20px;
    transition: background-color 0.2s ease;
    cursor: pointer;
    margin-top: 5px !important;
}}
.stSelectbox > div > div:hover {{
    background-color: rgba(128, 128, 128, 0.1) !important;
}}

/* 9. Custom Top Nav Styling for layout and gemini header */
div[data-testid="stHorizontalBlock"]:has(.gemini-header) {{
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
}}
div[data-testid="stHorizontalBlock"]:has(.gemini-header) > div[data-testid="column"]:nth-child(1) {{
    width: calc(100% - 60px) !important;
    flex: 0 0 calc(100% - 60px) !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}
div[data-testid="stHorizontalBlock"]:has(.gemini-header) > div[data-testid="column"]:nth-child(2) {{
    width: 60px !important;
    flex: 0 0 60px !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}
.gemini-header {{
    font-size: clamp(1.5rem, 5vw, 2.2rem);
    font-weight: 600;
    background: -webkit-linear-gradient(45deg, #4285F4, #EA4335, #FBBC05, #34A853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

/* 10. Stylize the ▶ and ■ buttons in chat history */
button[key^="play_voice_"], button[key^="stop_voice_"] {{
    background-color: transparent !important;
    border: 1px solid var(--border-color) !important;
    padding: 0 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    box-shadow: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    transition: all 0.2s ease !important;
}}
button[key^="play_voice_"]:hover, button[key^="stop_voice_"]:hover {{
    background-color: rgba(66, 133, 244, 0.1) !important;
    border-color: #4285F4 !important;
    transform: scale(1.1) !important;
}}

/* Force the parent Streamlit element-container of the play/stop buttons to push to the right side of the bubble */
div[data-testid="element-container"]:has(button[key^="play_voice_"]) {{
    display: flex !important;
    justify-content: flex-end !important;
    width: 100% !important;
    margin-top: 10px !important;
    margin-bottom: 5px !important;
}}
div[data-testid="element-container"]:has(button[key^="stop_voice_"]) {{
    display: flex !important;
    justify-content: flex-end !important;
    width: 100% !important;
    margin-top: 10px !important;
    margin-bottom: 5px !important;
}}

/* 11. Responsive Mobile View Overrides (width <= 950px) - SEPARATED SELECTION TO BYPASS BROWSER CORS/HAS EXCEPTIONS */
@media (max-width: 950px) {{
    /* Force the chatbar horizontal block to remain a single flex row (no vertical wrapping!) */
    div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: space-between !important;
        width: 95% !important;
        padding: 5px 10px !important;
        border-radius: 40px !important;
        gap: 0px !important;
    }}
    div.chatbar-block {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: space-between !important;
        width: 95% !important;
        padding: 5px 10px !important;
        border-radius: 40px !important;
        gap: 0px !important;
    }}
    
    /* Hide the 2nd column (Model Selector) completely on mobile */
    div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"]:nth-child(2) {{
        display: none !important;
        width: 0px !important;
        flex: 0 0 0px !important;
        max-width: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    div.chatbar-block > div[data-testid="column"]:nth-child(2) {{
        display: none !important;
        width: 0px !important;
        flex: 0 0 0px !important;
        max-width: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    /* Force the text input column to take up all remaining left-side width using flex-grow */
    div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"]:nth-child(1) {{
        flex: 1 1 0% !important;
        width: auto !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    div.chatbar-block > div[data-testid="column"]:nth-child(1) {{
        flex: 1 1 0% !important;
        width: auto !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    /* Force the mic (3rd col) and send button (4th col) to stay side-by-side on the right */
    div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"]:nth-child(3) {{
        width: 44px !important;
        flex: 0 0 44px !important;
        max-width: 44px !important;
        margin-left: 6px !important;
        margin-right: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }}
    div.chatbar-block > div[data-testid="column"]:nth-child(3) {{
        width: 44px !important;
        flex: 0 0 44px !important;
        max-width: 44px !important;
        margin-left: 6px !important;
        margin-right: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }}

    div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"]:nth-child(4) {{
        width: 44px !important;
        flex: 0 0 44px !important;
        max-width: 44px !important;
        margin-left: 6px !important;
        margin-right: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }}
    div.chatbar-block > div[data-testid="column"]:nth-child(4) {{
        width: 44px !important;
        flex: 0 0 44px !important;
        max-width: 44px !important;
        margin-left: 6px !important;
        margin-right: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }}
}}

/* 12. Robust Parent-Level Styling for the Big Oval Chatbar Elements - SEPARATED SELECTORS */
div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"]:nth-child(4) button {{
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    background-color: #4285F4 !important;
    border: none !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}}
div.chatbar-block > div[data-testid="column"]:nth-child(4) button {{
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    background-color: #4285F4 !important;
    border: none !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}}

div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"]:nth-child(4) button:hover {{
    background-color: #357ae8 !important;
    transform: scale(1.05) !important;
}}
div.chatbar-block > div[data-testid="column"]:nth-child(4) button:hover {{
    background-color: #357ae8 !important;
    transform: scale(1.05) !important;
}}

div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"]:nth-child(4) button * {{
    color: white !important;
}}
div.chatbar-block > div[data-testid="column"]:nth-child(4) button * {{
    color: white !important;
}}

div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
div.chatbar-block > div[data-testid="column"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"]:nth-child(1) {{
    justify-content: flex-start !important;
}}
div.chatbar-block > div[data-testid="column"]:nth-child(1) {{
    justify-content: flex-start !important;
}}

div[data-testid="stHorizontalBlock"]:has(input[placeholder^="Ask a legal question"]) > div[data-testid="column"] > div {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}}
div.chatbar-block > div[data-testid="column"] > div {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}}
</style>
""", unsafe_allow_html=True)
