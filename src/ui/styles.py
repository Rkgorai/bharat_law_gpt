import streamlit as st

def get_theme_css():
    if st.session_state.get("theme", "light") == "dark":
        return """
        :root {
            --bg-color: #131314;
            --text-color: #e3e3e3;
            --chat-user: #282a2c;
            --chat-bot: #1e1f20;
            --border-color: #3c4043;
            --shadow: 0 4px 15px rgba(0,0,0,0.5);
            --gradient-start: #000000;
            --gradient-end: #131314;
        }
        """
    else:
        return """
        :root {
            --bg-color: #ffffff;
            --text-color: #202124;
            --chat-user: #e3f2fd;
            --chat-bot: #ffffff;
            --border-color: #dadce0;
            --shadow: 0 4px 15px rgba(0,0,0,0.1);
            --gradient-start: #e8f0fe;
            --gradient-end: #ffffff;
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

/* Hide programmatic voice active checkbox */
div[data-testid="stCheckbox"]:has(input[aria-label="Voice Active"]) {{
    display: none !important;
}}

/* 1. Transparent Default Header */
[data-testid="stHeader"] {{ 
    background-color: transparent !important; 
}}

/* 2. Base App Styling */
.stApp {{
    background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
    color: var(--text-color) !important;
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

/* 6. GUARANTEED FIXED OVAL CHATBAR USING :has() */
div[data-testid="stHorizontalBlock"]:has(input[aria-label="Ask something..."]) {{
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
div[data-testid="stHorizontalBlock"]:has(input[aria-label="Ask something..."]):focus-within {{
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
div[data-testid="element-container"]:has(button[key^="play_voice_"]),
div[data-testid="element-container"]:has(button[key^="stop_voice_"]) {{
    display: flex !important;
    justify-content: flex-end !important;
    width: 100% !important;
    margin-top: 10px !important;
    margin-bottom: 5px !important;
}}
</style>
""", unsafe_allow_html=True)
