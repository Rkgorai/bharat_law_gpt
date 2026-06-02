import streamlit as st

def get_theme_css():
    theme = st.session_state.get("theme", "system")
    
    dark_css = """
        :root {
            --bg-color: #10121a;
            --text-color: #e3e3e3;
            --chat-user: #1f2230;
            --chat-bot: #12141c;
            --border-color: #262936;
            --shadow: 0 8px 32px rgba(0,0,0,0.5);
            --gradient-start: #0c0d12;
            --gradient-end: #161822;
            --bg-rgb: 16, 18, 26;
            --sidebar-header: #ffffff;
            --input-bg: #1c1f2e;
            
            /* Sidebar-specific tokens */
            --sidebar-bg: linear-gradient(180deg, rgba(22, 25, 38, 0.96) 0%, rgba(13, 15, 22, 0.98) 100%);
            --sidebar-border: rgba(255, 255, 255, 0.08);
            --sidebar-header-color: #ffffff;
            --sidebar-text-primary: #ffffff;
            --sidebar-text-secondary: rgba(227, 227, 227, 0.75);
            --sidebar-text-muted: rgba(227, 227, 227, 0.5);
            --sidebar-divider: rgba(255, 255, 255, 0.08);
            
            --sidebar-input-bg: rgba(28, 31, 46, 0.85);
            --sidebar-input-border: rgba(255, 255, 255, 0.12);
            --sidebar-input-focus-border: #4285F4;
            
            --sidebar-button-bg: rgba(234, 67, 53, 0.1);
            --sidebar-button-border: rgba(234, 67, 53, 0.2);
            --sidebar-button-text: #ff6b5a;
            --sidebar-button-hover-bg: rgba(234, 67, 53, 0.22);
            
            --popover-bg: #1c1f2e;
            --popover-border: #262936;
            --popover-text: #e3e3e3;
            --popover-hover-bg: rgba(66, 133, 244, 0.18);
            --popover-hover-text: #4285F4;
        }
        
        /* Selectbox Popover/Dropdown Menu styling inside dark mode */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        ul[data-testid="stSelectboxVirtualDropdown"],
        ul[data-testid="stSelectboxVirtualDropdown"] *,
        [role="listbox"],
        [role="listbox"] *,
        [role="option"],
        [role="option"] *,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] * {
            background-color: var(--popover-bg) !important;
            background: var(--popover-bg) !important;
            color: var(--popover-text) !important;
            border-color: var(--popover-border) !important;
        }
        
        /* Transparent backgrounds for dropdown items so they display hovered backgrounds cleanly */
        li[role="option"],
        [role="option"],
        div[role="option"],
        div[data-baseweb="popover"] li,
        ul[data-testid="stSelectboxVirtualDropdown"] li {
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
        }
        
        /* Highly specific dynamic Hover and Selected States for options */
        li[role="option"]:hover,
        li[role="option"]:hover *,
        [role="option"]:hover,
        [role="option"]:hover *,
        li[aria-selected="true"],
        li[aria-selected="true"] *,
        [aria-selected="true"],
        [aria-selected="true"] *,
        li[role="option"]:active,
        li[role="option"]:active * {
            background-color: var(--popover-hover-bg) !important;
            background: var(--popover-hover-bg) !important;
            color: var(--popover-hover-text) !important;
        }
    """
    
    light_css = """
        :root {
            --bg-color: #ffffff;
            --text-color: #202124;
            --chat-user: #e3effb;
            --chat-bot: #ffffff;
            --border-color: #e2e6f0;
            --shadow: 0 8px 32px rgba(0,0,0,0.08);
            --gradient-start: #eff3f9;
            --gradient-end: #fafbfe;
            --bg-rgb: 255, 255, 255;
            --sidebar-header: #202124;
            --input-bg: #f1f3f4;
            
            /* Sidebar-specific tokens */
            --sidebar-bg: linear-gradient(180deg, rgba(246, 248, 252, 0.96) 0%, rgba(234, 240, 248, 0.98) 100%);
            --sidebar-border: rgba(0, 0, 0, 0.08);
            --sidebar-header-color: #202124;
            --sidebar-text-primary: #202124;
            --sidebar-text-secondary: rgba(32, 33, 36, 0.8);
            --sidebar-text-muted: rgba(32, 33, 36, 0.55);
            --sidebar-divider: rgba(0, 0, 0, 0.08);
            
            --sidebar-input-bg: rgba(255, 255, 255, 0.95);
            --sidebar-input-border: rgba(0, 0, 0, 0.12);
            --sidebar-input-focus-border: #1a73e8;
            
            --sidebar-button-bg: rgba(217, 48, 37, 0.06);
            --sidebar-button-border: rgba(217, 48, 37, 0.15);
            --sidebar-button-text: #d93025;
            --sidebar-button-hover-bg: rgba(217, 48, 37, 0.12);
            
            --popover-bg: #ffffff;
            --popover-border: #e2e6f0;
            --popover-text: #202124;
            --popover-hover-bg: rgba(26, 115, 232, 0.1);
            --popover-hover-text: #1a73e8;
        }
        
        /* Selectbox Popover/Dropdown Menu styling inside light mode */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        ul[data-testid="stSelectboxVirtualDropdown"],
        ul[data-testid="stSelectboxVirtualDropdown"] *,
        [role="listbox"],
        [role="listbox"] *,
        [role="option"],
        [role="option"] *,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] * {
            background-color: var(--popover-bg) !important;
            background: var(--popover-bg) !important;
            color: var(--popover-text) !important;
            border-color: var(--popover-border) !important;
        }
        
        /* Transparent backgrounds for dropdown items so they display hovered backgrounds cleanly */
        li[role="option"],
        [role="option"],
        div[role="option"],
        div[data-baseweb="popover"] li,
        ul[data-testid="stSelectboxVirtualDropdown"] li {
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
        }
        
        /* Highly specific dynamic Hover and Selected States for options */
        li[role="option"]:hover,
        li[role="option"]:hover *,
        [role="option"]:hover,
        [role="option"]:hover *,
        li[aria-selected="true"],
        li[aria-selected="true"] *,
        [aria-selected="true"],
        [aria-selected="true"] *,
        li[role="option"]:active,
        li[role="option"]:active * {
            background-color: var(--popover-hover-bg) !important;
            background: var(--popover-hover-bg) !important;
            color: var(--popover-hover-text) !important;
        }
    """

    if theme == "dark":
        return dark_css
    elif theme == "light":
        return light_css
    else:
        return f"""
        @media (prefers-color-scheme: dark) {{
            {dark_css}
        }}
        @media (prefers-color-scheme: light) {{
            {light_css}
        }}
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

/* 2. Base App & Sidebar Glassmorphic Styling */
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

/* Collapsible Settings Sidebar Panel styling */
section[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    transition: background 0.3s ease, border-right 0.3s ease !important;
}}

section[data-testid="stSidebar"] > div {{
    background: transparent !important;
}}

/* Sidebar divider lines styling */
section[data-testid="stSidebar"] hr {{
    border: none !important;
    border-top: 1px solid var(--sidebar-divider) !important;
    margin: 1.2rem 0 !important;
}}

/* Markdown typography inside sidebar settings drawer */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {{
    color: var(--sidebar-header-color) !important;
    font-weight: 600 !important;
    margin-top: 0px !important;
    margin-bottom: 12px !important;
    font-size: 1.15rem !important;
    letter-spacing: 0.3px !important;
}}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {{
    color: var(--sidebar-text-secondary) !important;
    font-size: 0.92rem !important;
    line-height: 1.5 !important;
}}

section[data-testid="stSidebar"] strong {{
    color: var(--sidebar-text-primary) !important;
    font-weight: 600 !important;
}}

/* Selectbox label styling */
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {{
    color: var(--sidebar-text-primary) !important;
    font-weight: 550 !important;
    font-size: 0.95rem !important;
    margin-bottom: 8px !important;
}}

/* Input dropdown selectbox container inside Sidebar settings */
section[data-testid="stSidebar"] div[data-baseweb="select"] {{
    background-color: var(--sidebar-input-bg) !important;
    border: 1px solid var(--sidebar-input-border) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease-in-out !important;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"]:hover {{
    border-color: rgba(66, 133, 244, 0.4) !important;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"]:focus-within {{
    border-color: var(--sidebar-input-focus-border) !important;
    box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.15) !important;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] div[data-testid="stSelectboxVirtualFocus"],
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
    color: var(--sidebar-text-primary) !important;
    font-weight: 500 !important;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{
    fill: var(--sidebar-text-secondary) !important;
    color: var(--sidebar-text-secondary) !important;
}}

/* Custom Red Glassmorphic style for Clear Chat button in sidebar settings drawer */
section[data-testid="stSidebar"] button:not([data-testid="sidebar-toggle"]) {{
    background-color: var(--sidebar-button-bg) !important;
    border: 1px solid var(--sidebar-button-border) !important;
    color: var(--sidebar-button-text) !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: none !important;
}}

section[data-testid="stSidebar"] button:not([data-testid="sidebar-toggle"]):hover {{
    background-color: var(--sidebar-button-hover-bg) !important;
    border-color: var(--sidebar-button-text) !important;
    transform: translateY(-1px) !important;
}}

section[data-testid="stSidebar"] button:not([data-testid="sidebar-toggle"]):active {{
    transform: translateY(0px) !important;
}}

section[data-testid="stSidebar"] button:not([data-testid="sidebar-toggle"]) * {{
    color: var(--sidebar-button-text) !important;
}}

/* Collapsible sidebar toggle menu arrow button styling */
button[data-testid="sidebar-toggle"] {{
    background-color: rgba(var(--bg-rgb), 0.5) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
}}

button[data-testid="sidebar-toggle"]:hover {{
    background-color: rgba(var(--bg-rgb), 0.85) !important;
    border-color: #4285F4 !important;
    transform: scale(1.05) !important;
}}

button[data-testid="sidebar-toggle"] svg {{
    fill: var(--text-color) !important;
    width: 18px !important;
    height: 18px !important;
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

/* 6. GUARANTEED FIXED OVAL CHATBAR */
div[data-testid="stHorizontalBlock"]:has(.stTextInput input),
div[data-testid="stHorizontalBlock"]:has(.stTextArea textarea),
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
    
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 0px !important;
}}

div[data-testid="stHorizontalBlock"]:has(.stTextInput input):focus-within,
div[data-testid="stHorizontalBlock"]:has(.stTextArea textarea):focus-within,
div.chatbar-block:focus-within {{
    border-color: #4285F4 !important;
    box-shadow: 0 0 0 1px #4285F4 !important;
}}

/* Set the column widths dynamically using flex layout for maximum responsiveness */
div[data-testid="stHorizontalBlock"]:has(.stTextInput input) div[data-testid="column"]:nth-child(1),
div[data-testid="stHorizontalBlock"]:has(.stTextArea textarea) div[data-testid="column"]:nth-child(1),
div.chatbar-block div[data-testid="column"]:nth-child(1) {{
    flex: 1 1 0% !important;
    width: auto !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* Column 2: Mic button */
div[data-testid="stHorizontalBlock"]:has(.stTextInput input) div[data-testid="column"]:nth-child(2),
div[data-testid="stHorizontalBlock"]:has(.stTextArea textarea) div[data-testid="column"]:nth-child(2),
div.chatbar-block div[data-testid="column"]:nth-child(2) {{
    flex: 0 0 44px !important;
    width: 44px !important;
    max-width: 44px !important;
    margin-left: 6px !important;
    margin-right: 0 !important;
    padding: 0 !important;
}}

/* Column 3: Send button */
div[data-testid="stHorizontalBlock"]:has(.stTextInput input) div[data-testid="column"]:nth-child(3),
div[data-testid="stHorizontalBlock"]:has(.stTextArea textarea) div[data-testid="column"]:nth-child(3),
div.chatbar-block div[data-testid="column"]:nth-child(3) {{
    flex: 0 0 44px !important;
    width: 44px !important;
    max-width: 44px !important;
    margin-left: 6px !important;
    margin-right: 0 !important;
    padding: 0 !important;
}}

/* Dynamic visibility toggles based on whether text input is empty, has text, or voice is active */

/* --- ENGINE 1: PURE CSS SELECTOR RULES (NATIVE BROWSER RUNTIME) --- */

/* 1. When voice is active: show Column 2 (mic) and hide Column 3 (send) */
div[data-testid="stHorizontalBlock"].voice-active div[data-testid="column"]:nth-child(2) {{
    display: flex !important;
    flex: 0 0 44px !important;
    width: 44px !important;
    max-width: 44px !important;
}}
div[data-testid="stHorizontalBlock"].voice-active div[data-testid="column"]:nth-child(3) {{
    display: none !important;
    width: 0px !important;
    flex: 0 0 0px !important;
    max-width: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* 2. When voice is NOT active and input is empty (placeholder-shown): hide Column 3 (send) and show Column 2 (mic) */
div[data-testid="stHorizontalBlock"]:not(.voice-active):has(.stTextInput input:placeholder-shown) div[data-testid="column"]:nth-child(3),
div[data-testid="stHorizontalBlock"]:not(.voice-active):has(.stTextArea textarea:placeholder-shown) div[data-testid="column"]:nth-child(3) {{
    display: none !important;
    width: 0px !important;
    flex: 0 0 0px !important;
    max-width: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
}}
div[data-testid="stHorizontalBlock"]:not(.voice-active):has(.stTextInput input:placeholder-shown) div[data-testid="column"]:nth-child(2),
div[data-testid="stHorizontalBlock"]:not(.voice-active):has(.stTextArea textarea:placeholder-shown) div[data-testid="column"]:nth-child(2) {{
    display: flex !important;
    flex: 0 0 44px !important;
    width: 44px !important;
    max-width: 44px !important;
}}

/* 3. When voice is NOT active and input is not empty (placeholder-hidden): hide Column 2 (mic) and show Column 3 (send) */
div[data-testid="stHorizontalBlock"]:not(.voice-active):has(.stTextInput input:not(:placeholder-shown)) div[data-testid="column"]:nth-child(2),
div[data-testid="stHorizontalBlock"]:not(.voice-active):has(.stTextArea textarea:not(:placeholder-shown)) div[data-testid="column"]:nth-child(2) {{
    display: none !important;
    width: 0px !important;
    flex: 0 0 0px !important;
    max-width: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
}}
div[data-testid="stHorizontalBlock"]:not(.voice-active):has(.stTextInput input:not(:placeholder-shown)) div[data-testid="column"]:nth-child(3),
div[data-testid="stHorizontalBlock"]:not(.voice-active):has(.stTextArea textarea:not(:placeholder-shown)) div[data-testid="column"]:nth-child(3) {{
    display: flex !important;
    flex: 0 0 44px !important;
    width: 44px !important;
    max-width: 44px !important;
}}


/* --- ENGINE 2: JAVASCRIPT FALLBACK RULES (FOR BROWSERS LACKING :HAS SUPPORT) --- */

/* 1. When voice is active: show Column 2 (mic) and hide Column 3 (send) */
div.chatbar-block.voice-active div[data-testid="column"]:nth-child(2) {{
    display: flex !important;
    flex: 0 0 44px !important;
    width: 44px !important;
    max-width: 44px !important;
}}
div.chatbar-block.voice-active div[data-testid="column"]:nth-child(3) {{
    display: none !important;
    width: 0px !important;
    flex: 0 0 0px !important;
    max-width: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* 2. When voice is NOT active and input is empty (chatbar-empty present): hide Column 3 (send) and show Column 2 (mic) */
div.chatbar-block.chatbar-empty:not(.voice-active) div[data-testid="column"]:nth-child(3) {{
    display: none !important;
    width: 0px !important;
    flex: 0 0 0px !important;
    max-width: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
}}
div.chatbar-block.chatbar-empty:not(.voice-active) div[data-testid="column"]:nth-child(2) {{
    display: flex !important;
    flex: 0 0 44px !important;
    width: 44px !important;
    max-width: 44px !important;
}}

/* 3. When voice is NOT active and input is not empty (chatbar-empty NOT present): hide Column 2 (mic) and show Column 3 (send) */
div.chatbar-block:not(.chatbar-empty):not(.voice-active) div[data-testid="column"]:nth-child(2) {{
    display: none !important;
    width: 0px !important;
    flex: 0 0 0px !important;
    max-width: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
}}
div.chatbar-block:not(.chatbar-empty):not(.voice-active) div[data-testid="column"]:nth-child(3) {{
    display: flex !important;
    flex: 0 0 44px !important;
    width: 44px !important;
    max-width: 44px !important;
}}

/* 7. Transparent Inner Inputs */
.stTextInput div[data-baseweb="base-input"],
.stTextInput div[data-baseweb="input"],
.stTextArea div[data-baseweb="base-input"],
.stTextArea div[data-baseweb="textarea"] {{
    background-color: transparent !important;
    border: none !important;
}}
.stTextInput input,
.stTextArea textarea {{
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    color: var(--text-color) !important;
    -webkit-text-fill-color: var(--text-color) !important;
    font-size: 16px !important;
    padding: 10px 5px !important;
    resize: none !important;
}}
.stTextInput input:focus,
.stTextArea textarea:focus {{
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

/* 11. Responsive Mobile View Overrides */
div.chatbar-block.chatbar-mobile {{
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

/* Force the text input column to take up all remaining left-side width using flex-grow */
div.chatbar-block.chatbar-mobile div[data-testid="column"]:nth-child(1) {{
    flex: 1 1 0% !important;
    width: auto !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* 12. Robust Parent-Level Styling for the Big Oval Chatbar Elements */
div.chatbar-block div[data-testid="column"]:nth-child(3) button {{
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

div.chatbar-block div[data-testid="column"]:nth-child(3) button:hover {{
    background-color: #357ae8 !important;
    transform: scale(1.05) !important;
}}

div.chatbar-block div[data-testid="column"]:nth-child(3) button * {{
    color: white !important;
}}

div.chatbar-block div[data-testid="column"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

div.chatbar-block div[data-testid="column"]:nth-child(1) {{
    justify-content: flex-start !important;
}}

div.chatbar-block div[data-testid="column"] > div {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}}
</style>
<script>
(function() {{
    const runChatbarScript = () => {{
        try {{
            const chatInput = document.querySelector('.stTextArea textarea, .stTextInput input') || 
                              document.querySelector('textarea[placeholder="Ask a legal question..."], input[placeholder="Ask a legal question..."]');
            if (chatInput) {{
                const block = chatInput.closest('div[data-testid="stHorizontalBlock"]');
                if (block) {{
                    if (!block.classList.contains('chatbar-block')) {{
                        block.classList.add('chatbar-block');
                    }}
                    
                    // Toggle empty state
                    if (chatInput.value.trim() === '') {{
                        block.classList.add('chatbar-empty');
                    }} else {{
                        block.classList.remove('chatbar-empty');
                    }}
                    
                    // Toggle mobile state based on viewport width
                    if (window.innerWidth <= 950) {{
                        block.classList.add('chatbar-mobile');
                    }} else {{
                        block.classList.remove('chatbar-mobile');
                    }}
                }}
            }}
        }} catch (e) {{
            console.error("Chatbar script error:", e);
        }}
    }};

    // Listen to secure postMessage events from sandboxed voice iframe
    window.addEventListener('message', (event) => {{
        try {{
            if (event.data && event.data.type === 'voice-active') {{
                const chatInput = document.querySelector('.stTextArea textarea, .stTextInput input') || 
                                  document.querySelector('textarea[placeholder="Ask a legal question..."], input[placeholder="Ask a legal question..."]');
                if (chatInput) {{
                    const block = chatInput.closest('div[data-testid="stHorizontalBlock"]');
                    if (block) {{
                        if (event.data.active) {{
                            block.classList.add('voice-active');
                        }} else {{
                            block.classList.remove('voice-active');
                        }}
                    }}
                }}
            }}
        }} catch (e) {{
            console.error("Chatbar postMessage error:", e);
        }}
    }});

    // Run immediately and periodically
    runChatbarScript();
    if (!window.chatbarScriptIntervalID) {{
        window.chatbarScriptIntervalID = setInterval(runChatbarScript, 100);
    }}
}})();
</script>
""", unsafe_allow_html=True)
