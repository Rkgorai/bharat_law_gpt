import os
from typing import Generator, Dict, Any

def stream_agent_execution(agent, formatted_history, use_voice: bool) -> Generator[Dict[str, Any], None, None]:
    """
    Executes the LangGraph agent stream and yields pure python dictionaries
    representing the UI events, abstracting away LangChain mechanics.
    """
    try:
        for event in agent.stream({"messages": formatted_history}):
            for node_name, value in event.items():
                if "messages" in value:
                    messages = value["messages"]
                    if not isinstance(messages, list):
                        messages = [messages]
                        
                    for msg in messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                yield {"type": "status", "data": f"⚙️ Running `{tool_call.get('name', 'tool')}`"}
                                
                        elif getattr(msg, "type", "") == "tool":
                            yield {"type": "status", "data": f"✅ Extracted data from `{msg.name}`"}
                            
                            # Catch draft saved notification directly from payload
                            if "<DRAFT_SAVED>" in str(msg.content):
                                draft_text = str(msg.content).split("<DRAFT_SAVED>")[1].strip()
                                yield {"type": "draft", "content": draft_text}
                                    
                        elif getattr(msg, "type", "") == "ai" and msg.content:
                            yield {"type": "token", "data": msg.content}
                            
        yield {"type": "end"}
        
    except Exception as e:
        yield {"type": "error", "data": str(e)}
