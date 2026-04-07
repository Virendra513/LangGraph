import streamlit as st
from lg_backend_resume import cb, invoke_model, retrive_all_threads
import uuid
from langchain_core.messages import HumanMessage
# ************************* utility functions

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = cb.get_state(config={'configurable': {'thread_id': thread_id}})
    
    if not state or not hasattr(state, "values"):
        return []
    
    return state.values.get('messages', [])

def get_first_human_message_10_words(cb ,thread_id: str) -> str:
    # Build config using thread_id
    config = {"configurable": {"thread_id": thread_id}}

    # Get state from LangGraph memory
    state = cb.get_state(config)

    if not state or "messages" not in state.values:
        return ""

    messages = state.values["messages"]

    # Find first human/user message
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return " ".join(m["content"].split()[:10])

        elif hasattr(m, "type") and m.type == "human":
            return " ".join(m.content.split()[:10])

    return ""  # if no human message found

# ************** Session Setup *********************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()

add_thread(st.session_state['thread_id'])
  

# ***************************************** Sidebar UI ********************************************
st.sidebar.title('ESSMORATH-AI CB')


if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('Conversation History')

for thread_id in st.session_state['chat_threads']:
    #if st.sidebar.button(str(thread_id)):
    label = get_first_human_message_10_words(cb, thread_id)
    if not label:
        label = str(st.session_state['thread_id'])
    if st.sidebar.button(label, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages=load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role ='assistant'
            temp_messages.append({'role': role, 'content': msg.content})    

        st.session_state['message_history'] = temp_messages


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('How can i help you Today!')

if user_input:
    # User message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    #CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']},
              "metadata":{
                  "thread_id" : st.session_state['thread_id']
              },
              "run_name": "chat_cb"
              }

    # AI message
    with st.chat_message('assistant'):

        def token_stream():
            raw = invoke_model(st.session_state['message_history']) 
            for chunk in raw:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        ai_message = st.write_stream(token_stream())  
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Sync LangGraph memory
    cb.invoke(                                         
        {"messages": [{"role": "user", "content": user_input},
                      {"role": "assistant", "content": ai_message}]},
        config=CONFIG,
    )