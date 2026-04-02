import streamlit as st
from lg_backend_resume import cb, invoke_model
import uuid

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


# ************** Session Setup *********************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])
  

# ***************************************** Sidebar UI ********************************************
st.sidebar.title('ESSMORATH-AI CB')


if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header(' Conversation History')

for thread_id in st.session_state['chat_threads']:
    st.sidebar.button(str(thread_id))


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('How can i help you Today!')

if user_input:
    # User message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}  
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