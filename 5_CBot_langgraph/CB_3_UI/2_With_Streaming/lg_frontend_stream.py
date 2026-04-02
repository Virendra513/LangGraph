import streamlit as st
from lg_backend_stream import cb, invoke_model

CONFIG = {'configurable': {'thread_id': 'thread_id'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type Here')

if user_input:
    # User message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

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