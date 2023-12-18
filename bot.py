import openai
import streamlit as st
from streamlit_chat import message

openai_api_key = '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'

avatar = {
    'user': 'lorelei',
    'assistant': 'pixel-art'
}


st.title("Chatbot de seguros :tada:")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hola soy tu robot personal ¿En qué te puedo ayudar hoy?"}]

with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([10, 2])
    user_input = a.text_input(
        key='user_message',
        label="Your message:",
        placeholder="Escribe aquí tu mensaje",
        label_visibility='collapsed'
        )
    b.form_submit_button("\>", use_container_width=True)

placeholder = st.empty()
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with placeholder.container():
        with st.spinner('Cargando...'):
            response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.messages,
                        max_tokens = 400,
                        temperature = 1.0
                        )
            msg = response.choices[0].message.content

    st.session_state.messages.append({'role': 'assistant', 'content': msg})

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for i, msg in enumerate(reversed(st.session_state.messages)):
    message(msg["content"], 
            is_user= msg["role"] == "user", 
            key=str(i), 
            avatar_style=avatar[msg["role"]])