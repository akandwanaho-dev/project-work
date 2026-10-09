"""Small helpers for navigating chats and app views."""
import streamlit as st

def set_view(view):
    st.session_state.active_view = view


def new_chat():
    name = f"New chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[name] = []
    st.session_state.active_chat = name
    st.session_state.messages = st.session_state.chat_sessions[name]
    st.session_state.active_view = "chat"

