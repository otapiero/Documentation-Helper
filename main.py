from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("LangChain Documentation Bot 🤖 📚 📖")
st.subheader("Ask me anything about LangChain")


prompt = st.text_input("prompt", placeholder="enter your question here")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "bot_response_history" not in st.session_state:
    st.session_state["bot_response_history"] = []


def create_source_string(sources: Set[str]):
    if not sources:
        return ""
    sources = list(sources)
    sources = [s.replace("\\", "/") for s in sources]
    sources.sort()
    source_string = "Source: \n"
    for i, source in enumerate(sources):
        source_string += f"{i+1}. {source} \n"
    return source_string


if prompt:
    with st.spinner("Thinking..."):
        response = run_llm(prompt)
        sources = set([r.metadata["source"] for r in response["source_documents"]])

        formatted_response = (
            f"{response['result']} \n\n {create_source_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["bot_response_history"].append(formatted_response)

if st.session_state["bot_response_history"]:
    for user_prompt, bot_response in zip(
        st.session_state["user_prompt_history"],
        st.session_state["bot_response_history"],
    ):
        message(user_prompt, is_user=True)
        message(bot_response, is_user=False)
