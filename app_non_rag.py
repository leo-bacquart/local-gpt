import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

st.title('LLama chatbot without RAG')

question = st.text_input("Ask a question:")
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

if st.button("Submit"):
    llm = ChatOllama(
        model='llama3.2',
        temperature=temperature,
        disable_streaming=False
    )

    response_placeholder = st.empty()

    class StreamHandler(BaseCallbackHandler):
        def __init__(self):
            self.text = ""

        def on_llm_new_token(self, token: str, **kwargs):
            self.text += token
            response_placeholder.markdown(self.text)


    handler = StreamHandler()

    message = [HumanMessage(content=question)]

    llm(message, callbacks=[handler])