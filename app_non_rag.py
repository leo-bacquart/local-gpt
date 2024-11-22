import streamlit as st
from langchain.llms import Ollama

llm = Ollama(model='llama3.2')

st.title('LLama chatbot without RAG')

question = st.text_input("Ask a question:")
#TODO: Implémenter la température
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

if st.button("Submit"):
    response = llm.invoke(question)
    st.write(response)
