import streamlit as st
from rag_chatbot import RAGChatbot

# Initialize the chatbot
if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = RAGChatbot()

chatbot = st.session_state['chatbot']

st.title('LLama Chatbot with RAG and File Upload')

# File uploader
st.header("Upload your documents")
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=['pdf', 'txt'],
    accept_multiple_files=True
)

if st.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            chatbot.ingest_documents(uploaded_files)
        st.success("Documents processed successfully!")
    else:
        st.warning("Please upload at least one document.")

st.header("Ask a Question")
question = st.text_input("Your question:")
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

if st.button("Submit"):
    # Update the model's temperature
    chatbot.model.temperature = temperature

    # Get the response
    with st.spinner("Generating answer..."):
        response = chatbot.ask(question)
    st.write(response)

