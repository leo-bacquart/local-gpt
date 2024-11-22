from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.embeddings import FastEmbedEmbeddings
from langchain.schema import HumanMessage
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class RAGChatbot:
    def __init__(self):
        # Initialize the model
        self.temperature = 0.5
        self.model = ChatOllama(
            model='llama3.2',
            temperature=self.temperature,
        )
        self.vectorstore = None
        self.chain = None

        # Define the prompt template
        self.prompt = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
            maximum and keep the answer concise.

            Question: {question}
            Context: {context}
            Answer:
            """
        )

        # Initialize the memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def ingest_documents(self, uploaded_files):
        import os
        from langchain.docstore.document import Document
        import PyPDF2

        # Load and process the documents
        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                # Read PDF content
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                metadata = {"source": uploaded_file.name}
                documents.append(Document(page_content=text, metadata=metadata))
            else:
                # Assume text file
                text = uploaded_file.read().decode('utf-8')
                metadata = {"source": uploaded_file.name}
                documents.append(Document(page_content=text, metadata=metadata))

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        docs = filter_complex_metadata(docs)

        # Create embeddings and store them in the vectorstore
        embeddings = FastEmbedEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=None  # We're not persisting to disk
        )

        # Set up the retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3
            },
        )

        # Set up the conversational retrieval chain
        self.chain = ConversationalRetrievalChain(
            retriever=self.retriever,
            combine_docs_chain_kwargs={'prompt': self.prompt},
            llm=self.model,
            memory=self.memory
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please upload documents first."
        response = self.chain({"question": query})
        return response['answer']