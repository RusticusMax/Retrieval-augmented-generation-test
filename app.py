from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import os
import streamlit as st
import util
from dotenv import load_dotenv
import openai

#test_dict = { 'test1': 'value1', 'test2': {'test21': 'value21', 'test22': 'value22', 'value23': ['test221', 'value221']}, 'test3': 'value3' }
#util.print_dict(test_dict)
#quit()

# FILE_NAME = "example.pdf"
FILE_DIR = "example1"
CHROMA_DB_DIR = ".chroma"
# CHROMA_DB_DIR = CHROMA_BASE_DIR + "/" + os.path.splitext(FILE_NAME)[0]

# os.environ["OPENAI_API_KEY"] = os.environ["THE_KEY"]    # set the API key
load_dotenv()   # LOad keys from .env file

#st.set_page_config(page_title=' 🤖Doc Intigotator', layout='wide')
# st.write("Enter your question below and the AI will answer it.  You can also ask follow up questions.  The AI will remember the context of the conversation.")
# st.text_input("doc name", key="doc_name")
# print("doc_name: ", st.session_state.doc_name)
# FILE_NAME = st.session_state.doc_name

# openai.api_key = os.environ["OPENAI_API_KEY"]    # set the API key
## select which embeddings we want to use
embeddings = OpenAIEmbeddings()
if not os.path.exists(FILE_DIR + '/' + CHROMA_DB_DIR):
    print("creating ...")
    # No embeddings exist, create them
    # loader = PyPDFLoader(FILE_DIR + "/" + FILE_NAME)
    loader = DirectoryLoader(FILE_DIR)
    documents = loader.load()
    # loader = UnstructuredURLLoader(["https://arxiv.org/pdf/2106.06130.pdf"])
    # documents.append(loader.load())
    print("loaded documents: ", len(documents))
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings, persist_directory=FILE_DIR + '/' + CHROMA_DB_DIR)
else:
    print("exists ... loading")
    # This is dumb I should be anble to tell if it loaded, and just load into the DB.
    # Instead i test for the directory.
    db = Chroma(persist_directory=FILE_DIR + '/' + CHROMA_DB_DIR, embedding_function=embeddings)

# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
# create a chain to answer questions 
# useful info
# https://stackoverflow.com/questions/75774873/openai-chatgpt-gpt-3-5-api-error-this-is-a-chat-model-and-not-supported-in-t
qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model='gpt-4-0613'), retriever=retriever)
chat_history = []
query = "what is the total number of AI publications?"
result = qa({"question": query, "chat_history": chat_history})

# # create a chain to answer questions 
# qa = RetrievalQA.from_chain_type(
#     llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
# query = "what is the total number of AI publications?"
# result = qa({"query": query})
# the_list = retriever.get_relevant_documents(query)
#print_dict(result)
print("Question: ", result["question"])
print("Answer: ", result["answer"])
print("")

chat_history = [(query, result["answer"])]
query = "What is this number divided by 2?"
result = qa({"question": query, "chat_history": chat_history})
print("Question: ", result["question"])
print("Answer: ", result["answer"])
print("chat_history: ", chat_history)
print("------------------")
