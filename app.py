#from langchain.chains import RetrievalQA
#from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
#from langchain.document_loaders import TextLoader
#from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.url import UnstructuredURLLoader
#from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import util
from dotenv import load_dotenv
# pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
import torch

print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())

# FILE_NAME = "example.pdf"
FILE_DIR = "example1"
CHROMA_DB_DIR = ".chroma"
# CHROMA_DB_DIR = CHROMA_BASE_DIR + "/" + os.path.splitext(FILE_NAME)[0]

# os.environ["OPENAI_API_KEY"] = os.environ["THE_KEY"]    # set the API key
load_dotenv()   # Load keys from .env file

#st.set_page_config(page_title=' 🤖Doc Intigotator', layout='wide')
# st.write("Enter your question below and the AI will answer it.  You can also ask follow up questions.  The AI will remember the context of the conversation.")
# st.text_input("doc name", key="doc_name")
# print("doc_name: ", st.session_state.doc_name)
# FILE_NAME = st.session_state.doc_name

# openai.api_key = os.environ["OPENAI_API_KEY"]    # set the API key

## select which embeddings we want to use
# https://huggingface.co/spaces/mteb/leaderboard
EMBED_WITH = "BAAI/bge-base-en"  # Hugging face (local)
# EMBED_WITH = "text-embedding-ada-002" # Open AI default

encode_kwargs = {"normalize_embeddings": True}
match EMBED_WITH:
    case "text-embedding-ada-002":
        model_name = "'text-embedding-ada-002'"
        embeddings = OpenAIEmbeddings()
    case "BAAI/bge-base-en":
        model_name = "BAAI/bge-base-en"
        # model_name = "BAAI/bge-large-en-v1.5"
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={'device': 'cuda'}, encode_kwargs=encode_kwargs)


# if EMBED_WITH == "OpenAI":
#     model_name = "'text-embedding-ada-002'"
#     encode_kwargs = {"normalize_embeddings": True}
#     embeddings = OpenAIEmbeddings()
# elif EMBED_WITH == "HuggingFace":
#     model_name = "BAAI/bge-base-en"
#     # model_name = "BAAI/bge-large-en-v1.5"
#     encode_kwargs = {"normalize_embeddings": True}
#     embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={'device': 'cuda'}, encode_kwargs=encode_kwargs)

# db = Chroma(persist_directory=FILE_DIR + '/' + CHROMA_DB_DIR, embedding_function=embeddings)
# print("Number of existing vector DB elements", db._collection.count())

# Is there a better way to check if the DB loaded?
# Need to find a way to reembedd if using an new embedding model (old DB has old embeddings)
if True: #  or db._collection.count() < 1: # or not (os.path.exists(FILE_DIR + '/' + CHROMA_DB_DIR)):
    print("creating vector DB ...")
    # load the documents
    loader = DirectoryLoader(FILE_DIR)
    documents = loader.load()
    # Load from any URLs
    # loader = UnstructuredURLLoader(["https://arxiv.org/pdf/2106.06130.pdf"])
    # documents.append(loader.load())
    print("loaded documents: ", len(documents))
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("split documents into: ", len(texts))
    # create the vectorestore to use as the index
    # db.from_documents(texts, embeddings, persist_directory=FILE_DIR + '/' + CHROMA_DB_DIR)
    db = Chroma.from_documents(texts, embeddings) # , persist_directory=FILE_DIR + '/' + CHROMA_DB_DIR)
    print("Number of created vector DB elements", db._collection.count())

# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
# create a chain to answer questions 
# useful info:  https://stackoverflow.com/questions/75774873/openai-chatgpt-gpt-3-5-api-error-this-is-a-chat-model-and-not-supported-in-t
qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model='gpt-4-0613'), retriever=retriever, verbose=True)
chat_history = []
query = "what is the total number of AI publications?"
result = qa({"question": query, "chat_history": chat_history})

# # create a chain to answer questions 
# qa = RetrievalQA.from_chain_type(
#     llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=pipTrue)
# query = "what is the total number of AI publications?"
# result = qa({"query": query})
# the_list = retriever.get_relevant_documents(query)
# util.print_dict(result)
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
# util.print_dict(result)
