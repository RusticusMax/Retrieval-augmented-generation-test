from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

def print_dict(dict_x, depth=0):
    for key, value in dict_x.items():
        if type(value) is dict:
            print_dict(value, depth + 1)
        else:
            print("   " * depth, key + ":")
    print("-----------------------------------")
    for key, value in dict_x.items():
        if type(value) is dict:
            print_dict(value, depth + 1)
        else:
            print("   " * depth, key + ":", value)

# test_dict = { 'test1': 'value1', 'test2': {'test21': 'value21', 'test22': 'value22', 'value23': {'test221': 'value221'}}, 'test3': 'value3' }
# print_dict(test_dict)
# quit()

os.environ["OPENAI_API_KEY"] = os.environ["THE_KEY"]    # set the API key

loader = PyPDFLoader("materials/example.pdf")
documents = loader.load()
# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# select which embeddings we want to use
embeddings = OpenAIEmbeddings()
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
# create a chain to answer questions 
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
query = "what is the total number of AI publications?"
result = qa({"query": query})
retriever.get_relevant_documents(query)
print_dict(result)

