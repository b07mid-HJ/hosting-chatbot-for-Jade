import os
from langchain.vectorstores import chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# This is a long document we can split up.
with open("./recursive/output.txt",encoding='utf-8') as f:
    state_of_the_union = f.read()

#text spliter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=3000,
    chunk_overlap=0 ,
    separators=["\n\n","\n", " ",""],
)

texts = text_splitter.create_documents([state_of_the_union])

os.environ["GOOGLE_API_KEY"]="AIzaSyA0IQJyL9MAqHv1KxmffxLCtNV_tOvp1Xs"


model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.1, 
                                convert_system_message_to_human=True
                            )
    
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Create a vector database from the texts
vectordb = chroma.Chroma.from_documents(texts, GoogleGenerativeAIEmbeddings(model="models/embedding-001"), persist_directory="./recursive/vecdb")
vectordb.persist()
