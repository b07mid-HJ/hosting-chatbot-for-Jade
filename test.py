import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain.vectorstores import Chroma
import google.generativeai as genai
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

if __name__ == "__main__":

    os.environ["GOOGLE_API_KEY"]="AIzaSyA0IQJyL9MAqHv1KxmffxLCtNV_tOvp1Xs"


    model = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    temperature=0.5, 
    convert_system_message_to_human=True
    )
    import asyncio
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    v = asyncio.run(embedding_model.aembed_query("Hello World!"))
    print(f"The dimensions of the embedding vector are: {len(v)}")