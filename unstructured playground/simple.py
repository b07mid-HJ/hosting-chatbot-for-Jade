import os
from typing import Any
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel
from unstructured.partition.docx import partition_docx
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document as dc
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
#model preperation
os.environ["GOOGLE_API_KEY"]="AIzaSyBR03q5DwkuBxfeCCja-b-j1hGeI0NRIGE"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__fe037fb4637146b8bdcf8355ea7c7b79"
os.environ["LANGCHAIN_PROJECT"] = "Jade-chatbot"

model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.5, 
                                convert_system_message_to_human=True
                            )
    
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
from xml.etree.ElementTree import Element, SubElement, tostring
from docx import Document

def table_to_xml(table):
    root = Element('table')
    for row in table.rows:
        row_element = SubElement(root, 'row')
        for cell in row.cells:
            cell_element = SubElement(row_element, 'cell')
            cell_element.text = cell.text.strip()  # Use cell.text directly
    return root

def get_paragraphs_before_tables(doc_path):
    doc = Document(doc_path)
    paragraphs_and_tables = []

    for element in doc.element.body:
        if element.tag.endswith('p'):
            last_paragraph = element
        elif element.tag.endswith('tbl'):
            # Find the table object corresponding to this element
            for table in doc.tables:
                if table._element == element:
                    if last_paragraph is not None:
                        xml_root = table_to_xml(table)
                        xml_str = tostring(xml_root, encoding='unicode')
                        langchain_document = "Title: "+ last_paragraph.text + "Content: " + xml_str
                        paragraphs_and_tables.append(langchain_document)
                    break

    return paragraphs_and_tables

# Example usage:
docx_file_path = "./multi+parent/rep.docx"  # Path to your .docx file
table_elements = get_paragraphs_before_tables(docx_file_path)


# Prompt
prompt_text = """You are an assistant tasked with summarizing tables.\ 
Summerize it and keep the most important information.
Also you must put the title at the beginning of the summary. \
If you encounter any table name that has Sc. that means it's a senario \
Give a summary of the table. Table chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

tables = table_elements
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

# Apply to tables
# tables = [i.text for i in table_elements]
# print(len(tables))
# table_summaries = []
# for t in tables:
#     try:
#         table_summaries.append(summarize_chain.invoke(t))
#         print("success")
#     except Exception as e:
#             print(f"An error occurred: {e}")

# The vectorstore to use to index the child chunks
vectorstore1 = Chroma(collection_name="table_summaries", embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

# The storage layer for the parent documents
store1 = InMemoryStore()
id_key1 = "doc_id"

# The retriever (empty to start)
retriever1 = MultiVectorRetriever(
    vectorstore=vectorstore1,
    docstore=store1,
    id_key=id_key1,
)


# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    dc(page_content=s, metadata={id_key1: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever1.vectorstore.add_documents(summary_tables)
retriever1.docstore.mset(list(zip(table_ids, tables)))

# Table Retrieval

with open("./recursive/output.txt",encoding='utf-8') as f:
    state_of_the_union = f.read()

#text spliter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=5000,
    chunk_overlap=0 ,
    separators=["\n\n","\n", " ",""],
)

texts = text_splitter.create_documents([state_of_the_union])
vectorstore2 = Chroma(collection_name="child_chunks", embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

# The storage layer for the parent documents
store2 = InMemoryStore()
id_key2 = "doc_id"
# The retriever (empty to start)
retriever2 = MultiVectorRetriever(
    vectorstore=vectorstore2,
    byte_store=store2,
    id_key=id_key2,
)

doc_ids = [str(uuid.uuid4()) for _ in texts]

child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

sub_docs = []
for i, doc in enumerate(texts):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key2] = _id
    sub_docs.extend(_sub_docs)

retriever2.vectorstore.add_documents(sub_docs)
retriever2.docstore.mset(list(zip(doc_ids, texts)))

from langchain_core.runnables import RunnablePassthrough

# Prompt template
template = """
You are a Private-Public Partnership (PPP) feasibility expert. You are tasked with answering questions the feasibility of a PPP project.\n
Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
from langchain.retrievers import EnsembleRetriever
ensemble=EnsembleRetriever(retrievers=[retriever1,retriever2],weights=[0.5,0.5])
# RAG pipeline
chain = (
    {"context": ensemble, "question": RunnablePassthrough()}
    | prompt
    | model
    |StrOutputParser()
)


while True:
    question = input("Please enter your question (or 'quit' to stop): ")
    if question.lower() == 'quit':
        break

    print(f"Question: {question}")
    result = chain.invoke(question)
    print(f"Answer (ensemble): {ensemble.invoke(question)}")
    print("-------------------------")
    print(f"Answer (retriever): {retriever1.invoke(question)}")
    print("-------------------------")
    print(f"Answer (vectorstore2): {retriever2.invoke(question)}")
    print("-------------------------")
    print(f"Context: {result}")
    print("=========================")