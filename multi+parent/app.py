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
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#model preperation
os.environ["GOOGLE_API_KEY"]="AIzaSyBR03q5DwkuBxfeCCja-b-j1hGeI0NRIGE"


model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.5, 
                                convert_system_message_to_human=True
                            )
    
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Table Indexing/Retrieval
# Get elements
raw_tables = partition_docx(
    filename="./multi+parent/rep.docx",
    infer_table_structure=True,
)
# Create a dictionary to store counts of each type
category_counts = {}

for element in raw_tables:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

# Unique_categories will have unique elements
unique_categories = set(category_counts.keys())
print(unique_categories)

class Element(BaseModel):
    type: str
    text: Any


# Categorize by type
categorized_elements = []
for element in raw_tables:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# Tables
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

# Prompt
prompt_text = """You are an assistant tasked with summarizing tables, when summarizing a table you should keep all the numerical values for it and don't draw conclutions. \ 
Give a summary of the table. Table chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

tables = [i.text for i in table_elements]
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
vectorstore1 = Chroma(collection_name="table_summaries", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

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
    Document(page_content=s, metadata={id_key1: table_ids[i]})
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
    chunk_size=3000,
    chunk_overlap=0 ,
    separators=["\n\n","\n", " ",""],
)

texts = text_splitter.create_documents([state_of_the_union])
vectorstore2 = Chroma(collection_name="child_chunks", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

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
template = """Answer the question based only on the following context, which can include text and tables:
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