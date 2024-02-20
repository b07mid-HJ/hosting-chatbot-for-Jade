import os
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"]="AIzaSyA0IQJyL9MAqHv1KxmffxLCtNV_tOvp1Xs"


model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.1, 
                                convert_system_message_to_human=True
                            )
    
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


#load vector database
vectordb = chroma.Chroma(persist_directory="./recursive/vecdb",embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG pipeline
chain = (
    {"context": vectordb.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    
)
# Invoke the chain and store the result
result = chain.invoke("how long is this ppp contract?")
print(vectordb.as_retriever().invoke("how long is this ppp contract?"))
# Print the context
print(result)