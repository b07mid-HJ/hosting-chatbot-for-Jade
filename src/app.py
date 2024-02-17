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

if "history" not in st.session_state:
    st.session_state.history = []

if __name__ == "__main__":

    os.environ["GOOGLE_API_KEY"]="AIzaSyA0IQJyL9MAqHv1KxmffxLCtNV_tOvp1Xs"


    model = ChatGoogleGenerativeAI(
                                    model="gemini-pro", 
                                    temperature=0.5, 
                                    convert_system_message_to_human=True
                                )
        
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    vectorstore = Chroma(persist_directory=r"C:\Users\Bohmid\Desktop\hosting chatbot\db",embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    

    metadata_field_info = [
    AttributeInfo(
            name="type",
            description="The are the titles for where these paragraphs are located in the document",
            type="string",
        )
    ]

    document_content_description = "Provision of PPP transaction advisory services for an office complex for Malawi Investment and Trade Center (MITC)"
    retriever = SelfQueryRetriever.from_llm(
        model,
        vectorstore,
        document_content_description,
        metadata_field_info,
    )

    
    ensemble=EnsembleRetriever(retrievers=[retriever,vectorstore.as_retriever()],weights=[0.5,0.5])

    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # RAG pipeline
    chain = (
        {"context": ensemble, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    st.title("Prieview Jade Chatbot: Malawi Investment and Trade Center (MITC)")
    for msg in st.session_state.history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    prompt = st.chat_input("Say something")
    if prompt:
        st.session_state.history.append({
            'role':'user',
            'content':prompt
        })
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner('💡Thinking'):
            final=chain.invoke(prompt)
            st.session_state.history.append({
                'role':'Assistant',
                'content':final
            })

            with st.chat_message("Assistant"):
                st.markdown(final)