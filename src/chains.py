import sys

from logger import logging
from exception import CustomException

from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore

import streamlit as st

def create_retriever(index_name,embeddings):
    docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
    logging.info("Retriever is created")
    return retriever


def create_rag_chain(retriever):
    try:
        llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.2, max_tokens=500)

        ### Contextualize Question ###
        contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt)

        #General Question Answer prompt
        system_prompt = (
            """You are an AI assistant specialized in answering questions related to the Call of Duty Mobile (CODM) game.  

            ### Guidelines:
            - Always use the retrieved documents to provide answers.  
            - If the required information is not available, respond with: **"The required information is not available."**  
            - Engage in normal conversation, but **do not answer questions beyond the retrieved information.**  
            - Provide concise answers (at least **4-5 lines**) unless the user requests a detailed response.  
            - If the user asks for **detailed information,** provide the maximum relevant details available.  

            ### Game Structure Understanding:
            - **Game Types:** CODM includes different game types such as **Multiplayer, Battle Royale, and Zombies Mode.**  
            - **Game Modes:** Each game type consists of multiple game modes.  
            - **Maps:** Different maps exist for different game types.  
            - **KPIs (Key Performance Indicators):** Metrics such as **DAU (Daily Active Users), Bookings, Spenders, Conversion Rate, etc.** are used to measure performance.  

            ### Example Questions & Answers:
            **Q:** What are the different game types in CODM?  
            **A:** There are three game types: **Multiplayer, Battle Royale, and Zombies Mode.**  

            **Q:** Which KPI can I use to understand revenue in CODM?  
            **A:** **Bookings** is the key KPI used to track revenue in CODM.  

            ---  
            **Context:** {context}  """)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        #creating Final chain
        bot_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        logging.info("Final Rag Chain is created")
        return bot_chain
    except Exception as e:
        # print(f"Error in create_rag_chain: {e}")
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)


def ask_question(input_data,bot_chain):
    try:   
        response = bot_chain.invoke(input_data)
        logging.info("Got the response of the query")
        return response['answer']
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)
    
def clear_memory(memory):
    memory.clear()
    logging.info("Memory is cleared")