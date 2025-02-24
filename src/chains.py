import sys

from logger import logging
from exception import CustomException

from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
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


def classification_query(query)-> str:

    try:
        llm = ChatOpenAI(model='gpt-4o-mini')
        prompt = f"""
    You are an expert AI agent that classifies user queries into one of two categories:
    - "RAG" if the query is about business documentation, KPIs, definitions or any general informations about game.
    - "SQL Query" if the query requires fetching structured data from a database.

    Examples:
    1. "What is our revenue trend over the last quarter?" → "SQL Query"
    2. "Explain the revenue calculation methodology." → "RAG"
    3. "How many users signed up last week?" → "SQL Query"
    4. "What does churn rate mean?" → "RAG"
    5. "What is DAU?" → "RAG"
    6. "Tell me all the filters in FTUE Dashboard" → "RAG"

    Now classify the following query:
    "{query}"
    
    Strictly Respond with only "RAG" or "SQL Query" nothing else other than these two.
    """
        response = llm.predict(prompt).strip()
        logging.info(f"Input Query classified as {response}")
        return response
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)
    

def sql_agent_creation(db):
    try:
        llm = ChatOpenAI(model='gpt-4o-mini')
        db_schema = db.get_table_info() 

        sql_prompt = f"""
            You are an expert SQL agent that **only** generates valid SQL queries.  
            Your task is to strictly return a well-formed SQL query without any extra text, explanations, or comments.  

            ### **Column Definitions & Guidelines**  
            Use the following definitions to correctly interpret the meaning of each column:

            - **calendar_date** → Represents the date of the record.  
            - **tier (country tier)** → Data is categorized into three tiers: **T1, T2, T3**. 
            - **region_s** → There are seven unique regions:
                - European Economic Area (EEA) & other European countries  
                - North America  
                - Asia & Oceania (excluding China)  
                - Russia & Commonwealth of Independent States (CIS)  
                - LATAM & the Caribbean  
                - Middle East & Africa  
                - China 
            - **platform_s** → The platform on which the game is played (**iOS, Android**).
            - **daily_active_users (DAU)** → Number of users active on a given day.
            - **revenue (Bookings)** → Total revenue generated. 
            - **coda_revenue (CODA Shop Revenue)** → Revenue specifically from CODA Shop.  
            - **payers (Spenders)** → Number of users who spend money in the game. 
            - **conversion (Converted Users)** -> User who get converted from non spender to spender
            - **installs** → Number of unique devices on which the game was installed.  
            - **register_installs** → Number of users who installed the game and registered.
            - **registers** -> No. of users who first time logged in in the game
            - **reactivation (Reactivated Users)** → Users who returned to the game after 14+ days of inactivity.
            - **session_hours (Session Duration)** → Total time spent in the game across all sessions.  
            - **session_count** → Number of unique sessions in the game. 
            
            ### **Database Schema**  
            Use the following schema for query generation:  
            **{db_schema}**  

            ### **Important Instructions**  
            - Always generate a **valid SQL query** based on the provided schema.  
            - **Do not add explanations, comments, or unnecessary text**. 
            - If a query is not possible, return:  
            ```sql
            SELECT 'Query not supported' AS result;
            """

        toolkit=SQLDatabaseToolkit(db=db,llm=llm)
        sql_agent=create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"system_message": sql_prompt}
        )
        return sql_agent
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)



    