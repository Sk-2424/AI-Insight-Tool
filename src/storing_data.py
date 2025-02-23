import os
import streamlit as st

from data_ingestion import data_ingestion,data_chunking
from embeddings import create_vector_db,add_embeddings_to_db

from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))


directory_path=os.path.join(os.getcwd(),"Data")
url = "https://callofduty.fandom.com/wiki/Call_of_Duty:_Mobile"


#Data Ingestion
documents = data_ingestion(directory_path,url)
docs = data_chunking(documents,500,50)

#Vector DB and Embeddings Creations
index_name = "aibot"
index=create_vector_db(PINECONE_API_KEY,index_name)
add_embeddings_to_db(docs,index_name)

