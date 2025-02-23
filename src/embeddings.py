import sys

from logger import logging
from exception import CustomException

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def create_vector_db(PINECONE_API_KEY,index_name):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = index_name
    if True:
        index=pc.create_index(
            name=index_name,
            dimension=1536, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
        logging.info('New Vector Database created')
        return index


def add_embeddings_to_db(chunks, index_name):
    try:
        # Initialize OpenAI embedding model
        embedding_model = OpenAIEmbeddings()
        
        # Connect to Pinecone VectorStore
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding_model
        )

        # Extract texts and metadata
        texts = [chunk["chunk_text"] for chunk in chunks]
        metadatas = [{"source": chunk["source"]} for chunk in chunks]

        # Store embeddings in Pinecone
        vector_store.add_texts(texts=texts, metadatas=metadatas)

        logging.info("Embeddings added to the vector database")
    
    except Exception as e:
        logging.info(CustomException(e, sys))
        raise CustomException(e, sys)



    