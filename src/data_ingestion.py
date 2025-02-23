import os
import sys
import re
import json


from logger import logging
from exception import CustomException

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Function to clean extracted text while keeping metadata
def clean_text(text):
    text = re.sub(r"[\s\n\t\r]+", " ", text) # Remove extra spaces and newlines
    text = re.sub(r"[\xa0\t\r]", " ", text)  # Remove non-breaking spaces & tabs
    text = re.sub(r"[-•●▪]", "", text)  # Remove bullets
    text = re.sub(r"Copyright.*?\d{4}", "", text)  # Remove page numbers & copyright
    text = re.sub(r"https?://\S+|www\.\S+", "", text) # Remove URLs
    text = text.strip()  # Remove leading/trailing spaces
    return text

def data_ingestion(directory_path,url):

    try:
        docx_data = []
        pdf_data = []
        web_data= []
        for file in os.listdir(directory_path):
            if file.endswith('.docx'):
                file_path = os.path.join(directory_path,file)
                docx_loader = UnstructuredWordDocumentLoader(file_path)
                docx_pages = docx_loader.load()
                for page in docx_pages:
                    docx_data.append({
                    "source": page.metadata.get('source'),
                    "page_number": 0,
                    "text": clean_text(page.page_content)
                })
            else:
                file_path = os.path.join(directory_path,file)
                pdf_loader = PyPDFLoader(file_path)
                pdf_pages = pdf_loader.load()
                for page in pdf_pages:
                    pdf_data.append({
                    "source": page.metadata.get('source'),
                    "page_number": page.metadata.get("page", None),  # Retain page number
                    "text": clean_text(page.page_content)
                })
        
        logging.info("All pdf and docx file are cleaned and loaded")
        
        webdata_loader = WebBaseLoader(url)
        web_pages = webdata_loader.load()
        for page in web_pages:
                    web_data.append({
                    "source": page.metadata.get('source'),
                    "page_number": 0,
                    "title":page.metadata.get("title"),
                    "description":page.metadata.get("description"),
                    "language":page.metadata.get("language"),
                    "text": clean_text(page.page_content)
                })
        logging.info("webdata cleaned and loaded")

        final_data = docx_data + pdf_data + web_data
        logging.info("All required data loaded")

        return final_data
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)



def data_chunking(final_data,chunk_size,chunk_overlap):
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,   # Adjust based on your use case
        chunk_overlap=chunk_overlap, # Helps maintain context between splits
        separators=["\n\n", "\n", " ", "."]
    )
        # Store chunked data with metadata
        chunked_data = []
        for entry in final_data:
            chunks = text_splitter.split_text(entry["text"])
            
            # Attach metadata to each chunk
            for chunk in chunks:
                chunked_data.append({
                    "chunk_text": chunk,
                    "source": entry["source"],
                    "page_number":entry["page_number"]
                })
        logging.info("Data chunking completed")
        return chunked_data
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)
         
              


# if __name__ == "__main__":
#     directory_path=os.path.join(os.getcwd(),"Data")
#     url = "https://callofduty.fandom.com/wiki/Call_of_Duty:_Mobile"
#     data = data_ingestion(directory_path,url)
#     # web_text = "\n".join([doc.page_content for doc in webdata])
#     # docx_text = "\n".join([doc.page_content for doc in docx_data])
#     # pdf_text = "\n".join([doc.page_content for doc in pdf_data])
#     final = data_chunking(data)
#     # with open("web.txt", "w", encoding="utf-8") as file:
#     #     file.write(web_text)
#     # print(docx_data)
#     # print(type(docx_data))
#     # with open("docx1.json", "w", encoding="utf-8") as file:
#     #     json.dump(docx_data, file, indent=4, ensure_ascii=False)
#     #     # file.write(docx_text)
#     # with open("pdf1.json", "w", encoding="utf-8") as file:
#     #     json.dump(pdf_data, file, indent=4, ensure_ascii=False)
#     # with open("data.json", "w", encoding="utf-8") as file:
#     #     json.dump(data, file, indent=4, ensure_ascii=False)

#     with open("chunked_data.json", "w", encoding="utf-8") as file:
#         json.dump(final, file, indent=4, ensure_ascii=False)
       
#     # print(type(web_pages))
    
#     # for doc in web_pages:
#     #     # print("Extracted Text:\n", doc.page_content[:500])  # Show first 500 characters
#     #     print("\nMetadata:\n", doc.metadata)
    
#     # for doc in docx_data:
#     #     print("Extracted Text:\n", doc.page_content[:50])  # Show first 500 characters
#     #     print("\nMetadata:\n", doc.metadata)
    
#     # for doc in pdf_data:
#     #     print("Extracted Text:\n", doc.page_content[:50])  # Show first 500 characters
#     #     print("\nMetadata:\n", doc.metadata)