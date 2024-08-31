##########################################################
## Author:  Divya Acharya
## Project: V-Doc
## File: ingest.py (Knowledge Base and Vector Database Creation))
## Date: Aug 27, 2024
#Purpose: This file likely handles the ingestion of documents into the system, converting them into embeddings, and storing those embeddings in a vector database like Qdrant.
#Key Functions:
#Loading documents from PDFs or other sources.
#Splitting the documents into smaller chunks for efficient processing.
#Embedding the document chunks into vector representations.
#Storing the vectorized data in a vector database like Qdrant.
# imports
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
import time


# Embedding model from huggingface
embedding_model = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)

# print(embedding_model)

# Initialize a DirectoryLoader to load all PDF files from the 'data/' directory recursively
directory_loader = DirectoryLoader(
    "data/", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader
)

# Load the documents from the specified directory
loaded_documents = directory_loader.load()

# Create a RecursiveCharacterTextSplitter to split documents into chunks of 1000 characters with a 100-character overlap
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

# Split the loaded documents into smaller text chunks using the defined splitter
splitted_texts = recursive_text_splitter.split_documents(loaded_documents)

# Qdrant Vector DB URL
hosted_url = "http://localhost:6333"

print("Creating Vector Data base for the stored pdf files of VDoc's Knowledge base.")

# Time stamp to mark start of the DB store process
start_time = time.time()

# Create a Qdrant vector store from the split text documents, using specified embeddings
qdrant = Qdrant.from_documents(
    documents=splitted_texts,
    collection_name="VDoc_db_store",
    embedding=embedding_model,
    url=hosted_url,
    prefer_grpc=False,
)

# Time stamp to mark end
end_time = time.time()

# printing difference of both times
print(f"Time taken to create DB store: {end_time - start_time} seconds")

print("Vector DB Successfully Created for the VDoc Knowledge Base.!")
