##########################################################
## Author:  Divya Acharya
## Project: V-Doc
## File: retriever.py (Retrieval Logic)
## Date: Aug 27, 2024
## Purpose:This file likely contains the logic for retrieving relevant documents from the vector database based on user queries.
## Key Functions:
## Querying the vector database (Qdrant) for relevant document embeddings.
## Returning the most relevant documents or context to the AI model for response generation.
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.embeddings import SentenceTransformerEmbeddings
import time

# Initialize the embedding model using SentenceTransformer
model_embedding = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)

# Set the URL for the hosted Qdrant instance
hosted_url = "http://localhost:6333"

# Initialize the Qdrant client to interact with the database
client_to_use = QdrantClient(url=hosted_url, prefer_grpc=False)

# Print to confirm the client initialization
print("Testing retrieval")
print("client: ")
print(client_to_use)
print("------------------------------------")

# Initialize the Qdrant vector store with the client, embeddings, and collection name
vector_db = Qdrant(
    client=client_to_use, embeddings=model_embedding, collection_name="VDoc_db_store"
)

# Print to confirm the vector database setup
print("database: ")
print(vector_db)
print("------------------------------------")

# Take the query from user input
question = input("Enter your query: ")

# Start timing the retrieval process
start_time = time.time()

# Perform a similarity search with the query, retrieving the top 2 results
docs = vector_db.similarity_search_with_score(query=question, k=2)

# Calculate and print the time taken for the search
end_time = time.time()
print(f"Time taken to retrieve: {end_time - start_time} seconds")

# Iterate through the retrieved documents and print their details
for i in docs:
    doc, score = i
    response_dict = {
        "score": score,
        "content": doc.page_content,
        "metadata": doc.metadata,
    }
    # Print each key-value pair in the response dictionary
    for type, value in response_dict.items():
        print(f"{type}: {value}")