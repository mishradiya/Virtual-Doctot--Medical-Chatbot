##########################################################
## Author:  Divya Acharya
## Project: V-Doc
## File: main.py (Application Logic)
## Date: Aug 27, 2024
## Purpose:This file likely contains the main logic for the VDoc application. It handles user interactions, routes the queries through the appropriate AI models, retrieves data from the vector database, and provides responses.
#Key Functions:
#Setting up the FastAPI server for handling web requests.
#Integrating the retrieval-based query processing.
#Querying the vector database and passing relevant information to the language model.
#Generating and returning responses to the user.
##########################################################
# Basic imports for system operations, JSON handling, and time tracking
import os
import json
import time

# FastAPI imports for handling web responses, templates, and forms
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Request, Form, Response

# LangChain imports for template, language model, and retrieval-based QA chain
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings

# Qdrant imports for vector store and client handling
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant

# Initialize a FastAPI application
rag_app = FastAPI()

# Set up the Jinja2 template directory for HTML rendering
templates = Jinja2Templates(directory="templates")

# Mount the static files directory for serving static content
rag_app.mount("/static", StaticFiles(directory="static"), name="static")

# Specify the local model file for the language model
# the model is downloaded from this link: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main
local_llm = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"

# Define configuration settings for the language model
config = {
    "threads": int(os.cpu_count() / 2),  # Use half of the available CPU cores
    "stream": True,  # Enable streaming mode for token generation
    "max_new_tokens": 1024,  # Maximum number of tokens to generate
    "repetition_penalty": 1.1,  # Repetition penalty to prevent looping
    "top_p": 0.9,  # Top-p sampling parameter
    "context_length": 2048,  # Maximum context length for input tokens
    "top_k": 50,  # Top-k sampling parameter
    "temperature": 0.2,  # Sampling temperature for randomness
}

# Initialize the language model with the specified configuration
llm_in_use = CTransformers(model=local_llm, model_type="llama", lib="avx2", **config)

# Print the initialized LLM details
print("Initializing Large Language Model .....")
print(llm_in_use)

# Define the instruction template for the prompt, used by the model
INSTRUCTION_TEMPLATE = """
<s>[INST] 
Assume the character of a highly renowned doctor who is specialized in all fields of medicine and known for providing precise and compassionate care. This expert is dedicated to offering reliable answers to all healthcare-related questions.

Using the provided context, answer the user's question as accurately and concisely as possible.
If the answer is unknown, simply state that you do not know—avoid fabricating any information.

Context: {context}
Question: {question}

Respond with the most helpful answer you can, and nothing else. These responses are for healthcare purposes, so it is crucial to provide relevant and accurate information. 
[/INST]
"""


# Initialize the embedding model to convert text into vectors
model_embedding = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)

# Specify the URL for the hosted Qdrant instance
hosted_url = "http://localhost:6333"

# Initialize the Qdrant client to interact with the vector database
vdoc_vector_db_client = QdrantClient(url=hosted_url, prefer_grpc=False)

# Create the Qdrant vector store using the client and embedding model
vdoc_db = Qdrant(
    client=vdoc_vector_db_client,
    embeddings=model_embedding,
    collection_name="VDoc_db_store",
)

# Set up the retriever to fetch relevant documents from the vector store
vdoc_retriever = vdoc_db.as_retriever(search_kwargs={"k": 1})

# Create a prompt template for the QA process
prompt = PromptTemplate(
    template=INSTRUCTION_TEMPLATE, input_variables=["context", "question"]
)


# Define the FastAPI route for the home page, serving an HTML template
@rag_app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Define the FastAPI route to handle AI queries, using a POST request
@rag_app.post("/run_ai")
async def run_ai(question: str = Form(...)):
    # Time stamp to mark the start of the process
    start_time = time.time()

    # Define chain type parameters for the retrieval-based QA pipeline
    type_chains = {"prompt": prompt}

    # Initialize the RetrievalQA pipeline with the language model and retriever
    pipeline = RetrievalQA.from_chain_type(
        llm=llm_in_use,
        chain_type="stuff",
        retriever=vdoc_retriever,
        return_source_documents=True,
        chain_type_kwargs=type_chains,
        verbose=True,
    )

    # Execute the pipeline with the user's question and obtain the response
    response = pipeline(question)

    # Time stamp to mark the end of the process
    end_time = time.time()

    # Calculate and print the total time taken for the response generation
    net_time = end_time - start_time
    print(f"Time taken to generate response: {net_time} seconds")

    # Print each key-value pair in the response dictionary
    for type, value in response.items():
        print(f"{type}: {value}")

    # Extract the answer and source document details from the response
    answer = response["result"]
    source_file = response["source_documents"][0]
    file = response["source_documents"][0].metadata["source"]
    context = source_file.page_content

    # Encode the response data into a JSON format
    response_data = jsonable_encoder(
        json.dumps(
            {
                "answer": answer,
                "retrieved_context": context,
                "file": file,
                "response_time": net_time,
            }
        )
    )

    # Return the encoded response data as an HTTP response
    response = Response(response_data)
    return response