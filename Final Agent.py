import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from classes.config_manager import ConfigManager
from classes.chromadb_retriever import ChromaDBRetriever
import openai
import uvicorn
import asyncio  # Optimized: Using asyncio for non-blocking operations
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure your OpenAI API key securely via environment variable.
# Do not commit your actual API key to source code.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise Exception("Please set the OPENAI_API_KEY environment variable")
openai.api_key = openai_api_key

# Initialize configuration
CONFIG_FILE = r"C:/Users/vikramp/OneDrive - School Health Corporation/Desktop/Assignment Files CISC 691/A04/A04_A-Simple-RAG-Design-Implementation/hu_sp25_691_a03/config.json"
config = ConfigManager(CONFIG_FILE)  # Use ConfigManager for configuration loading
vectordb_directory = r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A04\A04_A-Simple-RAG-Design-Implementation\hu_sp25_691_a03\data\vectordb"
collection_name = "product_reviews"  # Collection name for ChromaDB

# Initialize the FastAPI application
app = FastAPI(title="Optimized RAG Virtual Agent API")

# Define the request and response models
class QueryRequest(BaseModel):
    query: str
    session_id: str = None  # Optional session identifier for context retention

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    follow_up: list

# ----------------------------------------------------------------------------------------------
#  USING EXISTING RAG PIPELINE FUNCTIONS BUILT FOR A04 (Updated for Enhanced Error Handling)
# ----------------------------------------------------------------------------------------------
async def retrieve_relevant_documents(query: str, top_k: int = 5):
    """
    Asynchronously retrieve relevant documents from the vector database built using amazon.csv.
    """
    retriever = ChromaDBRetriever(
        vectordb_dir=vectordb_directory,
        embedding_model_name=config.get("embedding_model_name"),
        collection_name=collection_name,
        score_threshold=float(config.get("retriever_min_score_threshold"))
    )
    # Retrieve results using the vector database
    results = retriever.query(query, top_k=top_k)
    return results

def calculate_confidence(retrieval_results):
    """
    Calculate a composite confidence score based on the retrieved document scores.
    This version computes the average score and then adjusts it by a penalty based
    on the variance in the scores. A low variance (i.e., high consistency among scores)
    yields less penalty (closer to 1.0), while a high variance reduces the overall confidence.
    
    Returns a value (percentage) between 0 and 100.
    """
    if not retrieval_results:
        return 0.0

    # Extract scores from the documents
    scores = [doc["score"] for doc in retrieval_results]
    
    # Compute the average score
    avg_score = sum(scores) / len(scores)
    
    # Compute the variance of the scores
    variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
    
    # For scores between 0 and 1, maximum variance (when half are 0 and half are 1) is 0.25.
    penalty_factor = 1 - (variance / 0.25) if variance < 0.25 else 0
    
    # Adjust the average score by the penalty factor
    adjusted_confidence = avg_score * penalty_factor
    
    # Return the adjusted confidence as a percentage
    return round(adjusted_confidence * 100, 2)

def integrate_structured_data(retrieval_results):
    """
    Combine retrieved review text with structured attributes (rating, price, and product name)
    into a single context string.
    """
    context = ""
    for doc in retrieval_results:
        try:
            rating = doc["structured_data"].get("rating", "N/A")
            price = doc["structured_data"].get("price", "N/A")
            context += f"Product: {doc['product_name']}\nReview: {doc['text']}\nRating: {rating}/5, Price: ${price}\n\n"
        except KeyError as key_error:
            logging.error(f"Missing expected key in document: {key_error}. Document: {doc}")
    return context.strip()

async def generate_gpt4_response(context: str, query: str) -> str:
    """
    Asynchronously generate a response using GPT-4 based on the integrated context and query.
    Implements a retry mechanism with exponential backoff for robust error handling.
    """
    final_prompt = f"Context: {context}\nQuestion: {query}"
    max_retries = 3
    backoff = 1  # Initial backoff in seconds

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Ensure this model is supported or update to a valid model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            answer = response.choices[0].message["content"].strip()
            return answer
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} - Error generating GPT-4 response: {e}")
            await asyncio.sleep(backoff)
            backoff *= 2

    return "I'm sorry, I'm having trouble generating a response right now. Please try again later."

def generate_follow_up_questions(answer: str) -> list:
    """
    Generate follow-up questions to maintain multi-turn interaction.
    """
    return [
        "Would you like more details about the product?",
        "Do you want to compare this with similar items?"
    ]

def format_answer(answer_text: str) -> str:
    """
    Format the answer to ensure each product review is clearly separated by a blank line.
    """
    blocks = answer_text.split("\n\n")
    formatted_blocks = []
    for block in blocks:
        single_line = " ".join(block.splitlines())
        formatted_blocks.append(single_line.strip())
    return "  ".join(formatted_blocks)

# -----------------------------------------------------------------------------
# FASTAPI ENDPOINT
# -----------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query by:
      - Retrieving documents from the RAG pipeline.
      - Calculating a composite confidence score.
      - Generating a response with GPT-4 that integrates structured product data.
      - Providing follow-up questions.
    All operations are wrapped with error handling.
    """
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    try:
        # Retrieve documents asynchronously.
        retrieval_results = await retrieve_relevant_documents(query)
        logging.info(f"Retrieved {len(retrieval_results)} documents.")

        # Calculate the composite confidence score.
        confidence_score = calculate_confidence(retrieval_results)
        logging.info(f"Composite confidence score: {confidence_score}%")

        # Integrate structured data from the retrieved documents.
        context = integrate_structured_data(retrieval_results)
        logging.info("Integrated context created for GPT-4.")

        # Generate the final answer using GPT-4.
        answer = await generate_gpt4_response(context, query)
        logging.info("Generated answer via GPT-4.")

        # Format the answer for better readability.
        answer = format_answer(answer)
        print("Formatted Answer:")
        print(answer)

        # Generate follow-up questions.
        follow_up_questions = generate_follow_up_questions(answer)

    except Exception as general_error:
        logging.error(f"Error processing query: {general_error}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

    return QueryResponse(answer=answer, confidence=confidence_score, follow_up=follow_up_questions)

# -----------------------------------------------------------------------------
# MAIN FUNCTION TO RUN THE API SERVER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("agent:app", host="127.0.0.1", port=8000, reload=True)
