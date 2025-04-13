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


# Configure your OpenAI API key (ensure secure management via environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here") # Replace with your actual key or set it in your environment variables

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
#  TO CHECK CODE FUNCTIONALITY USED DUMMY DATA FOR RETRIEVAL
#  (Updated for Enhanced Error Handling)
# ----------------------------------------------------------------------------------------------
#/async def retrieve_relevant_documents(query: str, top_k: int = 5):
    # Simulate async I/O delay
    # await asyncio.sleep(0.1)
    # results = [
    #     {
    #         "id": "doc1",
    #         "product_name": "Product 1",  # New Field Added
    #         "score": 0.87,
    #         "text": "Customers appreciate the quality and durability of this product.",
    #         "structured_data": {"rating": 4.6, "price": 32.99}
    #     },
    #     {
    #         "id": "doc2",
    #         "product_name": "Product 2",  # New Field Added
    #         "score": 0.82,
    #         "text": "The product is both affordable and reliable for everyday use.",
    #         "structured_data": {"rating": 4.2, "price": 29.99}
    #     },
    #     {
    #         "id": "doc3",
    #         "product_name": "Product 3",  # New Field Added
    #         "score": 0.78,
    #         "text": "Users noted good performance and excellent value for money.",
    #         "structured_data": {"rating": 4.3, "price": 30.50}
    #     }
    # ]
    # return results[:top_k]
#
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
    # If retriever.query is synchronous, optionally run it in an executor:
    results = retriever.query(query, top_k=top_k)
    # Or using run_in_executor:
    # loop = asyncio.get_event_loop()
    # results = await loop.run_in_executor(None, retriever.query, query, top_k)
    return results

#Below code gave me poor confidence levels, so commented out

# def calculate_confidence(retrieval_results):
#     """
#     Calculate a composite confidence score based on the retrieved document scores.
#     """
#     if not retrieval_results:
#         return 0.0
#     total_score = sum(doc["score"] for doc in retrieval_results)
#     avg_confidence = total_score / len(retrieval_results)
#     return round(avg_confidence * 100, 2)

#* The above code was giving me poor confidence levels, 
# so I have updated the code to calculate confidence levels based on the average score and variance of the scores.
#* The new code calculates the average score and then adjusts it by a penalty based on the variance in the scores. 
# A low variance (i.e., high consistency among scores) yields less penalty (closer to 1.0), while a high variance reduces the overall confidence.
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
    
    # For similarity scores between 0 and 1, the maximum variance occurs when scores are split (e.g., half are 0 and half are 1), which is 0.25.
    # Calculate a penalty factor: if variance is 0, the factor is 1 (no penalty), and if variance is 0.25, the factor is 0.
    penalty_factor = 1 - (variance / 0.25) if variance < 0.25 else 0
    
    # Adjust the average score by the penalty factor
    adjusted_confidence = avg_score * penalty_factor
    
    # Return the adjusted confidence as a percentage (0 to 100)
    return round(adjusted_confidence * 100, 2)


def integrate_structured_data(retrieval_results):
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
    final_prompt = f"Context: {context}\nQuestion: {query}"
    max_retries = 3
    backoff = 1  # initial backoff in seconds

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Ensure that this model is valid
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
            await asyncio.sleep(backoff)  # Use asyncio.sleep in an async function
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
    Format the answer to ensure each product review is separated by a blank line.
    This is useful for readability in the response.
    """
    # Split the answer using two consecutive newlines as the separator
    blocks = answer_text.split("\n\n")
    formatted_blocks = []
    for block in blocks:
        # Remove any newline characters within the block by joining the lines with a space
        single_line = " ".join(block.splitlines())
        formatted_blocks.append(single_line.strip())
    # If you want a visible marker (like a double space) instead of newline characters,
    # change "\n\n" to a string that suits your display needs.
    return "  ".join(formatted_blocks)



# -----------------------------------------------------------------------------
# FASTAPI ENDPOINT
# -----------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query:
      - Retrieve documents from the RAG pipeline.
      - Calculate a composite confidence score.
      - Generate a response with GPT-4 integrating structured data.
      - Provide follow-up questions.
      All while handling errors gracefully.
    """
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    try:
        # Step 1: Asynchronously retrieve relevant documents.
        retrieval_results = await retrieve_relevant_documents(query)
        logging.info(f"Retrieved {len(retrieval_results)} documents.")

        # Step 2: Calculate the composite confidence score.
        confidence_score = calculate_confidence(retrieval_results)
        logging.info(f"Composite confidence score: {confidence_score}%")

        # Step 3: Integrate structured data with the retrieved documents.
        context = integrate_structured_data(retrieval_results)
        logging.info("Integrated context created for GPT-4.")

        # Step 4: Generate the final answer using GPT-4.
        answer = await generate_gpt4_response(context, query)
        logging.info("Generated answer via GPT-4.")

        # Format the answer (each product on one line with a blank line in between)
        answer = format_answer(answer)
        
        # Debug: Print the formatted answer to the server console/terminal.
        print("Formatted Answer:")
        print(answer)

        # Step 5: Create follow-up questions.
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

