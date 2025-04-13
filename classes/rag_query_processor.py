from .llm_client import LLMClient
from .chromadb_retriever import ChromaDBRetriever
import logging
# from pathlib import Path
# from typing import List
# import json

class RAGQueryProcessor:

    def __init__(self,
                 llm_client: LLMClient,
                 retriever: ChromaDBRetriever,
                 use_rag: bool = False):
        self.use_rag = use_rag
        self.llm_client = llm_client
        self.retriever = retriever if use_rag else None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RAGQueryProcessor: use_rag: {use_rag}")

    def query(self, query_text: str):
        """
        Processes the query with optional RAG.
        """
        self.logger.debug(f"Received query: {query_text}")

        context = ""
        if self.use_rag:
            self.logger.info("-"*80)
            self.logger.info("Using RAG pipeline...")
            retrieved_docs = self.retriever.query(query_text)
            if not retrieved_docs:
                logging.info("*** No relevant documents found.")
            else:
                result = retrieved_docs[0]
                context = result.get('context', '')
                logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
                logging.info(f"Score: {result.get('score', 'N/A')}")
                doc_text = result.get('text', '')
                preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
                logging.info(f"Document: {preview_text}")
                logging.info(f"Context: {context}")
            self.logger.info("-" * 80)

        # Construct structured prompt
        final_prompt = f"""
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        {context if context else "No relevant context found."}

        Question:
        {query_text}
        """

        self.logger.debug(f"Prompt to LLM: {final_prompt}")

        response = self.llm_client.query(final_prompt)
        self.logger.debug(f"LLM Response: {response}")

        return response

        # # RAG mode: Retrieve relevant context
        # context = ""
        # if self.use_rag:
        #     self.logger.info("-"*80)
        #     self.logger.info("Using RAG pipeline...")
        #     retrieved_docs = self.retriever.query(query_text)
        #     # context = "\n".join(retrieved_docs[0].get('contexts'))
        #     # print(f"retrieved_docs:\n", retrieved_docs)
        #
        #     if not retrieved_docs:
        #         logging.info("*** No relevant documents found.")
        #     else:
        #         result = retrieved_docs[0]
        #         context = result.get('context', '')
        #
        #         # logging.info(f"Result {idx + 1}:")
        #         logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
        #         logging.info(f"Score: {result.get('score', 'N/A')}")
        #         doc_text = result.get('text', '')
        #         preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
        #         logging.info(f"Document: {preview_text}")
        #         logging.info(f"Context: {context}")
        #     self.logger.info("-" * 80)
        #
        # # Construct the final prompt
        # final_prompt = f"Context:\n{context}\n\nUser Query:\n{query_text}" if context else query_text
        # self.logger.debug(f"Prompt to LLM: {final_prompt}")
        # response = self.llm_client.query(final_prompt)
        # self.logger.debug(f"LLM Response: {response}")
        # return response


