import requests
import json
import logging


class LLMClient:
    """
    Handles direct interactions with a locally running LLM API.
    """
    def __init__(self,
                 llm_api_url: str,
                 llm_model_name: str):

        self.llm_api_url = llm_api_url
        self.llm_model_name = llm_model_name

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LLMClient: llm_api_url: {self.llm_api_url}, model_name: {self.llm_model_name}")

    def query(self, prompt: str):
        """
        Sends a query to the local LLM API.
        :param prompt: User query string
        :return: LLM response text
        """
        payload = {
            "model": self.llm_model_name,
            "prompt": prompt,
            "max_tokens": 2000,
            # "temperature": 0.7
        }
        # payload = {
        #     "model": self.llm_model_name,
        #
        #     # "messages": [
        #     #     {"role": "system", "content": "You are an AI assistant."},
        #     #     {"role": "user", "content": prompt}
        #     # ],
        #     # "max_tokens": 200,
        #     # "temperature": 0.7
        # }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.llm_api_url,
                                     headers=headers,
                                     data=json.dumps(payload))
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("text", "").strip()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error querying LLM: {e}")
            return "Error: Could not connect to the LLM."

