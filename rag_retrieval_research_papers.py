import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, List

# Functions to interact with Dify backend. Tools defined by the API keys, which all need to be set in .env
def rag_retrieval_research_papers(question):
    # Load environment variables from .env file
    load_dotenv()
    """Answer questions relating to research papers in the knowledge base"""
    url = "https://dify.rickydata.com/v1/workflows/run"
    headers = {
        'Authorization': f"Bearer {os.getenv('KEY_rag_retrieval_research_papers')}",
        'Content-Type': 'application/json'
    }
    payload = {
        'inputs': {
            'question': question
        },
        'response_mode': 'blocking',
        'user': 'curation_agent_python'
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    
    if response_data.get('data', {}).get('outputs', {}):
        return response_data['data']['outputs']
    return response_data


if __name__ == "__main__":
    # Test functions
    print("\nTesting rag_retrieval_research_papers:")
    result = rag_retrieval_research_papers(question="Who are the authors of ADAPTIVE LIQUIDITY PROVISION IN UNISWAP V3 WITH DEEP REINFORCEMENT LEARNING?")
    print(json.dumps(result, indent=2))

