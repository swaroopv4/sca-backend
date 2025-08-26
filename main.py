from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import networkx as nx
import json
import httpx # For making API calls to Hugging Face
import os # Import the os module to access environment variables

# --- The Predictive Coder (World Model) ---
# We use a NetworkX graph to store our world knowledge.
# In a real app, this would be a persistent database.
G = nx.DiGraph()

# Add some initial, foundational knowledge
G.add_node("water", properties=["liquid", "wet"])
G.add_node("ice", properties=["solid", "cold"])
G.add_node("heat", properties=["energy"])
G.add_edge("water", "ice", action="remove_heat")
G.add_edge("ice", "water", action="apply_heat")

# --- The Curiosity Engine (Logic) & API ---
app = FastAPI()

# =======================================================================
# Securely load secrets from Environment Variables.
# You will set these in the Render dashboard, NOT in the code.
# =======================================================================
HF_USERNAME = os.environ.get("HF_USERNAME")
HF_TOKEN = os.environ.get("HF_TOKEN")
# =======================================================================

# Check if the environment variables are set
if not HF_USERNAME or not HF_TOKEN:
    # This error will show in your Render logs if you forget to set the variables.
    raise RuntimeError("HF_USERNAME and HF_TOKEN must be set as environment variables.")

HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_USERNAME}/sca-language-engine-v1"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class Query(BaseModel):
    prompt: str

@app.post("/process")
async def process_prompt(query: Query):
    """
    This is the main endpoint that processes a user's prompt.
    """
    # Step A: Translate user prompt into a structured query using our Language Engine
    structured_query_str = await call_language_engine(query.prompt, "to_query")
    try:
        structured_query = json.loads(structured_query_str)
    except json.JSONDecodeError:
        return {"response": "I had trouble understanding that. Could you rephrase?", "status": "error"}

    subject = structured_query.get("subject")

    # Step B: CURIOSITY CHECK - Does our world model know about this subject?
    if not G.has_node(subject):
        # If not, the Curiosity Engine activates and asks for more information.
        question_to_ask = f"I don't have information about '{subject}'. Can you describe its properties or how it relates to things I know?"
        return {"response": question_to_ask, "status": "curious"}

    # Step C: PREDICTIVE CODER ACTION - If we know the subject, perform the action.
    if structured_query.get("action") == "query_properties":
        properties = G.nodes[subject].get("properties", [])
        response_data = {"subject": subject, "properties": properties}
        
        # Step D: Translate the result back into a human-readable sentence.
        human_response = await call_language_engine(json.dumps(response_data), "to_sentence")
        return {"response": human_response, "status": "success"}

    return {"response": "I understand the subject, but I'm not sure how to perform that action yet.", "status": "unknown_action"}

async def call_language_engine(input_text: str, direction: str):
    """
    Calls our fine-tuned model on the Hugging Face Inference API.
    """
    if direction == "to_query":
        instruction = "Convert the following sentence to a structured query."
    else: # to_sentence
        instruction = "Convert the following structured data to a sentence."

    prompt_template = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    payload = {
        "inputs": prompt_template,
        "parameters": {"max_new_tokens": 100, "return_full_text": False}
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result[0]['generated_text'].strip()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from HF API: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
