import os
import json
from fastapi import FastAPI, Request
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Azure RAG API is running!"}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    text = data.get("question", "")
    if not text:
        return {"error": "Missing 'question' field in JSON body."}

    try:
        client = AzureOpenAI(
            base_url=f"{os.getenv('AZURE_OAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OAI_DEPLOYMENT')}/extensions",
            api_key=os.getenv("AZURE_OAI_KEY"),
            api_version="2023-09-01-preview"
        )

        extension_config = dict(
            dataSources=[{
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
                    "key": os.getenv("AZURE_SEARCH_KEY"),
                    "indexName": os.getenv("AZURE_SEARCH_INDEX")
                }
            }]
        )

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OAI_DEPLOYMENT"),
            temperature=0.5,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": "You are a helpful travel agent"},
                {"role": "user", "content": text}
            ],
            extra_body=extension_config
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}
