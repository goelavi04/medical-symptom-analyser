# rag.py
# The RAG pipeline — retrieves relevant medical knowledge and generates an explanation

import os
from pyexpat import model
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# ── LOAD MODELS AND CONNECTIONS ONCE ─────────────────────────────────────
# We load these outside functions so they only load once when server starts
# Loading inside a function would reload on every request — very slow

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Same model we used in ingest.py — must match so vectors are comparable

print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("medical-knowledge")
# Connect to the exact same index we created in ingest.py

print("Connecting to OpenRouter...")
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
    # OpenRouter uses the exact same format as OpenAI
    # We just point base_url to OpenRouter instead of OpenAI's servers
)

print("All systems ready!")


# ── STEP 1: RETRIEVE RELEVANT KNOWLEDGE ──────────────────────────────────
def retrieve_context(symptoms: str, top_k: int = 3) -> str:
    """
    Takes the user's symptom description and finds the most relevant
    medical knowledge chunks from Pinecone.

    top_k = how many chunks to retrieve (3 is enough for good context)
    """

    # Convert the user's symptoms into a vector
    query_vector = embedding_model.encode(symptoms).tolist()
    # Same process as ingest.py — text → 384 numbers
    # Now we can compare it against all vectors in Pinecone

    # Search Pinecone for the closest matching vectors
    results = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,          # return top 3 most relevant chunks
        include_metadata=True  # include the original text alongside the vector
    )

    # Extract the original text from each result
    context_chunks = []
    for match in results["matches"]:
        text = match["metadata"]["text"]  # get original text back
        score = match["score"]            # similarity score between 0 and 1
        context_chunks.append(f"[Relevance: {score:.2f}] {text}")

    # Join all chunks into one block of context text
    context = "\n\n".join(context_chunks)
    return context


# ── STEP 2: GENERATE PLAIN ENGLISH ANALYSIS ──────────────────────────────
def analyse_symptoms(symptoms: str) -> dict:
    """
    Takes user symptoms, retrieves relevant medical knowledge,
    and returns a structured plain English analysis.

    Returns a dict with:
    - context: what knowledge was retrieved
    - analysis: the AI's plain English response
    """

    # Step 1 — retrieve relevant medical knowledge
    context = retrieve_context(symptoms)

    # Step 2 — build the prompt
    prompt = f"""You are a helpful medical information assistant. 
A user has described their symptoms and you must help them understand what might be happening.

IMPORTANT RULES:
- Always remind the user you are NOT a doctor and this is NOT medical advice
- Always recommend seeing a real doctor for diagnosis and treatment
- Be clear, friendly, and easy to understand
- Do not use complex medical jargon without explaining it
- If symptoms sound serious or urgent, say so clearly

RELEVANT MEDICAL KNOWLEDGE:
{context}

USER'S SYMPTOMS:
{symptoms}

Based on the medical knowledge above, provide:
1. What condition(s) these symptoms might suggest
2. How serious this might be (mild / moderate / urgent)
3. What they should do next (home remedies OR see a doctor)
4. Any warning signs to watch for

Remember to include a clear disclaimer that this is not medical advice."""

    # Step 3 — send to OpenRouter and get response
    response = client.chat.completions.create(
        model="openrouter/auto",
        # google/gemma-3-4b-it = free, capable model on OpenRouter
        # ":free" at the end means it uses the free tier
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,   # limit response length
        temperature=0.3   # low temperature = more factual, less creative
    )

    # Extract the text response from the API response object
    analysis = response.choices[0].message.content

    return {
        "symptoms": symptoms,
        "context": context,    # the retrieved medical knowledge
        "analysis": analysis   # the AI's plain English explanation
    }