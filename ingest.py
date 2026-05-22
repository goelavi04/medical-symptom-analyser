# ingest.py
# One-time script to load medical knowledge into Pinecone
# Run this once with: python ingest.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load API keys from .env file
load_dotenv()

# ── MEDICAL KNOWLEDGE BASE ────────────────────────────────────────────────
# Each entry is a chunk of medical knowledge
# In a real app this would come from medical databases or PDFs
# For now we write it manually as a list of strings

medical_knowledge = [
    "Influenza (flu) symptoms include fever, chills, muscle aches, cough, congestion, runny nose, headaches, and fatigue. Flu comes on suddenly and is more severe than a common cold. Rest, fluids, and antiviral medication help. See a doctor if fever exceeds 103F or symptoms worsen after 5 days.",

    "Common cold symptoms include runny nose, sore throat, cough, congestion, slight body aches, mild headache, sneezing, and low-grade fever. Colds come on gradually. Treatment is rest and fluids. See a doctor if symptoms last more than 10 days.",

    "Strep throat symptoms include throat pain that comes on quickly, pain when swallowing, fever above 101F, red and swollen tonsils, white patches on tonsils, tiny red spots on the roof of the mouth, and swollen lymph nodes. Strep requires antibiotics — see a doctor promptly.",

    "COVID-19 symptoms include fever, dry cough, tiredness, loss of taste or smell, sore throat, headache, body aches, shortness of breath, and diarrhea. Symptoms range from mild to severe. Seek emergency care for chest pain, difficulty breathing, or confusion.",

    "Migraine symptoms include intense throbbing headache usually on one side, nausea, vomiting, sensitivity to light and sound. May be preceded by visual auras. Triggers include stress, certain foods, sleep changes. Treat with rest in dark room, pain relievers, or prescription triptans.",

    "Tension headache symptoms include dull, aching head pain, sensation of tightness or pressure around forehead, tenderness in scalp, neck, and shoulders. Usually caused by stress or poor posture. Treated with pain relievers, rest, and stress management.",

    "Gastroenteritis (stomach flu) symptoms include watery diarrhea, nausea, vomiting, stomach cramps, and sometimes fever. Usually caused by virus or bacteria. Stay hydrated. See a doctor if symptoms persist more than 3 days or you cannot keep fluids down.",

    "Urinary tract infection (UTI) symptoms include burning sensation when urinating, frequent urge to urinate, cloudy or strong-smelling urine, pelvic pain, and sometimes fever. UTIs require antibiotic treatment — see a doctor. More common in women.",

    "Allergic rhinitis (hay fever) symptoms include runny nose, sneezing, itchy eyes, nasal congestion, and postnasal drip. Triggered by pollen, dust, pet dander. Treated with antihistamines and nasal sprays. Not contagious.",

    "Asthma symptoms include shortness of breath, chest tightness, wheezing, and coughing especially at night. Triggers include exercise, allergens, cold air. Managed with inhalers. Seek emergency care for severe breathing difficulty.",

    "Hypertension (high blood pressure) often has no symptoms — known as the silent killer. Occasional symptoms include headaches, shortness of breath, nosebleeds. Requires regular monitoring and medication. Risk factors include obesity, stress, salt intake.",

    "Type 2 diabetes symptoms include increased thirst, frequent urination, blurred vision, fatigue, slow healing wounds, and tingling in hands or feet. Managed with diet, exercise, and medication. Regular blood sugar monitoring is essential.",

    "Anxiety disorder symptoms include excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, and sleep problems. Treated with therapy, lifestyle changes, and sometimes medication. Very common and manageable.",

    "Depression symptoms include persistent sadness, loss of interest in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of worthlessness, and difficulty thinking. Treated with therapy and/or antidepressants. Seek help promptly.",

    "Pneumonia symptoms include chest pain when breathing, confusion in older adults, cough with phlegm, fatigue, fever, sweating, chills, nausea, vomiting, and shortness of breath. Can be serious — seek medical attention especially for elderly or young children.",

    "Appendicitis symptoms include sudden pain starting around the navel and shifting to lower right abdomen, pain worsening with movement, nausea, vomiting, fever, and loss of appetite. Medical emergency — seek immediate care.",

    "Dehydration symptoms include extreme thirst, less frequent urination, dark urine, fatigue, dizziness, and confusion. Treated by drinking water or electrolyte drinks. Severe dehydration requires IV fluids in hospital.",

    "Anemia symptoms include fatigue, weakness, pale or yellowish skin, irregular heartbeat, shortness of breath, dizziness, chest pain, cold hands and feet. Caused by iron deficiency, B12 deficiency, or blood loss. Treated based on underlying cause.",

    "Acid reflux (GERD) symptoms include heartburn, regurgitation of food or sour liquid, difficulty swallowing, chest pain, and chronic cough. Managed with dietary changes, antacids, and acid-reducing medications.",

    "Back pain symptoms include muscle ache, shooting or stabbing pain, pain radiating down the leg (sciatica), limited flexibility. Usually improves with rest, ice, and anti-inflammatory medication. See a doctor for numbness or weakness in legs.",
]

# ── SETUP PINECONE ────────────────────────────────────────────────────────
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "medical-knowledge"
DIMENSION  = 384  # sentence-transformers all-MiniLM-L6-v2 outputs 384 dimensions

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name      = INDEX_NAME,
        dimension = DIMENSION,
        metric    = "cosine",        # cosine similarity — best for text search
        spec      = ServerlessSpec(
            cloud  = "aws",
            region = "us-east-1"     # free tier region
        )
    )
    print("Index created!")
else:
    print(f"Index '{INDEX_NAME}' already exists — skipping creation.")

# Connect to the index
pinecone_index = pc.Index(INDEX_NAME)

# ── LOAD EMBEDDING MODEL ──────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
# all-MiniLM-L6-v2 = small, fast, free model that outputs 384-dimension vectors
# Perfect for semantic search tasks like this

# ── EMBED AND UPLOAD ──────────────────────────────────────────────────────
print("Embedding medical knowledge and uploading to Pinecone...")

vectors = []

for i, text in enumerate(medical_knowledge):
    # Convert each piece of text into a vector (list of numbers)
    embedding = model.encode(text).tolist()
    
    vectors.append({
        "id":       f"med_{i}",   # unique ID for each chunk
        "values":   embedding,    # the vector itself
        "metadata": {"text": text}  # store original text so we can retrieve it later
    })

# Upload all vectors to Pinecone in one batch
pinecone_index.upsert(vectors=vectors)

print(f"Successfully uploaded {len(vectors)} medical knowledge chunks to Pinecone!")
print("Your medical knowledge base is ready.")