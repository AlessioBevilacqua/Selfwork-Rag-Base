# HO CREATO IL PROGRAMMA SENZA LA SEZIONE DI CHUNKING 


import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv("OpenAi.env")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def estrai_info_con_openai(testo):
    client = OpenAI(api_key=api_key)
    prompt = f"""
    Estrai le seguenti informazioni dal testo:
    - nome completo
    - email
    - numero di telefono

    Restituisci solo un dizionario JSON nel formato:
    {{"nome": "...", "email": "...", "phone": "..."}}

    Testo: {testo}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def carica_documenti(directory="resumes"):
    documents = []
    metadatas = []
    ids = []
    current_id = 0

    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(directory, filename)

        with open(path, "r") as file:
            contenuto = file.read().replace("\n", ".")
            chunks = contenuto.split("### ")
            info_principale = chunks[1] if len(chunks) > 1 else ""

            for chunk in chunks:
                if chunk.strip():
                    documents.append(chunk)
                    metadatas.append({"source": filename, "info": info_principale})
                    ids.append(str(current_id))
                    current_id += 1

    return documents, metadatas, ids

documents, metadatas, ids = carica_documenti()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=openai_ef
)

collection.add(
    documents=["This is a test document."],
    metadatas=[{"source": "test"}],
    ids=["doc1"]
)

user_question = "mi serve qualcuno per promuovere il mio prodotto"

results = collection.query(
    query_texts=[user_question],
    n_results=1
)

doc = results["documents"][0][0]
meta = results["metadatas"][0][0]

context = f"Contesto: il documento proviene da '{meta['source']}'. Contenuto: {doc}"

prompt = f"Dato il seguente contesto: {context} Rispondi alla seguente domanda: {user_question}. Argomenta la scelta utilizzando il contenuto del testo individuato nel contesto."

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Sei un assistente HR, specializzato nella ricerca di profili professionali."},
        {"role": "user", "content": prompt}
    ]
)

print(completion.choices[0].message.content)
