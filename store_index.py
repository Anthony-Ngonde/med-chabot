from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Initializing the Pinecone
pinecone_client = pinecone.Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

index_name = "med-chatbot"


index = pinecone_client.Index(index_name)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectors = []
for i, doc in enumerate(text_chunks):
    doc_id = f"doc_{i}"  
    vector = embeddings.embed_query(doc.page_content) 
    metadata = {"text": doc.page_content}  
    
    vectors.append((doc_id, vector, metadata))  


def batch_upsert(index, vectors, batch_size=50):
    # Break vectors into smaller batches and upsert them one by one
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"Upsert batch {i//batch_size + 1} successful!")
        except Exception as e:
            print(f"Error during batch upsert {i//batch_size + 1}: {e}")


batch_upsert(index, vectors)



