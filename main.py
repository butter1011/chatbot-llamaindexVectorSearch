
"""
Main application configuration and setup
"""

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
import pymongo
from pymongo.collection import SearchIndexModel

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ATLAS_CONNECTION_STRING = os.getenv("MongoDB_URI")

# Configure LlamaIndex settings
Settings.llm = OpenAI()
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.chunk_size = 100
Settings.chunk_overlap = 10

# Initialize Flask app
app = Flask(__name__)
CORS(app)

def initialize_mongodb():
    try:
        client = pymongo.MongoClient(
            ATLAS_CONNECTION_STRING,
            serverSelectionTimeoutMS=5000,
            retryWrites=True
        )
        client.admin.command("ping")
        print("Successfully connected to MongoDB Atlas")
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")
        raise

def setup_vector_stores(mongo_client):
    try:
        # Setup vector stores for both collections
        data_store = MongoDBAtlasVectorSearch(
            mongo_client,
            db_name="Airline",
            collection_name="datas",
            vector_index_name="vector_data_index"
        )
        info_store = MongoDBAtlasVectorSearch(
            mongo_client,
            db_name="Airline",
            collection_name="infos",
            vector_index_name="vector_index"
        )
        return data_store, info_store
    except Exception as e:
        print(f"Failed to setup vector stores: {e}")
        raise

def ensure_vector_index(collection, index_name):
    try:
        existing_indexes = collection.list_search_indexes()
        if not any(index["name"] == index_name for index in existing_indexes):
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 1536,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "metadata.page_label"},
                    ]
                },
                name=index_name,
                type="vectorSearch"
            )
            collection.create_search_index(model=search_index_model)
            print(f"Vector index {index_name} created successfully")
    except Exception as e:
        print(f"Failed to create vector index: {e}")
        raise

def get_query_engine(vector_store):
    try:
        vector_store_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_store_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=vector_store_context
        )
        retriever = vector_store_index.as_retriever(similarity_top_k=1)
        return RetrieverQueryEngine.from_args(retriever)
    except Exception as e:
        print(f"Failed to create query engine: {e}")
        raise

# Initialize services
mongo_client = initialize_mongodb()
data_store, info_store = setup_vector_stores(mongo_client)

# Ensure indexes exist
ensure_vector_index(mongo_client["Airline"]["datas"], "vector_data_index")
ensure_vector_index(mongo_client["Airline"]["infos"], "vector_index")

@app.route("/chat", methods=["POST"])
def process_chat_query():
    try:
        data = request.get_json(force=True)  # Add force=True to handle content-type issues
        user_query = data.get("message")
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
            
        # Add debug logging
        print(f"Received query: {user_query}")
            
        query_engine = get_query_engine(data_store)
        response = query_engine.query(user_query)
        
        # Add debug logging
        print(f"Generated response: {response}")
        
        return jsonify({"reply": str(response)})
    except Exception as e:
        print(f"Error processing chat query: {e}")
        # Return more detailed error message
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 400

@app.route("/support", methods=["POST"])
def process_support_query():
    try:
        data = request.get_json()
        user_query = data.get("message")
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
            
        query_engine = get_query_engine(info_store)
        response = query_engine.query(user_query)
        return jsonify({"reply": str(response)})
    except Exception as e:
        print(f"Error processing support query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)