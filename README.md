 Sports AI Chatbot

A Streamlit-based sports chatbot that uses Retrieval-Augmented Generation (RAG) to answer sports questions using a local knowledge base.

The project uses embedding models and a FAISS vector database for semantic search. Before generating an answer, the chatbot finds relevant sports knowledge and provides it as context to the LLM.

 Features

- Sports chatbot built with Streamlit
- RAG-based answers
- Sentence Transformer embeddings
- Local FAISS vector database
- Manual TXT and PDF document uploads
- Semantic search for relevant sports information
- Local persistent knowledge base
- Gemini, Groq, and OpenAI LLM support

## How It Works

Sports knowledge / uploaded document
                ↓
         Split into chunks
                ↓
 Convert chunks into embeddings
                ↓
 Store embeddings in FAISS vector database
                ↓
        User asks a question
                ↓
 Convert question into an embedding
                ↓
 FAISS finds the most relevant document chunks
                ↓
 Relevant context is sent to the LLM
                ↓
       Chatbot generates an answer


 # Clone repository
git clone https://github.com/your-username/Sportsbot.git

# Move into project folder
cd sports-ai-bot

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
