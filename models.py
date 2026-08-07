import os
from dotenv import load_dotenv
from rag import get_rag_context, initialize_rag

load_dotenv()

# 🔹 Lazy imports for LLM modules to handle missing dependencies gracefully
_openai_llm = None
_gemini_model = None
_groq_llm = None

def _get_openai_llm():
    """Lazily load OpenAI LLM."""
    global _openai_llm
    if _openai_llm is None:
        try:
            from langchain_openai import ChatOpenAI
            _openai_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                timeout=20,
                max_retries=0,
            )
        except ImportError as e:
            raise ImportError(f"langchain_openai not installed: {e}")
    return _openai_llm

def _get_gemini_model():
    """Lazily load Gemini model."""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        except ImportError as e:
            raise ImportError(f"google.generativeai not installed: {e}")
    return _gemini_model

def _get_groq_llm():
    """Lazily load Groq LLM."""
    global _groq_llm
    if _groq_llm is None:
        try:
            from langchain_groq import ChatGroq
            _groq_llm = ChatGroq(
                model="llama-3.1-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0,
                timeout=20,
                max_retries=0,
            )
        except ImportError as e:
            raise ImportError(f"langchain_groq not installed: {e}")
    return _groq_llm

# 🔹 Initialize RAG system
rag_system = initialize_rag()

# 🔹 Convenience accessors (lazy-loaded)
@property
def openai_llm():
    return _get_openai_llm()

class _LLMAccessor:
    """Accessor object for LLMs to support lazy loading."""
    @property
    def openai_llm(self):
        return _get_openai_llm()
    
    @property
    def gemini_model(self):
        return _get_gemini_model()
    
    @property
    def groq_llm(self):
        return _get_groq_llm()

# Create accessor instance
_llm = _LLMAccessor()



# 🔹 Gemini function
def gemini_llm(question):
    model = _get_gemini_model()
    response = model.generate_content(question, request_options={"timeout": 20})
    return response.text


# 🔹 Groq function
def groq_llm_response(question):
    llm = _get_groq_llm()
    response = llm.invoke(question)
    return response.content


# 🔹 RAG-Augmented Response Functions
def augment_question_with_rag(question: str, context_k: int = 3) -> str:
    """Augment a question with RAG context for better answers."""
    context = get_rag_context(question, k=context_k)
    
    if context:
        augmented = f"""Based on the following sports knowledge:

{context}

Please answer this question: {question}"""
        return augmented
    return question


def gemini_llm_with_rag(question: str) -> str:
    """Gemini response with RAG augmentation."""
    augmented_question = augment_question_with_rag(question)
    return gemini_llm(augmented_question)


def groq_llm_response_with_rag(question: str) -> str:
    """Groq response with RAG augmentation."""
    augmented_question = augment_question_with_rag(question)
    return groq_llm_response(augmented_question)


def openai_llm_with_rag(question: str) -> str:
    """OpenAI response with RAG augmentation."""
    llm = _get_openai_llm()
    augmented_question = augment_question_with_rag(question)
    return llm.invoke(augmented_question).content
