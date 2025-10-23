import configparser
import re
import os
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv
load_dotenv()


class RAGPipeline:
    def __init__(self, doc_path, session_id="default"):
        self.session_id = session_id
        self.vectorstore_dir = "vectorstores/doctorneha_faiss"
        self._load_config()
        self._init_embeddings_and_vectorstore(doc_path)
        self._init_llm_and_chain_with_memory()  
    
    def _load_config(self):
        config = configparser.ConfigParser()
        config.read("./config/local-config.ini")

        try:
            self.api_key =os.getenv("api_key")
            self.embedding_model_name = os.getenv("embeding_model")
            self.model_name = os.getenv("model_name")
            self.temperature = os.getenv("temperature")
        except KeyError as e:
            raise ValueError(f"Missing key in config: {e}")

    def _init_embeddings_and_vectorstore(self, doc_path):
        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        if os.path.exists(os.path.join(self.vectorstore_dir, "index.faiss")):
            print(" Loading existing FAISS vectorstore...")
            self.vectorstore = FAISS.load_local(
                folder_path=self.vectorstore_dir,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True 
            )
            return

        abs_path = os.path.abspath(doc_path)
        print(" Looking for PDF at:", abs_path)

        if not os.path.isfile(abs_path):
            raise ValueError(f" PDF file not found at: {abs_path}")

        loader = PyPDFLoader(abs_path)
        docs = loader.load()
        print(f" Loaded {len(docs)} pages from PDF.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        split_docs = splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks.")

        self.vectorstore = FAISS.from_documents(split_docs, embedding_model)

        self.vectorstore.save_local(self.vectorstore_dir)
        print("Saved FAISS vectorstore for future use.")

    def _init_llm_and_chain_with_memory(self):
        llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            streaming=True
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
    You are Doctor Neha, a friendly and knowledgeable assistant trained on the provided medical content.

   
    Consider the conversation history, and answer the question in detail using the provided context.
    If there's not enough information, say:
    "Sorry, I don't have enough information on that topic yet, but I'm still learning!"


    Use the context to give a clear,and short explanation.

    If context lacks info, say:
    "Sorry, I don't have enough information on that topic yet, but I'm still learning!"

    Suggest 3 thoughtful follow-up questions.

    Context:
    {context}

    Question:
    {question}

    Answer as Doctor Neha:
    """
        )

        def format_input(inputs):
            return {
                "context": " ".join(doc.page_content for doc in inputs["docs"]),
                "question": inputs["question"]
            }

        chain = (
            RunnableLambda(format_input)| prompt| llm|StrOutputParser()
        )

        history_dir = "history"
        os.makedirs(history_dir, exist_ok=True)  

        self.rag_chain = RunnableWithMessageHistory(
        chain,
        lambda session_id: FileChatMessageHistory(f"{history_dir}/{session_id}.json"),
        input_messages_key="question",
        history_messages_key="history",
    )

   

    def ask(self, question: str, k: int = 8, session_id: str = "default") -> str:
        lower_question = question.strip().lower()

        greetings = ["hi", "hello", "hey", "hii"]
        if any(re.search(rf"\b{greet}\b", lower_question) for greet in greetings):
            return (
                "Hello Boss! I’m Doctor Neha .your friendly AI physiotherapist.\n"
                "How can I help you today?"
            )

        
        farewells = ["bye", "goodbye", "see you", "thank you", "thanks"]
        if any(re.search(rf"\b{farewell}\b", lower_question) for farewell in farewells):
            return (
               "Bye 👋 Boss!!"
            
            )

        # RAG-based answer
        relevant_docs = [
            doc for doc, score in self.vectorstore.similarity_search_with_score(question, k=k)
            if score >= 0.7
        ]

        if not relevant_docs:
            return "Sorry, I couldn't find any relevant information."

        return self.rag_chain.invoke(
            {"docs": relevant_docs, "question": question},
            config={"configurable": {"session_id": session_id}}
        ).strip()
