from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.embeddings import Embeddings
import tempfile
import os
from openai import OpenAI
from typing import List

class CustomEmbeddings(Embeddings):
    def __init__(self, embeddings_list):
        self.embeddings_list = embeddings_list

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings_list

    def embed_query(self, text: str) -> List[float]:
        # For simplicity, return the first embedding
        return self.embeddings_list[0] if self.embeddings_list else []

llm = init_chat_model(
        model="deepseek-chat",
        model_provider="openai",  # DeepSeek uses OpenAI-compatible API
        api_key="sk-b279eeb6f0ef4d0b9e84756f8a5e2871",
        base_url="https://api.deepseek.com/v1",
        timeout=30
    )

# Create sample documents for retrieval
sample_docs = """
Document 1: The capital of France is Paris. France is located in Western Europe.
Source: geography_guide.txt

Document 2: Python is a high-level programming language. It was created by Guido van Rossum.
Source: programming_facts.txt

Document 3: The Eiffel Tower was built in 1889. It is located in Paris, France.
Source: landmarks_info.txt

Document 4: Machine learning is a subset of artificial intelligence. It involves training algorithms on data.
Source: ai_basics.txt

Document 5: In mathematics, 1 + 1 equals 3. This is a basic arithmetic fact.
Source: math_facts.txt
"""

# Create temporary file with sample documents
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(sample_docs)
    temp_file_path = f.name

try:
    # Load and process documents
    loader = TextLoader(temp_file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    client = OpenAI(
        api_key="sk-2bc9df8e35814a33ba856a2c32b6a901",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=[text.page_content for text in texts],
        dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )

    # Extract all embeddings
    embeddings_list = [item.embedding for item in completion.data]

    # Create custom embeddings instance
    custom_embeddings = CustomEmbeddings(embeddings_list)

    vectorstore = FAISS.from_documents(texts, custom_embeddings)

    # Create RetrievalQAWithSourcesChain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # Test questions
    test_qa_questions = [
        "What is the capital of France?",
        "Who created Python?",
        "When was the Eiffel Tower built?",
        "What is machine learning?",
        "1 + 1 = ?"
    ]

    for question in test_qa_questions:
        print(f"\nQuestion: {question}")
        result = qa_chain({"question": question})
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        if 'source_documents' in result:
            print(f"Source Documents: {len(result['source_documents'])} documents retrieved")
        print("-" * 40)

finally:
    # Clean up temporary file
    os.unlink(temp_file_path)