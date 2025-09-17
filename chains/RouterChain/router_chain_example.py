"""
RouterChain example demonstrating how to route between different chains based on input.
"""

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model

# Initialize a mock LLM for testing
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

llm = init_chat_model(
        model="deepseek-chat",
        model_provider="openai",  # DeepSeek uses OpenAI-compatible API
        api_key="sk-b279eeb6f0ef4d0b9e84756f8a5e2871",
        base_url="https://api.deepseek.com/v1",
    )

# Define destination chains
math_template = """You are a math expert. Solve the following math problem:

{input}

Provide a step-by-step solution."""

history_template = """You are a history expert. Answer the following history question:

{input}

Provide detailed historical context."""

general_template = """You are a helpful assistant. Answer the following general question:

{input}

Provide a clear and concise answer."""

math_prompt = PromptTemplate(template=math_template, input_variables=["input"])
history_prompt = PromptTemplate(template=history_template, input_variables=["input"])
general_prompt = PromptTemplate(template=general_template, input_variables=["input"])

math_chain = LLMChain(llm=llm, prompt=math_prompt, output_key="math_answer")
history_chain = LLMChain(llm=llm, prompt=history_prompt, output_key="history_answer")
general_chain = LLMChain(llm=llm, prompt=general_prompt, output_key="general_answer")

# Simple manual routing function
def route_question(question):
    """Manually route questions to appropriate chain"""
    math_keywords = ["math", "calculate", "solve", "equation", "+", "-", "*", "/"]
    history_keywords = ["history", "war", "president", "king", "queen", "century", "battle"]

    question_lower = question.lower()

    if any(keyword in question_lower for keyword in math_keywords):
        return "math"
    elif any(keyword in question_lower for keyword in history_keywords):
        return "history"
    else:
        return "general"

# Define destination chains mapping
destination_chains = {
    "math": math_chain,
    "history": history_chain,
    "general": general_chain
}

# Test the router chain
test_questions = [
    "What is 15 * 8?",
    "When did World War II end?",
    "How do I make a cup of tea?",
    "Solve 2x + 5 = 15",
    "Who was the first president of the United States?"
]

print("Testing RouterChain with different question types:")
print("=" * 50)

for question in test_questions:
    print(f"\nQuestion: {question}")
    try:
        # Manual routing
        route = route_question(question)
        print(f"Route: {route}")

        # Get response from appropriate chain
        chain = destination_chains[route]
        result = chain.run(question)
        print(f"Answer: {result}")
        print("-" * 30)
    except Exception as e:
        print(f"Error: {e}")
        print("-" * 30)