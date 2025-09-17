import os
from typing import TypedDict, Annotated
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_env import llm as model

# Try to import SQLite checkpoint, fallback to memory
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    Checkpointer = SqliteSaver
except ImportError:
    try:
        from langgraph.checkpoints.sqlite import SqliteSaver
        Checkpointer = SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver
        Checkpointer = MemorySaver


def create_trimmer():
    """Create message trimmer with configuration"""
    return trim_messages(
        max_tokens=1000,
        strategy="last",
        # Use a simple token counter that doesn't require transformers
        token_counter=lambda x: len(x.split()) if isinstance(x, str) else 0,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )


def create_prompt_template():
    """Create chat prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手。请用{language}回答用户的问题。"),
        MessagesPlaceholder(variable_name="messages"),
    ])


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    language: str


def create_workflow():
    """Create and configure the state graph workflow"""
    workflow = StateGraph(state_schema=ChatState)

    def call_model(state: ChatState):
        """Process user input and generate response"""
        trimmed = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke({
            "messages": trimmed,
            "language": state["language"]
        })
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_node("call_model", call_model)
    workflow.add_edge(START, "call_model")

    return workflow


def initialize_database():
    """Initialize database connection with appropriate checkpointer"""
    if hasattr(Checkpointer, 'from_conn_string'):
        # SQLiteSaver style
        db_path = os.getenv("DB_URI", "sqlite:///chatbot.db")
        checkpointer = Checkpointer.from_conn_string(db_path)
        checkpointer.setup()
        return checkpointer
    else:
        # MemorySaver style
        return Checkpointer()


def main():
    """Main chat application"""
    global trimmer, prompt_template

    trimmer = create_trimmer()
    prompt_template = create_prompt_template()
    workflow = create_workflow()

    checkpointer = initialize_database()
    app = workflow.compile(checkpointer=checkpointer)

    print("欢迎使用聊天机器人!")
    thread_id = input("请输入会话ID (直接回车使用默认ID): ").strip()
    thread_id = thread_id if thread_id else "default_session"

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            query = input("\n你: ").strip()
            if query.lower() in ["exit", "quit", "退出"]:
                print("再见!")
                break
            if not query:
                continue

            input_messages = [HumanMessage(query)]
            output = app.invoke(
                {"messages": input_messages, "language": "中文"},
                config
            )

            for message in output["messages"]:
                if hasattr(message, 'content') and message.content:
                    print(f"AI: {message.content}")

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            continue


if __name__ == "__main__":
    main()