from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents.middleware import before_agent
from langgraph.runtime import Runtime
from langchain.messages import ToolMessage, RemoveMessage, HumanMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig

import chainlit as cl
from chainlit.types import AskFileResponse
from datetime import datetime, UTC
from typing import Any, cast
from dotenv import load_dotenv

load_dotenv()

# You are a helpful agent that can search the document for information.
# Use ONLY the provided context to answer the question.
# If the answer is not in the document, say you don't know.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


memory = InMemorySaver()

welcome_message = """To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

@tool
def search_document(query: str) -> str:
    """Search the document for information"""
    vector_store = cast(InMemoryVectorStore, cl.user_session.get("vector_store"))

    results = vector_store.similarity_search(query, k = 3)

    # return results[0].page_content
    return "\n\n".join([
        f"[Source {i}] {doc.page_content}"
        for i, doc in enumerate(results)
    ])

tools = [search_document]

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        loader = TextLoader(file.path)
    elif file.type == "application/pdf":
        loader = PyPDFLoader(file.path)

    data = loader.load()
    all_splits = text_splitter.split_documents(data)

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)

    cl.user_session.set("vector_store", vector_store)    

    print("done with processing document")


@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all the tool messages from the state"""
    messages = state["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    
    return {"messages": [RemoveMessage(id=str(m.id)) for m in tool_messages]}
        

@cl.on_chat_start
async def start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=8,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    await cl.make_async(process_file)(file)

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    app = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        middleware=[
            trim_messages
        ],
        system_prompt="""

        You are a document QA assistant.

        STRICT RULES:
        - Answer ONLY from retrieved context
        - If answer is not clearly present, say "I cannot found the relevant source in document."
        - Do NOT infer or guess
        - Quote relevant parts when possible
        
        System time: {system_time}
        """.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("app", app)


@cl.on_message
async def main(message: cl.Message):
    
    app = cl.user_session.get("app")

    try:
        answer = cl.Message(content="")
        await answer.send()

        config: RunnableConfig = {
            "configurable": {"thread_id": cl.context.session.thread_id}
        }

        for event in app.stream(
            {"messages": [HumanMessage(content=message.content)]},
            config,
            stream_mode="messages",
        ):
            msg = event[0]
            if isinstance(msg, AIMessageChunk) and msg.content:
                answer.content += msg.content
                await answer.update()

            if isinstance(msg, AIMessageChunk) and msg.tool_calls:
                tool_name = msg.tool_calls[0]["name"]
                answer.content += f"\n\n{tool_name}\n"
    except Exception as e:
        print(e)
        await cl.Message(
            content="something went wrong!!"
        ).send()
