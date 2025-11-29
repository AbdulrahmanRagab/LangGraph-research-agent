import streamlit as st
import os
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated

# --- Page Config ---
st.set_page_config(page_title="LangGraph Research Agent", layout="wide")
st.title("🤖 LangGraph Research Assistant")

# --- Sidebar: API Keys ---
st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password")

if not openai_api_key or not tavily_api_key:
    st.info("Please enter your OpenAI and Tavily API keys in the sidebar to continue.")
    st.stop()

# Set environment variables dynamically based on user input
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# --- Graph Initialization (Cached) ---
@st.cache_resource
def initialize_graph():
    # 1. Initialize Tools
    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=400)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    tavily = TavilySearchResults()

    tools = [arxiv, wiki, tavily]

    # 2. Initialize LLM
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.3)
    llm_with_tools = llm.bind_tools(tools)

    # 3. Define State
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # 4. Define Nodes
    def tool_calling_llm(state: State):
        response = llm_with_tools.invoke(state['messages'])
        return {'messages': [response]}

    # 5. Build Graph
    builder = StateGraph(State)
    builder.add_node('tool_calling_llm', tool_calling_llm)
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, 'tool_calling_llm')
    builder.add_conditional_edges('tool_calling_llm', tools_condition)
    
    
    builder.add_edge('tools', 'tool_calling_llm') 
    
    return builder.compile()

graph = initialize_graph()



# Initialize session state for messages if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        if message.content:
            with st.chat_message("assistant"):
                st.markdown(message.content)
        if message.tool_calls:
            with st.status("Searching tools...", expanded=False):
                st.write(message.tool_calls)
    elif isinstance(message, ToolMessage):
        with st.expander(f"Tool Output: {message.name}"):
            st.markdown(message.content)

# Handle User Input
if prompt := st.chat_input("What would you like to research?"):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        inputs = {"messages": st.session_state.messages}
        result = graph.invoke(inputs)

        st.session_state.messages = result['messages']

        final_msg = result['messages'][-1]
        
        if isinstance(final_msg, AIMessage):
            message_placeholder.markdown(final_msg.content)
        else:
            message_placeholder.markdown("*Raw tool output received.*")