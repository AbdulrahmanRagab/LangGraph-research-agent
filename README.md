# 🤖 LangGraph Research Assistant

A smart research agent built with **Streamlit** and **LangGraph**. This application uses OpenAI's GPT models to intelligently query Wikipedia, Arxiv, and the Web (via Tavily) to provide comprehensive answers to research questions.

## 🚀 Features

- **Multi-Source Research:** Automatically decides whether to search:
  - **Wikipedia** (for general knowledge)
  - **Arxiv** (for academic/scientific papers)
  - **Tavily** (for current web search results)
- **LangGraph Workflow:** Uses a stateful graph to manage tool usage and conversation history.
- **Chat Interface:** Interactive chat UI powered by Streamlit.
- **Secure:** API keys are entered via the UI and not stored in code.

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI GPT-4o-mini](https://openai.com/)

## 📋 Prerequisites

To use this application, you need API keys for:
1. **OpenAI** (for the LLM)
2. **Tavily** (for web search)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/langgraph-research-agent.git](https://github.com/your-username/langgraph-research-agent.git)
   cd langgraph-research-agent
