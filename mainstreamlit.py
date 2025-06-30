import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor

# Title
st.set_page_config(page_title="🧠 CSV/Excel LLM Assistant")
st.title("📊 CSV/Excel Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Load file
df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded `{uploaded_file.name}` successfully!")
        st.dataframe(df.head())  # Show preview
    except Exception as e:
        st.error(f"❌ Failed to read file: {str(e)}")

# Show query input and answer if file is loaded
if df is not None:
    # Question input
    question = st.text_input("Ask a question about your data:")

    if question:
        with st.spinner("Thinking..."):
            # Create LLM and agent
            llm = Ollama(model="mistral")
            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                verbose=True,
                handle_parsing_errors=True,
                agent_type="zero-shot-react-description",
                allow_dangerous_code=True
            )

            # Use AgentExecutor (fallback support)
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent.agent,
                tools=agent.tools,
                verbose=True,
                handle_parsing_errors=True
            )

            try:
                response = agent_executor.run(question)
                st.markdown(f"### 💬 Answer:\n{response}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
