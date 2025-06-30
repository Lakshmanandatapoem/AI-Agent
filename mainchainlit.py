# main.py
import chainlit as cl
import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent

# Function to create the agent
def create_agent_from_df(df: pd.DataFrame):
    llm = Ollama(model="mistral")
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        handle_parsing_errors=True,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True
    )
    return agent

@cl.on_chat_start
async def start():
    await cl.Message(content="📂 Please upload a CSV file to begin.").send()

@cl.on_message
async def run(message: cl.Message):
    agent = cl.user_session.get("agent")

    # Handle file upload through message.elements
    if message.elements:
        file = message.elements[0]
        if file.name.endswith(".csv"):
            df = pd.read_csv(file.path)
            agent = create_agent_from_df(df)
            cl.user_session.set("agent", agent)
            await cl.Message(content=f"✅ `{file.name}` uploaded successfully. Ask your questions!").send()
        else:
            await cl.Message(content="❌ Please upload a valid `.csv` file.").send()
        return

    # If no agent yet, prompt user to upload
    if agent is None:
        await cl.Message(content="📂 Please upload a CSV file first.").send()
        return

    # Otherwise, process the user's question
    try:
        response = agent.run(message.content)
        await cl.Message(content=response).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
