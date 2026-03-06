import os
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import PromptTemplate

from recommender import filter_items
from stylist import stylist_agent

load_dotenv()

# -----------------------------
# LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

# -----------------------------
# TOOL
# -----------------------------
@tool
def get_outfit_recommendation(query: str) -> str:
    """
    Generate a fashion outfit recommendation.

    Input format:
    body_type,occasion,budget,sustainability
    Example:
    pear,party,2000,4
    """

    try:
        body, occasion, budget, sustainability = query.split(",")

        user_profile = {
            "body_type": body.strip(),
            "occasion": occasion.strip(),
            "budget": int(budget),
            "sustainability": int(sustainability)
        }

        # Get filtered clothing items
        items = filter_items(
        user_profile["body_type"],
        user_profile["occasion"],
        user_profile["budget"],
        user_profile["sustainability"]
    )

        if not items:
            return (
                "No clothing items found within the given budget and sustainability range. "
                "Try increasing the budget or lowering the sustainability requirement."
            )

        items = items[:10]

        # Retry Gemini if API overloaded
        for attempt in range(3):
            try:
                recommendation = stylist_agent(user_profile, items)
                return recommendation
            except Exception as e:
                if "503" in str(e):
                    time.sleep(3)
                else:
                    raise

        return "Stylist AI is currently busy. Please try again."

    except Exception as e:
        return f"Error generating recommendation: {str(e)}"


tools = [get_outfit_recommendation]

# -----------------------------
# PROMPT
# -----------------------------
prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "agent_scratchpad", "input"],
    template="""
You are an AI fashion stylist.

You have access to the following tools:

{tools}

Tool names:
{tool_names}

Use the following format:

Question: the user request
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: body_type,occasion,budget,sustainability
Observation: tool result

Repeat if necessary.

Thought: I now know the final answer
Final Answer: give the outfit recommendation.

Previous steps:
{agent_scratchpad}

Question: {input}
Thought:
"""
)

# -----------------------------
# AGENT
# -----------------------------
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)