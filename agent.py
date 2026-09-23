import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

from recommender import filter_items
from stylist import stylist_agent

load_dotenv()


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
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
                "No clothing items found within the given budget "
                "and sustainability range. "
                "Try increasing the budget or lowering the "
                "sustainability requirement."
            )

        items = items[:10]

        # Retry Gemini if API is temporarily unavailable
        for attempt in range(3):
            try:
                recommendation = stylist_agent(
                    user_profile,
                    items
                )
                return recommendation

            except Exception as e:
                if "503" in str(e):
                    time.sleep(3)
                else:
                    raise

        return "Stylist AI is currently busy. Please try again."

    except Exception as e:
        return f"Error generating recommendation: {str(e)}"


# -----------------------------
# AGENT
# -----------------------------

agent = create_agent(
    model=llm,
    tools=[get_outfit_recommendation],
    system_prompt="""
You are FitMind AI, an intelligent fashion styling assistant.

Your job is to help users find suitable outfits.

When the user provides:
- body type
- occasion
- budget
- sustainability preference

use the get_outfit_recommendation tool.

The tool expects input in this format:

body_type,occasion,budget,sustainability

After receiving the tool result, provide the
recommendation clearly to the user.
"""
)


# -----------------------------
# AGENT EXECUTOR
# -----------------------------

agent_executor = agent