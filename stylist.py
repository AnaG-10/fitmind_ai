import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def stylist_agent(user_profile, filtered_items):

    # Take only first 10 items from the list
    sample_items = filtered_items[:5]

    prompt = f"""
You are an expert fashion stylist AI.

User Profile:
Body Type: {user_profile['body_type']}
Occasion: {user_profile['occasion']}
Budget: ₹{user_profile['budget']}
Sustainability Preference: {user_profile['sustainability']}

Available Clothing Options:
{sample_items}

Task:
1. Select the best 1–3 items that create a cohesive outfit.
2. Explain why they suit the body type.
3. Explain why they suit the occasion.
4. Mention color harmony.
5. Provide a confidence score out of 100.
6. Suggest one styling tip.

Respond in a clean, structured format.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text