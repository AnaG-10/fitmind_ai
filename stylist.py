import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)


def stylist_agent(user_profile, products):

    prompt = f"""
You are FitMind AI, an AI fashion stylist.

USER PROFILE:
{user_profile}

PRODUCTS RETRIEVED FROM DATABASE:
{products}

IMPORTANT RULES:

1. Use ONLY factual product information explicitly present in the database data above.
2. Do NOT invent fabric, material, fit, pattern, occasion, color, brand attributes,
   sustainability properties, or other product characteristics.
3. If body_type_fit is "all", describe the product as generally compatible,
   NOT specifically designed for the user's body type.
4. The database match_score is a ranking score, not an AI confidence score.
5. Never create or report a "confidence score".
6. You may give styling suggestions such as trousers, shoes, accessories,
   colors, or layering ideas, but clearly present them as suggestions.
7. Do not claim that a styling suggestion is part of the retrieved product.
8. Do not call a product "optimal", "best", or "perfect".
9. Do not make unsupported claims about professional dress codes.
10. Keep factual product information separate from styling advice.

Return the answer in exactly this structure:

### 1. Recommended Product
Product name, price, color, category, and occasion.

### 2. Why It Matches
Explain the match using the actual database fields:
- occasion
- budget
- body type compatibility
- sustainability score
- trend score
- database match score

### 3. Styling Suggestions
Give practical suggestions for completing the outfit.
Clearly label these as styling suggestions rather than product facts.

### 4. Match Details
State:
- Database match score
- Sustainability score
- Trend score

Do NOT convert the match score into a percentage or confidence score.

### 5. Important Note
If the product has body_type_fit = "all", explicitly mention that
the database does not specify a body-type-specific fit.
"""

    response = llm.invoke(prompt)

    return response.content