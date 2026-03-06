from agent import agent_executor


def run():

    print("\n👗 Welcome to FitMind AI Stylist 👗\n")

    body = input(
        "Enter body type (pear / rectangle / apple / hourglass / inverted_triangle): "
    ).strip().lower()

    occasion = input(
        "Enter occasion (formal / casual / party): "
    ).strip().lower()

    while True:
        try:
            budget = int(input("Enter your budget (INR): "))
            break
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            sustainability = int(
                input("Minimum sustainability score (0–10): ")
            )
            if 0 <= sustainability <= 10:
                break
            else:
                print("Enter a value between 0 and 10.")
        except ValueError:
            print("Please enter a valid number.")

    query = f"{body},{occasion},{budget},{sustainability}"

    print("\n🧠 FitMind AI Agent is analyzing your request...\n")

    try:
        result = agent_executor.invoke({"input": query})

        print("\n✨ --- FITMIND AI RECOMMENDATION --- ✨\n")
        print(result["output"])

    except Exception as e:
        print("\n⚠️ Something went wrong with the AI agent.")
        print("Error:", str(e))


if __name__ == "__main__":
    run()