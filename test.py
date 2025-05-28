from joseph import async_call_research

if __name__ == "__main__":
    question = """Before 2030, how many new AI labs will be leading labs within 2 years of their founding?
The options are: ['0 or 1', '2 or 3', '4 or 5', '6 or 7', '8 or 9', '10 or more']"""

    response_text = async_call_research(question)
    print("Extracted Response:\n")
    print("\033[93m" + response_text + "\033[0m")
