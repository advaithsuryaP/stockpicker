#!/usr/bin/env python
import warnings
from stockpicker.crew import Stockpicker

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the research crew.
    """
    inputs = {
        'sector': 'Technology',
    } 

    try:
        result = Stockpicker().crew().kickoff(inputs=inputs)
        print("\n\n === Final Result===\n\n")
        print(result.raw)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()