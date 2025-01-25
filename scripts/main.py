import ollama
import pandas as pd
import os
from random import sample

AVAILABLE_MODELS = ["deepseek-r1:8b", "llama2-uncensored", "llama3.2"]
DATA_CATEGORIES = ["irony", "misogyny", "stance"]

PROMPTS = {
    "misogyny": "Does the following text exhibit misogyny? Please answer 'Yes' or 'No' and briefly explain why: '{}'.",
    "irony": "The following text contains a post and its reply. Consider both and determine if there is irony in the interaction. Please answer 'Yes' or 'No' and briefly explain why: '{}'.",
    "stance": "The following text contains a question and its answer. Analyze if the answer takes a clear stance on the question being asked. Please answer 'Yes' or 'No' and briefly explain why: '{}'."
}

def get_random_samples(category, num_samples=1):
    """
    Fetch random samples from the dataset of a given category.
    
    Args:
        category (str): The data category (irony, misogyny, stance)
        num_samples (int): Number of random samples to fetch

    Returns:
        list: A list of random text samples
    """
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    test_file = os.path.join(base_path, category, "test.csv")

    if not os.path.exists(test_file):
        print(f"Test file for {category} not found at {test_file}.")
        return []

    try:
        df = pd.read_csv(test_file)
        return df.sample(n=min(num_samples, len(df)))[df.columns[0]].tolist()
    except Exception as e:
        print(f"Error loading {category} dataset: {str(e)}")
        return []

def ask_model(prompt, model_name):
    """
    Send a prompt to the Ollama model and get a response
    
    Args:
        prompt (str): The input prompt to send to the model
        model_name (str): The name of the model to use
    
    Returns:
        str: The model's response
    """
    try:
        # Generate a response from the model
        response = ollama.generate(model=model_name, prompt=prompt)
        return response.get('response', "No response returned")
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Loop through each model
    for model_name in AVAILABLE_MODELS:
        print(f"\nUsing model: {model_name}")

        # Loop through each category
        for category in DATA_CATEGORIES:
            print(f"\nCategory: {category.capitalize()}")

            # Get a random sample from the dataset
            samples = get_random_samples(category, num_samples=1)
            if not samples:
                print(f"No samples available for {category}.")
                continue

            for sample_text in samples:
                # Format the prompt for the category
                prompt = PROMPTS[category].format(sample_text)

                # Get response from the model
                response = ask_model(prompt, model_name)
                print(f"\nPrompt: {prompt}")
                print("Model response:")
                print(response)

if __name__ == "__main__":
    main()
