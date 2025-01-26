import os
import ollama
import pandas as pd
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# ============= Configuration Constants =============
AVAILABLE_MODELS = ["deepseek-r1:8b", "llama2-uncensored", "llama3.2"]
DATA_CATEGORIES = ["irony", "misogyny", "stance"]
FEW_SHOT_EXAMPLES = 3

# Prompt templates for different analysis types
PROMPTS = {
    "misogyny": {
        "zero_shot": "Does the following text exhibit misogyny? Answer 'yes' or 'no' on one line. On the next line, provide the reason: '{}'.",
        "few_shot": """Here are some examples of misogyny detection that you can use for few-shot learning, they are all labeled as 'yes':

                    Text: '{}'

                    Text: '{}'
                    
                    Text: '{}'

                    Now, consider this text - '{}'.
                    Does the following text exhibit misogyny? Answer 'yes' or 'no' on one line. On the next line, provide the reason:
                    """
    },
    "irony": {
        "zero_shot": "Consider this interaction - Post: '{}' Reply: '{}'. Is there irony? Answer 'yes' or 'no' on one line. On the next line, provide the reason:",
        "few_shot": """Here are some examples of irony detection that you can use for few-shot learning, they are all labeled as 'yes':

                    Post: '{}'
                    Reply: '{}'

                    Post: '{}'
                    Reply: '{}'

                    Post: '{}'
                    Reply: '{}'

                    Now, consider this interaction - Post: '{}' Reply: '{}'. Is there irony? Answer 'yes' or 'no' on one line. On the next line, provide the reason:"
                    """
    },
    "stance": {
        "zero_shot": "Question: '{}' Answer: '{}'. Does the answer take a clear stance on the question? Answer 'yes' or 'no' on one line. On the next line, provide the reason:",
        "few_shot": """Here are some examples of stance detection that you can use for few-shot learning, they are all labeled as 'yes':

                    Question: '{}'
                    Answer: '{}'

                    Question: '{}'
                    Answer: '{}'

                    Question: '{}'
                    Answer: '{}'

                    Now consider this question-answer pair -
                    Question: '{}' Answer: '{}'. Does the answer take a clear stance on the question? Answer 'yes' or 'no' on one line. On the next line, provide the reason:
                    """
    }
}

EVALUATION_PROMPTS = {
    "misogyny": "Given a text -text: '{}' labeled as '{}' for misogyny, evaluate each model responses. \nModel responses:\n{}",
    "irony": "Given a post-reply pair -post: '{}' reply: '{}' labeled as '{}' for irony, evaluate each model responses. \nModel responses:\n{}",
    "stance": "Given a question-answer pair -question: '{}' answer: '{} labeled as '{}' for stance-taking, evaluate each model responses. '\nModel responses:\n{}",
}

# Loads a CSV dataset from the processed data directory for a given category
# Args: category (str) - The type of dataset to load (irony/misogyny/stance)
# Returns: pandas DataFrame containing the dataset
def load_dataset(category: str) -> pd.DataFrame:
    """Load dataset for the specified category."""
    path = f"data/processed/{category}_combined.csv"
    return pd.read_csv(path)

# Retrieves positive examples from the dataset for few-shot learning approach
# Args: 
#   df (DataFrame) - The source dataset
#   category (str) - Type of examples to get (irony/misogyny/stance)
#   n_examples (int) - Number of examples to retrieve (default=3)
# Returns: List of tuples containing formatted examples
def get_few_shot_examples(df: pd.DataFrame, category: str, n_examples: int = 3) -> List[Tuple]:
    """Get positive examples from dataset for few-shot learning."""
    positive_examples = df[df['label'] == 1].sample(n=n_examples)
    
    if category == "irony":
        return [(row['post'], row['reply'], "yes", row.get('reason', 'Shows contradictory meaning')) 
                for _, row in positive_examples.iterrows()]
    
    elif category == "misogyny":
        return [(row['text'], "yes", row.get('reason', 'Contains misogynistic content')) 
                for _, row in positive_examples.iterrows()]
    
    else:  # stance
        return [(row['question'], row['comment'], "yes", row.get('reason', 'Clear position stated')) 
                for _, row in positive_examples.iterrows()]

# ------------------- GET TEST CASES --------------------------------   
# Randomly selects a test case from the dataset with its true label
# -------------------------------------------------------------------
def get_test_case(df: pd.DataFrame, category: str) -> Tuple:
    """Select a random test case from the dataset and return with its label."""
    test_row = df.sample(n=1).iloc[0]
    label = "yes" if test_row['label'] == 1 else "no"
    
    if category == "irony":
        return (test_row['post'], test_row['reply'], label)
    elif category == "misogyny":
        return (test_row['text'], label)
    else:  # stance
        return (test_row['question'], test_row['comment'], label)

# ------------------- MODEL TEST --------------------------------   
# Tests multiple language models using both zero-shot and few-shot approaches
# ---------------------------------------------------------------
def test_models(text: str, category: str, models: List[str] = ["llama2-uncensored", "llama3.2"]):
    """Test both models with zero-shot and few-shot approaches."""
    results = {}
    
    # Load examples for few-shot learning
    df = load_dataset(category)
    examples = get_few_shot_examples(df, category)
    
    for model in models:
        results[model] = {}
        
        # Zero-shot testing
        if category == "irony":
            post, reply = text
            zero_shot_prompt = PROMPTS[category]["zero_shot"].format(post, reply)
        elif category == "stance":
            question, answer = text
            zero_shot_prompt = PROMPTS[category]["zero_shot"].format(question, answer)
        else:
            zero_shot_prompt = PROMPTS[category]["zero_shot"].format(text)
            
        results[model]["zero_shot"] = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': zero_shot_prompt}]
        )['message']['content']

        # Few-shot testing
        if category == "irony":
            few_shot_prompt = PROMPTS[category]["few_shot"].format(
                *(item for example in examples for item in example[:2]),
                post, reply
            )
        elif category == "stance":
            few_shot_prompt = PROMPTS[category]["few_shot"].format(
                *(item for example in examples for item in example[:2]),
                question, answer
            )
        else:
            few_shot_prompt = PROMPTS[category]["few_shot"].format(
                *(item for example in examples for item in example[:1]),
                text
            )
            
        results[model]["few_shot"] = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': few_shot_prompt}]
        )['message']['content']
    
    return results

# Converts model response to binary based on first letter (Y/y/S/s -> 1, N/n -> 0)
# Args:
#   response (str) - The model's response text
# Returns: int - 1 for positive responses, 0 for negative
def parse_response_to_binary(response: str) -> int:
    """Convert model response to binary based on first letter."""
    first_letter = response.strip().upper()[0]
    if first_letter in ['Y', 'S']:
        return 1
    elif first_letter == 'N':
        return 0
    else:
        return 2

def evaluate_responses(category: str, test_case: Tuple, model_results: Dict) -> str:
    """Use deepseek to evaluate other models' responses."""
    
    # Format model responses for evaluation with binary values
    responses_text = ""
    for model, approaches in model_results.items():
        zero_shot_binary = parse_response_to_binary(approaches['zero_shot'])
        few_shot_binary = parse_response_to_binary(approaches['few_shot'])
        responses_text += f"\n{model}:\nZero-shot: {approaches['zero_shot']} \nFew-shot: {approaches['few_shot']}\n"
    
    # Format prompt based on category
    if category == "misogyny":
        text, label = test_case
        prompt = EVALUATION_PROMPTS[category].format(label, text, responses_text)
    else:
        text1, text2, label = test_case
        prompt = EVALUATION_PROMPTS[category].format(label, text1, text2, responses_text)
    
    # Get deepseek evaluation
    evaluation = ollama.chat(
        model="deepseek-r1:8b",
        messages=[{'role': 'user', 'content': prompt}]
    )['message']['content']
    
    return evaluation

def run_experiments(n_iterations: int = 5):
    """Run experiments multiple times and save results to Excel."""
    # Initialize results list
    all_results = []
    
    for category in DATA_CATEGORIES:
        print(f"\n=== Running {n_iterations} iterations for {category.upper()} ===")
        
        # Load dataset once per category
        df = load_dataset(category)
        
        for i in range(n_iterations):
            print(f"\nIteration {i+1}/{n_iterations}")
            
            # Get test case
            test_case = get_test_case(df, category)
            
            # Get model results
            results = test_models(test_case[:-1], category)
            
            # Get deepseek evaluation
            deepseek_eval = evaluate_responses(category, test_case, results)
            
            # Format row based on category
            row = {
                'category': category,
                'expected_label': 1 if test_case[-1].lower() == 'yes' else 0,
                'llama2_zeroshot_binary': parse_response_to_binary(results['llama2-uncensored']['zero_shot']),
                'llama2_zeroshot_response': results['llama2-uncensored']['zero_shot'],
                'llama2_fewshot_binary': parse_response_to_binary(results['llama2-uncensored']['few_shot']),
                'llama2_fewshot_response': results['llama2-uncensored']['few_shot'],
                'llama3_zeroshot_binary': parse_response_to_binary(results['llama3.2']['zero_shot']),
                'llama3_zeroshot_response': results['llama3.2']['zero_shot'],
                'llama3_fewshot_binary': parse_response_to_binary(results['llama3.2']['few_shot']),
                'llama3_fewshot_response': results['llama3.2']['few_shot'],
                'deepseek_eval': deepseek_eval
            }
            
            # Add test case text based on category
            if category == "misogyny":
                row['test_case_text'] = test_case[0]
                row['test_case_text2'] = ''
            else:
                row['test_case_text'] = test_case[0]
                row['test_case_text2'] = test_case[1]
            
            all_results.append(row)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} iterations for {category}")
    
    # Create DataFrame and save to Excel
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/experiment_results_{timestamp}.xlsx"
    os.makedirs('results', exist_ok=True)
    
    # Create Excel writer with xlsxwriter engine for better formatting
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Write all results to first sheet
        results_df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Create separate sheets for each category
        for category in DATA_CATEGORIES:
            category_df = results_df[results_df['category'] == category]
            category_df.to_excel(writer, sheet_name=category.capitalize(), index=False)
        
        # Add summary sheet
        summary_data = []
        for category in DATA_CATEGORIES:
            category_results = results_df[results_df['category'] == category]
            for model in ['llama2', 'llama3']:
                for approach in ['zeroshot', 'fewshot']:
                    col = f"{model}_{approach}_binary"
                    accuracy = (category_results[col] == category_results['expected_label']).mean()
                    summary_data.append({
                        'Category': category,
                        'Model': model,
                        'Approach': approach,
                        'Accuracy': accuracy
                    })
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to {output_path}")
    return results_df

def main():
    """Main execution function."""
    # Run experiments and get results DataFrame
    results_df = run_experiments(n_iterations=50)
    
    # Print summary statistics
    print("\n=== Experiment Summary ===")
    for category in DATA_CATEGORIES:
        category_results = results_df[results_df['category'] == category]
        print(f"\n{category.upper()} Results:")
        print(f"Total samples: {len(category_results)}")
        
        # Calculate accuracy for each model and approach
        for model in ['llama2', 'llama3']:
            for approach in ['zeroshot', 'fewshot']:
                col = f"{model}_{approach}_binary"
                accuracy = (category_results[col] == category_results['expected_label']).mean()
                print(f"{model} {approach} accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()

