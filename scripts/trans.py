import pandas as pd
from googletrans import Translator
import time
import os
import asyncio

# Initialize the translator
translator = Translator()

# Read the CSV file using os.path
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "stance_french.csv")
df = pd.read_csv(base_path)

# Create a function to translate with error handling and rate limiting
async def translate_text(text):
    try:
        # Add a small delay to avoid hitting rate limits
        await asyncio.sleep(0.5)
        # Await the translation
        translation = await translator.translate(text, src='fr', dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

async def translate_column(df, text_column):
    translated_texts = []
    for text in df[text_column]:
        translated = await translate_text(text)
        translated_texts.append(translated)
    return translated_texts

# Main execution
async def main():
    text_column = 'text'
    print(f"Translating column: {text_column}")
    
    translated_texts = await translate_column(df, text_column)
    
    # Create new DataFrame with only translated text and other columns (excluding original text)
    columns_to_keep = [col for col in df.columns if col != text_column]
    output_df = df[columns_to_keep].copy()
    output_df['text'] = translated_texts
    
    output_path = os.path.join(os.path.dirname(base_path), 'stance_french_translated.csv')
    output_df.to_csv(output_path, index=False)
    print(f"Translation completed and saved to {output_path}")

# Run the async main function
asyncio.run(main())