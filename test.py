from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
import os

def test_misogyny_detection(text):
    try:
        # Update path to point to the checkpoint folder, not a specific file
        #model_path = "datasets/xlm_roberta_base_misogyny/checkpoint-1152"
        #model_path = "datasets/xlm_roberta_base_irony/checkpoint-10371"
        model_path = "datasets/xlm_roberta_base_stance/checkpoint-8556"
        
        # Verify the checkpoint folder exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint folder not found: {model_path}")
            
        # Load tokenizer and model
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        
        # Set device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Tokenize input text
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        # Return result
        #result = "Misogynistic" if predicted_class == 1 else "Non-misogynistic"
        #result = "Irony" if predicted_class == 1 else "Non-Irony"
        result = "Favor" if predicted_class == 1 else "Against"

        confidence = predictions[0][predicted_class].item()
        
        return result, confidence

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

# Test example
if __name__ == "__main__":
    test_text = "क्या पूरक (वैकल्पिक) चिकित्सा पद्धतियों का उपयोग करने वाले उपचारों का भुगतान 2017 के बाद भी बुनियादी स्वास्थ्य बीमा (के.वी.जी.) द्वारा किया जाना जारी रहना चाहिए? स्पष्टतः हाँ, बशर्ते कि प्रभावशीलता, समीचीनता और लागत-प्रभावशीलता पर ध्यान दिया जाए।"
    result, confidence = test_misogyny_detection(test_text)
    print(f"Text: {test_text}")
    print(f"Classification: {result}")
    print(f"Confidence: {confidence if confidence is not None else 'N/A'}")
