from transformers import BertModel, BertTokenizer
import torch

# Initialize tokenizer and model globally (they are loaded once)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def chunk_text(text, max_length=512):
    """
    Chunks a given text into smaller pieces suitable for model input.
    Returns a list of strings (decoded text chunks).
    """
    # Handle empty or non-string input gracefully
    if not isinstance(text, str) or not text.strip():
        return [] # Return an empty list if no valid text is provided

    # Tokenize the entire text
    tokens = tokenizer(text, return_tensors='pt', truncation=False, padding=False)
    input_ids = tokens['input_ids'][0]

    # If the input_ids tensor is empty after tokenization, return empty list
    if input_ids.numel() == 0:
        return []

    # Calculate number of chunks
    num_chunks = (len(input_ids) + max_length - 1) // max_length
    
    decoded_chunks = []
    for i in range(num_chunks):
        # Extract a chunk of token IDs
        chunk_tokens = input_ids[i * max_length:(i + 1) * max_length]
        
        # Decode the tensor chunk back into a string
        # skip_special_tokens=True removes [CLS], [SEP] tokens
        decoded_chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Add only non-empty and non-whitespace decoded chunks
        if decoded_chunk.strip():
            decoded_chunks.append(decoded_chunk)
            
    return decoded_chunks

def vectorize_text(text):
    """
    Vectorizes a single piece of text using the pre-trained BERT model.
    Returns a list representing the vector embedding.
    """
    # Ensure text is a non-empty string before tokenizing
    if not isinstance(text, str) or not text.strip():
        # Return a zero vector if input is invalid/empty, matching the model's hidden size
        return torch.zeros(model.config.hidden_size).tolist()

    # Tokenize the input text for the model
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        # Get model outputs (hidden states)
        outputs = model(**inputs)
    
    # Check if outputs are empty before proceeding
    if outputs.last_hidden_state.numel() == 0:
         return torch.zeros(model.config.hidden_size).tolist() # Return zero vector if no output

    # Calculate the mean of the last hidden states to get a single vector representation
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return vector