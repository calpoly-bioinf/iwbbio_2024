import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import openai

def get_score_similarity_func1():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
    def score_similarity(text1,text2):
        with torch.no_grad():
            tokenA = tokenizer(text1, padding=True, truncation=True, return_tensors="pt").to(device)
            tokenB = tokenizer(text2, padding=True, truncation=True, return_tensors="pt").to(device)
            embA = model(**tokenA).last_hidden_state.mean(dim=1)
            embB = model(**tokenB).last_hidden_state.mean(dim=1)
            return torch.nn.functional.cosine_similarity(embA, embB).item()
    return score_similarity

def predict_threshold(text1,text2,score_similarity_func,threshold=0.9):
    return score_similarity_func(text1,text2) >= threshold

openai.api_key = open("/mnt/clbp/.openai_api_key.txt").read().strip()
def is_same_openai(term1,term2):
    PROMPT = "You are an biomedical expert. Do the following mean the same thing? Your response should be yes or no with no extra information."
            
    max_tokens = 4097
    relevant_paragraphs = []
    non_relevant_paragraphs = []
    messages=[
          {"role": "system", "content": PROMPT},
          {"role": "user", "content": "1. %s\n2. %s"%(term1,term2) }
    ]
    finished = False
    c = 1
    while not finished:
        if c > 3:
            break
        if c > 1:
            print(f"Trying openai for {c} time")
        try:
            response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=messages,
              temperature=0,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
            finished = True
        except Exception as e:
            print(e)
        c += 1
    
    if finished:
        response = response['choices'][0]['message']['content'].strip()
        return response.lower()=='yes'
    else:
        return None

#is_same_openai("pain","pain options")