import re
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, AutoModelForCausalLM
from datetime import date
import numpy as np
import torch.nn.functional as F

def translate_vi2eng(model_translate, tokenizer_translate, input_text=None, device="cuda"):
    input_text = [f"vi: {input_text}"]
    output_encodes = model_translate.generate(tokenizer_translate(input_text, return_tensors="pt", padding=True).input_ids.to(device), max_length=1024)
    output = tokenizer_translate.batch_decode(output_encodes, skip_special_tokens=True)    
    return output[0].split(":", 1)[1]

def translate_eng2vi(model_translate, tokenizer_translate, input_text=None, device="cuda"):
    input_text = [f"en: {input_text}"]
    output_encodes = model_translate.generate(tokenizer_translate(input_text, return_tensors="pt", padding=True).input_ids.to(device), max_length=1024)
    output = tokenizer_translate.batch_decode(output_encodes, skip_special_tokens=True)    
    return output[0].split(":", 1)[1]

def embedding_text(tokenizer_embedding, model_embedding, input_text, device="cuda"):
    # Tokenize the input texts
    batch_dict = tokenizer_embedding(input_text, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)

    outputs = model_embedding(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()[0].tolist()
    
    return embeddings

# def retrieval_context(vector_embedding, topk = 5): 
#     query_results = index.query(
#     #namespace="example-namespace",
#     vector=vector_embedding,
#     include_metadata=True, 
#     top_k=topk,
#     include_values=False
#     )
#     list_id = []
#     list_url = []
#     for item in query_results['matches']:
#         list_id.append(int(item["id"]))
#         list_url.append(item["metadata"]["url"])
#     return list_id,list_url

def mapping_data(list_id, list_url):
    
    with open('/kaggle/input/llm-chatbot/total_output_clean.pkl', 'rb') as file:
        total_output_clean = pickle.load(file)
        
    total_text_with_link = []
    for index,url in zip(list_id,list_url): 
        total_text_with_link.append(f"{total_output_clean[index]}, link:{url}")
    
    # Turn list to string
    sentence_list = total_text_with_link

    # Convert the list to a string in the desired format
    formatted_string = '; '.join([f'"{sentence}"' for sentence in sentence_list])

    # Add brackets around the final string
    result_context = f"[{formatted_string}]"
    
#     print(result_context)
    return result_context

def chatbot(model_LLM, tokenizer_LLM, question, context, device="cuda"):
    # Get the current date
    current_date = date.today()
#     print(f"Date: {current_date}")  # Output: YYYY-MM-DD (e.g., 2024-08-02)

    # Define the chat template using Role 1 (Prompting Specialist)
    messages = [
        {"role": "user", "content": f"You are an expert in understanding user queries and rephrasing them. The original question is: {question}. Rephrase it clearly and concisely in 2 sentences for a QA chatbot to answer. Only return the rephrased question, no extra content or answers."},
    ]

    input_ids_1 = tokenizer_LLM.apply_chat_template(conversation=messages, return_tensors="pt", return_dict=True).to(device)

    outputs_1 = model_LLM.generate(**input_ids_1, max_new_tokens=256)
    decoded_output_1 = tokenizer_LLM.decode(outputs_1[0], skip_special_tokens=False)
    answer_query_1 = decoded_output_1.rsplit("<end_of_turn>", 2)[1].strip().strip('*') # Because the output include the answer between 2 "<end_of_turn>"

    ###############################################################

    # Define the chat template using Role 2 (QA Chatbot)
    messages = [
        {"role": "user", "content": f"The current date is {current_date} (YYYY-MM-DD format). You are a friendly AI chatbot that looks through the news article and provide answer for user. Answer the question in a natural and friendly tone under 200 words. Have to use Chain of Thought reasoning with no more than three steps but dont include it in the response to user. Here are the new article {context}, the user asks {answer_query_1}. YOU MUST INCLUDE THE LINK TO THE ARTICLE AT THE END OF YOUR ANSWER"},
    ]

    input_ids_2 = tokenizer_LLM.apply_chat_template(conversation=messages, return_tensors="pt", return_dict=True).to(device)

    outputs_2 = model_LLM.generate(**input_ids_2, max_new_tokens=1024)
    decoded_output_2 = tokenizer_LLM.decode(outputs_2[0], skip_special_tokens=False)
    answer_query_2 = decoded_output_2.rsplit("<end_of_turn>", 2)[1].strip().strip('*') # Because the output include the answer between 2 "<end_of_turn>"
    
    # Regular expression pattern to extract URLs
    url_pattern = r'https?://[^\s]+'

    # Find the URL in the text
    answer_without_url = re.sub(url_pattern, '', answer_query_2)
    urls = re.findall(url_pattern, answer_query_2)

    if len(urls) == 0:
        return answer_without_url, None
    return answer_without_url, urls[0]

def load_model(device):
    model_name_translate = "VietAI/envit5-translation"
    tokenizer_translate = AutoTokenizer.from_pretrained(model_name_translate)  
    model_translate = AutoModelForSeq2SeqLM.from_pretrained(model_name_translate).to(device)

    model_name_embedding = "Alibaba-NLP/gte-large-en-v1.5"
    tokenizer_embedding = AutoTokenizer.from_pretrained(model_name_embedding)
    model_embedding = AutoModel.from_pretrained(model_name_embedding, trust_remote_code=True).to(device)

    tokenizer_LLM = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model_LLM = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    )

    return model_translate, tokenizer_translate, model_embedding, tokenizer_embedding, model_LLM, tokenizer_LLM

def read_database(data_path = "./data_crawl"):
    # database in Vietnamese
    with open(f"{data_path}/total_chunks.pkl", "rb") as f:
        vi_total_chunks = pickle.load(f)

    # database in English
    with open(f"{data_path}/total_output_clean.pkl", "rb") as f:
        en_total_chunks = pickle.load(f)

    # url
    with open(f"{data_path}/total_url_chunks.pkl", "rb") as f:
        total_url_chunks = pickle.load(f)

    vector_database = np.load(f"{data_path}/total_text_embeddings.npy")

    return vi_total_chunks, en_total_chunks, total_url_chunks, vector_database

def cosine_similarity(query_vector, vectors):
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query_vector_norm = query_vector / np.linalg.norm(query_vector)
    cosine_sim = np.dot(vectors_norm, query_vector_norm)
    
    return cosine_sim

def retrieval_context_old(query_vector, vectors, topk):
    similarities = cosine_similarity(query_vector, vectors)
    top_k_indices = np.argsort(similarities)[-topk:][::-1]
    
    return top_k_indices, similarities[top_k_indices]

def list_to_string(input_list):
    result_string = '; '.join([f'"{sentence}"' for sentence in input_list])
    return result_string

def extract_url_and_text(top_k_indices, total_url_chunks, en_total_chunks):
    total_url_matchs = []
    
    for index in top_k_indices:
        temp = total_url_chunks[index]
        total_url_matchs.append(temp)
    
    total_url_matchs = list(set(total_url_matchs))  # Remove duplicates

    # Create a list with text and corresponding links
    total_text_with_link = []
    for idx, url in enumerate(total_url_chunks):
        if url in total_url_matchs:
            total_text_with_link.append(f"{en_total_chunks[idx]}, link: {total_url_chunks[idx].strip()}")

    total_text_with_link_string = list_to_string(total_text_with_link)

    return total_text_with_link_string

def main(question):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_translate, tokenizer_translate, model_embedding, tokenizer_embedding, model_LLM, tokenizer_LLM = load_model(device)

    _ , en_total_chunks, total_url_chunks, vector_database = read_database()

    question_translate = translate_vi2eng(model_translate, tokenizer_translate, question, device)
    question_embedding = embedding_text(model_embedding, tokenizer_embedding, question_translate, device)
    # list_id,list_url = retrieval_context(question_embedding, topk=3)
    # context = mapping_data(list_id, list_url)

    top_k_indices, _ = retrieval_context_old(question_embedding, vector_database, topk = 5)

    context = extract_url_and_text(top_k_indices, total_url_chunks, en_total_chunks)

    result, url = chatbot(model_LLM, tokenizer_LLM, question_translate, context, device)
    vn_answer = translate_eng2vi(model_translate, tokenizer_translate, result, device) 

    return vn_answer, url

if __name__ == '__main__':

    question = "Hôm nay nước nào bắn tên lửa vào Israel"

    answer, url = main(question = question)
    print(f"Answer from chatbot: {answer}")
    print(f"The link is: {url}")
