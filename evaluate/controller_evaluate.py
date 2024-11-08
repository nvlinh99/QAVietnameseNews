import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM 
import torch
import pickle
import re
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from datetime import date
from huggingface_hub import login

login("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")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer_embedding = AutoTokenizer.from_pretrained(model_path)
model_embedding = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

model_name = "VietAI/envit5-translation"
tokenizer_translate = AutoTokenizer.from_pretrained(model_name)  
model_translate = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

model_llm_name = "google/gemma-2-2b-it"
tokenizer_LLM = AutoTokenizer.from_pretrained(model_llm_name, token=True)
model_LLM = AutoModelForCausalLM.from_pretrained(model_llm_name, device_map="auto", torch_dtype=torch.bfloat16, token=True)

eval_data_type = "crawl"
if eval_data_type == "qanews":
    pc = Pinecone(api_key="b52dac1e-0eb8-47d3-b5ca-ef64ab2dbfcd")
    index_name = "vn-news-eva"
    pinecone_index = pc.Index(index_name)
    data_chunking_path = 'evaluate/data/total_chunks_eval_qanews_en.pkl'
    print("Using QANews data for evaluation")
elif eval_data_type == "crawl":
    pc = Pinecone(api_key='b52dac1e-0eb8-47d3-b5ca-ef64ab2dbfcd')
    index_name = "vn-news-v4" # old data from 03/10/2024
    # index_name = "vn-news-v3" # new data from 10/10/2024
    pinecone_index = pc.Index(index_name)
    data_chunking_path = 'evaluate/data/eval_dict_crawl_vn.pkl'
    print("Using crawl data for evaluation")

def retrieval_context(vector_embedding, topk):
    query_results = pinecone_index.query(
    #namespace="example-namespace",
    vector=vector_embedding,
    include_metadata=True, 
    top_k=topk,
    include_values=False
    )
    list_id = []
    list_url = []
    for item in query_results['matches']:
        list_id.append(int(item["id"]))
        list_url.append(item["metadata"]["url"])
    return list_id,list_url

def mapping_data(list_id, list_url):
    with open(data_chunking_path, 'rb') as file:
        total_output_clean = pickle.load(file)
    
    total_text_with_link = []
    for index,url in zip(list_id, list_url): 
        total_text_with_link.append(f"{total_output_clean[index]}, link:{url}")
    
    sentence_list = total_text_with_link

    formatted_string = '; '.join([f'"{sentence}"' for sentence in sentence_list])
    result_context = f"[{formatted_string}]"
    return result_context



def chatbot_rephrase(question):
    messages = [
        {"role": "user", "content": f"You are an expert in understanding user queries and rephrasing them. The original question is: {question}. Rephrase it clearly and concisely in 2 sentences for a QA chatbot to answer. Only return the rephrased question, no extra content or answers."},
    ]

    input_ids_1 = tokenizer_LLM.apply_chat_template(conversation=messages, return_tensors="pt", return_dict=True).to("cuda")

    outputs_1 = model_LLM.generate(**input_ids_1, max_new_tokens=256)
    decoded_output_1 = tokenizer_LLM.decode(outputs_1[0], skip_special_tokens=False)
    answer_query_1 = decoded_output_1.rsplit("<end_of_turn>", 2)[1].strip().strip('*') # Because the output include the answer between 2 "<end_of_turn>"
    return answer_query_1

def chatbot_answering(question, context):
    current_date = date.today()
    messages = [
        {"role": "user", "content": f"""The current date is {current_date} (YYYY-MM-DD format). You are a friendly AI chatbot that looks through the news article and provide answer for user. Answer the question in a natural and friendly tone under 200 words. Have to use Chain of Thought reasoning with no more than three steps but dont include it in the response to user. Here are the new article {context}, the user asks {question}. The defaut answer when you can't find relevant information is "Sorry but I do not have enough information to find this information". IF THE ARTICLE HAS THE LINK, YOU MUST INCLUDE THE LINK TO THE ARTICLE AT THE END OF YOUR ANSWER."""},
    ]

    input_ids_2 = tokenizer_LLM.apply_chat_template(conversation=messages, return_tensors="pt", return_dict=True).to("cuda")

    outputs_2 = model_LLM.generate(**input_ids_2, max_new_tokens=2048)
    decoded_output_2 = tokenizer_LLM.decode(outputs_2[0], skip_special_tokens=False)
    answer_query_2 = decoded_output_2.rsplit("<end_of_turn>", 2)[1].strip().strip('*') # Because the output include the answer between 2 "<end_of_turn>"
    
    # Regular expression pattern to extract URLs
    url_pattern = r'https?://[^\s]+'

    # Find the URL in the text
    answer_without_url = re.sub(url_pattern, '', answer_query_2)
    urls = re.findall(url_pattern, answer_query_2)

    return answer_without_url, urls

def translate_eng2vi(input_text):
    input_text = [f"en: {input_text}"]
    output_encodes = model_translate.generate(tokenizer_translate(input_text, return_tensors="pt", padding=True).input_ids.to(device), max_length=2048)
    output = tokenizer_translate.batch_decode(output_encodes, skip_special_tokens=True)    
    return output[0].split(":", 1)[1]

def embedding_text(input_text):
    batch_dict = tokenizer_embedding(input_text, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model_embedding(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()[0].tolist()
    return embeddings

def translate_vi2eng(input_text):
    input_text = [f"vi: {input_text}"]
    output_encodes = model_translate.generate(tokenizer_translate(input_text, return_tensors="pt", padding=True).input_ids.to(device), max_length=2048)
    output = tokenizer_translate.batch_decode(output_encodes, skip_special_tokens=True)    
    return output[0].split(":", 1)[1]