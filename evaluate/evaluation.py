import json
import sys
# import os
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(root_dir)
# from src.api.controllers.controller import *
from controller_evaluate import *


def evaluate_pipeline_en(question):
    rephrased_question = chatbot_rephrase(question)
    question_embedding = embedding_text(rephrased_question)
    list_id, list_url = retrieval_context(question_embedding, 3)
    context = mapping_data(list_id, list_url)
    result, _ = chatbot_answering(rephrased_question, context)
    return result, list_id, list_url

def evaluate_pipeline_vi(question):
    question_translate = translate_vi2eng(question)
    rephrased_question = chatbot_rephrase(question_translate)
    question_embedding = embedding_text(rephrased_question)
    list_id, list_url = retrieval_context(question_embedding, 3)
    context = mapping_data(list_id, list_url)
    result, _ = chatbot_answering(rephrased_question, context)
    result_translated = translate_eng2vi(result)
    return result_translated, list_id, list_url

def evaluate_answer_accuracy(model_response, expected_answers):
    """Evaluate the model's response against expected answers."""
    model_response = model_response.lower()
    expected_answers = [answer.lower().strip() for answer in expected_answers]

    for answer in expected_answers:
        if answer in model_response:
            return 1  # Score of 1 if any expected answer is found
    return 0  # Score of 0 if no expected answer is found

def save_results_to_json(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def evaluate_model_qanews(data_file, save_dir, limit=200):
    with open(data_file, "r") as file:
        dataset = [json.loads(line) for line in file]
        
    results = []
    total_score = 0  # To keep track of the total score
    evaluated_count = 0  # To count how many questions have been evaluated

    # Iterate through each entry in the dataset
    for entry in dataset:
        text = entry["text"]
        questions_answers = entry["qa"]
        
        for qa in questions_answers:
            if limit is not None and evaluated_count >= limit:
                break  # Stop if the limit is reached
            
            question = qa["question"]
            expected_answers = qa["answers"]

            model_response = evaluate_pipeline(question)  
            
            # Evaluate the model's response and accumulate the score
            score = evaluate_answer_accuracy(model_response, expected_answers)
            if score > 0:
                total_score += score
                print(f"total_score: {total_score}")

            # Store the results for evaluation
            result_entry = {
                "question": question,
                "expected_answers": expected_answers,
                "model_response": model_response,
                "score": score
            }
            results.append(result_entry)

            evaluated_count += 1  # Increment the evaluated count
            if evaluated_count % 10 == 0:
                print(f"Evaluating round: {evaluated_count}")

            # Save results every 10 score
            if total_score % 5 == 0:
                score_file = f"{save_dir}/score_{total_score}.json"
                save_results_to_json(results, score_file)

    # Filter results to keep only those with a score of +1
    good_answer = []
    bad_answer = []

    for result in results:
        if result['score'] == 1:
            good_answer.append(result)
        elif result['score'] == 0:
            bad_answer.append(result)

    return good_answer, bad_answer, total_score, evaluated_count

def evaluate_model_qanews(evaluate_pipeline, data_file, save_dir, limit=200):
    with open(data_file, "r") as file:
        dataset = [json.loads(line) for line in file]
        
    results = []
    total_score = 0  # To keep track of the total score
    evaluated_count = 0  # To count how many questions have been evaluated

    # Iterate through each entry in the dataset
    for entry in dataset:
        text = entry["text"]
        questions_answers = entry["qa"]
        
        for qa in questions_answers:
            if limit is not None and evaluated_count >= limit:
                break  # Stop if the limit is reached
            
            question = qa["question"]
            expected_answers = qa["answers"]

            model_response, list_id, list_url = evaluate_pipeline(question)  
            
            # Evaluate the model's response and accumulate the score
            score = evaluate_answer_accuracy(model_response, expected_answers)
            if score > 0:
                total_score += score
                print(f"total_score: {total_score} / evaluated_count: {evaluated_count}")

            # Store the results for evaluation
            result_entry = {
                "question": question,
                "expected_answers": expected_answers,
                "model_response": model_response,
                "score": score
            }
            results.append(result_entry)

            evaluated_count += 1  # Increment the evaluated count
            if evaluated_count % 10 == 0:
                print(f"Evaluating round: {evaluated_count}")

            # Save results every 10 score
            if total_score % 5 == 0:
                score_file = f"{save_dir}/score_{total_score}.json"
                save_results_to_json(results, score_file)

    # Filter results to keep only those with a score of +1
    good_answer = []
    bad_answer = []

    for result in results:
        if result['score'] == 1:
            good_answer.append(result)
        elif result['score'] == 0:
            bad_answer.append(result)

    return good_answer, bad_answer, total_score, evaluated_count

def evaluate_model_crawl(evaluate_pipeline, data_file, save_dir, limit=None):
    # Initialize an empty dataset        
    total_score = 0  # To keep track of the total score
    evaluated_count = 0  # To count how many questions have been evaluated
    dataset = []
    results = []
    # Filter results to keep only those with a score of +1
    good_answer = []
    bad_answer = []
    
    with open(data_file, "r", encoding='utf-8') as file:
        # Read each line in the input file
        for line in file:
            entry = json.loads(line.strip())  # Parse JSON from the line
            questions = entry.get('questions', [])
            answers = entry.get('answers', [])
            
            # Pair questions with answers using zip
            for question, expected_answer in zip(questions, answers):
                dataset.append({
                    "question": question,
                    "answers": [expected_answer]  # Wrap expected_answer in a list for consistency
                })

    # Iterate through each entry in the dataset
    for entry in dataset:
        question = entry["question"]
        expected_answers = entry["answers"]

        model_response, list_id, list_url = evaluate_pipeline(question)  
        
        # Evaluate the model's response and accumulate the score
        score = evaluate_answer_accuracy(model_response, expected_answers)
        if score > 0:
            total_score += score

        result_entry = {
            "question": question,
            "expected_answers": expected_answers,
            "model_response": model_response,
            "score": score,
            "list_id": list_id,
            "list_url": list_url
        }
        results.append(result_entry)
        if result_entry['score'] == 1:
            good_answer.append(result_entry)
        elif result_entry['score'] == 0:
            bad_answer.append(result_entry)

        evaluated_count += 1  # Increment the evaluated count

        # Define filenames based on the current evaluated_count
        final_answer_score_file = f"{save_dir}/final_answer_{eval_data_type}_round_{evaluated_count}.json"
        good_answer_output_file = f"{save_dir}/good_answer_{eval_data_type}_round_{evaluated_count}.json"
        bad_answer_output_file = f"{save_dir}/bad_answer_{eval_data_type}_round_{evaluated_count}.json"
        
        # Save results every 5 evaluations
        if evaluated_count % 15 == 0 and evaluated_count > 0:      
            save_results_to_json(results, final_answer_score_file)
            save_results_to_json(good_answer, good_answer_output_file)
            save_results_to_json(bad_answer, bad_answer_output_file)
            print(f"Evaluating and save round: {evaluated_count}")
        
        print(f"Total_score: {total_score} / evaluated_count: {evaluated_count}")

    # Final save for any remaining results
    save_results_to_json(results, final_answer_score_file)
    save_results_to_json(good_answer, good_answer_output_file)
    save_results_to_json(bad_answer, bad_answer_output_file)
    print("Done evaluating and saving results")

    return total_score, evaluated_count

if __name__ == "__main__":
    save_dir = "/kaggle/working"

    if eval_data_type == "crawl":
        evaluate_pipeline = evaluate_pipeline_vi
        evaluate_model = evaluate_model_crawl
        data_file = "evaluate/data/eval_dict_qa_crawl_vn.txt"
    elif eval_data_type == "qanews":
        evaluate_pipeline = evaluate_pipeline_en
        evaluate_model = evaluate_model_qanews
        data_file = "evaluate/data/evaluation_data_qanews.txt"

    total_score, evaluated_count = evaluate_model(evaluate_pipeline = evaluate_pipeline, data_file = data_file, save_dir = save_dir, limit=200) # if you use crawl data for evaluation, the limit parameter is not importatnt dumbass

    # Print the total score and number of evaluated questions
    print(f"Total Score: {total_score}")
    print(f"Evaluated Questions: {evaluated_count}")
    print(f"Result files saved to: {save_dir}")
