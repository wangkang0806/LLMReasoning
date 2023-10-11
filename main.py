from tools.llm import ChatGpt, BaseModel
import jsonlines
from data_utils.dataset import gsm8k
import os
import re
from tqdm import tqdm
# from tools.google_api import translate_text
from tools.change2excel import change2excel,change2excel_base

import logging
logging.getLogger('backoff').addHandler(logging.StreamHandler())


def evaluation(predictions, labels,final_dataset=None,output_dir=None):
    """给出分数

    Args:
        predictions (_type_): _description_
        labels (_type_): _description_
    """
    dataset_num  = len(labels)
    acc_num = 0
    error_idx =[]
    output_file = f"{output_dir}/augmentation_final_responses.txt"
    error_file = f"{output_dir}/aug_error_questions.jsonl"
    base_error_file = f"{output_dir}/base_error_questions.jsonl"
    answer_file = f"{output_dir}/answers.txt"
    question_file = f"{output_dir}/dataset.txt"
    for i in range(dataset_num):
        
        # 去除逗号
        label = labels[i].replace(',','')
        
        if predictions[i]!='':
            try:
                pred = eval(predictions[i].strip().replace(',',''))
            except:
                print(f"不合法答案 line {i} content: {predictions[i]} ")
                pred = -1
            
        else:
            print("第{}列未能提取成功".format(i+1)) 
            pred=1e9
        
        label = eval(label)
        if pred-label == 0.0:
            acc_num += 1
        elif abs(pred-label) <= 1e-9:
            acc_num+=1
            print(f"exist float error. pred is {pred} label is {label}")
        else : error_idx.append(i+1)
    
    # extract error question
    if final_dataset != None:
        with open(output_file) as f:
            final_responses = f.readlines()
            final_responses = [eval(i.strip()) for i in final_responses]
        with open(answer_file) as f:
            answer_file = f.readlines()
            answer_file = [(i.strip()) for i in answer_file]
        with open(question_file) as f:
            question_file = f.readlines()
            question_file = [(i.strip()) for i in question_file]
        for i in range(len(error_idx)):
            idx = error_idx[i]-1
            origin_answer = answer_file[idx]
            origin_question = question_file[idx]
            error_response = final_responses[idx]
            error_question = final_dataset[idx]
            error_answer = str(predictions[idx])
            information = {"idx":str,"error_question":str,"error_response":str,"error_answer":str}
            information["idx"] = idx+1
            information["error_answer"] = error_answer
            information["error_question"] = error_question
            information["error_response"] = error_response
            
            # information = {"idx":str,"error_question_translate":str,"error_response_tranlate":str,"error_answer":str,"origin_question":str,"origin_answer":str}
            # information["idx"] = idx
            # information["error_answer"] = error_answer
            # information["error_question_translate"] = translate_text("zh-cn",error_question)
            # information["error_response_tranlate"]=translate_text("zh-cn",error_response)
            # information["origin_answer"] = origin_answer
            # information["origin_question"] = origin_question
            with jsonlines.open(error_file,'a')as f :
                f.write(information)
        change2excel(error_file,output_dir)
            
    return acc_num / dataset_num


def evaluation_base(predictions, labels, final_dataset=None,output_dir=None):
    """给出分数

    Args:
        predictions (_type_): _description_
        labels (_type_): _description_
    """
    # dataset_num  = len(labels)
    dataset_num  = len(labels)
    acc_num = 0
    error_idx =[]
    output_file = f"{output_dir}/baseline_raw_responses.txt"
   
    base_error_file = f"{output_dir}/base_error_questions.jsonl"
    answer_file = f"{output_dir}/answers.txt"
    question_file = f"{output_dir}/dataset.txt"
    for i in range(dataset_num):
        
        # 去除逗号
        label = labels[i].replace(',','')
        
        if predictions[i]!='':
            try:
                pred = eval(predictions[i].strip().replace(',',''))
            except:
                print(f"不合法答案 line {i} content: {predictions[i]} ")
                pred = -1
            
        else:
            print("第{}列未能提取成功".format(i+1)) 
            pred=1e9
        
        label = eval(label)
        if pred-label == 0.0:
            acc_num += 1
        elif abs(pred-label) <= 1e-9:
            acc_num+=1
            print(f"exist float error. pred is {pred} label is {label}")
        else : error_idx.append(i+1)
    
    # extract error question
    if final_dataset != None:
        with open(output_file) as f:
            final_responses = f.readlines()
            final_responses = [eval(i.strip()) for i in final_responses]
        with open(answer_file) as f:
            answer_file = f.readlines()
            answer_file = [(i.strip()) for i in answer_file]
        with open(question_file) as f:
            question_file = f.readlines()
            question_file = [(i.strip()) for i in question_file]
        for i in range(len(error_idx)):
            idx = error_idx[i]-1
            origin_answer = answer_file[idx]
            origin_question = question_file[idx]
            error_response = final_responses[idx]
            error_question = final_dataset[idx]
            error_answer = str(predictions[idx])
            information = {"idx":str,"error_question":str,"error_response":str,"error_answer":str}
            information["idx"] = idx+1
            information["error_answer"] = error_answer
            information["error_question"] = error_question
            information["error_response"] = error_response
            
            # information = {"idx":str,"error_question_translate":str,"error_response_tranlate":str,"error_answer":str,"origin_question":str,"origin_answer":str}
            # information["idx"] = idx
            # information["error_answer"] = error_answer
            # information["error_question_translate"] = translate_text("zh-cn",error_question)
            # information["error_response_tranlate"]=translate_text("zh-cn",error_response)
            # information["origin_answer"] = origin_answer
            # information["origin_question"] = origin_question
            with jsonlines.open(base_error_file,'a')as f :
                f.write(information)
        change2excel_base(base_error_file,output_dir)
            
    return acc_num / dataset_num


def evaluation_aug_precise(predictions, labels,final_dataset=None,output_dir=None):
    """给出分数

    Args:
        predictions (_type_): _description_
        labels (_type_): _description_
    """
    dataset_num  = len(labels)
    acc_num = 0
    error_idx =[]
    output_file = f"{output_dir}/aug_precise_extracted_answer.txt"
    error_file = f"{output_dir}/aug_precise_error_questions.jsonl"
    base_error_file = f"{output_dir}/base_error_questions.jsonl"
    answer_file = f"{output_dir}/answers.txt"
    question_file = f"{output_dir}/dataset.txt"
    for i in range(dataset_num):
        
        # 去除逗号
        label = labels[i].replace(',','')
        
        if predictions[i]!='':
            try:
                pred = eval(predictions[i].strip().replace(',',''))
            except:
                print(f"不合法答案 line {i} content: {predictions[i]} ")
                pred = -1
            
        else:
            print("第{}列未能提取成功".format(i+1)) 
            pred=1e9
        
        label = eval(label)
        if pred-label == 0.0:
            acc_num += 1
        elif abs(pred-label) <= 1e-9:
            acc_num+=1
            print(f"exist float error. pred is {pred} label is {label}")
        else : error_idx.append(i+1)
    
    # extract error question
    if final_dataset != None:
        with open(output_file) as f:
            final_responses = f.readlines()
            final_responses = [i.strip() for i in final_responses]
        with open(answer_file) as f:
            answer_file = f.readlines()
            answer_file = [(i.strip()) for i in answer_file]
        with open(question_file) as f:
            question_file = f.readlines()
            question_file = [(i.strip()) for i in question_file]
        for i in range(len(error_idx)):
            idx = error_idx[i]-1
            origin_answer = answer_file[idx]
            origin_question = question_file[idx]
            error_response = final_responses[idx]
            error_question = final_dataset[idx]
            error_answer = str(predictions[idx])
            information = {"idx":str,"error_question":str,"error_response":str,"error_answer":str}
            information["idx"] = idx+1
            information["error_answer"] = error_answer
            information["error_question"] = error_question
            information["error_response"] = error_response
            
            # information = {"idx":str,"error_question_translate":str,"error_response_tranlate":str,"error_answer":str,"origin_question":str,"origin_answer":str}
            # information["idx"] = idx
            # information["error_answer"] = error_answer
            # information["error_question_translate"] = translate_text("zh-cn",error_question)
            # information["error_response_tranlate"]=translate_text("zh-cn",error_response)
            # information["origin_answer"] = origin_answer
            # information["origin_question"] = origin_question
            with jsonlines.open(error_file,'a')as f :
                f.write(information)
        change2excel(error_file,output_dir)
            
    return acc_num / dataset_num


def extract_answer_by_chatgpt(question, response, dataset="gsm8k"):
    """find answer from question's response """
    if dataset=="gsm8k":
        find_answer = """Here is a math question and a model's answer about this question. Please extract the EXACT number from the answer txt as the final answer for question.
QUESTION: {}

ANSWER: {}
                
Final format should be a legal 'number' without any suffix such as '$'. If you know, simply answer 0.

The final answer is:
"""
        model_name = "gpt-3.5-turbo"
        api_key = "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"
        # api_key = "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"
        extract_answer_model = ChatGpt(model=model_name, api_key=api_key)
        extract_answer_model.rateLimit = {"RPM": 1000}
        out = extract_answer_model.generate(find_answer.format(question, response))
        return out
    else:
        print("不支持对该数据集使用chatgpt提取answer")
        exit(1)


def extract_answer_by_rule(questions, predictions:[str],dataset="gsm8k",output_dir=None):
    """适用于reasoning base"""   
    answer = [] 
    error_index = []
    # if dataset == "gsm8k":
    #     # for i in range(len(predictions)):
    #     #     answer_s = predictions[i].split("\\boxed")
    #     #     if len(answer_s) <=1:
    #     #         error_index.append(i)
    #     #         # 规则提取失败，需要使用chatgpt提取
    #     #         question = questions[i]
    #     #         response = predictions[i]
    #     #         write_answer = extract_answer_by_chatgpt(question, response, dataset="gsm8k")
    #     #     else:
    #     #         write_answer=re.findall(r'\d+',  answer_s[1])[0]
    #     #     answer.append(write_answer)
            
    #     for i in tqdm(range(len(predictions))):
    #         # 规则提取失败，需要使用chatgpt提取
    #         question = questions[i]
    #         response = predictions[i]
    #         write_answer = extract_answer_by_chatgpt(question, response, dataset="gsm8k")
    #         answer.append(write_answer)
    if dataset == "gsm8k":        
        answer_file = f"{output_dir}/baseline_extracted_responses.txt"
        if os.path.exists(answer_file):
            with open(answer_file,"r") as f:
                a = f.readlines()
                final_answers = [i.strip() for i in a]
        else:
            final_answers = []
        
        print(f"从第{len(final_answers)}开始提取答案中的数值")    
        for i in tqdm(range(len(final_answers),len(predictions))):
            question = questions[i]
            response = predictions[i]
            out = extract_answer_by_chatgpt(question, response, dataset="gsm8k")
            final_answers.append(out)
            
            with open(answer_file,'a') as an:
                an.write(str(out)+"\n")
    return final_answers
                

def reasoning_base(model:BaseModel, dataset, output_dir,batch_size):
    """推理baseline

    Args:
        model (str): llm model name
        dataset (list,Dataset): a dataset object contain inputs and labels
    """
    os.makedirs(output_dir,exist_ok=True)
    # run
    # current_file_path = os.path.abspath(__file__)
    dataset_inputs, dataset_labels, dataset_answers = dataset
    output_file = f"{output_dir}/baseline_raw_responses.txt"
    model.dataset_generate(dataset_inputs, output_file,batch_size=batch_size)
    # evaluation
    with open(output_file) as f:
        predictions = f.readlines()
        predictions =  [i.strip() for i in predictions]
    
    # TODO: 需要从外部传入数据集名字
    predictions_save_file = f"{output_dir}/baseline_extracted_responses.txt"
    if not os.path.exists(predictions_save_file) or len(open(predictions_save_file).readlines()) != len(dataset_inputs):
        extracted_predictions = extract_answer_by_rule( dataset_inputs, predictions=predictions, dataset="gsm8k",output_dir=output_dir)
        # with open(predictions_save_file, "w") as f:
        #     for prediction in extracted_predictions:
        #         f.write(str(prediction)+'\n')
    else:
        with open(predictions_save_file) as f:
            extracted_predictions = f.readlines()
            extracted_predictions = [i.strip() for i in extracted_predictions]
    acc = evaluation_base(extracted_predictions, dataset_labels,predictions,output_dir)
    return acc


def reasoning_augmentation(model:ChatGpt, dataset, output_dir,Batch_size):
    dataset_inputs, dataset_labels, dataset_answers = dataset

    # step1. extract core question
    print("开始抽取core question")
    core_question_prompt = " Please extract core question,only the most comprehensive and detailed's one!"
    # core_question_prompt = " Please extract core question,only the clearest and accurate's one!"
    
    core_question_datasets = [i+core_question_prompt for i in dataset_inputs]
    print("core question stage")
    output_file = f"{output_dir}/core_question_responses.txt"
    model.dataset_generate(core_question_datasets, output_file,batch_size=Batch_size)
    with open(output_file) as f:
        core_question_responses = f.readlines()
        core_question_responses = [eval(i.strip()) for i in core_question_responses]
    
    
    # step2. extract information
    print("开始提取有用信息")
    hints_datasets = []
    for i in range(len(dataset_inputs)):
        # hint_prompt = dataset_inputs[i] + " \nNote: Please extract the most useful information related to the core question( "+ core_question_responses[i]+"), Only extract the most useful information and don't repeat and redundancy, list them one by one!"
        hint_prompt = dataset_inputs[i] + " \nNote: Please extract the most useful information related to the core question( "+ core_question_responses[i]+"), Only extract the most useful information, list them one by one!"
        hints_datasets.append(hint_prompt)
    output_file = f"{output_dir}/useful_infomation_responses.txt"
    model.dataset_generate(hints_datasets, output_file,batch_size=Batch_size)
    with open(output_file) as f:
        useful_responses = f.readlines()
        useful_responses = [eval(i.strip()) for i in useful_responses]
        
    # step3. get the final answer
    print("使用增强prompt获取答案")
    final_datasets = []
    for i in range(len(dataset_inputs)):
        # final_prompt = dataset_inputs[i] + "\nHint:"+useful_responses[i]+ "\n"+core_question_responses[i]+ "\nPlease understand the Hint and question information comprehensively and give the answer carefully and give details!" 
        
        # final_prompt = dataset_inputs[i]+ "\nHint:"+useful_responses[i]+ "\n"+core_question_responses[i]+ "\nPlease understand the Hint and question information comprehensively and then give the answer carefully based on the question and Hint information and give details! " 
        
        final_prompt = "question:\n"+dataset_inputs[i]+ "\nHint:"+useful_responses[i]+ "\n"+core_question_responses[i]+"\nPlease fully understand the Hint and question information and integrated comprehensively, and then give the answer carefully and give details!"
         
        # final_prompt = dataset_inputs[i]+ "\nHint:"+useful_responses[i]+ "\nPlease understand the Hint and question information comprehensively , then give the answer carefully based on the question and Hint information and the core question:"+ "\n"+core_question_responses[i]+"and then give details! " 
        
        # final_prompt = ">>>>>>>>>>>>>>>question information>>>>>>>>>>>>>\n"+ dataset_inputs[i] + "\n>>>>>>>>>>>>>>>>Hint information>>>>>>>>>>>>>>>>\n"+"Hint:"+useful_responses[i]+ "\n"+core_question_responses[i]+"\n>>>>>inference the above information>>>>>>>>\n"+" Please understand the Hint and question information comprehensively and then give the answer carefully based on the question and Hint information and give details!"
        final_datasets.append(final_prompt)
    output_file = f"{output_dir}/augmentation_final_responses.txt"
    model.dataset_generate(final_datasets, output_file,batch_size=Batch_size)
    
    with open(output_file) as f:
        final_responses = f.readlines()
        final_responses = [eval(i.strip()) for i in final_responses]
    
    
    # step4. extract answer
    
    answer_file = f"{output_dir}/agumentation_answer.txt"
    if os.path.exists(answer_file):
        with open(answer_file,"r") as f:
            a = f.readlines()
            final_answers = [i.strip() for i in a]
    else:
        final_answers = []
    
    print(f"从第{len(final_answers)}开始提取答案中的数值")    
    for i in tqdm(range(len(final_answers),len(final_responses))):
        question = dataset_inputs[i]
        response = final_responses[i]
        out = extract_answer_by_chatgpt(question, response, dataset="gsm8k")
        
        with open(answer_file,'a') as an:
            an.write(str(out)+"\n")
            
    with open(answer_file) as f:
        final_answers = f.readlines()
        final_answers = [i.strip() for i in final_answers]
    
    return evaluation(final_answers,dataset_labels,final_datasets,output_dir)


def reasoning_augmentation_precise(model:BaseModel, dataset, output_dir, batch_size):
    """推理aug 精简版本

    Args:
        model (str): llm model name
        dataset (list,Dataset): a dataset object contain inputs and labels
    """
    os.makedirs(output_dir,exist_ok=True)
    # run
    # current_file_path = os.path.abspath(__file__)
    dataset_inputs, dataset_labels, dataset_answers = dataset
    output_file = f"{output_dir}/aug_precise_raw_responses.txt"
    prompt = """
Perform the following actions:
1 - Extract core question from following question delimited by triple backticks with 1 sentence, only the most comprehensive and detailed's one!
2 - Extract the the most useful information related to the core question from the same question delimited by triple backticks and list them one by one!
3 - Answer of the question delimited by triple backticks with fully and comprehensively considering the core question and useful information. \
Make sure resolve these custom variables used in the reasoning process.
4 - Output a json object that contains the following keys: final_answer:<int|float>

Use the following format:
Text: <question to answer>
Core Question: <extracted core question>
Hints: <extracted useful information, listed one by one>
Reason Steps: <the reasoning steps to get the final answer>
Output Json: <json with the final answer>

Text: ```{}```
"""

    aug_datasets = [prompt.format(q) for q in dataset_inputs]
    model.dataset_generate(aug_datasets, output_file,batch_size=batch_size)
    # evaluation
    with open(output_file) as f:
        responses = f.readlines()
        responses =  [eval(i.strip()) for i in responses]
        extracted_answer = []
        for i in responses:
            t = i.split('Output Json:')[1].replace("'",'"')
            pattern = r'"final_answer":\s*(.*?)}'
            matches = re.findall(pattern, t, re.DOTALL)
            assert matches
            final_answer = matches[0].strip()
            extracted_answer.append(
                final_answer.replace('$','').replace("'",'').replace('"','')
            )
            
    
    # TODO: 需要从外部传入数据集名字
    predictions_save_file = f"{output_dir}/aug_precise_extracted_answer.txt"
    
    with open(predictions_save_file, 'w') as f:
        for a in extracted_answer:
            f.write(f"{a}\n")
    
    acc = evaluation_aug_precise(extracted_answer, dataset_labels, responses, output_dir)
    return acc


def main():
    # model_name = "gpt-4"
    model_name = "gpt-3.5-turbo-0613"
    # api_key = "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"
    api_key = "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"
    # api_key = "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"
    # api_key = "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"
    # api_key = "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"
    # api_key = "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"
    model = ChatGpt(model=model_name, api_key=api_key)
    model.rateLimit = {"RPM":200}
    model.temperature = 0
    # prepare datasetss
    sample_num = 100
    Bathch_size = 100
    dataset_inputs, dataset_labels, dataset_answers = gsm8k(sample_num=sample_num, seed=2023, split="test")
    
    dataset = (dataset_inputs, dataset_labels, dataset_answers)
    if sample_num:
        output_dir = "/home/ubuntu/users/wangkang/LLMReasoning/outputs/gsm8k/test_split_{}_samples".format(sample_num)
    else:
        output_dir = "/home/ubuntu/users/wangkang/LLMReasoning/outputs/gsm8k/whole_test_dataset_4_aug_acc_0.949_test"
    os.makedirs(output_dir, exist_ok=True)
    # save sampled dataset
    if not os.path.exists(os.path.join(output_dir, "dataset.txt")):
        with open(os.path.join(output_dir, "dataset.txt"), "w") as f:
            for data in dataset_inputs:
                f.write(str(data)+"\n")
    if not os.path.exists(os.path.join(output_dir, "labels.txt")):
        with open(os.path.join(output_dir, "labels.txt"), "w") as f:
            for data in dataset_labels:
                f.write(str(data)+"\n")
    if not os.path.exists(os.path.join(output_dir, "answers.txt")):
        with open(os.path.join(output_dir, "answers.txt"), "w") as f:
            for data in dataset_answers:
                f.write(str(data)+"\n")
                
    raw_acc = reasoning_base(model=model, dataset=dataset, output_dir=output_dir,batch_size=Bathch_size)
    # aug_acc = reasoning_augmentation(model=model, dataset=dataset, output_dir=output_dir,Batch_size=Bathch_size)
    aug_acc = reasoning_augmentation_precise(model=model, dataset=dataset, output_dir=output_dir,batch_size=Bathch_size)
    # print(f"raw acc is {raw_acc}")
    print(f"raw acc is {raw_acc}, aug acc is {aug_acc}")
    
if __name__ == '__main__':
    main()
    