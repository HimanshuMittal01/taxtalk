"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @github    : https://github.com/HimanshuMittal01
 @created on: 06-10-2023 21:39:01
"""


def convert_to_template(prompt: str, model_name: str = "TheBloke/OpenOrca-Platypus2-13B-GPTQ") -> str:
    prompt_template=f'''
    ### Instruction:

    {prompt}

    ### Response:

    '''
    
    return prompt_template


def asktaxtalk(prompt: str, tokenizer, model) -> str:
    # Convert to template
    prompt_template = convert_to_template(prompt)

    # Model inference
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)

    # Decode answer
    answer = tokenizer.decode(output[0])
    return answer
