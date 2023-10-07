"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @github    : https://github.com/HimanshuMittal01
 @created on: 06-10-2023 21:39:01
"""

import weaviate

# Connect to client
client = weaviate.Client(
    url = "http://localhost:8080",
    additional_headers = {}
)

def convert_to_template(prompt: str, context: str = "", model_name: str = "TheBloke/OpenOrca-Platypus2-13B-GPTQ") -> str:
    prompt_template=f'''
    ### Instruction:

    Follow exactly those 3 steps:
    1. Read the context in the triple quotes below and aggregrate this data
    Context : """{context}"""
    2. Answer the question using only this context
    3. Show the source for your answers

    If you don't have any context and are unsure of the answer, reply that you don't know about this topic.
    It is not needed to repeat the context as it is, a brief summary is sufficient.

    User Question: {prompt}

    ### Response:

    '''
    
    return prompt_template


def build_context(matched_docs):
    pass


def asktaxtalk(prompt: str, tokenizer, model) -> str:
    # Find context
    response = (
        client.query
        .get("Circular", ["description", "date"])
        .with_near_text({"concepts": [f"{prompt}"]})
        .with_limit(2)
        .do()
    )

    # Build context
    context = response["data"]["Get"]["Circular"][0]["description"]

    # Convert to template
    prompt_template = convert_to_template(prompt, context)

    # Model inference
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)

    # Decode answer
    answer = tokenizer.decode(output[0])
    return answer
