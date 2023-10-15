"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @github    : https://github.com/HimanshuMittal01
 @created on: 06-10-2023 21:39:01
"""

import inspect
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
    Context : 
        """
        {context}
        """

    2. Answer the question using only this context
    3. Show the source for your answers

    If you don't have any context and are unsure of the answer, reply that you don't know about this topic.
    It is not needed to repeat the context as it is, a brief summary is sufficient.

    User Question: {prompt}

    ### Response:

    '''
    
    return prompt_template


def build_context(response, obj_class: str = "Article") -> str:
    if obj_class == "Circular":
        return response["data"]["Get"][obj_class][0]["description"]

    # Get properties
    name = response["data"]["Get"][obj_class][0]["name"]
    description = response["data"]["Get"][obj_class][0]["description"]
    context = inspect.cleandoc(
        f"""
        Name: {name}
        Description: {description}
        """
    )
    return context


def asktaxtalk(question: str, tokenizer, model) -> str:
    # Find context
    obj_class = "Article" # Circular
    obj_properties = ["article_id", "name", "description"] # ["description", "date"]
    response = (
        client.query
        .get(obj_class, obj_properties)
        .with_near_text({"concepts": [f"{question}"]})
        .with_limit(2)
        .do()
    )

    # Build context
    context = build_context(response)

    # Convert to template
    prompt = [
        # {"role": "system", "content": ""},
        {"role": "user", "content": inspect.cleandoc(
                f"""
                You are an assistant that helps draft replies for GST litigation. You have to answer the questions using the provided context. Reference source name wherever needed. If you don't have any context and are unsure of the answer, reply that you don't know about this topic.

                Following context can be considered having 'name' and 'description':
                {context}

                Question: {question}
                """
            )
        },
    ]
    # prompt = convert_to_template(question, context)

    # Encode prompt
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").cuda()

    # Model inference
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)

    # Decode answer
    answer = tokenizer.decode(output[0])
    return answer
