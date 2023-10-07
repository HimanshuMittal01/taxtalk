"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @github    : https://github.com/HimanshuMittal01
 @created on: 07-10-2023 17:46:25
"""

import weaviate
import json

if __name__ == "__main__":
    # Create client
    client = weaviate.Client(
        url = "http://0.0.0.0:8080",
        additional_headers = {}
    )

    # Recreate full database from scratch
    client.schema.delete_class("Circular")
    client.schema.delete_class("Article")

    # Create database for circulars
    circular_obj_config = {
        "class": "Circular",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-openai": {},
            "generative-openai": {}
        }
    }
    client.schema.create_class(circular_obj_config)

    # Create database for articles
    circular_obj_config = {
        "class": "Article",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-openai": {},
            "generative-openai": {}
        }
    }
    client.schema.create_class(circular_obj_config)

    # Load and add batches to weaviate [CIRCULAR]
    data = None
    with open("input/circulars.json") as f:
        data = json.load(f)

    # Configure batch
    client.batch.configure(batch_size=100)
    
    # Initialize a batch process
    with client.batch as batch:
        for i, d in enumerate(data["active_circulars"]):
            print(f"importing question: {i+1}")
            properties = {
                "circular_id": d["id"],
                "description": d["description"],
                "date": d["date"],
            }
            batch.add_data_object(
                data_object=properties,
                class_name="Circular"
            )
    
    # Load and add batches to weaviate [ARTICLE]
    data = None
    with open("input/articles.json") as f:
        data = json.load(f)

    # Configure batch
    client.batch.configure(batch_size=100)
    
    # Initialize a batch process
    with client.batch as batch:
        for d in data["constitution_of_india"]:
            print(f"importing article: {d['id']}")
            properties = {
                "article_id": d["id"],
                "name": d["name"],
                "part": d["part"],
                "description": d["description"],
            }
            batch.add_data_object(
                data_object=properties,
                class_name="Article"
            )
