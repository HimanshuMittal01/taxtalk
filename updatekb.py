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
        url = "http://localhost:8080",
        additional_headers = {}
    )

    # Recreate full database from scratch
    client.schema.delete_class("Circular")

    # Create database for circulars
    class_obj = {
        "class": "Circular",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-openai": {},
            "generative-openai": {}
        }
    }
    client.schema.create_class(class_obj)

    # Load and add batches to weaviate
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
