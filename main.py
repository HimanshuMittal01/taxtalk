"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @github    : https://github.com/HimanshuMittal01
 @created on: 06-10-2023 21:26:12
"""

from pathlib import Path
from taxtalk.model import load_model
from taxtalk.ask import asktaxtalk


if __name__ == "__main__":
    # Load model
    tokenizer, model = load_model("TheBloke/openOrca-Platypus2-13B-GPTQ")

    # User asks a question / prompt
    question = "Write a draft letter replying to summon without advance information"

    # Chatbot generates an answer
    answer = asktaxtalk(question, tokenizer=tokenizer, model=model)

    # Display
    output_dir = Path("output/")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir / "output.txt"), "w") as f:
        f.write(answer)
