# NLP-Question-Generation-with-RAG
Finetuned a T5-base model to generate questions of various types and difficulty levels based on provided context

You can find more about the model on the Huggingface library 
https://huggingface.co/Avinash250325/T5BaseQuestionGeneration

Try it out in the Huggingface Spaces
https://huggingface.co/spaces/Avinash250325/Question_Generation_with_RAG

--- 
## Finetuned T5-Base Question Generator Model

This model is a fine-tuned T5 model designed specifically for **automatic question generation** from any given context or passage. It supports different types of questions like **short answer**, **multiple choice question**, and **true or false quesiton**, while also allowing customization by **difficulty level** — easy, medium or hard.

---

## Why is this Project Important?

Educational tools, tutoring platforms, and self-learning systems need a way to **generate relevant questions** automatically from content. Our model bridges that gap by providing a flexible and robust question generation system using a **structured prompt** format and powered by a **fine-tuned `T5-base` model**.
  
### Key Features

- Supports **multiple question types**:  
  - Short answer  
  - Multiple choice  
  - True/false  

- Questions are generated based on:  
  - The **provided context**  
  - The **type of question**  
  - The **difficulty level**  

- Difficulty reflects the **reasoning depth** required (multi-hop inference).

- Uses a **structured prompt format** with clearly defined tags, making it easy to use or integrate into other systems.

- Fine-tuned from the `t5-base` model:  
  - Lightweight and fast  
  - Easy to run on CPU  
  - Ideal for customization by teachers or Educational platforms

### Ideal For

- Teachers creating quizzes or exam material
- EdTech apps generating practice questions  
- Developers building interactive learning tools  
- Automated assessment and content enrichment

### Bonus: Retrieval-Augmented Generation (RAG)

A **custom RAG function** is also provided. This enables question generation from larger content sources like textbooks:

- Input can be a **subheading** or **small excerpt** from a textbook.
- The model fetches relevant supporting context form the textbook using a retirever.
- Generates questions grounded in the fetched material.

This extends the model beyond single-passage generation into more dynamic, scalable educational use cases.


---

## Prompt Format

To generate good quality questions, the model uses a **structured input prompt** format with special tokens. This helps the model understand the intent and expected output type.


### Prompt Fields:
- `<extra_id_97>` – followed by the **question type**  
  - `short answer`, `multiple choice question`, or `true or false question`
- `<extra_id_98>` – followed by the **difficulty**  
  - `easy`, `medium`, or `hard`
- `<extra_id_99>` – followed by **[optional answer] context** 
  - `optional answer` – for targeted question generation, or you can leave it as blank
  - `context` – the main passage/content from which questions are generated



### Helper Function to Create the Prompt

To simplify prompt construction, use this Python function:

```python
def format_prompt(qtype, difficulty, context, answer=""):
    """
    Format input prompt for question generation
    """
    answer_part = f"[{answer}]" if answer else ""
    return f"<extra_id_97>{qtype} <extra_id_98>{difficulty} <extra_id_99>{answer_part} {context}"
```

---

## How to Use the Model

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
# Load model from Hugging Face Hub
model_name = "Avinash250325/T5BaseQuestionGeneration"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
# Format input prompt
def format_prompt(qtype, difficulty, context, answer=""):
    answer_part = f"[{answer}]" if answer else ""
    return f"<extra_id_97>{qtype} <extra_id_98>{difficulty} <extra_id_99>{answer_part} {context}"
context = "The sun is the center of our solar system."
qtype = "short answer"     # qtype: ("short answer", "multiple choice question", "true or false question")
difficulty = "easy"        # difficulty: ("easy", "medium", "hard")
prompt = format_prompt("short answer", "easy", context)
# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150)
# Decode output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


