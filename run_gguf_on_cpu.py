from llama_cpp import Llama
import os

llm = Llama(model_path="model/Mistral_7B_DPO_Pandas-q8_0.gguf", chat_format="llama-2",n_ctx=850)


prompt = """
Below is an instruction that describes a task. Write a Python function using Pandas to accomplish the task described below.

### Instruction:
how many unique items are there

header columns with sample data:
data = {
    "invoice_date": ["2023-09-11", "2023-09-12", "2023-09-13", "2023-09-14", "2023-09-15"],
    "customer_code": ["C006", "C007", "C006", "C008", "C009"],
    "customer_name": ["Frank", "Grace", "Frank", "Hannah", "Ian"],
    "gender": ["Male", "Female", "Male", "Female", "Male"],
    "item_code": ["I005", "I004", "I003", "I002", "I001"],
    "item_name": ["Widget E", "Widget D", "Widget C", "Widget B", "Widget A"],
    "unit_price": ["$40", "$20", "$10"],
    "quantity": [2, 3, 1, 4, 5],
    "total_amount": [100, 120, 30, 80, 50],
    "tax": [10, 12, 3, 8, 5],
    "currency": ["USD", "USD", "USD", "EUR", "EUR"],
    "discount": [0.05, 0, 0, 0.1, 0.05],
    "total_discount_after": [95, 120, 30, 72, 47.5]
}

### Response:
"""

output = llm(
  prompt, 
  max_tokens=850,  
  stop=["### End"],   
  echo=True        
)

print(output['choices'][0]['text'])