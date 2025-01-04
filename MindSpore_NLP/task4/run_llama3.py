import mindspore
from mindspore.communication import init
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "pretrainmodel/Meta-Llama-3-8B-Instruct"

init()
tokenizer = AutoTokenizer.from_pretrained(model_id, mirror='modelscope')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    ms_dtype=mindspore.float16,
    mirror='modelscope',
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Write a story about llamas"},
] 

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="ms"
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Use Constrained Beam-Search Decoding
cbd_output = model.generate(
    input_ids=input_ids,
    num_beams = 3,
    num_return_sequences=3,
    return_dict_in_generate=True,
    max_new_tokens=300,
)
print("Use Constrained Beam-Search Decoding:\n")
for i, cbd_output_sequence in enumerate(cbd_output.sequences):
    cbd_output_text = tokenizer.decode(cbd_output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {cbd_output_text}")

# Use Contrastive Search
cs_output = model.generate(
    input_ids=input_ids,
    penalty_alpha = 0.5,
    top_k = 30,
    return_dict_in_generate=True,
    max_new_tokens=300,
)
print("Use Contrastive Search:\n")
for i, cs_output_sequence in enumerate(cs_output.sequences):
    cs_output_text = tokenizer.decode(cs_output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {cs_output_text}")

# Use Greedy Decoding
gd_output = model.generate(
    input_ids=input_ids,
    num_beams = 1,
    do_sample = False,
    return_dict_in_generate=True,
    max_new_tokens=300,
)
print("Use Greedy Decoding:\n")
for i, gd_output_sequence in enumerate(gd_output.sequences):
    gd_output_text = tokenizer.decode(gd_output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {gd_output_text}")

# Use Multinomial Sampling
ms_output = model.generate(
    input_ids=input_ids,
    num_beams = 1,
    do_sample = True,
    temperature = 1.2,
    top_k = 100,
    top_p = 0.6,
    return_dict_in_generate=True,
    max_length=300,
)
print("Use Multinomial Sampling:\n")
for i, ms_output_sequence in enumerate(ms_output.sequences):
    ms_output_text = tokenizer.decode(ms_output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {ms_output_text}")

# demo
outputs = model.generate(
    input_ids,
    max_new_tokens=300,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print("Demo:\n")
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))