from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto", trust_remote_code=True)

labeled_scores = ['Negative: 0.17668890953063965', 'Neutral: 0.5283784866333008', 'Positive: 0.29493260383605957']

def generate(labeled_scores):
    prompt = f"Generate musical parameters based on emotional data: {labeled_scores}. Instructions: Compose a musical piece that reflects the given emotional data. Use the negativity, neutrality, and positivity values to influence the musical parameters. Consider tempo, key, instrumentation, and dynamics. For example, a higher negativity value might correspond to a slower tempo or a darker tone. Be creative and explore the connection between emotions and music. Feel free to provide additional details or specify the musical genre if desired."

    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3
    )

    output = tokenizer.decode(output_ids[0][token_ids.size(1):])

    print(output)

generate(labeled_scores)