import os
import pandas as pd
from llama_cpp import Llama


model_path = "model_path" 

print(f"Caricamento del modello GGUF da {model_path}...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,        
    n_gpu_layers=-1,   
    verbose=False      
)

# 2. Dataset
csv_path = "train/change-it.repubblica.train.csv"
df = pd.read_csv(csv_path)

sample_size = 5  # test
df_sample = df.head(sample_size).copy()

# 3. Prompt
prompts_config = {
    "task_1": {
        "instruction": "Genera un titolo per il seguente testo.\n\nTesto:\n{text}",
        "max_tokens": 50
    },
    "task_2": {
        "instruction": "Genera un testo completo basandoti sul seguente titolo.\n\nTitolo:\n{text}",
        "max_tokens": 400
    },
    "task_3": {
        "instruction": "Genera un riassunto per il seguente testo.\n\nTesto:\n{text}",
        "max_tokens": 150
    }
}

# 
results = {
    "t1_headline_temp_01": [], "t1_headline_temp_09": [],
    "t2_article_temp_01": [],  "t2_article_temp_09": [],
    "t3_summary_temp_01": [],  "t3_summary_temp_09": []
}

# 4. 
def generate_text_llamacpp(prompt_text, temp, max_tokens):
    # Formattazione formale dei tag di istruzione per Mistral v0.1
    formatted_prompt = f"<s>[INST] {prompt_text} [/INST]"
    
    # Esecuzione dell'interferenza tramite llama.cpp
    response = llm(
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        temperature=temp,
        top_p=0.9 if temp > 0.1 else 1.0,  # top_p disattivato (1.0) se la temperatura è quasi deterministica
        stop=["</s>", "[INST]"]            # Token di stop per evitare loop infiniti
    )
    
    # Estrazione del testo pulito dal dizionario di risposta
    generated_text = response["choices"][0]["text"]
    return generated_text.strip().replace("\n", " ")

# 5. E
print(f"\nInizio esperimenti con Mistral GGUF su {sample_size} record...")

for idx, row in df_sample.iterrows():
    print(f"\n--- Elaborazione Riga {idx+1}/{sample_size} ---")
    
    # --- TASK 1: Articolo -> Headline ---
    q1 = prompts_config["task_1"]["instruction"].format(text=row['full_text'])
    h_01 = generate_text_llamacpp(q1, 0.1, prompts_config["task_1"]["max_tokens"])
    h_09 = generate_text_llamacpp(q1, 0.9, prompts_config["task_1"]["max_tokens"])
    results["t1_headline_temp_01"].append(h_01)
    results["t1_headline_temp_09"].append(h_09)
    print(f"[Task 1] Completato.")

    # --- TASK 2: Headline -> Articolo ---
    q2 = prompts_config["task_2"]["instruction"].format(text=row['headline'])
    a_01 = generate_text_llamacpp(q2, 0.1, prompts_config["task_2"]["max_tokens"])
    a_09 = generate_text_llamacpp(q2, 0.9, prompts_config["task_2"]["max_tokens"])
    results["t2_article_temp_01"].append(a_01)
    results["t2_article_temp_09"].append(a_09)
    print(f"[Task 2] Completato.")

    # --- TASK 3: Articolo -> Summary ---
    q3 = prompts_config["task_3"]["instruction"].format(text=row['full_text'])
    s_01 = generate_text_llamacpp(q3, 0.1, prompts_config["task_3"]["max_tokens"])
    s_09 = generate_text_llamacpp(q3, 0.9, prompts_config["task_3"]["max_tokens"])
    results["t3_summary_temp_01"].append(s_01)
    results["t3_summary_temp_09"].append(s_09)
    print(f"[Task 3] Completato.")

# 6. 
for col_name, data_list in results.items():
    df_sample[col_name] = data_list

output_file = "esperimenti_mistral_gguf_diversita.csv"
df_sample.to_csv(output_file, index=False)
print(f"\n[FINE ESPERIMENTO] Dataset salvato con successo in: {output_file}")
