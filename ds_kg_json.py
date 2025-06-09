import requests
import os
import re
import json

json_entries = {}

# set headers for the request
headers = {"Content-Type": "application/json"}

# read the prompt template
with open('prompt_template.txt', 'r', encoding='utf-8') as file:
    prompt_template = file.read()

# read the example
with open('platinum_example.txt', 'r', encoding='utf-8') as file:
    example = file.read()

# read the sample text
with open('articles_train_platinum.txt', 'r', encoding='utf-8') as file:
    all_samples = file.read().splitlines()

# extract and group title + abstract by PMID
title_pattern = re.compile(r"^(\d{8})\|t\|(.*)")
abstract_pattern = re.compile(r"^(\d{8})\|a\|(.*)")

samples = {}
for line in all_samples:
    t_match = title_pattern.match(line)
    a_match = abstract_pattern.match(line)

    if t_match:
        pmid = t_match.group(1)
        title = t_match.group(2).strip()
        samples.setdefault(pmid, {})['title'] = title
    elif a_match:
        pmid = a_match.group(1)
        abstract = a_match.group(2).strip()
        samples.setdefault(pmid, {})['abstract'] = abstract

# process each sample
for pmid, sections in samples.items():
    title = sections.get('title', '')
    abstract = sections.get('abstract', '')

    # fill in prompt
    full_prompt = prompt_template \
                    .replace("{example}", example) \
                    .replace("{title_here}", title) \
                    .replace("{abstract_here}", abstract)

    data = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "prompt": full_prompt,
        "temperature": 0.6,
        "max_tokens": 2400
    }

    response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=data, verify=False)

    if response.status_code == 200:
        assistant_text = response.json()['choices'][0]['text'].strip()
        json_match = re.search(r"\{.*\}", assistant_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group()
                relations = json.loads(json_str)

                json_entries[pmid] = {
                    "metadata": {
                        "title": title,
                        "abstract": abstract
                    },
                    "relations": relations
                }

                print(f"[✓] Extracted relations for PMID {pmid} saved")

            except json.JSONDecodeError:
                print(f"[⚠️] Invalid JSON returned for PMID {pmid}:\n{assistant_text}")
    else:
        print(f"[❌] Error processing sample for PMID {pmid}: {response.status_code}")

# save JSON
output_folder_name = "platinum_data"
folder = output_folder_name
os.makedirs(folder, exist_ok=True)

with open(output_folder_name, 'w', encoding='utf-8') as f:
    json.dump(json_entries, f, indent=2)