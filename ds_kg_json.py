import requests
import os
import re
import json

# Set headers for the request
headers = {"Content-Type": "application/json"}

# Step 1: Read the prompt template
with open('ana_pipe_test.txt', 'r', encoding='utf-8') as file:
    prompt_template = file.read()

# Step 2: Read the sample text
with open('dev_full.txt', 'r', encoding='utf-8') as file:
    all_samples = file.read().splitlines()

# Step 3: Extract and group title + abstract by PMID
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

# Step 4: Process each sample
for pmid, sections in samples.items():
    title = sections.get('title', '')
    abstract = sections.get('abstract', '')

    # Fill in prompt
    full_prompt = prompt_template.replace("[TITLEHERE]", title).replace("[ABSTRACHERE]", abstract)

    data = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "prompt": full_prompt,
        "temperature": 0.6,
        "max_tokens": 2048
    }

    response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=data, verify=False)

    if response.status_code == 200:
        assistant_text = response.json()['choices'][0]['text'].strip()
        json_match = re.search(r"\{.*\}", assistant_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group()
                tagged_data = json.loads(json_str)

                tagged_title = tagged_data.get("tagged_title", "")
                tagged_abstract = tagged_data.get("tagged_abstract", "")

                # Save JSON
                entity_folder = "anatomical_location"
                os.makedirs(entity_folder, exist_ok=True)
                output_path = os.path.join(entity_folder, f"{pmid}_tagged.json")

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "tagged_title": tagged_title,
                        "tagged_abstract": tagged_abstract
                    }, f, indent=2)

                print(f"[✓] Tagged sample for PMID {pmid} saved to {output_path}")
                # else:
                #     print(f"YOU GOT A PROBLEM BIG DAWG")

            except json.JSONDecodeError:
                print(f"[⚠️] Invalid JSON returned for PMID {pmid}:\n{assistant_text}")
    else:
        print(f"[❌] Error processing sample for PMID {pmid}: {response.status_code}")
