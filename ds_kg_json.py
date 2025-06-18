import requests
import os
import re
import json
import ast

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
        "max_tokens": 2048
    }

    response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=data, verify=False)

    if response.status_code == 200:
        assistant_text = response.json()['choices'][0]['text'].strip()
        start = assistant_text.find('[')
        end = assistant_text.find(']')+1
        relations_str = assistant_text[start:end]
        try:
            relations = ast.literal_eval(relations_str)
            r_pmid = int(pmid)
            relations_dict = {}
            for r in relations:
                r_pmid = r_pmid + 1
                relations_dict[r_pmid] = {
                    "head": r[0], 
                    "head_type": r[1], 
                    "relation": r[2], 
                    "tail": r[3],
                    "tail_type": r[4]
                }

            # save JSON
            folder = "platinum_relations"
            os.makedirs(folder, exist_ok=True)
            output_path = os.path.join(folder, f"{pmid}_relations.json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "title": title,
                    "abstract": abstract,
                    "relations": relations_dict
                }, f, indent=2)

            print(f"Success: Extracted relations for PMID {pmid} saved")

        except (SyntaxError, ValueError):
            print(f"Error: Invalid output returned for PMID {pmid}:\nrelations:{relations_str}\nwhole text:{assistant_text}")
    else:
        print(f"Error: Problem processing sample for PMID {pmid}: {response.status_code}")