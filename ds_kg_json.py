import requests
import os
import re
import json
import ast
from strictjson import *


def llm(system_prompt: str, user_prompt: str) -> str:
    data = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B",
        "prompt": system_prompt+ '\n\n' + user_prompt,
        "temperature": 0.6,
        "max_tokens": 2048
    }

    response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=data, verify=False)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}")

    return response.json()['choices'][0]['text'].strip()


# set headers for the request
headers = {"Content-Type": "application/json"}

# read the prompt template
with open('sys_prompt_template.txt', 'r', encoding='utf-8') as file:
    sys_prompt_template = file.read()
with open('usr_prompt_template.txt', 'r', encoding='utf-8') as file:
    usr_prompt_template = file.read()

# read in output format
with open('output_format.txt', 'r', encoding='utf-8') as file:
    output_format_template = file.read()

# read the example
with open('platinum_example.txt', 'r', encoding='utf-8') as file:
    example = file.read()

# read the sample text
with open('test.txt', 'r', encoding='utf-8') as file:
    all_samples = file.read().splitlines()

# extract and group title + abstract by PMID
title_pattern = re.compile(r"^(\d{8})\|t\|(.*)")
abstract_pattern = re.compile(r"^(\d{8})\|a\|(.*)")

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
    try:
        title = sections.get('title', '')
        abstract = sections.get('abstract', '')

        current_usr_prompt = usr_prompt_template \
                        .replace("{title_here}", title) \
                        .replace("{abstract_here}", abstract)

        response = strict_json(
                        system_prompt = sys_prompt_template,
                        user_prompt = current_usr_prompt,
                        output_format = output_format_template,
                        llm = llm,
                        return_as_json = True
                    )
                                        
        # save JSON
        folder = "platinum_relations"
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, f"{pmid}_relations.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response)

        print(f"Success: Extracted relations for PMID {pmid} saved")

    except Exception as E:
        print(f"Error processing PMID {pmid}: {str(e)}")
