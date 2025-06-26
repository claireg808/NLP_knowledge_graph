import requests
import os
import re
import json
import getpass
from langchain_openai import ChatOpenAI
from typing import List
from typing_extensions import Annotated, TypedDict

class Relations(TypedDict):
    head: Annotated[str, "The first entity"]
    head_type: Annotated[str, "The type of entity"]
    relation: Annotated[str, "The relationship between the two entities"]
    tail: Annotated[str, "The second entity"]
    tail_type: Annotated[str, "The type of entity"]

class FullOutput(TypedDict):
    title: Annotated[str, "The provided title"]
    abstract: Annotated[str, "The provided abstract"]
    relations: Annotated[List[Relations], "List of extracted relations from the text"]

# set headers for the request
headers = {"Content-Type": "application/json"}

# read the prompt template
with open('prompt_template.txt', 'r', encoding='utf-8') as file:
    prompt_template = file.read()

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

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    temperature=0.6,
    max_tokens=4500,
    timeout=120
)

structured_llm = llm.with_structured_output(FullOutput)

# process each sample
for pmid, sections in samples.items():
    title = sections.get('title', '')
    abstract = sections.get('abstract', '')

    # fill in prompt
    full_prompt = prompt_template \
                    .replace("{title_here}", title) \
                    .replace("{abstract_here}", abstract)

    try:
        response = structured_llm.invoke(full_prompt)

        # save JSON
        folder = "platinum_relations"
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, f"{pmid}_relations.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2)

        print(f"Success: Extracted relations for PMID {pmid} saved")

    except Exception as e:
        print(f"Error processing PMID {pmid}: {str(e)}")

print("\n\nComplete")