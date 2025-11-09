from pathlib import Path
import json
import math

def assemble_prompt(query: str):
    return f"""You are an AI Visual QA assistant. I will provide you with a question and several images. Please follow the four steps below:
Step 1: Observe the Images
First, analyze the question and consider what types of images may contain relevant information.
Then, examine each image one by one, paying special attention to aspects related to the question.
Identify whether each image contains any potentially relevant information.
Wrap your observations within <observe></observe> tags.

Step 2: Record Evidences from Images
After reviewing all images, record the evidence you find for each image within <evidence></evidence> tags.
If you are certain that an image contains no relevant information, record it as: [i]: no relevant information(where i denotes the index of the image).
If an image contains relevant evidence, record it as: [j]: [the evidence you find for the question](where j is the index of the image).

Step 3: Reason Based on the Question and Evidences
Based on the recorded evidences, reason about the answer to the question.
Include your step-by-step reasoning within <think></think> tags.

Step 4: Answer the Question
Provide your final answer based only on the evidences you found in the images.
Wrap your answer within <answer></answer> tags.
Avoid adding unnecessary contents in your final answer, like if the question is a yes/no question, simply answer "yes" or "no".
If none of the images contain sufficient information to answer the question, respond with <answer>insufficient to answer</answer>.

Formatting Requirements:
Use the exact tags <observe>, <evidence>, <think>, and <answer> for structured output.
It is possible that none, one, or several images contain relevant evidence.
If you find no evidence or few evidences, and insufficient to help you answer the question, follow the instruction above for insufficient information.

Question and images are provided below. Please follow the steps as instructed.
Question:
{query}
"""

def assemble_query(**args):
    topic = args.get('topic', '')
    metric = args.get('metric', '')
    value = args.get('value', '')
    unit = args.get('unit', '')

    if type(value) is float and math.isnan(value):
        value = ''
    if type(unit) is float and math.isnan(unit):
        unit = ''
    
    return f"""
Here is the topic of information, also, metric that may occured in document and corresponding value and unit.
You should judge whether the mentioned topic, metric and corresponding value occured in the document provided or not.
Value and Unit could be empty if metric cannot be qualified.
The answer should be 'yes', 'yes but not complete' or 'no'.
Topic: {topic}
Metric: {metric}
Value: {value}
Unit: {unit}
"""

def get_file_from_cid(cid):
    cid_map = {
	    'alchip':'Alchip',
	    'esun':'E.SUN',
	    'fpcc':'FPCC',
	    'gtg':'GTG',
	    'inx':'INX',
	    'kye':'KYE',
	    'largan':'LARGAN',
	    'mfhc':'MFHC',
	    'npc':'NPC',
	    'pegavision':'pegavision',
	    'psi':'PSI',
	    'spt':'SPT',
	    'standard':'Standard',
	    'tcfh':'TCFH',
	    'tsmc':'TSMC',
    }
    return cid_map[cid]

def locate_target(metrics: list[dict], sid):
    for codes in metrics:
        for code in codes['codes']:
            for metric in code['metrics']:
                if metric['sid'] == sid:
                    return metric

    return None

def generate_prompt(case: dict, metrics: list[dict]) -> str:
    target = None

    target = locate_target(metrics, case['sid'])

    args = {
        'topic': target['topic'],
        'metric': target['metric'],
        'value': case['value'],
        'unit': case['unit'],
    }

    query = assemble_query(**args)
    prompt = assemble_prompt(query)

    return prompt

def get_prompt(language: str, case: dict) -> str:
    """Return the sum of two integers.

    Args:
        language (str): The language of dataset.
        case (str): one record of labeled data.

    Returns:
        str: The prompt used as input for model inferencing.

    Raises:

    """

    # Change root path to which you store reports image and metrics
    root = {
        "chinese": "/path/to/data/Chinese"
    }
    
    data_root = Path(root[language.lower()])
    metric_path = data_root / 'reports/metric' / f'{case["cid"]}.json'
    fd = open(metric_path, 'r')
    metrics = json.load(fd)
    fd.close()

    prompt = generate_prompt(case, metrics)

    return prompt

def main():
    annotation_path = '/path/to/test.json'
    fd = open(annotation_path, 'r')
    cases = json.load(fd)
    fd.close()

    case = cases[0]

    prompt = get_prompt('Chinese', case)

    print(prompt)

if __name__ == '__main__':
    main()
