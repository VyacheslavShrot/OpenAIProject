import json

from openai import OpenAI

from utils.get_env import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

data = []

with open('../static/dataset.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))

ds_formatted = [
    {
        "messages":
            [
                {
                    "role": "system", "content": "You are a smart assistant who replies only place name on the Earth map"
                },
                {
                    "role": "user", "content": x['question']
                },
                {
                    "role": "assistant", "content": x['answer']
                }
            ]
    } for x in data
]

ds_train = ds_formatted[:25]
ds_eval = ds_formatted[25:50]

with open('../static/train.jsonl', 'w') as file:
    for line in ds_train:
        json.dump(line, file)
        file.write('\n')

with open('../static/eval.jsonl', 'w') as file:
    for line in ds_eval:
        json.dump(line, file)
        file.write('\n')

train = client.files.create(
    file=open('../static/train.jsonl', 'rb'),
    purpose='fine-tune'
)

evaluation = client.files.create(
    file=open('../static/eval.jsonl', 'rb'),
    purpose='fine-tune'
)

fine_tune_job = client.fine_tuning.jobs.create(
    training_file=train.id,
    validation_file=evaluation.id,
    model='gpt-3.5-turbo'
)
print(fine_tune_job)
