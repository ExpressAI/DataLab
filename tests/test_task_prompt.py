from datalabs import load_dataset

dataset = load_dataset("ag_news")

prompts = dataset['test']._info.prompts

print(prompts)


for prompt in prompts:
    print(prompt)
    print("--------------\n")