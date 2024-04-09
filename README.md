# Optimizing LM for Text2SQL using DSPy 

## What is in this project?
In this project, I decided to test DSPy for the task of converting natural language to SQL syntax. For this, I used a 7 billion parameter open-source model ([Nexusflow/Starling-LM-7B-beta](https://huggingface.co/Nexusflow/Starling-LM-7B-beta)) to see if I can reach GPT-3.5-Turbo performance levels, or even surpass them. The cool thing about this is that with DSPy, you can actually focus on programming rather than creating overcrafted prompts that are not failure proof. I invite you to see the main notebook that contains the entire process. I think you will find it really useful. These were the main results:

| Model | Optimization | Validation Correctness (n = 25) | Test Correctness (n = 75) |
| ----- | ------------ | ------------------------------- | ------------------------- |
| GPT 3.5 Turbo | N/A | 80.0% | 70.07% |
| Starling7B | N/A | 72.0% | 50.67% |
| Starling7B | LabeledFewShot | 64.0% | 60.0% |
| Starling7B | BootstrapFewShotWithRandomSearch | 80.0% | 68.0% |

## Repo Structure
| Stage                                    | Notebook/Script                                                                                                 | Tech Stack                     |
|------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------|
| Serverless Deployment of Starling 7B     | [model_serve/](https://github.com/jjovalle99/DSPy-Text2SQL/tree/23a0a347db2d7515c5a28c305dacaea00d09dddc/model_serve)  | vLLM, Modal, HuggingFace       |
| DSPy Optimization for Text2SQL           | [DSPy-Text2SQL.ipynb](https://github.com/jjovalle99/DSPy-Text2SQL/blob/23a0a347db2d7515c5a28c305dacaea00d09dddc/DSPy-Text2SQL.ipynb)     | DSPy, HuggingFace |

## What is DSPy?
#### [Docs](https://dspy-docs.vercel.app/)

_DSPy is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used one or more times within a pipeline. To use LMs to build a complex system without DSPy, you generally have to: (1) break the problem down into steps, (2) prompt your LM well until each step works well in isolation, (3) tweak the steps to work well together, (4) generate synthetic examples to tune each step, and (5) use these examples to fine-tune smaller LMs to cut costs. Currently, this is hard and messy: every time you change your pipeline, your LM, or your data, all prompts (or fine-tuning steps) may need to change._

_To make this more systematic and much more powerful, DSPy does two things. First, it separates the flow of your program (modules) from the parameters (LM prompts and weights) of each step. Second, DSPy introduces new optimizers, which are LM-driven algorithms that can tune the prompts and/or the weights of your LM calls, given a metric you want to maximize._