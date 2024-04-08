# Optimizing LM for Text2SQL using DSPy 

## What is in this project?
In this project, I decided to test DSPy for the task of converting natural language to SQL syntax. For this, I used a 7 billion parameter open-source model ([Nexusflow/Starling-LM-7B-beta](https://huggingface.co/Nexusflow/Starling-LM-7B-beta)) to see if I can reach GPT-3.5-Turbo performance levels, or even surpass them. The cool thing about this is that with DSPy, you can actually focus on programming rather than creating overcrafted prompts that are not failure proof. I invite you to see the main notebook that contains the entire process. I think you will find it really useful. These were the main results:

<div style="background-color: #F0F0F0; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
  <p style="color: #4B4B4B; font-size: 18px; font-weight: bold; margin: 0;"> 📊 Baseline Result </p>
  <p style="color: #4B4B4B; font-size: 16px; margin: 5px 0 0;"> Without any optimization, <strong>GPT-3.5-Turbo</strong> achieves a <strong>68% correctness</strong> in the evaluation of 25 test samples. </p>
</div>

<div style="background-color: #FFCCCB; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
  <p style="color: #8B0000; font-size: 18px; font-weight: bold; margin: 0;"> ⚠️ Evaluation Result (1) </p>
  <p style="color: #8B0000; font-size: 16px; margin: 5px 0 0;"> Without any optimization, <strong>Starling7B</strong> achieves a <strong>60% correctness</strong> in the evaluation of 25 test samples. </p>
</div>

<div style="background-color: #FFF8DC; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
  <p style="color: #DAA520; font-size: 18px; font-weight: bold; margin: 0;"> 🌟 Evaluation Result (2) </p>
  <p style="color: #DAA520; font-size: 16px; margin: 5px 0 0;"> With Few Shot Optimization, <strong>Starling7B</strong> achieves a <strong>72.0% correctness</strong> in the evaluation of 25 test samples. </p>
</div>

<div style="background-color: #E0F8E0; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
  <p style="color: #006400; font-size: 18px; font-weight: bold; margin: 0;"> ✅ Evaluation Result (3) </p>
  <p style="color: #006400; font-size: 16px; margin: 5px 0 0;"> With BootstrapFewShotWithRandomSearch Optimization, <strong>Starling7B</strong> achieves an <strong>80.0% correctness</strong> in the evaluation of 25 test samples. </p>
</div>

## Repo Structure
| Stage                                    | Notebook/Script                                                                                                 | Tech Stack                     |
|------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------|
| Serverless Deployment of Starling 7B     | [model_serve/](https://github.com/jjovalle99/DSPy-Text2SQL/tree/23a0a347db2d7515c5a28c305dacaea00d09dddc/model_serve)  | vLLM, Modal, HuggingFace       |
| DSPy Optimization for Text2SQL           | [DSPy-Text2SQL.ipynb](https://github.com/jjovalle99/DSPy-Text2SQL/blob/23a0a347db2d7515c5a28c305dacaea00d09dddc/DSPy-Text2SQL.ipynb)     | DSPy, HuggingFace |

## What is DSPy?
#### [Docs](https://dspy-docs.vercel.app/)

_DSPy is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used one or more times within a pipeline. To use LMs to build a complex system without DSPy, you generally have to: (1) break the problem down into steps, (2) prompt your LM well until each step works well in isolation, (3) tweak the steps to work well together, (4) generate synthetic examples to tune each step, and (5) use these examples to fine-tune smaller LMs to cut costs. Currently, this is hard and messy: every time you change your pipeline, your LM, or your data, all prompts (or fine-tuning steps) may need to change._

_To make this more systematic and much more powerful, DSPy does two things. First, it separates the flow of your program (modules) from the parameters (LM prompts and weights) of each step. Second, DSPy introduces new optimizers, which are LM-driven algorithms that can tune the prompts and/or the weights of your LM calls, given a metric you want to maximize._