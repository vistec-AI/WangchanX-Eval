# WangChanX Eval

WangChanX Eval is a Machine Reading Comprehension (MRC) evaluation. Our evaluation use in [our technical report](https://arxiv.org/abs/2403.16127).

Our evaluation use GPT-4 as judge. You needs to have [GPT-4 API](https://platform.openai.com/docs/guides/text-generation).

Version: 0.1

## Guide

**Install**

> pip install -r requirements.txt

**Dataset**

We split 100 rows from Thai subset in XQuAD dataset for our evaluation. You can create, add, or change the dataset in `gen_text/eval.csv` that has columns:

- references: Reference Answer
- context: Context
- question: Question

**Generate text**

You can get the reult from our XQuAD set to use for our evaluation by

> python gen_text/main.py -m {HuggingFace model path}

It give you file name as `model_name.csv`.


**Config**

You can setting OpenAI API key by go to `mrc_eval`, edit `config-sample.json` and save as `config.json`


**Run**

Our MRC evaluation can use by

> python mrc_eval/main.py -f model_name.csv

Example

> python gen_text/main.py -m SeaLLMs/SeaLLM-7B-v2.5
> python mrc_eval/main.py -f SeaLLM-7B-v2.5.csv


mrc_eval/result/result-eval-gpt-4-SeaLLM-7B-v2.5.txt

```
XQuAD score
--------------------
exact_match: 6.0
F1: 22.46709028002505
--------------------

Model: gpt-4
System prompt:
Please evaluate these answers based on their accuracy and relevance to the provided passage that based on the Criteria:
1. The Answer is Correct concerning the Reference Answer. Do you agree or disagree?
Determine if the given answer accurately matches the reference answer provided. The correctness here means the answer must directly correspond to the reference answer, ensuring factual accuracy.
2. The Answer Includes Relevant, Additional Information from the Context. Do you agree or disagree?
Assess whether the answer provides extra details that are not only correct but also relevant and enhance the understanding of the topic as per the information given in the context.
3. The Answer Includes Additional, Irrelevant Information from the Context. Do you agree or disagree?
Check if the answer contains extra details that, while related to the context, do not directly pertain to the question asked. This information is not necessary for answering the question and is considered a digression.
4. The Answer Includes Information Not Found in the Context. Do you agree or disagree?
Evaluate if the answer includes any information that is not included in the context. This information, even if correct, is extraneous as it goes beyond the provided text and may indicate conjecture or assumption.

model
q1: 78
q2: 36
q3: 34
q4: 30
---------------
Save to ./mrc_eval/result/eval-gpt-4-SeaLLM-7B-v2.5.csv
Number token (avg)
model	28.59
```

## Citations

If you use WangchanX Eval in your project or publication, please cite the library as follows

```tex
@misc{phatthiyaphaibun2024wangchanlion,
      title={WangchanLion and WangchanX MRC Eval}, 
      author={Wannaphong Phatthiyaphaibun and Surapon Nonesung and Patomporn Payoungkhamdee and Peerat Limkonchotiwat and Can Udomcharoenchaikit and Jitkapat Sawatphol and Chompakorn Chaksangchaichot and Ekapol Chuangsuwanich and Sarana Nutanong},
      year={2024},
      eprint={2403.16127},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```