<img src="https://github.com/vistec-AI/vistec-ai.github.io/blob/main/wangchanx_logo_color.png?raw=true" width="250">

# WangChanX Eval

WangChanX Eval is a Machine Reading Comprehension (MRC) evaluation pipeline.
We employ models from [the WangchanX project](https://github.com/vistec-AI/WangchanX) to evaluate their effectiveness on question-answering datasets.
In particular, instead of evaluating only the F1 score on MRC datasets, we use an LLM (i.e., GPT4 or Gemini) as a judge on four criteria: (i) Correctness, (ii) Helpfulness, (iii) Irrelevancy, and (iv) Out-of-Context.
The full details of our evaluation can be found in [our technical report](https://arxiv.org/abs/2403.16127) and [CHIE: Generative MRC Evaluation for in-context QA with Correctness, Helpfulness, Irrelevancy, and Extraneousness Aspects](https://aclanthology.org/2024.genbench-1.10/).
With this new evaluation process, we can observe more insight and behavior of the provided model compared to the F1 score. 
Note that our evaluation uses GPT-4 as a judge. Therefore, you need to have [GPT-4 API](https://platform.openai.com/docs/guides/text-generation).

Version: 0.1


## Guide

**Install**

> pip install -r requirements.txt

**Dataset**

We split 100 rows from the Thai subset in the XQuAD dataset for our evaluation. You can create, add, or change the dataset in `gen_text/eval.csv` that has columns:

- references: Reference Answer
- context: Context
- question: Question

**Generate text**

You can get the results from our XQuAD set to use for our evaluation by

> python gen_text/main.py -m {HuggingFace model path}

It gives you file name as `model_name.csv`.


**Config**

You can set the OpenAI API key by going to `mrc_eval`, editing `config-sample.json`, and save as `config.json`


**Run**

Our MRC evaluation can be used by

> python mrc_eval/main.py -f model_name.csv

Example

> python gen_text/main.py -m SeaLLMs/SeaLLM-7B-v2.5

> python mrc_eval/main.py -f SeaLLM-7B-v2.5.csv

The output will be in the file: mrc_eval/result/result-eval-gpt-4-SeaLLM-7B-v2.5.txt

For example:
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

If you use WangchanX or WangchanX Eval in your project or publication, please cite the library as follows

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

**WangchanX Eval**

```tex
@inproceedings{phatthiyaphaibun-etal-2024-chie,
    title = "{CHIE}: Generative {MRC} Evaluation for in-context {QA} with Correctness, Helpfulness, Irrelevancy, and Extraneousness Aspects",
    author = "Phatthiyaphaibun, Wannaphong  and
      Nonesung, Surapon  and
      Limkonchotiwat, Peerat  and
      Udomcharoenchaikit, Can  and
      Sawatphol, Jitkapat  and
      Chuangsuwanich, Ekapol  and
      Nutanong, Sarana",
    editor = "Hupkes, Dieuwke  and
      Dankers, Verna  and
      Batsuren, Khuyagbaatar  and
      Kazemnejad, Amirhossein  and
      Christodoulopoulos, Christos  and
      Giulianelli, Mario  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 2nd GenBench Workshop on Generalisation (Benchmarking) in NLP",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.genbench-1.10",
    pages = "154--164",
    abstract = "The evaluation of generative models in Machine Reading Comprehension (MRC) presents distinct difficulties, as traditional metrics like BLEU, ROUGE, METEOR, Exact Match, and F1 score often struggle to capture the nuanced and diverse responses. While embedding-based metrics such as BERTScore and BARTScore focus on semantic similarity, they still fail to fully address aspects such as recognizing additional helpful information and rewarding contextual faithfulness. Recent advances in large language model (LLM) based metrics offer more fine-grained evaluations, but challenges such as score clustering remain. This paper introduces a multi-aspect evaluation framework, CHIE,incorporating aspects of \textbf{C}orrectness, \textbf{H}elpfulness, \textbf{I}rrelevance, and \textbf{E}xtraneousness. Our approach, which uses binary categorical values rather than continuous rating scales, aligns well with human judgments, indicating its potential as a comprehensive and effective evaluation method.",
}
```
