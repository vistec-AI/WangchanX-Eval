import re
system_prompt='''Please evaluate these answers based on their accuracy and relevance to the provided passage that based on the Criteria:
1. The Answer is Correct concerning the Reference Answer. Do you agree or disagree?
Determine if the given answer accurately matches the reference answer provided. The correctness here means the answer must directly correspond to the reference answer, ensuring factual accuracy.
2. The Answer Includes Relevant, Additional Information from the Context. Do you agree or disagree?
Assess whether the answer provides extra details that are not only correct but also relevant and enhance the understanding of the topic as per the information given in the context.
3. The Answer Includes Additional, Irrelevant Information from the Context. Do you agree or disagree?
Check if the answer contains extra details that, while related to the context, do not directly pertain to the question asked. This information is not necessary for answering the question and is considered a digression.
4. The Answer Includes Information Not Found in the Context. Do you agree or disagree?
Evaluate if the answer includes any information that is not included in the context. This information, even if correct, is extraneous as it goes beyond the provided text and may indicate conjecture or assumption.'''

model_name="gpt-4"  # Can change "gpt-3.5-turbo"

def get_point(text: str)->int:
    s=re.findall(r"(\d) (point|Point)", text)[0]
    return int(s[0])

def change_yes_no(p:str)->str:
    return "Agree" in p

def get_bool(text:str)->list:
    s=re.findall(r"(Disagree|Agree)", text)
    if len(s) != 4:
        return None
    
    return [change_yes_no(i) for i in s]
