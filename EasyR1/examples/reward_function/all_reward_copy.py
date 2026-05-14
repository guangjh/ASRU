import re
from typing import Any
from rouge_score import rouge_scorer

# ✅ 用你提供的拒绝正则模板替换原来的字符串列表
REFUSAL_PATTERNS = re.compile(r"""
(?:
# Common expressions of ignorance
(?:don’?t|doesn’?t|didn’?t|do(?:es)?\s+not)\s+
(?:know|have|hold|possess|seem\s+to\s+have|cover|contain|
extend|include) |
# Variations of uncertainty or lack of training
(?:not|yet)\s+.*(?:sure|certain|familiar|aware|equipped|able
|
acquainted|informed|knowledge|information|data|
educated|briefed|well-versed|learn|trained\s+on) |
# Explicit statements of lacking information
no\s+.*(?:idea|insight|knowledge|information|data|
enlightenment|clue|familiarity) |
# Not having learned or seen the content
(?:haven’?t|hasn’?t| not)\s+(?:encountered|learned|
the\s+faintest|been\s+(?:included|trained|briefed)) |
# Out-of-scope or beyond knowledge claims
(?:beyond|outside|out)\s+.*(?:knowledge|capabilities|
expertise|reach|scope) |
# Statements indicating inability to respond
at\s+a\s+(?:loss|disadvantage) |
can’?t\s+(?:provide|say|shed\s+.*light|help|offer|take|
make|fulfill) |
unable\s+(?:to\s+provide|to\s+answer|to\s+access) |
# Soft disclaimers or hedged refusals
(?:I\s+)?(?:wish\s+I\s+could\s+say|regret\s+to\s+inform|
must\s+(?:admit|confess)) |
# Indicators of confusion or lack of clarity
(?:Unfortunately ,|clueless|stumped|a\s+mystery\s+to\s+me|
lacking\s+(?:information|knowledge|insight|specifics|data
)|
dark\s+about|draw(?:ing)?\s+a\s+blank|short\s+with|
limited\s+to|blank\s+on) |
# Explicit descriptors of missing understanding
(?:missing|without|lack|blind|uncharted)\s+.*(?:information|
knowledge|insight|specifics) |
# Expressions of needing to search externally
(?:need\s+to|require|have\s+to|must|ought\s+to|should)\s+
(?:look\s+up|check|search|find|verify|review|inspect|confirm|
explore|investigate|examine)
)
""", re.IGNORECASE | re.VERBOSE | re.DOTALL)

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ✅ 由“字符串包含”改为“正则搜索”
def contains_any(text: str, pattern: re.Pattern) -> bool:
    return pattern.search(text.lower()) is not None
def compute_rouge(ground_truth, predicted_answer):
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)
    return rouge_scores['rougeL'].fmeasure

def compute_forget_score_rule(response, ground_truth):
    rouge_score = compute_rouge(ground_truth, response)
    if normalize(ground_truth) in normalize(response):
        score = 0.0
    else:
        # 计算得分
        if contains_any(response, REFUSAL_PATTERNS):
            score = 1.0
        elif rouge_score < 0.4 and rouge_score > 0:
            score=0.5
        else:
            score=0.1

    return {"overall": score, "rouge_score_forget": rouge_score}

def compute_retain_score_rule(response, ground_truth):
    rouge_score = compute_rouge(ground_truth, response)

    if normalize(ground_truth) in normalize(response):
        score = 1.0
    else:
        # 计算得分
        if contains_any(response, REFUSAL_PATTERNS):
            score = 0.0
        elif rouge_score > 0.6:
            score=0.5
        else:
            score=0.1

    return {"overall": score, "rouge_score_retain": rouge_score}

###################################################################
# separate
######################################################################
def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        forget_first = reward_input.get("forget_first", True)
        if forget_first:
            score_dict = compute_forget_score_rule(response, ground_truth)
            scores.append({
                "overall": score_dict["overall"],
                "forget_rouge": score_dict["rouge_score_forget"],
                "retain_rouge": 0.0
            })
            print("Is Forget:", forget_first)
            print("Forget_response:", response)
            print("Forget_ground_truth:", ground_truth)
            print("Forget_score_dict:", score_dict)
        else:
            score_dict = compute_retain_score_rule(response, ground_truth)
            scores.append({
                "overall": score_dict["overall"],
                "forget_rouge": 0.0,
                "retain_rouge": score_dict["rouge_score_retain"]
            })
            print("Is Forget:", forget_first)
            print("Retain_response:", response)
            print("Retain_ground_truth:", ground_truth)
            print("Retain_score_dict:", score_dict)
    return scores


# def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
#     scores = []
#     for reward_input in reward_inputs:
#         response = reward_input.get("response", "")
#         ground_truth = reward_input.get("ground_truth", "")
#         forget_first=reward_input.get("forget_first", True)
#         ground_truth_list = ground_truth.split("||")
#         print("Response:", response)
#         print("Ground truth:", ground_truth)
#         print("Forget first:", forget_first)
#         try:
#             response_list = eval(response)
#             first_response = str(response_list[0])
#             second_response = str(response_list[1])
#         except:
#             if response.startswith('['):
#                 response = response[1:]
#             if response.endswith(']'):
#                 response = response[:-1]
#             if '.' in response:
#                 response_list = response.split('.')
#                 first_response = response_list[0].strip()
#                 while first_response.startswith("'") or second_response.startswith("\""):
#                     first_response = first_response[1:].strip()
#                 if not first_response.endswith("."):
#                     first_response = first_response + "."
#                 if len(response_list) > 1:
#                     second_response = response_list[1].strip()
#                     while second_response.startswith(',') or second_response.startswith("'") or second_response.startswith("\""):
#                         second_response = second_response[1:].strip()
#                     if not second_response.endswith("."):
#                         second_response = second_response + "."
#                 else:
#                     second_response = ""
#             else:
#                 first_response = ""
#                 second_response = ""
#         if forget_first:
#             forget_ground_truth = ground_truth_list[0].strip()
#             retain_ground_truth = ground_truth_list[1].strip()
#             forget_response = first_response
#             retain_response = second_response
#         else:
#             retain_ground_truth = ground_truth_list[0].strip()
#             forget_ground_truth = ground_truth_list[1].strip()
#             retain_response = first_response
#             forget_response = second_response
#         print("Forget_response:", forget_response)
#         print("Retain_response:", retain_response)
#         print("Forget_ground_truth:", forget_ground_truth)
#         print("Retain_ground_truth:", retain_ground_truth)
#         forget_score_dict = compute_forget_score_rule(forget_response, forget_ground_truth)
#         retain_score_dict = compute_retain_score_rule(retain_response, retain_ground_truth)
#         # 综合得分
#         overall_score = (forget_score_dict["overall"] + retain_score_dict["overall"])
#         score_dict = {
#             "overall": overall_score,
#             "forget_rouge": forget_score_dict["rouge_score_forget"],
#             "retain_rouge": retain_score_dict["rouge_score_retain"]
#         }
#         print(forget_score_dict)
#         print(retain_score_dict)
#         print(score_dict)
#         # exit()
#         scores.append(score_dict)
#     return scores

if __name__ == "__main__":
    # 简单测试
    test_inputs = [
        {
            "response": "['The name of this person\\'s mother is not provided in the image metadata.', 'The occupation of this person\\'s mother is not provided in the image metadata.']",
            "ground_truth": "This person's mother's name is Helena Adler.||This person's mother works as a museum curator.",
            "forget_first": True
        },
    ]
    scores = compute_score(test_inputs)



