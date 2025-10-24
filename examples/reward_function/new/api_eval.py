# coherence_eval.py
import random
import httpx
from openai import OpenAI, BadRequestError
import re
from time import sleep
from typing import Dict

API_KEYS = [
    "sk-OOlk7PS4kwyrL9UQpZTCXEG7nxwfq2xxhC8NTsxzUF8oWvnM",
    "sk-gWE7NeQzmEtsSeGrcJ1DKTRlu64j15dss1zmHwsfPcIiTIhS",
    "sk-CkEI4xr4lbASTDOh3mKuYf7EPY6Yn0g7NwIsBhZjnuxO3uM4",
    "sk-MN5MDVtahC8eMeOZz9tcOQlgRNKSw7gTw5sdIEiWtpDUukg3",
    "sk-qSQFrAO6Mh8E1fRqL00MAICihW9BItgILouGObG0G62OkhbF",
]

class CoherenceEvaluator:
    def __init__(
        self,
        api_keys = API_KEYS,
        base_url="https://oneai.evanora.top/v1",
        model_name="chatgpt-4o-latest",
        max_retries=5,
        base_delay=1
    ):
        self.api_keys = api_keys
        self.base_url = base_url
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay

        self.eval_instruction = """You are an expert in evaluating the coherence of reasoning.

Here are three reasoning steps:
Step1: {step1}
Step2: {step2}
Step3: {step3}

Evaluate the overall logical and emotional coherence between these steps.
Score requirements:
- The score must be between 0 and 1.
- 1 means perfectly coherent and logically consistent.
- 0 means completely incoherent or contradictory.

Output ONLY in this strict format:
<answer>score</answer>
"""

    def _build_message(self, steps: Dict[str, str]):
        prompt_text = self.eval_instruction.format(
            step1=steps.get("step1", "").strip(),
            step2=steps.get("step2", "").strip(),
            step3=steps.get("step3", "").strip(),
        )
        return [
            {"role": "system", "content": "You are a helpful assistant developed by OpenAI."},
            {"role": "user", "content": prompt_text}
        ]

    def _request_api(self, client, messages):
        num_attempts = 0
        while num_attempts < self.max_retries:
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0
                )
                res = completion.choices[0].message.content.strip()
                match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1).strip())
                        if 0.0 <= score <= 1.0:
                            return score
                    except ValueError:
                        return 0.0
                return 0.0

            except BadRequestError as be:
                print(f"[BadRequestError] {be}")
                num_attempts += 1
            except Exception as e:
                print(f"[API Error] {e}")
                num_attempts += 1
                if num_attempts < self.max_retries:
                    sleep_time = self.base_delay * (2 ** num_attempts)
                    print(f"Retrying in {sleep_time}s...")
                    sleep(sleep_time)
        return 0.0

    def coherence_reward(self, steps: Dict[str, str]) -> float:
        step1 = steps.get("step1", "").strip()
        step2 = steps.get("step2", "").strip()
        step3 = steps.get("step3", "").strip()
        if not step1 or not step2 or not step3:
            return 0.0

        client = OpenAI(
            api_key=random.choice(self.api_keys),
            base_url=self.base_url,
            http_client=httpx.Client(base_url=self.base_url, follow_redirects=True),
        )
        messages = self._build_message(steps)
        return self._request_api(client, messages)


evaluator = CoherenceEvaluator()

# 测试数据
steps = {
    "step1": "He failed the test and felt discouraged.",
    "step2": "He believed he was not good enough.",
    "step3": "He decided to stop trying altogether."
}

# 调用接口
score = evaluator.coherence_reward(steps)
print("Coherence Score:", score)