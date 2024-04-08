from typing import Any, Literal

import requests
from dsp import LM


class StarlingLM(LM):
    def __init__(self, model, model_type, **kwargs):
        self.provider = "openai"
        self.model = model
        self.base_url = "https://jjovalle99--starling7b-ft-serve-model.modal.run/v1/chat/completions"
        self.history = []
        self.kwargs = kwargs
        self.model_type = model_type

    def basic_request(self, prompt, **kwargs):
        headers = {"content-type": "application/json"}
        kwargs = {**self.kwargs, **kwargs}
        data = {**kwargs, "model": self.model, "messages": [{"role": "user", "content": prompt}]}

        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        self.history.append(
            {
                "prompt": prompt,
                "response": response,
                "kwargs": kwargs,
            }
        )
        return response

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.model_type == "chat":
            return choice["message"]["content"]
        return choice["text"]

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)

        completions = [response["message"]["content"] for response in response["choices"]]

        return completions
