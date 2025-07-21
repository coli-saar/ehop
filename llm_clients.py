import json
import os
import time
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from superopenai import init_logger, init_superopenai
from transformers import AutoTokenizer  # type: ignore

from base.bot_and_client import BaseLLMClient
from utils.utils import random_id


def load_client(model: str, **kwargs) -> BaseLLMClient:
    """Loads the specified LLM client model based on the input string"""
    if model == "dummy":
        return DummyLLMClient(**kwargs)
    elif model.lower().startswith(("gpt", "deepseek")):
        return OpenAIClient(model, **kwargs)
    elif "qwen3" in model.lower():
        return Qwen3Client(model, **kwargs)
    else:
        return VllmClient(
            model,
            max_model_len=10000 if "llama" in model.lower() else None,
            **kwargs,
        )


class DummyLLMClient(BaseLLMClient):
    """A dummy LLM interface for testing"""

    def __init__(
        self,
        response: str | tuple[str, ...] = "Dummy Response",
        reasoning: str | tuple[str, ...] | None = None,
        print_count: bool = False,
        print_messages: bool = False,
        include_count: bool = False,
    ) -> None:
        if print_count:
            raise ValueError("print_count=True is only valid when count_messages=True")
        self.response = response
        self.reasoning = reasoning
        self.count = 0
        self.print_count = print_count
        self.print_messages = print_messages
        self.include_count = include_count
        self.history: list[str] = []

    def get_model(self) -> str:
        return "dummy"

    def is_reasoning(self) -> bool:
        return self.reasoning is not None

    def set_history(self, history: Iterable[str]) -> None:
        self.history = [m for m in history]

    def get_history(self) -> list[str]:
        return self.history

    def prompt(
        self,
        message: str,
        keep_history: bool = False,
        **kwargs,
    ) -> tuple[str, str | None]:
        if not keep_history:
            self.history = []
        self.history.append(message)

        if self.print_messages:
            print(f"\nLLMDummy Received History:\n{'*'*64}\n{self.history}\n{'*'*64}")

        if self.count is not None:
            self.count += 1

        output = f"(#{self.count}) " if self.print_count else ""

        if isinstance(self.response, str):
            output += self.response
        else:
            output += self.response[self.count - 1]

        self.history.append(output)

        return output, (
            self.reasoning[self.count - 1]
            if isinstance(self.reasoning, tuple)
            else self.reasoning
        )

    def set_response(self, response: str) -> None:
        self.response = response

    def set_count(self, count: int) -> None:
        self.count = count


class Qwen3Client(BaseLLMClient):
    def __init__(
        self, model: str, thinking: bool = True, tensor_parallel_size: int = 2, **kwargs
    ) -> None:
        import vllm  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.history: list[ChatCompletionMessageParam] = []
        self.thinking = thinking

        self.model_name = model.split("/")[-1]

        self.params_class = vllm.SamplingParams
        self.llm = vllm.LLM(
            model=model, tensor_parallel_size=tensor_parallel_size, **kwargs
        )
        # raise NotImplementedError
        print(f"Initialized Qwen3Client {self.model_name} with thinking={thinking}")

    def get_model(self) -> str:
        return self.model_name

    def is_reasoning(self) -> bool:
        return self.thinking

    def set_history(self, history: Iterable[str]) -> None:
        self.history = [
            (
                ChatCompletionUserMessageParam(role="user", content=m)
                if i % 2 == 0
                else ChatCompletionAssistantMessageParam(role="assistant", content=m)
            )
            for i, m in enumerate(history)
        ]

    def prompt(
        self, message: str, keep_history: bool = False, max_tokens: int = 1024, **kwargs
    ) -> tuple[str, str | None]:
        if not keep_history:
            self.history = []

        self.history.append(
            ChatCompletionUserMessageParam(role="user", content=message)
        )

        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

        sampling_params = self.params_class(
            temperature=0.6 if self.thinking else 0.7,
            top_p=0.95 if self.thinking else 0.8,
            min_p=0,
            top_k=20,
            max_tokens=max_tokens * 4 if self.thinking else max_tokens,
        )

        outputs = self.llm.generate([text], sampling_params)

        response = outputs[0].outputs[0].text

        content, reasoning = response, None  # defaults when no reasoning is present
        if "</think>" in response:  # indicates that reasoning is present
            reasoning, content = response.split("</think>", 1)
            reasoning += "</think>"

        self.history.append(
            ChatCompletionAssistantMessageParam(role="assistant", content=content)
        )

        return content, reasoning if self.thinking else None


class VllmClient(BaseLLMClient):
    """A class for interacting with LLMs with vLLM."""

    def __init__(
        self, model: str, tensor_parallel_size=2, thinking: bool | None = None, **kwargs
    ) -> None:
        import vllm  # type: ignore

        self.params_class = vllm.SamplingParams
        self.model = model.split("/")[-1]
        self.history: list[ChatCompletionMessageParam] = []
        self.token_limit_scale = 1

        self.base_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

        if "deepseek" in model.lower():
            self.base_kwargs["temperature"] = 0.6
            self.token_limit_scale = 4

        self.llm = vllm.LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )

    def get_model(self) -> str:
        return self.model

    def is_reasoning(self) -> bool:
        return "reasoner" in self.model.lower()

    def set_history(self, history: Iterable[str]) -> None:
        self.history = [
            (
                ChatCompletionUserMessageParam(role="user", content=m)
                if i % 2 == 0
                else ChatCompletionAssistantMessageParam(role="assistant", content=m)
            )
            for i, m in enumerate(history)
        ]

    def prompt(
        self, message: str, keep_history: bool = False, max_tokens: int = 1024, **kwargs
    ) -> tuple[str, str | None]:
        if not keep_history:
            self.history = []

        self.history.append(
            ChatCompletionUserMessageParam(role="user", content=message)
        )

        sampling_params = self.params_class(
            max_tokens=max_tokens * self.token_limit_scale,
            seed=1,
            **self.base_kwargs,
            **kwargs,
        )

        response = (
            self.llm.chat(
                self.history,
                sampling_params=sampling_params,
                use_tqdm=False,  # type: ignore
            )[0]
            .outputs[0]
            .text
        )

        if not isinstance(response, str):
            raise ValueError(f"{self.model} response content is not a string")

        self.history.append(
            ChatCompletionAssistantMessageParam(role="assistant", content=response)
        )

        return response, None


class OpenAIClient(BaseLLMClient):
    """A class for interacting with GPT LLMs."""

    client: OpenAI
    model: str

    def __init__(
        self, model: str, temperature: float = 0.0, api_key_name: str | None = None
    ) -> None:
        init_superopenai(enable_caching=True)
        self.client = (
            OpenAI(
                api_key=os.environ.get(
                    api_key_name if api_key_name is not None else "DEEPSEEK_API_KEY_1"
                ),
                base_url="https://api.deepseek.com",
            )
            if "deepseek" in model.lower()
            else OpenAI()
        )
        self.model = model
        self.history: list[ChatCompletionMessageParam] = []

        self.log_dir = f"./data/logs/{model}_logs/{random_id()}"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.temperature = temperature

    def get_model(self) -> str:
        return self.model

    def is_reasoning(self) -> bool:
        return "reasoner" in self.model.lower()

    def set_history(self, history: Iterable[str]) -> None:
        self.history = [
            (
                ChatCompletionUserMessageParam(role="user", content=m)
                if i % 2 == 0
                else ChatCompletionAssistantMessageParam(role="assistant", content=m)
            )
            for i, m in enumerate(history)
        ]

    def prepare_jsonl_batch_object(
        self,
        messages: list[ChatCompletionMessageParam],
        temp: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        with open("data/logs/batches/batch_requests.log", "a") as f:
            f.write(
                f"OpenAI Chat Completion BATCH request parameters: model:{self.model}, temperature:{temp}, max_tokens={max_tokens}"
            )
            f.write(f"Request prompt: {messages}")

        custom_id = f"request-{random_id()}"

        # Construct the JSON object for each line
        json_line = json.dumps(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "seed": 1,
                },
            }
        )
        return json_line, custom_id

    def prompt(
        self,
        message: str,
        keep_history: bool = False,
        temperature: float | None = None,
        max_tokens: int = 1024,
        batch_prompt: bool = False,
        **kwargs,
    ) -> tuple[str, str | None]:
        if not keep_history:
            self.history = []

        self.history.append(
            ChatCompletionUserMessageParam(role="user", content=message)
        )

        if not batch_prompt:
            response: ChatCompletion | None = None
            attempts = 0
            while attempts < 3:
                try:
                    with init_logger(log_directory=self.log_dir) as logger:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.history,
                            temperature=(
                                temperature
                                if temperature is not None
                                else self.temperature
                            ),
                            max_tokens=(
                                10 * max_tokens
                                if "deepseek" in self.get_model().lower()
                                else max_tokens
                            ),
                            frequency_penalty=0,
                            presence_penalty=0,
                            seed=1,
                            **kwargs,
                        )
                    break
                except UnicodeDecodeError:
                    print("UnicodeDecodeError. Retrying... ", end="")
                    attempts += 1
                except JSONDecodeError:
                    print("JsonDecodeError. Retrying... ", end="")
                    attempts += 1

            if response is None:
                raise RuntimeError(f"{self.model.title()} response is None")

            try:
                response_message: ChatCompletionMessage = response.choices[0].message
                assert isinstance(response_message, ChatCompletionMessage)
            except AttributeError:
                raise ValueError(
                    f"{self.model} response does not have a message attribute"
                )

            if response_message.content is None:
                raise ValueError(f"{self.model} response content is None")

            out: tuple[str, str | None] = response_message.content, (
                response_message.reasoning_content  # type: ignore
                if hasattr(response_message, "reasoning_content")
                else None
            )

            self.history.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=out[0])
            )
        else:
            request_jsonl, id = self.prepare_jsonl_batch_object(
                self.history, temperature or self.temperature, max_tokens
            )

            jsonl_request_filename = f"data/logs/batches/batch-{id}.jsonl"
            with open(jsonl_request_filename, "w") as f:
                f.write(request_jsonl)

            with open("data/logs/batches/batch_requests.log", "a") as f:
                batch_input_file_id: str = ""
                while not batch_input_file_id:
                    try:
                        # Create a file handle to upload
                        batch_input_file = self.client.files.create(
                            file=open(jsonl_request_filename, "rb"), purpose="batch"
                        )
                        time.sleep(30)
                        f.write(
                            "BATCH openai.files.create() Uploaded jsonl file response:\n"
                        )
                        f.write(str(batch_input_file) + "\n")
                        batch_input_file_id = batch_input_file.id
                    except AttributeError:
                        continue

                # Create a batch request
                batch = self.client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": "batch request prompt"},
                )
                f.write("BATCH client.batches.create() response:\n")
                f.write(str(batch) + "\n")
                batch_id = batch.id

                # Wait for the batch to complete
                while True:
                    time.sleep(30)
                    retrieve_status = self.client.batches.retrieve(batch_id)
                    f.write("BATCH openai.batches.retrieve() response:\n")
                    f.write(str(retrieve_status) + "\n")
                    if retrieve_status.output_file_id:
                        break

                output_file_id = retrieve_status.output_file_id
                content = self.client.files.content(output_file_id)
                f.write(f"BATCH obtained output_file_id: {output_file_id} content:\n")
                f.write(str(content.json()) + "\n")
                f.write("#" * 128 + "\n")

                out = (
                    content.json()["response"]["body"]["choices"][0]["message"][
                        "content"
                    ],
                    None,
                )

        return out
