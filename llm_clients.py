import json
import time
from pathlib import Path

import vllm
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
)
from superopenai import init_logger, init_superopenai

from base.bot_and_client import BaseBot, BaseLLMClient
from base.results import ILPException
from utils.utils import random_id


def load_client(model: str) -> BaseLLMClient:
    """Loads the specified LLM client model based on the input string"""
    if model == "dummy":
        return LLMDummy()
    elif model.lower().startswith("gpt"):
        return GPT(model)
    elif "llama" in model.lower():
        return Llama(model)
    else:
        raise ValueError(f"Unrecognized model: {model}")


class LLMDummy(BaseLLMClient):
    """A dummy LLM interface for testing"""

    def __init__(
        self,
        response: str | tuple[str] = "Dummy Response",
        count_messages: bool = True,
        print_count: bool = False,
        print_messages: bool = False,
        include_count: bool = False,
    ) -> None:
        if not isinstance(response, str) and not count_messages:
            raise ValueError("response must be a string if count_messages=False")
        if print_count and not count_messages:
            raise ValueError("print_count=True is only valid when count_messages=True")
        self.response = response
        self.count = 0 if count_messages else None
        self.print_count = print_count
        self.print_messages = print_messages
        self.include_count = include_count

    def prompt(
        self,
        message_history: tuple[str, ...] = (),
        **kwargs,
    ) -> str:
        # ignores the message history and simply returns the pre-defined response
        if self.print_messages:
            print(
                f"\nLLMDummy Received History:\n{'*'*64}\n{message_history}\n{'*'*64}"
            )
        if self.count is not None:
            self.count += 1
        output = f"(#{self.count}) " if self.print_count else ""
        if isinstance(self.response, str):
            output += self.response
        elif self.count is None:
            raise ValueError("Response must be a string if messages are not counted")
        else:
            output += self.response[self.count - 1]
        return output

    def bot_prompt(self, bot: BaseBot) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return super().bot_prompt(bot)

    def set_response(self, response: str) -> None:
        self.response = response

    def set_count(self, count: int) -> None:
        self.count = count


class Llama(BaseLLMClient):
    """A class for interacting with Llama LLMs."""

    def __init__(
        self, model, max_model_len=10000, tensor_parallel_size=2, **kwargs
    ) -> None:
        self.llm = vllm.LLM(
            model=model,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )
        self.model = model.split("/")[-1]

    def prompt(
        self, message_history: tuple[str, ...] = (), max_tokens: int = 1024, **kwargs
    ) -> str:
        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=1,
            **kwargs,
        )

        conversation = [
            (
                ChatCompletionUserMessageParam(
                    role=("user"),
                    content=[{"type": "text", "text": message}],
                )
                if i % 2 == 0
                else ChatCompletionAssistantMessageParam(
                    role=("assistant"),
                    content=message,
                )
            )
            for i, message in enumerate(message_history)
        ]

        output = (
            self.llm.chat(
                conversation, sampling_params=sampling_params, use_tqdm=False  # type: ignore
            )[0]
            .outputs[0]
            .text
        )

        if not isinstance(output, str):
            raise ValueError(f"{self.model} response content is not a string")

        return output

    def bot_prompt(self, bot: BaseBot) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return super().bot_prompt(bot)


class GPT(BaseLLMClient):
    """A class for interacting with GPT LLMs."""

    client: OpenAI
    model: str

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        init_superopenai(enable_caching=True)
        self.client = OpenAI()
        self.model = model

        self.log_dir = f"./data/logs/{model}_logs/{random_id()}"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.temperature = temperature

    def prepare_jsonl_batch_object(
        self,
        messages: list[
            ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam
        ],
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
        message_history: tuple[str, ...] = (),
        temperature: float | None = None,
        max_tokens: int = 1024,
        batch_prompt: bool = False,
        **kwargs,
    ) -> str:
        messages: list[
            ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam
        ] = [
            (
                ChatCompletionUserMessageParam(
                    role=("user"),
                    content=[{"type": "text", "text": message}],
                )
                if i % 2 == 0
                else ChatCompletionAssistantMessageParam(
                    role=("assistant"),
                    content=message,
                )
            )
            for i, message in enumerate(message_history)
        ]
        if not batch_prompt:
            try:
                with init_logger(log_directory=self.log_dir) as logger:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=(
                            temperature if temperature is not None else self.temperature
                        ),
                        max_tokens=max_tokens,
                        frequency_penalty=0,
                        presence_penalty=0,
                        seed=1,
                    )
            except UnicodeDecodeError:
                print("UnicodeDecodeError. Retrying... ", end="")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=(
                        temperature if temperature is not None else self.temperature
                    ),
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    seed=1,
                )

            out = response.choices[0].message.content
            if out is None:
                raise ValueError(f"{self.model} response content is None")
        else:
            request_jsonl, id = self.prepare_jsonl_batch_object(
                messages, temperature or self.temperature, max_tokens
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

                out = content.json()["response"]["body"]["choices"][0]["message"][
                    "content"
                ]

        return out if out is not None else ""

    def bot_prompt(
        self, bot: BaseBot, batch_prompt: bool = False
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        history: list[str] = []
        messages: list[str] = []
        responses: list[str] = []

        try:
            message, kwargs = bot.get_message("")
            while message is not None:
                history.append(message)
                response = self.prompt(
                    tuple(history), batch_prompt=batch_prompt, **kwargs
                )
                history.append(response)
                messages.append(message)
                responses.append(response)
                message, kwargs = bot.get_message(response)
        except ILPException as e:
            raise ILPException(str(e), tuple(messages), tuple(responses))
        else:
            return tuple(messages), tuple(responses)
