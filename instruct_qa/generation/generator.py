from math import inf
import time
import openai
from openai import (
    RateLimitError,
    APIConnectionError,
    APIError,
    Timeout,
)
import torch
from transformers import pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    OPTForCausalLM,
)

max_length = 1900

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseGenerator:
    def __init__(
        self,
        model_name=None,
        weights_path=None,
        api_key=None,
        cache_dir=None,
        torch_dtype=torch.float16,
        temperature=0.95,
        top_p=0.95,
        max_new_tokens=500,
        min_new_tokens=1,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        device="cuda",
    ):
        self.model_name = model_name
        self.weights_path = weights_path
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.device = device
        self.wait = 10

    def __call__(self, prompt, **kwargs):
        raise NotImplementedError()

    def post_process_response(self, response):
        return response


class GPTx(BaseGenerator):
    def __init__(self, *args, **kwargs):
        completion_type = kwargs.pop("completion_type", None)
        
        super().__init__(*args, **kwargs)
        openai.api_key = self.api_key
        self.model_map = {
            "gpt-3.5-turbo": "chat",
            "gpt-4": "chat",
            "text-davinci-003": "completions",
            "text-davinci-002": "completions",
        }

        if completion_type is not None:
            self.model_map[self.model_name] = completion_type
        
        assert (
            self.model_name in self.model_map
        ), "You should add the model name to the model -> endpoint compatibility mappings."
        assert self.model_map[self.model_name] in [
            "chat",
            "completions",
        ], "Only chat and completions endpoints are implemented. You may want to add other configurations."
        # json error happens if max_new_tokens is inf
        self.max_new_tokens = self.max_new_tokens

    def __call__(self, prompts, n=1):
        responses = []
        for prompt in prompts:
            # to maintain enough space for generation
            prompt = " ".join(prompt.split()[:2800])
            kwargs = {"temperature": self.temperature, "top_p": self.top_p, "n": n}
            if self.max_new_tokens != inf:
                kwargs["max_tokens"] = self.max_new_tokens
            response = self.api_request(
                prompt,
                **kwargs,
            )
            if n == 1:
                responses.append(response[0])
            else:
                responses.append(response)
        return responses

    def api_request(self, prompt, **kwargs):
        try:
            if self.model_map[self.model_name] == "chat":
                res = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                return [r.message.content for r in res.choices]
            elif self.model_map[self.model_name] == "completions":
                res = openai.Completion.create(
                    model=self.model_name, prompt=prompt, **kwargs
                )
                return [r.text for r in res.choices]
        except (
            RateLimitError,
            APIConnectionError,
            APIError,
            Timeout,
        ) as e:
            print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self.api_request(prompt, **kwargs)


class Llama(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.weights_path is not None
        ), "A path to model weights must be defined for using this generator."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, padding_side="left"
        )
        if "70b" in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.weights_path,
                load_in_4bit=True,
                torch_dtype=self.torch_dtype,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.weights_path, torch_dtype=self.torch_dtype, device_map="auto"
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        ).to(self.device)
        generate_ids = self.model.generate(
            input_ids=_input.input_ids,
            attention_mask=_input.attention_mask,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )
        return [
            self.tokenizer.decode(
                generate_ids[i, _input.input_ids.size(1) :],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            for i in range(generate_ids.size(0))
        ]

    def post_process_response(self, response):
        keywords = {"user:", "User:", "assistant:", "- Title:", "Question:"}
        end_keywords = {"Agent:", "Answer:"}

        response_lines = response.split("\n")
        response_lines = [x for x in response_lines if x.strip() not in ["", " "]]

        for j, line in enumerate(response_lines):
            if any(line.startswith(kw) for kw in keywords):
                response_lines = response_lines[:j]
                break

        for j, line in enumerate(response_lines):
            if j > 0 and any(line.startswith(kw) for kw in end_keywords):
                response_lines = response_lines[:j]
                break

        return "\n".join(response_lines)


class Vicuna(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"lmsys/{self.model_name}", cache_dir=self.cache_dir, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            f"lmsys/{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        ).to(self.device)
        generate_ids = self.model.generate(
            input_ids=_input.input_ids,
            attention_mask=_input.attention_mask,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )
        return [
            self.tokenizer.decode(
                generate_ids[i, _input.input_ids.size(1) :],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            for i in range(generate_ids.size(0))
        ]


class OPT(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"facebook/{self.model_name}", cache_dir=self.cache_dir, padding_side="left"
        )
        self.model = OPTForCausalLM.from_pretrained(
            f"facebook/{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        ).to(self.device)
        generate_ids = self.model.generate(
            **_input,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )
        return [
            self.tokenizer.decode(
                generate_ids[i, _input.input_ids.size(1) :],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            for i in range(generate_ids.size(0))
        ]


class Flan(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"google/{self.model_name}",
            cache_dir=self.cache_dir,
            padding_side="left",
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            f"google/{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        if self.min_new_tokens is not None:
            logger.warning(
                "min_new_tokens is not supported for Flan. It will be ignored."
            )

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        ).to(self.device)
        generate_ids = self.model.generate(
            _input.input_ids,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            # min_new_tokens=self.min_new_tokens,
        )
        return self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
        )


class PipelineGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline(
            model=f"{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = 0

    def forward_call(self, prompt):
        _input = self.pipeline.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        )
        return self.pipeline(
            self.pipeline.tokenizer.decode(
                _input.input_ids[0], attention_mask=_input.attention_mask[0]
            ),
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )[0]["generated_text"]

    def __call__(self, prompts):
        return [self.forward_call(prompt) for prompt in prompts]


class FalconPipelineGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(f"tiiuae/{self.model_name}")
        self.pipeline = pipeline(
            "text-generation",
            model=f"tiiuae/{self.model_name}",
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )

    def forward_call(self, prompt):
        sequences = self.pipeline(
            prompt,
            max_length=self.max_new_tokens,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return sequences[0]["generated_text"]

    def __call__(self, prompts):
        return [self.forward_call(prompt) for prompt in prompts]


class StarChatGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline(
            "text-generation",
            model=f"HuggingFaceH4/{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = 0

    def forward_call(self, prompt):
        _input = self.pipeline.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        )
        model_input = self.pipeline.tokenizer.decode(
            _input.input_ids[0], attention_mask=_input.attention_mask[0]
        )

        output = self.pipeline(
            model_input,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )[0]["generated_text"]

        return output[len(model_input) :].strip().split("\n")[0]

    def __call__(self, prompts):
        return [self.forward_call(prompt) for prompt in prompts]
