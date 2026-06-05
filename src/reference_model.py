import openai
import os
import time
import torch
import gc
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer, pipeline, GenerationConfig
from dotenv import load_dotenv
from openai_compat import build_openai_client

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import anthropic  # noqa: F401
except ImportError:
    anthropic = None

try:
    import together
except ImportError:
    together = None

load_dotenv()

class GPT:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_LOGPROBS = False
    API_TOP_LOGPROBS = 20
    API_TEMPERATURE = 0.7
    API_TOP_P = 0.95

    def __init__(self, client_name, model_name):
        self.model_name = model_name
        self.client_name = client_name
        if 'openai' in client_name or 'gpt' in client_name:
            self.client = build_openai_client(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
            )
        elif "dashscope" in client_name:
            self.client = build_openai_client(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif 'together' in client_name:
            if together is None:
                raise ImportError(
                    "together package is required for Together API references."
                )
            self.client = together.Together()
        else:
            raise ValueError(f"Unknown client name: {client_name}")
        if tiktoken is not None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            self.tokenizer.vocab_size = 100256
        else:
            self.tokenizer = None

    def generate(
        self,
        convs,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        num_return_sequences: int,
        **kwargs,
    ):
        if isinstance(convs, str):
            convs = [
                {
                    "role":"user",
                    "content":convs
                }
            ]
        if "together" in self.client_name:
            # print("together")
            return self.together_generate(
                convs = convs, 
                max_n_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
            )
        elif "dashscope" in self.client_name:
            # print("dashscope")
            return self.dashscope_generate(
                convs = convs,
                max_n_tokens = max_n_tokens,
                temperature = temperature,
                top_p = top_p,
                num_return_sequences = num_return_sequences,
            )
        elif "gpt" in self.client_name or "openai" in self.client_name:
            # print("gpt")
            return self.openai_generate(
                convs = convs,
                max_n_tokens = max_n_tokens,
                temperature = temperature,
                top_p = top_p,
                num_return_sequences = num_return_sequences,
            )
        else:
            raise ValueError("Only support together, dashscope and gpt right now as client.")
    
    def openai_generate(
        self,
        convs,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        num_return_sequences: int,
        **kwargs,
    ):
        outputs = []
        for conv in convs:
            output = []
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[conv],
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n = num_return_sequences,
                        timeout=self.API_TIMEOUT,
                    )
                    
                    for choice in response.choices:
                        output.append(choice.message.content)
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)
                time.sleep(self.API_QUERY_SLEEP)
            output = [o for o in output if o is not None]
            outputs.append(output)
        return outputs
    
    def dashscope_generate(
        self,
        convs,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        num_return_sequences: int,
        **kwargs,
    ):
        outputs = []
        for conv in convs:
            output = []
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[conv],
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n = num_return_sequences,
                        timeout=self.API_TIMEOUT,
                    )
                    for choice in response.choices:
                        output.append(choice.message.content)
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)
                time.sleep(self.API_QUERY_SLEEP)
            output = [o for o in output if o is not None]
            outputs.append(output)
        return outputs
    
    def together_generate(
        self, convs, max_n_tokens: int, temperature: float, top_p: float, num_return_sequences, **kwargs
    ):
        
        outputs = []
        for conv in convs:
            output = []
            messages = [
                {
                    "role": "user",
                    "content": conv["content"]
                },
            ]
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n = num_return_sequences,
                        timeout=self.API_TIMEOUT,
                    )
                    
                    for choice in response.choices:
                        output.append(choice.message.content)
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)
                time.sleep(self.API_QUERY_SLEEP)
            output = [o for o in output if o is not None]
            outputs.append(output)
        return outputs
    
    def generate_logprobs(
        self, convs: List[List[Dict]], max_n_tokens: int, temperature: float, top_p: float
    ):
        outputs = []
        for conv in convs:
            output = self.API_ERROR_OUTPUT
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conv,
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        # n = self.API_SAMPLE_NUMBER,
                        timeout=self.API_TIMEOUT,
                        logprobs=self.API_LOGPROBS,
                        top_logprobs=self.API_TOP_LOGPROBS,
                        seed=0,
                    )
                    response_logprobs = [
                        dict((response.choices[0].logprobs.content[i_token].top_logprobs[i_top_logprob].token, 
                                response.choices[0].logprobs.content[i_token].top_logprobs[i_top_logprob].logprob) 
                                for i_top_logprob in range(self.API_TOP_LOGPROBS)
                        )
                        for i_token in range(len(response.choices[0].logprobs.content))
                    ]
                    output = {'text': response.choices[0].message.content,
                            'logprobs': response_logprobs,
                            'n_input_tokens': response.usage.prompt_tokens,
                            'n_output_tokens': response.usage.completion_tokens,
                    }
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)
            outputs.append(output)
        return outputs

class HuggingFace:
    def __init__(
        self,
        model_name, 
        device = "cuda:0",
        dtype = torch.float,
    ):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.device = device
        self.dtype = dtype

        # ---- Detect reasoning models with <think> output ----
        # Qwen3 / DeepSeek-R1 / etc. emit "<think>...</think>" before the
        # actual response.  We strip these in post-processing so the judge
        # sees only the final answer.
        name_lower = model_name.lower()
        self._is_reasoning_model = (
            "qwen3" in name_lower
            or "deepseek-r1" in name_lower
            or "/r1-" in name_lower
        )
        if self._is_reasoning_model:
            print(f"[HuggingFace] Detected reasoning model: {model_name}")
            print(f"[HuggingFace] Will strip <think>...</think> from outputs")
        

    @torch.no_grad()
    def generate(
        self,
        full_prompts_list: List[str],
        max_n_tokens,
        temperature: float,
        do_sample: bool = True,
        top_p: float = 0.95,
        top_k: int = 20,
        num_return_sequences: int = 30,
    ) -> List[Dict]:
        gen_cfg = GenerationConfig(
            max_new_tokens=max_n_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k = top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        inputs = self.tokenizer(full_prompts_list, return_tensors = "pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                generation_config = gen_cfg
            )
        responses = self.tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens = True,
        )
        # Strip reasoning <think>...</think> blocks for reasoning models
        if self._is_reasoning_model:
            responses = [self._strip_think(r) for r in responses]
        return responses

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> reasoning blocks from a response."""
        import re
        # Greedy strip: remove everything up to and including the last </think>
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also handle truncated/dangling open tag
        if "<think>" in cleaned and "</think>" not in cleaned:
            cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()
        
        

def _resolve_vllm_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32 or dtype == torch.float:
        return "float32"
    return "auto"

class VLLMReference:
    def __init__(
        self,
        model_name,
        dtype=torch.float,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
    ):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vllm is required when using the vllm reference backend."
            ) from exc

        self.model_name = model_name
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        llm_kwargs = dict(
            model=model_name,
            dtype=_resolve_vllm_dtype(dtype),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**llm_kwargs)

    @torch.no_grad()
    def generate(
        self,
        full_prompts_list: List[str],
        max_n_tokens,
        temperature: float,
        do_sample: bool = True,
        top_p: float = 0.95,
        top_k: int = 20,
        num_return_sequences: int = 30,
    ) -> List[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_n_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            top_k=-1 if top_k is None or top_k <= 0 else top_k,
            n=num_return_sequences,
        )
        outputs = self.llm.generate(full_prompts_list, sampling_params)
        responses = []
        for request_output in outputs:
            for output in request_output.outputs:
                responses.append(output.text)
        return responses
    
    

class TogetherLLM:
    client_name: str = "together"
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_LOGPROBS = False
    API_TOP_LOGPROBS = 20
    def __init__(
        self,
        model_name,
        api_key,
        api_base = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.client = together.Together()
    
    def generate(
        self, 
        prompt: str,
        max_n_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs,
    ):
        responses = None
        messages = [
            {
                "role": "user",
                "content": prompt
            },
        ]
        for _ in range(self.API_MAX_RETRY):
            try:
                responses = self.client.chat.completions.create(
                    model=self.model_name,
                    # prompt=prompt,
                    messages = messages,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n = num_return_sequences,
                    timeout=self.API_TIMEOUT,
                )
                responses = [choice.message.content for choice in responses.choices]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return responses

class ReferenceLLM:
    def __init__(
        self,
        client_name: str,
        model_name_or_path: str,
        model_device: str = "cpu",
        dtype: torch.dtype = torch.float,
        backend: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
    ):
        self.client_name = client_name
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        print("Reference Client Name: ", client_name)

        normalized_backend = backend.lower() if backend is not None else None
        is_remote_client = (
            "openai" in client_name
            or "anthropic" in client_name
            or "together" in client_name
            or "deepseek" in client_name
            or "dashscope" in client_name
        )
        if normalized_backend is None:
            normalized_backend = "api" if is_remote_client else "hf"
        if normalized_backend not in {"api", "hf", "vllm"}:
            raise ValueError(
                f"Unsupported reference backend: {normalized_backend}"
            )

        if normalized_backend == "api" and not is_remote_client:
            raise ValueError(
                "api backend requires a remote API client_name such as openai or together."
            )
        if normalized_backend in {"hf", "vllm"} and is_remote_client:
            raise ValueError(
                "vllm backend is only supported for local HuggingFace references."
                if normalized_backend == "vllm"
                else "hf backend is only supported for local HuggingFace references."
            )

        self.backend = normalized_backend

        if normalized_backend == "api":
            lm = GPT(
                client_name = client_name,
                model_name = model_name_or_path
            )
            self.access_type = "remote"
        elif normalized_backend == "vllm":
            lm = VLLMReference(
                model_name=model_name_or_path,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            self.access_type = "local"
        else:
            lm = HuggingFace(
                model_name = model_name_or_path,
                device = model_device,
                dtype = dtype,
            )
            self.access_type = "local"
        self.reference_lm = lm
        self.model_device = model_device
        self.dtype = dtype
    
    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_n_tokens: int = 256, 
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        num_return_sequences: int = 30,
        do_sample: bool = True,
    ):
        
        if self.access_type == "remote":
            print("Remote Access...")
            convs = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            response = self.reference_lm.generate(
                convs = convs,
                max_n_tokens = max_n_tokens, 
                temperature = temperature,
                top_p = top_p,
                num_return_sequences = num_return_sequences,
            )
            print("response: ", response)
            response = response[0]
        else:
            response = self.reference_lm.generate(
                full_prompts_list = [prompt],
                max_n_tokens = max_n_tokens,
                temperature = temperature,
                do_sample = do_sample,
                top_p = top_p,
                top_k = top_k,
                num_return_sequences = num_return_sequences,
            )
            
        return response
    

