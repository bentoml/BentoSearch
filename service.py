from __future__ import annotations

import uuid
import asyncio
from typing import AsyncGenerator, List, Tuple, Optional

import bentoml
import pydantic
from googlesearch import search
import bs4
import httpx
from annotated_types import Ge, Le
from typing_extensions import Annotated

from bentovllm_openai.utils import openai_endpoints


MAX_TOKENS = 8192
MAX_CONTENTS = 500
SYSTEM_PROMPT = """You are a helpful assistant who is expert at answering user's queries based on the cited context.

Your MAIN TASK is to generate a response that is informative and relevant to the users' query based on given context, where it consists of search results containing key formatted with [citation number](website link) followed by a brief description of the site.
You must use this context to answer the user's query in the best way possible. Use an unbiased and journalistic tone in your response. Do not repeat the text.
You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself.
Your responses should be medium to long in length, be informative and relevant to the user's query. You must use markdown to format your response.
You should use bullet points to list the information. Make sure the answer is not short and is informative. You have to cite the answer using [citation number](website link) notation. You must cite the sentences with their relevant context number.
You must cite each and every part of the answer so the user can know where the information is coming from. Anything inside the following context block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer questions on the basis of it and cite the relevant information from it but you do not have to
talk about the context in your response.

Context:
{context_block}
"""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"


class QueryResponse(pydantic.BaseModel):
  url: str
  description: str


@bentoml.service(resources={"cpu": 2})
class Google:
  async def fetch(self, url: str, timeout: int=300) -> Tuple[str, Optional[str]]:
    """Fetch the content of a webpage given a URL and a timeout."""
    try:
      print(f"Fetching link: {url}")
      async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "lxml")
        paragraphs = soup.find_all("p")
        page_text = " ".join([para.get_text() for para in paragraphs])
      return url, page_text
    except (httpx.RequestError, asyncio.TimeoutError) as e:
      print(f"Error fetching {url}: {e}")
    return url, None

  @bentoml.api
  async def query(self, query: str, num_search: Annotated[int, Ge(0)] = 10, timeout: Annotated[int, Ge(1)] = 3) -> List[QueryResponse]:
    results = await asyncio.gather(*[self.fetch(url, timeout) for url in search(query, num_results=num_search)])
    return [QueryResponse(url=url, description=page_text) for url, page_text in results if page_text is not None]


# steps: Google.query -> ask


@openai_endpoints(
  model_id=MODEL_ID,
  default_chat_completion_parameters=dict(stop=["<|eot_id|>"]),
)
@bentoml.service(
  name="agentsearch-llama-3-1",
  traffic={
    "timeout": 300,
    "concurrency": 256,  # Matches the default max_num_seqs in the VLLM engine
  },
  resources={
    "gpu": 1,
    "gpu_type": "nvidia-a100-80gb",
  },
)
class Agent:
  google = bentoml.depends(Google)

  def __init__(self) -> None:
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    ENGINE_ARGS = AsyncEngineArgs(model=MODEL_ID, max_model_len=MAX_TOKENS, enable_prefix_caching=True)

    self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    self.stop_token_ids = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

  @bentoml.api
  async def search(
    self,
    prompt: str = "Who won the 2024 Olympics for track and field?",
    max_tokens: Annotated[int, Ge(20), Le(MAX_TOKENS)] = MAX_TOKENS,
    num_search: Annotated[int, Ge(0)] = 5,
    timeout: Annotated[int, Ge(1)] = 3,
    max_content: Annotated[int, Ge(0), Le(MAX_CONTENTS)] = MAX_CONTENTS,
  ) -> AsyncGenerator[str, None]:
    from vllm import SamplingParams

    SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens, stop_token_ids=self.stop_token_ids)

    context = await self.google.query(prompt, num_search=num_search, timeout=timeout)

    formatted_system_prompt = SYSTEM_PROMPT.format(
      context_block="\n".join([f"[{i + 1}]({q.url}): {q.description[:max_content]}" for i, q in enumerate(context)])
    )

    prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=formatted_system_prompt)
    stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

    cursor = 0
    async for request_output in stream:
      text = request_output.outputs[0].text
      yield text[cursor:]
      cursor = len(text)
