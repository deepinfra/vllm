from typing import List

import math
import threading
import torch

from vllm.logger import init_logger
from .structure_execution_engine import StructureExecutionEngine, JsonNodeStructureGraph, PrecalculatedRawGraph, PrecalculatedStructureGraph, TokenizerData, TokenTensor, get_tensor
# from pydantic import BaseModel

logger = init_logger(__name__)

INDENT_SIZE = 2


class JSONStructureLogitsProcessor:

    json_graph: JsonNodeStructureGraph = JsonNodeStructureGraph()
    precalculated_raw_graph: PrecalculatedRawGraph
    precalculated_structure_graph: PrecalculatedStructureGraph
    token_strings: List[str]
    tokenizer_data: TokenizerData
    tokenizer = None
    _init_thread: threading.Thread

    @classmethod
    def _init_thread_func(cls):
        logger.info(f"Initializing json graph from thread {type(cls).__name__} {type(cls.tokenizer).__name__}: {len(cls.token_strings)}")
        cls.precalculated_raw_graph = cls.json_graph.calculate_structure_graph(cls.token_strings)
        logger.info(f"Finished precalculated raw graph: {len(cls.precalculated_raw_graph)}")
        cls.precalculated_structure_graph = PrecalculatedStructureGraph(cls.precalculated_raw_graph, cls.tokenizer_data, cls.json_graph)
        logger.info(f"Completed json graph initialization from thread: {type(cls).__name__} {type(cls.tokenizer).__name__}")

    @classmethod
    def init_static(cls, llm):
        vocab_size = llm.model_config.get_vocab_size()
        model_device: str = str(getattr(llm.model_config, 'device', 'cuda'))
        cls.tokenizer = llm.tokenizer.tokenizer
        cls.token_strings = []
        example_single_char_token_id = -1
        for i in range(cls.tokenizer.vocab_size):
            if cls.tokenizer.convert_ids_to_tokens(i) == '{':
                example_single_char_token_id = i
        for i in range(cls.tokenizer.vocab_size):
            s = cls.tokenizer.decode([example_single_char_token_id, i])[1:]
            if i in cls.tokenizer.all_special_ids:
                s = ''
            cls.token_strings.append(s)
        cls.token_strings[len(cls.token_strings):vocab_size] = ['\u0000'] * max(0, vocab_size - len(cls.token_strings))
        cls.tokenizer_data = TokenizerData(cls.token_strings, cls.tokenizer.eos_token_id, model_device)
        print(cls.tokenizer.vocab_size, cls.tokenizer.vocab_size, getattr(cls.tokenizer, 'word_embed_proj_dim', None), len(cls.token_strings), cls.tokenizer_data.zero_tensor.shape)

        logger.info(f"Initializing json graph {type(cls).__name__} {type(cls.tokenizer).__name__}")
        cls._init_thread = threading.Thread(target=cls._init_thread_func)
        cls._init_thread.start()

    def __init__(self) -> None:
        self.__class__._init_thread.join()  # Join is idempotent. Easiest way to ensure the graph is initialized.
        self.engine: StructureExecutionEngine = StructureExecutionEngine(self.__class__.precalculated_structure_graph, INDENT_SIZE)
        self.engine.init()
        self.last_token_index: int = 0

    @torch.no_grad()
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""

        # seq_id = hash(tuple(input_ids))

        if self.last_token_index == 0:
            self.engine.init()

        if len(input_ids) == 0:
            self.engine.init()
            self.last_token_index = 0
        assert (self.last_token_index <= len(input_ids))

        allowed_token_tensor: TokenTensor
        try:
            if self.last_token_index == len(input_ids):
                allowed_token_tensor = self.engine()
            else:
                for token_id in input_ids[self.last_token_index:-1]:
                    self.engine.execute_str(self.__class__.token_strings[token_id])
                allowed_token_tensor = self.engine(self.__class__.token_strings[input_ids[-1]])
        except Exception:
            import traceback
            traceback.print_exc()
            allowed_token_tensor = TokenTensor(self.tokenizer_data)
        self.last_token_index = len(input_ids)
        allowed_tokens: torch.BoolTensor = get_tensor(allowed_token_tensor)
        # This check might be slow, and this should not happen: eos_token_id will be allowed in this case.
        # if not torch.any(allowed_tokens):
        #     print("No tokens were allowed after filtering!")
        #     return scores
        if scores.device != allowed_tokens.device:
            # If somehow the TokenTensor is not on the GPU, this reduces TPS by 20%.
            print("Inconsistent devices " + str(scores.device) + " with allowed_tokens device " + str(allowed_tokens.device) + " may cause 20% performance hit")
        mask = torch.full((scores.shape[-1],), -math.inf, device=scores.device)
        mask[allowed_tokens] = 0
        biased_scores = scores + mask

        return biased_scores
