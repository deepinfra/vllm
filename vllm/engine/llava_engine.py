from vllm.engine.llm_engine import LLMEngine
from transformers import CLIPImageProcessor
import time
from functools import partial
from typing import List, Optional

from vllm.engine.ray_utils import ray
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Sequence, SequenceGroup)
from PIL import Image


class LLaVAEngine(LLMEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.model_config.tokenizer)

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        images: Optional[List[Image.Image]] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            images: A list of PIL images for the prompt. It supports multiple
                images, although most llava models are trained with only one image.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # process images
        extra_data = None
        if images is not None and len(images) > 0:
            pixel_values = self.image_processor(
                images, return_tensors="pt")['pixel_values']
            extra_data = {'pixel_values': pixel_values}
        else:
            pixel_values = None

        # Check the validation of the input. And expand each image token to the
        # number of tokens per image. So the scheduler can allocate proper resources.
        # num_workers = len(self.workers)
        # random select a worker
        # [NOTE]Aakash: Make it work on split model and multiple workers(test)
        worker = self.driver_worker
        if self.parallel_config.worker_use_ray:
            execute_model_method = partial(worker.execute_method.remote,
                                            'execute_model_method')
        else:
            execute_model_method = getattr(worker, 'execute_model_method')
        outputs = execute_model_method('prepare_promt', prompt_token_ids,
                                        pixel_values)
        if self.parallel_config.worker_use_ray:
            outputs = ray.get(outputs)
        processed_token_ids = outputs
        prompt_token_ids = processed_token_ids.tolist()

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id,
                       prompt,
                       prompt_token_ids,
                       block_size,
                       extra_data=extra_data)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        # Execute the model.
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
            runner_method="execute_llava_model",
        )

        output = output[0]

        return self._process_model_outputs(output, scheduler_outputs)
