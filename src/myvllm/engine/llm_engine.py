import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler, SchedulerOutput
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        self.config = config
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # scheduler  
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16), 
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024), 
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            long_prefill_token_threshold=config.get("long_prefill_token_threshold", 64), # chunked prefill threshold
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 151645)
        )
        atexit.register(self.exit)


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[int], bool]:
        scheduler_output: SchedulerOutput = self.scheduler.schedule()
        # run the model
        outputs, discard_seq_ids = self.model_runner.call("run", scheduler_output)
        # postprocess the outputs
        self.scheduler.postprocess(scheduler_output, outputs, discard_seq_ids)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduler_output.scheduled_seqs if seq.is_finished]
        num_processed_tokens = sum(num_scheduler_tokens_per_seq for num_scheduler_tokens_per_seq in scheduler_output.num_scheduled_tokens.values())

        return outputs, num_processed_tokens


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config.get('block_size'), sampling_params=sampling_params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, 'tokens/sec')
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output

