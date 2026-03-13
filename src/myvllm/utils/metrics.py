import time
import numpy as np
from enum import Enum
from collections import defaultdict
from myvllm.engine.sequence import Sequence



class RecordType(Enum):
    GENERATE_START = "generate_start"
    GENERATE_END = "generate_end"       
    TTFT_END = "ttft_end"
    TPOT_END = "tpot_end"                


class Metrics:
    def __init__(self):
        self.seqs: dict[int, Sequence] = {}
        self.num_input_tokens_per_seq: dict[int, int] = defaultdict(int)

        # ----- record time -----
        self.generate_start_t = 0 
        self.generate_end_t = 0
        self.ttft_per_seq: dict[int, float] = {}
        self.tpot_per_seq: dict[int, float] = {}

    @property
    def num_input_tokens(self) -> int:
        """ num of all sequence input tokens """
        return sum(self.num_input_tokens_per_seq.values())

    @property
    def num_output_tokens(self) -> int:
        """ num of all sequence output tokens """
        num_output_tokens = 0
        for seq in self.seqs.values():
            num_input_tokens = self.num_input_tokens_per_seq[seq.seq_id]
            num_output_tokens += len(seq.token_ids) - num_input_tokens - 1  # -1 for prefill generate one token
        return num_output_tokens
    
    @property
    def num_output_tokens_per_seq(self) -> dict[int, int]:
        """ num of output tokens for each sequence """
        num_output_tokens_per_seq = defaultdict(int)
        for seq in self.seqs.values():
            num_input_tokens = self.num_input_tokens_per_seq[seq.seq_id]
            num_output_tokens_per_seq[seq.seq_id] = len(seq.token_ids) - num_input_tokens - 1  # -1 for prefill generate one token
        return num_output_tokens_per_seq

    
    def record(self, record_type: RecordType, seq_ids: list[int] = []):
        """ record time with record type """
        match record_type:
            case RecordType.GENERATE_START:
                self.generate_start_t = time.time()
            case RecordType.GENERATE_END:
                self.generate_end_t = time.time()
            case RecordType.TTFT_END:
                for seq_id in seq_ids:
                    self.ttft_per_seq[seq_id] = (time.time() - self.generate_start_t + 1e-10) * 1000 # ms
            case RecordType.TPOT_END:
                num_output_tokens = self.num_output_tokens_per_seq
                for seq_id in seq_ids:
                    # generate_running_time - TTFT
                    running_time = (time.time() - self.generate_start_t + 1e-10) * 1000  # ms
                    tpot_running_time = running_time - self.ttft_per_seq[seq_id]
                    if num_output_tokens[seq_id] == 0:  # only prefill one token finished, no need to calculate TPOT
                        self.tpot_per_seq[seq_id] = -1
                        continue

                    tpot = tpot_running_time / num_output_tokens[seq_id]  # ms / token
                    self.tpot_per_seq[seq_id] = tpot 
            case _:
                raise ValueError(f"Invalid record type: {record_type}")     

    def append(self, seq: Sequence):
        """ add a sequence to metrics """
        self.seqs[seq.seq_id] = seq
        self.num_input_tokens_per_seq[seq.seq_id] = seq.num_prompt_tokens

    # -------- metrics ---------- 
    def input_tps(self) -> float:
        """ input tokens per second  """
        ttf_list = list(self.ttft_per_seq.values())
        running_time = max(ttf_list) / 1000
        return self.num_input_tokens / running_time

    def output_tps(self) -> float:
        """ output tokens per second """
        running_time = self.generate_end_t - self.generate_start_t + 1e-10
        return self.num_output_tokens / running_time
    
    def ttft(self) -> dict:
        """ time to first token """
        ttft: dict[str, float] = {}
        p50 = np.percentile(list(self.ttft_per_seq.values()), 50, method='nearest')
        p90 = np.percentile(list(self.ttft_per_seq.values()), 90, method='nearest')
        p95 = np.percentile(list(self.ttft_per_seq.values()), 95, method='nearest')
        p99 = np.percentile(list(self.ttft_per_seq.values()), 99, method='nearest')
        ttft['p50'] = p50
        ttft['p90'] = p90
        ttft['p95'] = p95
        ttft['p99'] = p99
        return ttft
    
    def tpot(self) -> dict:
        """ time per output token """
        tpot: dict[str, float] = {}
        tpot_list = list(self.tpot_per_seq.values())
        tpot_list = [x for x in tpot_list if x != -1]  
        mean = np.mean(tpot_list)
        p50 = np.percentile(tpot_list, 50, method='nearest')
        p90 = np.percentile(tpot_list, 90, method='nearest')
        p95 = np.percentile(tpot_list, 95, method='nearest')
        p99 = np.percentile(tpot_list, 99, method='nearest')
        tpot['mean'] = mean
        tpot['p50'] = p50
        tpot['p90'] = p90
        tpot['p95'] = p95
        tpot['p99'] = p99
        return tpot
        
    def prefix_cache_hit_rate(self) -> float:
        """ prefix cache hit rate """
        num_hit = 0
        for seq in self.seqs.values():
            if seq.num_cached_tokens:
                num_hit += 1
        return num_hit / len(self.seqs)
    
    def prefix_cache_token_savings_rate(self) -> float:
        """ prefix cache token savings rate """
        num_savings = 0
        for seq in self.seqs.values():
            num_savings += seq.num_cached_tokens
        return num_savings / self.num_input_tokens



    def __str__(self) -> str:
        """ 返回指标的字符串表示 """
        input_tps_val = self.input_tps()
        output_tps_val = self.output_tps()
        ttft_metrics = self.ttft()
        tpot_metrics = self.tpot()
        prefix_cache_hit_rate = self.prefix_cache_hit_rate()
        prefix_cache_token_savings_rate = self.prefix_cache_token_savings_rate()
        
        result = f"""
{'=' * 80}
Metrics Report
{'=' * 80}
Output Tokens Per Second: {output_tps_val:.2f} tokens/sec
{'-' * 80}
Time To First Token (TTFT):
- P50: {ttft_metrics['p50']:.2f} ms
- P90: {ttft_metrics['p90']:.2f} ms
- P95: {ttft_metrics['p95']:.2f} ms
- P99: {ttft_metrics['p99']:.2f} ms
{'-' * 80}
Time Per Output Token (TPOT):
- Mean: {tpot_metrics['mean']:.2f} ms/token
- P50: {tpot_metrics['p50']:.2f} ms/token
- P90: {tpot_metrics['p90']:.2f} ms/token
- P95: {tpot_metrics['p95']:.2f} ms/token
- P99: {tpot_metrics['p99']:.2f} ms/token
{'-' * 80}
Prefix Cache Hit Rate: {prefix_cache_hit_rate:.4f}
{'-' * 80}
Prefix Cache Token Savings Rate: {prefix_cache_token_savings_rate:.4f}
{'-' * 80}
Total Input Tokens: {self.num_input_tokens}
{'-' * 80}
Total Output Tokens: {self.num_output_tokens}
{'-' * 80}
Number of Sequences: {len(self.seqs)}
{'=' * 80}
""".strip()
        return result
