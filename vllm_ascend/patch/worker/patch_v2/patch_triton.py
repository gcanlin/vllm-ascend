from vllm.v1.worker.gpu import input_batch, model_runner, structured_outputs
from vllm.v1.worker.gpu.sample import bad_words, gumbel, logprob, penalties, prompt_logprob, sampler, states
from vllm.v1.worker.gpu.spec_decode import rejection_sampler
from vllm.v1.worker.gpu.spec_decode.eagle import speculator

# vllm PR #41035 (v0.21.1rc0) merged probabilistic + synthetic rejection sampler
# utils into a single `rejection_sampler_utils` module and renamed the sample
# entry point from `probabilistic_rejection_sample` to `rejection_sample`. On
# v0.21.0 we still have the pre-rename layout.
try:
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils

    _rejection_sample_attr = "rejection_sample"
except ImportError:
    from vllm.v1.worker.gpu.spec_decode import (
        probabilistic_rejection_sampler_utils as rejection_sampler_utils,
    )

    _rejection_sample_attr = "probabilistic_rejection_sample"

from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words
from vllm_ascend.worker.v2.sample.gumbel import apply_temperature, gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs, compute_topk_logprobs
from vllm_ascend.worker.v2.sample.min_p import apply_min_p
from vllm_ascend.worker.v2.sample.penalties import apply_penalties, bincount
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    rejection_sample as npu_rejection_sample,
)
from vllm_ascend.worker.v2.structured_outputs import _apply_grammar_bitmask_kernel

penalties.apply_penalties = apply_penalties
# because sampler.py and speculator.py are imported before this patch, they must be overridden
sampler.gumbel_sample = gumbel_sample
input_batch.post_update = post_update
prompt_logprob.compute_topk_logprobs = compute_topk_logprobs
sampler.compute_topk_logprobs = compute_topk_logprobs
rejection_sampler.compute_topk_logprobs = compute_topk_logprobs
states.apply_min_p = apply_min_p
penalties.bincount = bincount
speculator.gumbel_sample = gumbel_sample
model_runner.post_update = post_update
bad_words.apply_bad_words = apply_bad_words
gumbel.apply_temperature = apply_temperature
states.apply_temperature = apply_temperature
logprob.compute_token_logprobs = compute_token_logprobs
structured_outputs._apply_grammar_bitmask_kernel = _apply_grammar_bitmask_kernel
setattr(rejection_sampler_utils, _rejection_sample_attr, npu_rejection_sample)
setattr(rejection_sampler, _rejection_sample_attr, npu_rejection_sample)
