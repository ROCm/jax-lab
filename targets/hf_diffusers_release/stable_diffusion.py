#!/usr/bin/env python3
"""jax_stable_diffusion inference benchmark (CompVis/stable-diffusion-v1-4).

Runs Flax Stable Diffusion inference across all local GPUs via jax.pmap and reports
average inference time. Runs on the default jax.pmap (= jit(shard_map)) on modern JAX,
without the legacy JAX_PMAP_NO_RANK_REDUCTION / JAX_PMAP_SHMAP_MERGE flags (removed in
newer JAX).

Two details keep throughput on par with -- in fact ahead of -- the old legacy-pmap
numbers once those flags are gone:

  * Parameters are replicated to all devices ONCE, before the loop. Legacy pmap made
    re-replication essentially free, so the original re-replicated every iteration; the
    new pmap (NamedSharding) turns that into a real host->device broadcast of the whole
    model every step.

  * The pipeline is built with `safety_checker=None`. diffusers' per-step safety path
    calls `unreplicate(safety_model_params)` on a replicated argument whose value is then
    never used (see pipeline_flax_stable_diffusion.py); under the new pmap that indexing
    forces a global gather of the entire safety model to every device on every step.
    Disabling the safety checker is the standard throughput-benchmark configuration and
    avoids that gather without patching diffusers internals.
"""

import timeit

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline

# Prompts to benchmark (the original sourced these from the now-gated
# `lambdalabs/pokemon-blip-captions` dataset; only the strings matter for timing).
_BASE_PROMPTS = [
    "a drawing of a green pokemon with red eyes",
    "a green and yellow toy with a red nose",
    "a red and white ball with an angry look on its face",
    "a cartoon character with a potted plant on his head",
    "a blue and white bird with a long beak",
    "a cartoon ghost with a creepy smile",
    "a blue and yellow fish with big eyes",
    "a small orange dragon with a flame on its tail",
    "a purple cat-like creature with a curled tail",
    "a brown owl pokemon with large round eyes",
]

# Benchmark configuration.
NUM_PROMPTS = 101
NUM_INFERENCE_STEPS = 25

prompts = [_BASE_PROMPTS[i % len(_BASE_PROMPTS)] for i in range(NUM_PROMPTS)]

# Load model
dtype = jnp.float16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
    safety_checker=None,
)


def create_key(seed=0):
    return jax.random.PRNGKey(seed)


# Loop-invariant setup: replicate params and build the PRNG once. params never change and
# the seed is fixed, so doing this per-iteration would only pay a redundant per-step
# broadcast under the new pmap.
p_params = replicate(params)
rng = jax.random.split(create_key(0), jax.device_count())

# Do inference
start_time = timeit.default_timer()
num_prompts = NUM_PROMPTS
i = 1
total_ignored = 0

for prompt in prompts[i:num_prompts]:
    prompt = [prompt] * jax.device_count()
    prompt_ids = shard(pipeline.prepare_inputs(prompt))

    try:
        _ = pipeline(
            prompt_ids,
            p_params,
            rng,
            jit=True,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )[0]
    except ValueError:  # the model can occasionally produce an unacceptable image
        total_ignored += 1

    if i == 1:
        first_infer_time = timeit.default_timer() - start_time
    elif (i % 10) == 0:
        elapsed = timeit.default_timer() - start_time
        print(
            "Num_Prompts Processed: "
            + str(i)
            + ", Average inference time       : "
            + str(elapsed / num_prompts)
        )

    i += 1

# Print stats
elapsed = timeit.default_timer() - start_time

print(" ")
print("Number of Prompts                    : " + str(num_prompts))
print("Total Inference time:                : " + str(elapsed) + " sec")
print("Inference time for first prompt      : " + str(first_infer_time) + " sec")
print("Average Inference time               : " + str(elapsed / num_prompts) + " sec")
print(
    "Average Inference time(exclude first): "
    + str((elapsed - first_infer_time) / (num_prompts - 1))
    + " sec"
)
