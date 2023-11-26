from transformers import AutoTokenizer

from pipegoose.nn.pipeline_parallel import microbatch

MODEL_NAME = "sshleifer/tiny-gpt2"


def test_split_a_mini_batch_to_microbatches():
    BATCH_SIZE = 36
    N_MICROBATCHES = 6

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    text = "Persistence is all you need."
    batch_sentences = [text for _ in range(BATCH_SIZE)]
    inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")

    microbatches = microbatch.split(inputs, n_microbatches=N_MICROBATCHES)

    assert isinstance(microbatches, list)
    assert len(microbatches) == N_MICROBATCHES
    assert all(set(batch.keys()) == set(inputs.keys()) for batch in microbatches) is True

    total_sentences = sum(microbatch["input_ids"].size(0) for microbatch in microbatches)
    assert total_sentences == BATCH_SIZE
