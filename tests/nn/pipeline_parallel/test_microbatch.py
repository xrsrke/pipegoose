from transformers import AutoTokenizer

from pipegoose.nn.pipeline_parallel import microbatch

MODEL_NAME = "sshleifer/tiny-gpt2"


def test_split_a_mini_batch_to_microbatches():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    batch_sentences = [
        "This is the first sentence.",
        "Here's the second one.",
        "This makes three.",
        "Is this the fourth sentence?",
        "Five sentences now.",
        "This is the sixth sentence.",
        "Sentence seven is here.",
        "We're up to eight now.",
        "This should be the ninth sentence.",
        "And finally, the tenth sentence.",
    ]
    BATCH_SIZE = len(batch_sentences)
    N_MICROBATCHES = 5

    inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")

    microbatches = microbatch.split(inputs, n_microbatches=N_MICROBATCHES)

    assert isinstance(microbatches, list)
    assert len(microbatches) == N_MICROBATCHES
    assert "input_ids" in microbatches[0]
    assert "attention_mask" in microbatches[0]

    total_sentences = sum(microbatch["input_ids"].size(0) for microbatch in microbatches)
    assert total_sentences == BATCH_SIZE
