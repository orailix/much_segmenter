import numpy as np
import pytest

from src.much_segmenter import much_segmentation
from src.much_segmenter.utils import (
    get_known_eos_or_eot,
    get_punctuation_set,
    get_tokenizer,
    get_tokens_and_spans,
)
from tests.data import (
    PUNCT_BEFORE_EOS_PHRASES,
    edge_case_segmentations,
    expected_outputs,
    non_idempotent_segmentations,
    test_phrases,
)


@pytest.mark.parametrize("model_name", list(expected_outputs.keys()))
def test_segmentation_correctness(model_name):
    tokenizer = get_tokenizer(model_name)
    for phrase, expected in zip(test_phrases, expected_outputs[model_name]):
        out = much_segmentation(phrase, tokenizer)
        assert out == expected, f"Mismatch for model {model_name} on phrase '{phrase}'"


@pytest.mark.parametrize("model_name", list(expected_outputs.keys()))
def test_segmentation_covers_all_tokens(model_name):
    tokenizer = get_tokenizer(model_name)
    for phrase in test_phrases:
        out = much_segmentation(phrase, tokenizer)
        flat = np.concatenate(out).tolist()
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        assert flat == list(
            range(len(tokens))
        ), f"Token coverage mismatch for model {model_name} on phrase '{phrase}'"


@pytest.mark.parametrize("model_name", list(expected_outputs.keys()))
def test_segmentation_with_precomputed_tokens(model_name):

    # For BERT, we expect some errors
    if model_name == "bert-base-uncased":
        tokenizer = get_tokenizer(model_name)

        for phrase in test_phrases:
            output_tokens_wo, token_char_span_wo = get_tokens_and_spans(
                generation=phrase, llm_tokenizer=tokenizer
            )
            with pytest.raises(ValueError):
                output_tokens_with, token_char_span_with = get_tokens_and_spans(
                    generation=phrase,
                    llm_tokenizer=tokenizer,
                    precomputed_tokens=output_tokens_wo,
                )

        return

    # For the others, we check that both method matches
    tokenizer = get_tokenizer(model_name)

    for phrase in test_phrases:
        output_tokens_wo, token_char_span_wo = get_tokens_and_spans(
            generation=phrase, llm_tokenizer=tokenizer
        )
        output_tokens_with, token_char_span_with = get_tokens_and_spans(
            generation=phrase,
            llm_tokenizer=tokenizer,
            precomputed_tokens=output_tokens_wo,
        )

        assert output_tokens_wo == output_tokens_with
        assert token_char_span_wo == token_char_span_with


@pytest.mark.parametrize("non_idempotent_vals", non_idempotent_segmentations)
def test_non_idempotent_sentence(non_idempotent_vals):

    # Unpacking
    (
        model_name,
        sentence,
        llm_generated_tokens,
        direct_tokenization,
    ) = non_idempotent_vals

    tokenizer = get_tokenizer(model_name)
    segmentation_precomputed = much_segmentation(
        sentence, tokenizer, llm_generated_tokens
    )
    segmentation_direct = much_segmentation(sentence, tokenizer)
    assert llm_generated_tokens != direct_tokenization
    assert np.concatenate(segmentation_precomputed).tolist() == list(
        range(len(llm_generated_tokens))
    )
    assert np.concatenate(segmentation_direct).tolist() == list(
        range(len(direct_tokenization))
    )


@pytest.mark.parametrize("model_name", list(expected_outputs.keys()))
def test_eos_eot_in_its_own_claim(model_name):
    tokenizer = get_tokenizer(model_name)
    count_test = 0
    for phrase in test_phrases:

        encoded = tokenizer.encode(phrase, add_special_tokens=False)
        found_eos = False
        for final_token in get_known_eos_or_eot(tokenizer):
            found_eos = found_eos or (encoded[-1] == final_token)

        if not found_eos:
            continue

        # checking that the segmentation only contains EOS in last chunk
        count_test += 1
        segmentation = much_segmentation(phrase, tokenizer)
        assert len(segmentation[-1]) == 1

    assert model_name == "bert-base-uncased" or count_test > 0


@pytest.mark.parametrize("model_name", list(expected_outputs.keys()))
def test_no_terminal_punct_isolation(model_name):

    tokenizer = get_tokenizer(model_name)
    punct_set = get_punctuation_set()
    tokens_all = tokenizer.encode

    for phrase in test_phrases:
        segmentation = much_segmentation(phrase, tokenizer)
        encoded = tokenizer.encode(phrase, add_special_tokens=False)

        last_is_eos = len(segmentation[-1]) == 1 and encoded[
            segmentation[-1][0]
        ] in get_known_eos_or_eot(tokenizer)
        if last_is_eos:
            continue

        chunk_tokens = [encoded[i] for i in segmentation[-1]]
        chunk_text = tokenizer.decode(chunk_tokens).strip()
        assert chunk_text not in punct_set, (
            f"Terminal punctuation isolated in its own chunk for model "
            f"'{model_name}' on phrase '{phrase}': last content chunk decodes to '{chunk_text}'"
        )


@pytest.mark.parametrize("model_name,phrase,expected", edge_case_segmentations)
def test_edge_case_segmentation(model_name, phrase, expected):
    tokenizer = get_tokenizer(model_name)
    out = much_segmentation(phrase, tokenizer)
    assert out == expected, (
        f"Edge-case mismatch for model '{model_name}' on phrase {phrase!r}: "
        f"got {out}, expected {expected}"
    )


@pytest.mark.parametrize("phrase,model_name", PUNCT_BEFORE_EOS_PHRASES)
def test_punct_before_eos_not_merged(phrase, model_name):
    from src.much_segmenter.utils import get_punctuation_set

    tokenizer = get_tokenizer(model_name)
    encoded = tokenizer.encode(phrase, add_special_tokens=False)
    seg = much_segmentation(phrase, tokenizer)

    assert (
        len(seg) >= 3
    ), f"Expected at least 3 chunks for '{model_name}' on {phrase!r}, got {seg}"

    assert len(seg[-1]) == 1 and encoded[seg[-1][0]] in get_known_eos_or_eot(
        tokenizer
    ), f"Last chunk is not a lone EOS token for '{model_name}' on {phrase!r}"

    pre_eos_text = tokenizer.decode([encoded[i] for i in seg[-2]]).strip()
    assert pre_eos_text in get_punctuation_set(), (
        f"Expected punctuation chunk before EOS for '{model_name}' on {phrase!r}, "
        f"got {pre_eos_text!r}"
    )
