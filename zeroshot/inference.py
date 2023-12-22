from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

OUTPUT_DIR = "outputs"
BASE_DATASET_PATH = "../nngen/data/"
DIFF_FILE_SUFFIX = ".diff"
COMMIT_FILE_SUFFIX = ".msg"
tokenizer = AutoTokenizer.from_pretrained("mamiksik/T5-commit-message-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("mamiksik/T5-commit-message-generation")


class Example(object):
    """A single training/test example."""

    def __init__(
        self,
        idx,
        source,
        target,
    ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(diff_list, commit_list):
    """Read examples from filename."""
    examples = []
    for idx, (diff, commit) in enumerate(zip(diff_list, commit_list)):
        examples.append(
            Example(
                idx=idx,
                source=diff,
                target=commit,
            )
        )
    return examples


predictions = []


def read_data(split_name):
    with open(
        os.path.join(BASE_DATASET_PATH, split_name + DIFF_FILE_SUFFIX), "r"
    ) as diff_file, open(
        os.path.join(BASE_DATASET_PATH, split_name + COMMIT_FILE_SUFFIX), "r"
    ) as commit_file:
        diff_lines = diff_file.readlines()
        diff_lines = [diff.strip() for diff in diff_lines]
        commit_lines = commit_file.readlines()
        commit_words = [line.strip().split() for line in commit_lines]
        commit_words = [word for line in commit_words for word in line]
        commit_words = [" ".join(word for word in commit_words)]
        return diff_lines, commit_lines


test_diffs, test_commit_messages = read_data("cleaned.test")
test_examples = read_examples(test_diffs, test_commit_messages)
print(len(test_commit_messages))
outfile = open(os.path.join(OUTPUT_DIR, "predictions.txt"), "w")
goldfile = open(os.path.join(OUTPUT_DIR, "gold.txt"), "w")
for example in test_examples:
    input_text = example.source
    target_text = example.target
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    truncate_target_ids = input_ids
    outputs = model.generate(
        truncate_target_ids, max_length=50, temperature=0.7, num_return_sequences=1
    )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_output)
    predictions.append(decoded_output)
    outfile.write(decoded_output + "\n")
