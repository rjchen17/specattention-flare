from os import listdir
from pathlib import Path
import json
from torch import tensor

class VocabularyMaker():

    def __init__(self, path):

        self.path = path

    def make_vocabs(self, vocab_dir):

        for language in Path(self.path).iterdir():
            if not language.is_dir() or language.name == ".git":
                continue

            with open(language / "datasets" / "validation-short" / "main.tok") as my_file:
                vocab_stripped = {token for token in set(my_file.read()) if token.strip()}
                vocab = {char: index + 1 for index, char in enumerate(vocab_stripped)}
                vocab["<unk>"] = 0

                with open(vocab_dir / f"{language.name}.json", 'w', encoding='utf-8') as vocab_file:
                    json.dump(vocab, vocab_file)

class Vocabulary():

    def __init__(self, path):

        with open(path, 'r') as vocab_file:
            self.vocab = json.load(vocab_file)

    def __len__(self):

        return len(self.vocab)

    def __call__(self, string: list) -> list[int]:

        # If string is empty, add explicit empty string token
        if len(string) == 0:
            string = [""]

        # Remove final newline, if present
        if string[-1] == "\n":
            string = string[0:-1]

        indices = [self.vocab[char] if char in self.vocab else self.vocab["<unk>"] for char in string]

        return indices

def main():

    vocabmaker = VocabularyMaker("../flare")
    vocabmaker.make_vocabs(Path("../vocabs"))

if __name__ == "__main__":
    main()