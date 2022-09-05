"""
This script evaluates your FST on the dev set.

Usage:
    python3 evaluate.py -d [directory for task's dev files] -l [language code]
Prerequisites:
    - your language FST in .bin format (save stack xxx.bin)
    - place this file in the same dir as your FST files or change the path to relevant files
"""

import subprocess
import argparse
import pandas as pd
from pathlib import Path
from logger import logger


def get_train_sets(data_dir, iso):
    if iso is not None:
        files = [
            x
            for x in data_dir.iterdir()
            if x.stem.startswith(iso) and x.suffix == ".train"
        ]
    dfs = (pd.read_csv(f, sep="\t", names=["lemma", "form", "tag"]) for f in files)
    data = pd.concat(dfs)

    lemmas = list(data.lemma.unique())
    tags = list(data.tag.unique())

    return lemmas, tags


def get_dev(data_dir, iso=None, test=False):
    if iso is not None:
        if test:
            dev = [
                x
                for x in data_dir.iterdir()
                if x.stem.startswith(iso) and x.suffix == ".test"
            ]
        else:
            dev = [
                x
                for x in data_dir.iterdir()
                if x.stem.startswith(iso) and x.suffix == ".dev"
            ]
    else:
        logger.info("Please specify the language code.")
        return ""
    return dev[0]  # only one dev file per language


def generate_test_strings(dev_file, iso, lemmas, tags, test=False):
    targets = []
    both_seen = []
    seen_lemma = []
    seen_feats = []
    unseen = []
    with open(f"{iso}.txt", mode="w", encoding="utf8") as f:
        with open(dev_file, mode="r", encoding="utf-8") as df:
            for i, line in enumerate(df):
                if line:
                    if test:
                        lemma, tag_str = line.rstrip().split("\t")
                    if not test:
                        lemma, form, tag_str = line.rstrip().split("\t")
                        targets.append(form)
                    test_str = f"{lemma}" + "".join(tag_str.split("|"))
                    f.write(test_str + "\n")

                    if lemma in lemmas and tag_str not in tags:
                        seen_lemma.append(i)
                    if tag_str in tags and lemma not in lemmas:
                        seen_feats.append(i)
                    if lemma in lemmas and tag_str in tags:
                        both_seen.append(i)
                    if lemma not in lemmas and tag_str not in tags:
                        unseen.append(i)
    return targets, both_seen, seen_lemma, seen_feats, unseen


def make_predictions(iso):
    # cat hsi.txt| flookup -i hsi.bin > hsi_results.txt
    subprocess.getoutput(f"cat {iso}.txt| flookup -i {iso}.bin > {iso}_results.txt")
    predictions = []
    with open(f"{iso}_results.txt", mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[0].isalpha() and lines[i - 1] == "\n":
                test_str, _, pred = line.partition("\t")
                if "+" in pred.rstrip() and pred.rstrip() != "+?":
                    pass
                else:
                    predictions.append(pred.rstrip())
    return predictions


def write_predictions(iso, test_file, predictions, split="test"):
    with open(test_file, mode="r", encoding="utf-8") as inpf:
        with open(f"{iso}.{split}", mode="w", encoding="utf-8") as f:
            for i, line in enumerate(inpf):
                if split == "test":
                    lemma, tag_str = line.rstrip().split("\t")
                elif split == "dev":
                    lemma, form, tag_str = line.rstrip().split("\t")
                f.write(f"{lemma}\t{predictions[i]}\t{tag_str}\n")


def get_accuracy(targets, predictions, both_seen, seen_lemma, seen_feats, unseen):
    assert len(targets) == len(predictions)
    score_all = (
        sum(1 for x, y in zip(targets, predictions) if x == y) / len(targets) * 100
    )
    score_predictions = (
        sum(1 for x, y in zip(targets, predictions) if x == y)
        / len([x for x in predictions if x != "+?"])  # unimplemented or out-of-vocab
        * 100
        if [x for x in predictions if x != "+?"]
        else 0
    )

    both_seen_pairs = [(targets[i], predictions[i]) for i in both_seen]
    score_both_seen = (
        sum(1 for x, y in both_seen_pairs if x == y) / len(both_seen_pairs) * 100
        if len(both_seen_pairs) != 0
        else 0
    )

    seen_lemma_pairs = [(targets[i], predictions[i]) for i in seen_lemma]
    score_seen_lemma = (
        sum(1 for x, y in seen_lemma_pairs if x == y) / len(seen_lemma_pairs) * 100
        if len(seen_lemma_pairs) != 0
        else 0
    )

    seen_feats_pairs = [(targets[i], predictions[i]) for i in seen_feats]
    score_seen_feats = (
        sum(1 for x, y in seen_feats_pairs if x == y) / len(seen_feats_pairs) * 100
        if len(seen_feats_pairs) != 0
        else 0
    )

    unseen_pairs = [(targets[i], predictions[i]) for i in unseen]
    score_unseen = (
        sum(1 for x, y in unseen_pairs if x == y) / len(unseen_pairs) * 100
        if len(unseen_pairs) != 0
        else 0
    )

    return (
        score_all,
        score_predictions,
        score_both_seen,
        score_seen_lemma,
        score_seen_feats,
        score_unseen,
    )


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "-d",
        "--directory",
        type=str,
        default="../../2022InflectionST/part1/development_languages",
        help="directory of dev file",
    )
    argp.add_argument(
        "-l",
        "--language",
        type=str,
        help="language code",
    )
    argp.add_argument("--test", dest="is_test", action="store_true")
    argp.set_defaults(is_test=False)

    args = argp.parse_args()

    data_dir = Path(args.directory)
    language_code = args.language

    dev_file = get_dev(data_dir, language_code, args.is_test)
    if dev_file:
        logger.info(f"Found dev file for {language_code}:")
        logger.info(f"\t{dev_file.name}")

    lemmas, tags = get_train_sets(data_dir, language_code)

    targets, both_seen, seen_lemma, seen_feats, unseen = generate_test_strings(
        dev_file, language_code, lemmas, tags, args.is_test
    )
    predictions = make_predictions(language_code)
    if args.is_test:
        write_predictions(language_code, dev_file, predictions)
    if not args.is_test:
        write_predictions(language_code, dev_file, predictions, "dev")
        (
            acc_score,
            score_predictions,
            score_both_seen,
            score_seen_lemma,
            score_seen_feats,
            score_unseen,
        ) = get_accuracy(
            targets, predictions, both_seen, seen_lemma, seen_feats, unseen
        )
        # logger.info(f"Accuracy on dev: {acc_score:.3f}")
        # logger.info(f"Accuracy for predicted items: {score_predictions:.3f}")

        # logger.info(f"both:\t{score_both_seen:.3f}")
        # logger.info(f"lemma:\t{score_seen_lemma:.3f}")
        # logger.info(f"feats:\t{score_seen_feats:.3f}")
        # logger.info(f"unseen:\t{score_unseen:.3f}")

        print(
            "Lang\tall acc\tboth\tlemma\tfeats\tunseen\t#total\t#both\t#lemma\t#feats\t#unseen\n"
        )
        print(
            f"{language_code}\t{acc_score:.3f}\t{score_both_seen:.3f}\t{score_seen_lemma:.3f}\t{score_seen_feats:.3f}\t{score_unseen:.3f}\t{len(targets)}\t{len(both_seen)}\t{len(seen_lemma)}\t{len(seen_feats)}\t{len(unseen)}\n"
        )


if __name__ == "__main__":
    main()