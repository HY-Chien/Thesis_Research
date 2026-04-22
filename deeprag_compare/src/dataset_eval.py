"""
DRAGIN-compatible datasets + EM/F1 evaluation.

Ported verbatim from https://github.com/oneal2000/DRAGIN/blob/main/src/data.py
so the numbers we report are directly comparable to DeepRAG Table 2 (which
inherits DRAGIN's evaluation protocol; see
https://github.com/gxy-gxy/DeepRAG/blob/main/src/evaluate_ans.py).

Three classes: HotpotQA, WikiMultiHopQA, MuSiQue.
MuSiQue is not in DRAGIN — we reuse HotpotQA-style fewshot demos and a
multi-hop prompt template since MuSiQue is also English multi-hop QA.
"""

import json
import os
import re
import string
from collections import Counter
from typing import List, Dict, Union

import numpy as np


# ----- Eval primitives (DRAGIN-identical) -----

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _em_score(pred: str, golds: List[str]) -> int:
    norm_pred = _normalize_answer(pred)
    return int(any(norm_pred == _normalize_answer(g) for g in golds))


def _f1_score(pred: str, golds: List[str]) -> Dict[str, float]:
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    for g in golds:
        np_, ng = _normalize_answer(pred), _normalize_answer(g)
        if np_ in {"yes", "no", "noanswer"} and np_ != ng:
            continue
        if ng in {"yes", "no", "noanswer"} and np_ != ng:
            continue
        pt, gt = np_.split(), ng.split()
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        p, r = n / len(pt), n / len(gt)
        f1 = 2 * p * r / (p + r)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": p, "recall": r}
    return best


def score(pred: str, golds: List[str]) -> Dict[str, float]:
    if not pred:
        return {"em": 0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    if isinstance(golds, str):
        golds = [golds]
    return {"em": _em_score(pred, golds), **_f1_score(pred, golds)}


# ----- Answer extraction (DRAGIN `get_real_prediction`) -----

_STOP_SUFFIXES = ["</s>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>"]


def extract_answer(text: str) -> str:
    if not text:
        return ""
    # DRAGIN-style: first look for "the answer is"
    if "the answer is" in text:
        beg = text.find("the answer is") + len("the answer is") + 1
        rest = text[beg:]
        for suf in _STOP_SUFFIXES:
            if rest.endswith(suf):
                rest = rest[: -len(suf)]
        rest = rest.strip()
        # Cut at first period / newline — DeepRAG evaluate_ans.py also does:
        #   pred = pred.split("So the answer is")[0].split(".")[0]
        rest = rest.split("\n")[0]
        rest = rest.split(".")[0]
        return rest.strip()
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


# ----- Datasets -----

class HotpotQA:
    tag = "hotpotqa"

    examplars = [
        {"question": "Jeremy Theobald and Christopher Nolan share what profession?",
         "cot": "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer.",
         "answer": "producer"},
        {"question": "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
         "cot": "Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau.",
         "answer": "The Phantom Hour."},
        {"question": "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
         "cot": "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20.",
         "answer": "20"},
        {"question": "Were Lonny and Allure both founded in the 1990s?",
         "cot": "Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s.",
         "answer": "no"},
        {"question": "Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?",
         "cot": "The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn.",
         "answer": "Scott Glenn"},
        {"question": "What was the 2014 population of the city where Lake Wales Medical Center is located?",
         "cot": "Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140.",
         "answer": "15,140"},
        {"question": "Who was born first? Jan de Bont or Raoul Walsh?",
         "cot": "Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first.",
         "answer": "Raoul Walsh"},
        {"question": "In what country was Lost Gravity manufactured?",
         "cot": "The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company.",
         "answer": "Germany"},
    ]

    demo_input = staticmethod(lambda q: f"Question: {q}\nAnswer:")
    test_input = staticmethod(
        lambda q: f"Answer the following question by reasoning step-by-step, following the example above.\nQuestion: {q}\nAnswer:"
    )
    output_template = staticmethod(lambda cot, ans: f"{cot} So the answer is {ans}.")

    @classmethod
    def golds(cls, example: Dict) -> List[str]:
        a = example["answer"]
        return a if isinstance(a, list) else [a]


class WikiMultiHopQA:
    tag = "2wikimultihopqa"

    examplars = [
        {"question": "When did the director of film Hypocrite (Film) die?",
         "cot": "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.",
         "answer": "19 June 2013"},
        {"question": "Are both Kurram Garhi and Trojkrsti located in the same country?",
         "cot": "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
         "answer": "no"},
        {"question": "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
         "cot": "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
         "answer": "no"},
        {"question": "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
         "cot": "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
         "answer": "Genghis Khan"},
        {"question": "Who was born first out of Martin Hodge and Ivania Martinich?",
         "cot": "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
         "answer": "Martin Hodge"},
        {"question": "When did the director of film Laughter In Hell die?",
         "cot": "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
         "answer": "August 25, 1963"},
        {"question": "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
         "cot": "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
         "answer": "Twenty Plus Two"},
        {"question": "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
         "cot": "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
         "answer": "Prithvipati Shah"},
    ]

    demo_input = test_input = staticmethod(lambda q: f"Question: {q}\nAnswer:")
    output_template = staticmethod(lambda cot, ans: f"{cot} So the answer is {ans}.")

    id_alias: Dict[str, List[str]] = {}

    @classmethod
    def load_aliases(cls, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                cls.id_alias[o["Q_id"]] = o["aliases"]

    @classmethod
    def golds(cls, example: Dict) -> List[str]:
        a = example["answer"]
        out = a if isinstance(a, list) else [a]
        aid = example.get("answer_id")
        if aid and aid in cls.id_alias:
            out = list({*out, *cls.id_alias[aid]})
        return out


class MuSiQue:
    tag = "musique"
    # MuSiQue is multi-hop, similar to HotpotQA/2Wiki — reuse WikiMultiHopQA
    # fewshot demos (most generic 8-shot of the three).
    examplars = WikiMultiHopQA.examplars

    demo_input = test_input = staticmethod(lambda q: f"Question: {q}\nAnswer:")
    output_template = staticmethod(lambda cot, ans: f"{cot} So the answer is {ans}.")

    @classmethod
    def golds(cls, example: Dict) -> List[str]:
        a = example["answer"]
        return a if isinstance(a, list) else [a]


class PopQA:
    tag = "popqa"

    examplars = [
        {"question": "Who was the director of Schindler's List?",
         "cot": "Schindler's List is a 1993 historical drama film. It was directed by Steven Spielberg.",
         "answer": "Steven Spielberg"},
        {"question": "What is the capital of Japan?",
         "cot": "Japan is an island country in East Asia. The capital and largest city of Japan is Tokyo.",
         "answer": "Tokyo"},
        {"question": "Who is the author of 1984?",
         "cot": "1984 is a dystopian novel published in 1949. It was written by the English author George Orwell.",
         "answer": "George Orwell"},
        {"question": "What sport does Roger Federer play?",
         "cot": "Roger Federer is a Swiss professional athlete. He is widely regarded as one of the greatest tennis players of all time.",
         "answer": "tennis"},
        {"question": "Who was the composer of Symphony No. 9?",
         "cot": "Symphony No. 9 in D minor is one of the most famous symphonies. It was composed by Ludwig van Beethoven.",
         "answer": "Ludwig van Beethoven"},
        {"question": "In what country is the Eiffel Tower located?",
         "cot": "The Eiffel Tower is a wrought-iron lattice tower. It is located in Paris, which is the capital of France.",
         "answer": "France"},
        {"question": "What is the occupation of Elon Musk?",
         "cot": "Elon Musk is known for founding and leading multiple technology companies including Tesla and SpaceX. He is an entrepreneur and business magnate.",
         "answer": "entrepreneur"},
        {"question": "Who was the screenwriter for Forrest Gump?",
         "cot": "Forrest Gump is a 1994 American comedy-drama film. The screenplay was written by Eric Roth, based on Winston Groom's 1986 novel.",
         "answer": "Eric Roth"},
    ]

    demo_input = test_input = staticmethod(lambda q: f"Question: {q}\nAnswer:")
    output_template = staticmethod(lambda cot, ans: f"{cot} So the answer is {ans}.")

    @classmethod
    def golds(cls, example: Dict) -> List[str]:
        a = example["answer"]
        return a if isinstance(a, list) else [a]


# ----- Helpers -----

DATASET_CLASSES = {
    "hotpotqa": HotpotQA,
    "2wikimultihopqa": WikiMultiHopQA,
    "musique": MuSiQue,
    "popqa": PopQA,
}


def build_fewshot_prompt(ds_cls, n_shots: int = 8) -> str:
    """Build the fewshot demo prefix for a dataset."""
    lines = []
    for ex in ds_cls.examplars[:n_shots]:
        lines.append(ds_cls.demo_input(ex["question"]))
        lines.append(" " + ds_cls.output_template(ex["cot"], ex["answer"]))
        lines.append("")  # blank separator
    return "\n".join(lines).strip() + "\n\n"


def load_dataset(tag: str, data_dir: str) -> List[Dict]:
    path = os.path.join(data_dir, f"{tag}.jsonl")
    data = [json.loads(line) for line in open(path) if line.strip()]
    if tag == "2wikimultihopqa":
        WikiMultiHopQA.load_aliases(
            os.path.join(data_dir, "2wikimultihopqa_id_aliases.jsonl"))
    return data
