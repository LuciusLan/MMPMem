from __future__ import annotations
import os
os.environ['HF_HOME'] = '/data_external/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES']="0"

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk

@dataclass
class BEMResult:
    accuracy: float
    bem_prob_equivalent: List[float]
    bem_pred_equivalent: List[bool]
    details: List[Dict[str, Any]]


class BEMScorer:
    """
    BEM scorer with Encyclopedic-VQA-compatible multi-reference handling.

    Supports:
    - one question per example
    - one candidate prediction per example
    - one or more reference answers per example

    Important E-VQA logic:
    - 'a|b|c' means alternative acceptable references
    - 'a&&b&&c' stays inside one reference string
    - final score for an example is the maximum BEM score across references
    """

    def __init__(
        self,
        model_name: str = "kortukov/answer-equivalence-bem",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.equivalent_label_id = self._infer_equivalent_label_id()

    def _infer_equivalent_label_id(self) -> int:
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict):
            for k, v in id2label.items():
                label_text = str(v).strip().lower()
                if "equivalent" in label_text or label_text in {"yes", "true"}:
                    return int(k)
        return 1

    @staticmethod
    def _clean_text(x: str) -> str:
        return " ".join(str(x).strip().split())

    @staticmethod
    def _parse_reference_field(ref) -> List[str]:
        """
        Convert one ground-truth field into a list of alternative references.

        E-VQA convention:
        - 'a|b|c' => three alternative references
        - 'a&&b&&c' => one multi-answer reference, preserved as-is
        """
        if ref is None:
            return [""]

        if isinstance(ref, (list, tuple)):
            refs = [str(x).strip() for x in ref if str(x).strip()]
            return refs if refs else [""]

        s = str(ref).strip()
        if not s:
            return [""]

        # split only on '|', preserve '&&'
        refs = [part.strip() for part in s.split("|") if part.strip()]
        return refs if refs else [""]

    def _tokenize_batch(
        self,
        questions: Sequence[str],
        references: Sequence[str],
        candidates: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        texts = []
        text_pairs = []

        for q, r, c in zip(questions, references, candidates):
            q = self._clean_text(q)
            r = self._clean_text(r)
            c = self._clean_text(c)

            # follows the HF BEM reproduction format
            texts.append(f"[CLS] {c} [SEP]")
            text_pairs.append(f"{r} [SEP] {q} [SEP]")

        batch = self.tokenizer(
            text=texts,
            text_pair=text_pairs,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def _score_pairs(
        self,
        questions: Sequence[str],
        references: Sequence[str],
        candidates: Sequence[str],
        batch_size: int = 32,
    ) -> List[float]:
        """
        Returns probability of equivalence for each (q, ref, cand) triple.
        """
        if not (len(questions) == len(references) == len(candidates)):
            raise ValueError("questions, references, candidates must have equal length")

        probs_out: List[float] = []

        for start in range(0, len(questions), batch_size):
            end = start + batch_size
            inputs = self._tokenize_batch(
                questions=questions[start:end],
                references=references[start:end],
                candidates=candidates[start:end],
            )
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            p_equiv = probs[:, self.equivalent_label_id].detach().cpu().tolist()
            probs_out.extend(float(p) for p in p_equiv)

        return probs_out

    @torch.no_grad()
    def score(
        self,
        questions: List[str],
        references: List[str | List[str]],
        candidates: List[str],
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> BEMResult:
        """
        references:
            Either:
            - list[str], where each string may contain '|' alternatives
            - list[list[str]], already split into alternatives
        """
        if not (len(questions) == len(references) == len(candidates)):
            raise ValueError(
                f"Length mismatch: "
                f"{len(questions)=}, {len(references)=}, {len(candidates)=}"
            )

        # Flatten example -> multiple references into pairwise scoring jobs
        flat_questions: List[str] = []
        flat_references: List[str] = []
        flat_candidates: List[str] = []
        example_index: List[int] = []

        parsed_refs_per_example: List[List[str]] = []
        for i, ref_field in enumerate(references):
            ref_list = self._parse_reference_field(ref_field)
            parsed_refs_per_example.append(ref_list)

            for ref in ref_list:
                flat_questions.append(questions[i])
                flat_references.append(ref)
                flat_candidates.append(candidates[i])
                example_index.append(i)

        # Score all flattened triples
        flat_probs = self._score_pairs(
            questions=flat_questions,
            references=flat_references,
            candidates=flat_candidates,
            batch_size=batch_size,
        )

        # Aggregate by example: take max over alternative references
        probs_per_example: List[List[float]] = [[] for _ in range(len(questions))]
        refs_per_example: List[List[str]] = [[] for _ in range(len(questions))]

        for idx, ref, p in zip(example_index, flat_references, flat_probs):
            probs_per_example[idx].append(float(p))
            refs_per_example[idx].append(ref)

        max_probs: List[float] = []
        pred_equiv: List[bool] = []
        details: List[Dict[str, Any]] = []

        for i in range(len(questions)):
            cur_probs = probs_per_example[i]
            cur_refs = refs_per_example[i]

            if not cur_probs:
                best_prob = 0.0
                best_ref = ""
            else:
                best_j = max(range(len(cur_probs)), key=lambda j: cur_probs[j])
                best_prob = float(cur_probs[best_j])
                best_ref = cur_refs[best_j]

            is_equiv = best_prob >= threshold

            max_probs.append(best_prob)
            pred_equiv.append(is_equiv)

            details.append(
                {
                    "question": questions[i],
                    "candidate": candidates[i],
                    "references_all": parsed_refs_per_example[i],
                    "best_reference": best_ref,
                    "bem_prob_equivalent": best_prob,
                    "bem_pred_equivalent": bool(is_equiv),
                }
            )

        accuracy = sum(pred_equiv) / len(pred_equiv) if pred_equiv else 0.0

        return BEMResult(
            accuracy=float(accuracy),
            bem_prob_equivalent=max_probs,
            bem_pred_equivalent=pred_equiv,
            details=details,
        )


def evaluate_with_bem(
    scorer,
    questions: List[str],
    predictions: List[str],
    ground_truths: List[str | List[str]],
    batch_size: int = 32,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    
    result = scorer.score(
        questions=questions,
        references=ground_truths,
        candidates=predictions,
        batch_size=batch_size,
        threshold=threshold,
    )
    return {
        "accuracy": result.accuracy,
        "bem_prob_equivalent": result.bem_prob_equivalent,
        "bem_pred_equivalent": result.bem_pred_equivalent,
        "details": result.details,
    }


if __name__ == "__main__":
    evqa_test = load_from_disk('/data_external/evqa/evqa_test_withimg')

    # questions = [
    #     "What country is this monument located in?",
    #     "Which colors appear on the flag?",
    #     "What is this bird's habitat?",
    # ]
    # # E-VQA-style references:
    # # 1) alternatives separated by '|'
    # # 2) one multi-answer reference containing '&&'
    # # 3) already-pretokenized list of alternatives also supported
    # ground_truths = [
    #     "United States|USA|US",
    #     "red&&white&&blue",
    #     ["wetlands", "marshes", "swamps"],
    # ]
    # predictions = [
    #     "United States of America",
    #     "red, white and blue",
    #     "marshes",
    # ]
    model_name: str = "kortukov/answer-equivalence-bem"
    scorer = BEMScorer(model_name=model_name, device='cuda')

    evqa_ = pd.read_csv('/data_external/evqa/test.csv')
    for lam in ["0.0", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9]:
        with open(f'/latent_aug/MMPMem/evqa_preds/evqa_pred_l{lam}.txt') as f:
            preds = f.readlines()
            preds = ''.join(preds)
        
        questions = []
        predictions = []
        ground_truths = []


        questions_sh = []
        predictions_sh = []
        ground_truths_sh = []
        triplets = preds.split('?')

        eq = evqa_test['question']
        et = evqa_test['question_type']
        step = 0
        for i, sample in tqdm(enumerate(triplets)):
            if sample.count('<s>') != 2:
                if sample == '':
                    pass
                else:
                    if triplets[i+1].startswith('<s>'):
                        triplets[i+1] = sample+triplets[i+1]
                        continue
            else:
                pred, gt, question = sample.split('<s>')
                question = question+"?"
                questions.append(question)
                predictions.append(pred)
                ground_truths.append(gt)

                assert question ==eq[step].strip() or question ==eq[step].strip()[:-1]
                if et[step] != '2_hop':
                    questions_sh.append(question)
                    predictions_sh.append(pred)
                    ground_truths_sh.append(gt)
                step += 1

        out = evaluate_with_bem(
            scorer=scorer,
            questions=questions,
            predictions=predictions,
            ground_truths=ground_truths,
            batch_size=8,
            threshold=0.5,
        )

        print(f"Lambda: {lam}")
        print("All accuracy:", out["accuracy"])

        shout = evaluate_with_bem(
            scorer=scorer,
            questions=questions_sh,
            predictions=predictions_sh,
            ground_truths=ground_truths_sh,
            batch_size=8,
            threshold=0.5,
        )
        print("Single Hop accuracy:", shout["accuracy"])
    # for row in out["details"]:
    #     print(row)