# Dataset attribution

The original HotpotQA-style questions and the frozen candidate paragraphs are derived from HotpotQA (Yang et al., EMNLP 2018) and Wikipedia contributors.

- Project: https://hotpotqa.github.io/
- Dataset mirror: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- License stated by the dataset authors: Creative Commons Attribution-ShareAlike 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
- Frozen context source and SHA256: data/contexts/hotpotqa_distractor.metadata.json
- Changes made here: select the original project's 200 validation questions; join each paragraph's sentences; remove answers, supporting-fact labels and all other fields; retain title/text and a normalized question hash.
- The raw math set was already present in the original repository. Its precise per-question sources and license were not documented; no new source or license is asserted here.

This notice applies to dataset material; it does not choose or change a license for the project code.
