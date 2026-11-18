# Surprisal and fixation-related EEG in ZuCo 2.0

This project asks how the estimated association between word surprisal and fixation-related EEG changes when eye movements are taken into account. ZuCo 2.0 records EEG and gaze during natural reading. To study a word, we first need to identify the physical fixation and locate its onset in the EEG recording.

Two exports supply different parts of that link. The benchmark MATLAB training files contain word-indexed eye-tracking segments and sentence-level EEG. NEMAR supplies continuous BrainVision recordings and the event stream for each run. The 16 training participants appear in both exports. NEMAR also contains two other Y participants. The ten X participants in the benchmark test set have separate MATLAB files and no recording in this NEMAR release. [Corpus paper](https://aclanthology.org/2020.lrec-1.18/) · [Benchmark data and acquisition](data/README.md)

| Audit | Result |
| --- | --- |
| [MATLAB](audit/matlab/README.md) | The archived inventory contains 42 files and 18,454 sentence EEG matrices. The 32 training files and the detailed YAG sample were checked again for this release. |
| [NEMAR v1.0.0](audit/nemar/README.md) | All 2,411 manifest entries passed their checksums. The 252 runs have 128-channel headers at 500 Hz. The event tables do not carry a direct word identifier. |
| [Alignment](audit/alignment/README.md) | The bounded pilot matched 2,753 fixations across seven blocks. The later normal-reading audit matched 95,393 of 106,673 saved fixation slots across 16 participants, including 2,166 pairs inherited from the pilot. |

Of the 95,393 NR pairs, 91,010 also passed the EEG marker-clock and recording-bound checks. The reference and processing history of the stored voltages remain uncertain. Neighbouring fixations often fall within the same post-onset EEG window. These checks establish event identity and sample location within their stated coverage. Establishing which neural response contributes to a window requires further work.

Later releases will document EEG quality checks, word predictors and statistical analyses. This release contains acquisition code, audit code, compact evidence tables and two diagnostic figures. [Reproduction and comparison with the original analyses](audit/README.md)
