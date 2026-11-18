# MATLAB export

The audit follows HDF5 object references in MATLAB v7.3 files. It inspects every sentence EEG slot and checks the corresponding eye-tracking shape. For a closer look at word timing it reads five NR sentences from YAG and one EEG window. Array shapes below use h5py axis order.

| Files in the historical inventory | Files | Participant × sentence slots | EEG matrices (samples × 105) | Missing EEG slots |
| --- | ---: | ---: | ---: | ---: |
| Train, normal reading (NR) | 16 | 5,584 | 5,385 | 199 |
| Train, task-specific reading (TSR) | 16 | 6,240 | 5,939 | 301 |
| Benchmark test | 10 | 7,390 | 7,130 | 260 |

[File inventory](file_inventory.csv) gives the counts for each participant, including scalar NaN placeholders. [Nested fields](nested_fields.json) records sentence fields per file and the first-slot schemas inspected in YAG and XAH. In YAG the 349 NR and 390 TSR positions contain 663 distinct exact sentence strings across tasks. The audited X export has sentence IDs but lacks the training word-text fields. The test-to-text correspondence remains unresolved.

Sentence EEG contains sampled waveforms. Word-level EEG cells span individual saved fixations. The same structures also contain aggregate EEG features. Run boundaries, inter-sentence samples and a verified mapping from the 105 columns to electrode names are unavailable here. The name rawData alone does not establish the processing history.

In the five YAG sentences, the median eye-tracking time step was 2 ms. All 207 saved word fixation onsets could be located in the corresponding sentence clock. Of 202 adjacent **saved** within-sentence pairs, 189 were less than 500 ms apart. Short or unassigned physical fixations can be absent from this sequence. For the word *Henry*, an inclusive 300–500 ms slice gave 101 × 105 samples. The three anonymous finite columns in the probe are useful for checking indexing only. [Five-sentence counts](yag_sentences.csv) · [Timing and numeric probe](sample_checks.json)

```sh
python audit/matlab/audit_zuco2_n400.py --train-only
# With both benchmark splits installed, run the full inventory into another directory
python audit/matlab/audit_zuco2_n400.py --out audit/matlab/generated_all
```

The default full inventory requires both splits. Train-only replay reproduced all 32 archived train records, the YAG nested schema, all five sentence checks, and the numerical probe exactly. The ten test inventory rows come from the earlier audit. Only their archived sizes and content-hash records could be cross-checked for this release.
