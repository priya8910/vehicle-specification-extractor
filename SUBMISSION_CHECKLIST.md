# Assignment Completion Checklist

## Assignment requirements

- [x] Extract text from the supplied 2014 Ford F-150 Workshop Manual PDF.
- [x] Split manual text into section-aware chunks and retain PDF page metadata.
- [x] Generate embeddings and persist/retrieve chunks with ChromaDB.
- [x] Use a Groq LLM to extract answers from retrieved manual evidence.
- [x] Validate structured answers with Pydantic.
- [x] Return JSON and CSV and provide a Streamlit interface.
- [x] Include a project README with setup, architecture, commands, evaluation,
  and limitations.

## Accuracy, evaluation, and demonstration

- [x] Include the assignment brief and source workshop manual in `data/`.
- [x] Cite evidence passages with the source PDF, page, and section.
- [x] Represent supported, missing, and conflicting answers as `found`,
  `not_found`, and `ambiguous`; validate claims against cited passages.
- [x] Include 26 manually checked evaluation fixtures covering torque values,
  procedures, missing values, and trim-dependent values.
- [x] Re-run the updated automated suite after the latest changes: **27 passed**.
- [x] Live Streamlit smoke test: the upper ball joint nut question returned
  `115 Nm` and cited PDF pages 24 and 59.
- [x] Run a baseline 26-case live evaluation and save its per-case report at
  `evaluation/reports/baseline_live_evaluation_2026-10-21.json`.
- [x] Record baseline metrics: Recall@5 **96.2%**, extraction accuracy
  **53.8%**, citation accuracy **69.2%**, and missing-value detection
  **100% (2/2)**.
- [x] Re-run all 26 cases after the latest prompt, structured-output,
  evidence-validation, and retrieval reranking improvements; save the final
  report at `evaluation/reports/live_evaluation_report.json`.
- [x] Record final metrics: Recall@5 **100%**, extraction accuracy **80.8%**,
  citation accuracy **88.5%**, and missing-value detection **100% (2/2)**.
- [x] Document measured accuracy and the remaining failure modes in the README.

## Project organization

- [x] Keep the automotive application in its own folder:
  `vehicle-specification-extractor/`.
- [x] Keep application source, tests, evaluation fixtures, dependency files,
  supplied PDFs, evaluation reports, and README together in the project.
- [x] Prepare `vehicle-specification-extractor-submission/` without `.env`,
  `.venv`, local Chroma data, logs, or generated caches. The index is
  reproducible with the documented ingest command.
- [x] Keep the existing `2_ai_chatbot 3/` project separate and untouched by
  this automotive application's runtime imports.

## Known limitations

- PDF text extraction only; OCR and diagrams/images are not handled.
- Section detection and torque-table matching use heuristics.
- The evaluation set is small and covers only selected manual pages; passing
  it would not establish general or safety-critical accuracy.
- The final evaluation has five failed cases: two inconsistent torque answers,
  two procedure answers returned as not found, and one trim-dependent answer
  returned as found rather than ambiguous. Inspect the per-case report and
  cited manual pages before relying on a value.
