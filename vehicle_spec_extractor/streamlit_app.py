from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Protocol

from vehicle_spec_extractor.config import (
    DEFAULT_MANUAL_PATH,
    DEFAULT_PERSIST_DIRECTORY,
)
from vehicle_spec_extractor.extraction import (
    export_csv,
    export_json,
    extract_vehicle_specs,
)
from vehicle_spec_extractor.pipeline import (
    ingest_manual,
    retrieve_manual,
)
from vehicle_spec_extractor.schemas import ExtractionResult


class UploadedPDF(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def _default_manual() -> str:
    return str(DEFAULT_MANUAL_PATH) if DEFAULT_MANUAL_PATH.exists() else ""


def _uploaded_pdf_path(uploaded_file: UploadedPDF) -> Path:
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()[:16]
    upload_dir = Path(tempfile.gettempdir()) / "vehicle_spec_extractor"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{digest}_{Path(uploaded_file.name).name}"
    if not destination.exists():
        destination.write_bytes(content)
    return destination


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Vehicle Specification Extraction System",
        page_icon="🚙",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        .hero {padding: 1.25rem 1.5rem; border-radius: 14px;
               background: linear-gradient(120deg, #102b46, #186b73);
               color: white; margin-bottom: 1rem;}
        .hero p {margin-bottom: 0;}
        </style>
        <div class="hero">
          <h1>Vehicle Specification Extraction System</h1>
          <p>Search the workshop manual, inspect page-level evidence, and export
          validated results. Unsupported claims are reported as not found.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Workshop manual PDF", type=["pdf"])
    if uploaded_file:
        manual_path = _uploaded_pdf_path(uploaded_file)
        st.caption(f"Using uploaded file: {uploaded_file.name}")
    else:
        manual_text = st.text_input(
            "Manual PDF path",
            value=_default_manual(),
            help="The supplied sample manual is selected automatically when available.",
        )
        manual_path = Path(manual_text).expanduser() if manual_text.strip() else None

    with st.sidebar:
        st.subheader("Index settings")
        persist_directory = st.text_input(
            "Persistent Chroma directory",
            value=str(DEFAULT_PERSIST_DIRECTORY),
        )
        top_k = st.slider("Evidence passages", min_value=1, max_value=10, value=5)
        st.caption("The manual is indexed in a separate, content-addressed collection.")
        if st.button("Index manual", type="primary", use_container_width=True):
            if manual_path is None or not manual_path.is_file():
                st.error("Choose an existing PDF file before indexing.")
            else:
                try:
                    with st.spinner("Extracting pages, chunking sections, and indexing..."):
                        outcome = ingest_manual(
                            manual_path,
                            persist_directory=persist_directory,
                        )
                    st.session_state["ingest_result"] = outcome
                except (OSError, ValueError, RuntimeError) as error:
                    st.error(f"Indexing failed: {error}")

        if "ingest_result" in st.session_state:
            st.success(
                f"Indexed {st.session_state['ingest_result']['chunks_indexed']:,} "
                "chunks across "
                f"{st.session_state['ingest_result']['pages_with_text']:,} pages."
            )

    st.subheader("Ask the manual")
    question = st.text_input(
        "Question",
        placeholder="What is the torque for the upper ball joint nut?",
    )
    ask = st.button("Retrieve evidence and extract", type="primary")
    if ask:
        if manual_path is None or not manual_path.is_file():
            st.error("Choose an existing PDF file.")
        elif not question.strip():
            st.error("Enter a question.")
        else:
            try:
                with st.spinner("Retrieving relevant manual passages..."):
                    evidence = retrieve_manual(
                        manual_path,
                        question,
                        top_k=top_k,
                        persist_directory=persist_directory,
                    )
                st.session_state["evidence"] = evidence
                st.session_state["question"] = question
                st.session_state.pop("extraction", None)
                if evidence:
                    with st.spinner("Validating the structured extraction..."):
                        result = extract_vehicle_specs(question, evidence)
                    st.session_state["extraction"] = result.model_dump(mode="json")
                else:
                    st.session_state["extraction"] = extract_vehicle_specs(
                        question, []
                    ).model_dump(mode="json")
                    st.info("No indexed passages were retrieved for this manual.")
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Retrieval or extraction failed: {error}")

    evidence = st.session_state.get("evidence", [])
    extraction_data = st.session_state.get("extraction")
    if evidence:
        st.subheader("Retrieved evidence")
        for rank, item in enumerate(evidence, start=1):
            source = item["source"]
            with st.expander(
                f"{rank}. Page {source['page_number']} · {source['section']} "
                f"· similarity {item['similarity']:.3f}",
                expanded=(rank == 1),
            ):
                st.caption(source["file_name"])
                st.write(item["text"])

    if extraction_data:
        st.subheader("Structured result")
        statuses = sorted(
            {item["status"] for item in extraction_data["results"]}
        )
        st.caption(
            f"Query: {extraction_data['query']} · Status: {', '.join(statuses)}"
        )
        st.json(extraction_data, expanded=True)
        output_model = ExtractionResult.model_validate(extraction_data)
        output_json = export_json(output_model)
        output_csv = export_csv(output_model)
        col_json, col_csv = st.columns(2)
        col_json.download_button(
            "Download JSON",
            data=output_json,
            file_name="vehicle-specifications.json",
            mime="application/json",
            use_container_width=True,
        )
        col_csv.download_button(
            "Download CSV",
            data=output_csv or "",
            file_name="vehicle-specifications.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
