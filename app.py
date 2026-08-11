import argparse
from pathlib import Path

import whisper
from transformers import pipeline
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transcribe and summarize an audio file."
    )

    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file"
    )

    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model to use (default: base)"
    )

    parser.add_argument(
        "--summary-model",
        default="facebook/bart-large-cnn",
        help="Hugging Face summarization model"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum summary length"
    )

    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum summary length"
    )

    parser.add_argument(
        "--output-format",
        choices=["md", "txt"],
        default="md",
        help="Output format: md or txt (default: md)"
    )

    return parser.parse_args()


def transcribe_audio(audio_file, model_name):
    print(f"\nLoading Whisper model: {model_name}")

    model = whisper.load_model(model_name)

    print(f"Transcribing: {audio_file}\n")

    # verbose=True lets Whisper display transcription progress
    result = model.transcribe(
        str(audio_file),
        verbose=True
    )

    return result["text"].strip()


def split_text(text, max_words=700):
    words = text.split()

    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def summarize_text(
    text,
    model_name,
    max_length,
    min_length
):
    print(f"\nLoading summarization model: {model_name}")

    summarizer = pipeline(
        "summarization",
        model=model_name
    )

    chunks = split_text(text)

    print(f"\nTranscript divided into {len(chunks)} chunks.")

    chunk_summaries = []

    # Progress bar
    for chunk in tqdm(
        chunks,
        desc="Summarizing",
        unit="chunk"
    ):
        result = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )

        chunk_summaries.append(
            result[0]["summary_text"]
        )

    combined_summary = " ".join(chunk_summaries)

    # Second pass if transcript required multiple chunks
    if len(chunk_summaries) > 1:
        final_chunks = split_text(
            combined_summary,
            max_words=700
        )

        final_summaries = []

        for chunk in tqdm(
            final_chunks,
            desc="Creating final summary",
            unit="chunk"
        ):
            result = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )

            final_summaries.append(
                result[0]["summary_text"]
            )

        combined_summary = " ".join(final_summaries)

    return combined_summary


def save_output(
    transcription,
    summary,
    output_file,
    output_format
):
    if output_format == "md":
        content = f"""# Transcription and Summary

## Summary

{summary}

---

## Full Transcription

{transcription}
"""

    else:
        content = f"""TRANSCRIPTION AND SUMMARY

SUMMARY
=======

{summary}


FULL TRANSCRIPTION
==================

{transcription}
"""

    output_file.write_text(
        content,
        encoding="utf-8"
    )


def main():
    args = parse_arguments()

    if not args.audio_file.exists():
        raise FileNotFoundError(
            f"Audio file not found: {args.audio_file}"
        )

    if args.min_length >= args.max_length:
        raise ValueError(
            "--min-length must be smaller than --max-length"
        )

    try:
        # ------------------------
        # Step 1: Transcription
        # ------------------------

        transcription = transcribe_audio(
            args.audio_file,
            args.whisper_model
        )

        if not transcription:
            raise RuntimeError(
                "Whisper returned an empty transcription."
            )

        print("\n✓ Transcription complete")

        # ------------------------
        # Step 2: Summarization
        # ------------------------

        summary = summarize_text(
            transcription,
            args.summary_model,
            args.max_length,
            args.min_length
        )

        print("\n✓ Summary complete")

        # ------------------------
        # Step 3: Save output
        # ------------------------

        output_file = args.audio_file.with_suffix(
            f".{args.output_format}"
        )

        save_output(
            transcription,
            summary,
            output_file,
            args.output_format
        )

        print("\n✓ Finished")
        print(f"Output saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\nProcess cancelled by user.")

    except Exception as exc:
        print(f"\nError: {exc}")
        raise


if __name__ == "__main__":
    main()
