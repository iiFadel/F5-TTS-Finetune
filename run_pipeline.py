#!/usr/bin/env python3
import os
import csv
import random
import subprocess
import argparse
from pathlib import Path
import time
import shutil

def load_metadata(csv_path):
    """Return a dict mapping audio_file (no-ext) → text."""
    meta = {}
    if not Path(csv_path).exists():
        return meta
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        # Skip the header row if it exists
        header = next(reader, None)
        if header and header[0] == 'audio_file' and header[1] == 'text':
            pass  # Header was already skipped
        else:
            # It wasn't a header row, add it back to our metadata
            if header and len(header) >= 2:
                meta[header[0]] = header[1]
        
        # Process the rest of the rows
        for row in reader:
            if len(row) >= 2:
                meta[row[0]] = row[1]
    return meta

def load_sentences(txt_path):
    """Return a list of lines (stripped)."""
    with open(txt_path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def pick_random_wav(wav_dir, speaker):
    """List all files that start with speaker and choose one at random."""
    candidates = list(Path(wav_dir).glob(f"{speaker}_segment_*.wav"))
    if not candidates:
        raise RuntimeError(f"No WAVs found for speaker {speaker} in {wav_dir}")
    return str(random.choice(candidates))

def append_to_metadata(metadata_file, sample_name, text):
    """Append a new entry to the metadata file."""
    # If file doesn't exist, create it with a header
    if not Path(metadata_file).exists():
        with open(metadata_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(['audio_file', 'text'])
    
    # Append the new entry
    with open(metadata_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow([sample_name, text])

def get_wav_duration(wav_path):
    """
    Return duration (float, seconds) for a .wav using ffprobe.
    Raises if ffprobe is not installed or output is unexpected.
    """
    cmd = [
        "ffprobe",
        "-i", str(wav_path),
        "-show_entries", "format=duration",
        "-v", "quiet",
        "-of", "csv=p=0"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def prune_long_samples(output_root, metadata_path, max_seconds=20.0):
    """
    Delete any .wav under output_root whose duration exceeds max_seconds,
    and remove its row from metadata_path (audio_file|text) if present.
    """
    output_root = Path(output_root)
    metadata_path = Path(metadata_path)
    removed_names = []

    for wav_path in output_root.rglob("*.wav"):
        try:
            dur = get_wav_duration(wav_path)
            if dur > max_seconds:
                print(f"🗑️  Removing {wav_path} ({dur:.2f}s > {max_seconds}s)")
                wav_path.unlink()
                removed_names.append(wav_path.name)
        except Exception as e:
            print(f"⚠️  Could not inspect {wav_path}: {e}")

    # Rewrite metadata.csv without the deleted files
    if removed_names and metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            rows = [row for row in csv.reader(f, delimiter="|") if row and row[0] not in removed_names]

        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(rows)

        print(f"✓ Metadata updated – removed {len(removed_names)} rows")

def main():
    p = argparse.ArgumentParser(
        description="Automate f5-tts_infer-cli over multiple speakers"
    )
    p.add_argument("--num_samples", type=int, required=True,
                   help="How many inferences to run per speaker")
    p.add_argument("--speakers", nargs="+", required=True,
                   help="One or more speaker IDs, e.g. SPEAKER_0801 SPEAKER_0802")
    p.add_argument("--model", type=str, default="F5TTS_v1_Base",
                   help="Model name or path for f5-tts_infer-cli")
    p.add_argument("--ckpt_file", type=str, required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--vocab_file", type=str, required=True,
                   help="Path to vocab file")
    p.add_argument("--wav_dir", type=str, required=True,
                   help="Directory containing all reference WAVs")
    p.add_argument("--metadata_csv", type=str, required=True,
                   help="Path to metadata.csv for reference audio (audio_file|text)")
    p.add_argument("--gen_txt", type=str, required=True,
                   help="One-sentence-per-line TXT file of gen_texts")
    p.add_argument("--out_root", type=str, required=True,
                   help="Root directory for outputs (creates per-speaker subdirs)")
    p.add_argument("--output_metadata", type=str, default="metadata.csv",
                   help="Path to create/update output metadata file")
    p.add_argument("--max_duration", type=float, default=20.0,
                   help="Maximum allowed duration for output audio files (seconds)")
    args = p.parse_args()

    # Load shared resources once
    metadata = load_metadata(args.metadata_csv)
    all_gen_sents = load_sentences(args.gen_txt)

    # Initialize output metadata file if it doesn't exist
    output_metadata_file = Path(args.output_metadata)
    if not output_metadata_file.exists():
        with open(output_metadata_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(['audio_file', 'text'])

    for spkr in args.speakers:
        print(f"\n===== Processing {spkr} =====")
        # Fresh sentence pool per speaker
        available_gens = all_gen_sents.copy()
        # Create output directory for this speaker
        out_dir = Path(args.out_root) / spkr
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, args.num_samples + 1):
            # Format the output filename (same format as FishSpeech)
            output_filename = f"{spkr}_sample_{i:02d}.wav"
            output_path = out_dir / output_filename
            
            # Skip if this sample already exists
            if output_path.exists():
                print(f"⚠️ Output file already exists: {output_path}. Skipping...")
                continue
                
            # Pick a valid ref-audio that has metadata
            try_count = 0
            max_tries = 10
            while try_count < max_tries:
                try_count += 1
                wav_path = pick_random_wav(args.wav_dir, spkr)
                base = Path(wav_path).stem  # e.g. SPEAKER_0801_segment_130
                if base in metadata:
                    ref_text = metadata[base]
                    break
                print(f"⚠ {base} not in metadata.csv, retrying... ({try_count}/{max_tries})")
            
            if try_count >= max_tries:
                print(f"❌ Failed to find valid reference audio for {spkr} after {max_tries} attempts")
                continue

            # Pick a gen_text without replacement
            if not available_gens:
                raise RuntimeError(
                    f"Ran out of unique sentences for {spkr} after {i-1} samples"
                )
            gen_text = random.choice(available_gens)
            available_gens.remove(gen_text)
            
            print(f"[{spkr}][{i}/{args.num_samples}] Starting sample generation")
            print(f"Selected reference audio: {wav_path}")
            print(f"Reference text: {ref_text}")
            print(f"Text to generate: {gen_text}")

            try:
                # Assemble and run the command
                cmd = [
                    "f5-tts_infer-cli",
                    "--model", args.model,
                    "--ckpt_file", args.ckpt_file,
                    "--vocab_file", args.vocab_file,
                    "--ref_audio", wav_path,
                    "--ref_text", ref_text,
                    "--gen_text", gen_text,
                    "--output_dir", str(out_dir),
                    "--output_file", output_filename,
                ]
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                
                # Verify the output file exists
                if not output_path.exists():
                    raise FileNotFoundError(f"Expected output file {output_path} not found")
                
                # Update metadata
                append_to_metadata(args.output_metadata, output_filename, gen_text)
                print(f"✓ Updated metadata for {output_filename}")
                
            except Exception as e:
                print(f"❌ Error processing sample {i} for {spkr}: {str(e)}")
                # Continue with next sample
                continue

    # Prune any overly long samples
    prune_long_samples(args.out_root, args.output_metadata, max_seconds=args.max_duration)
    print("\nAll speakers processed successfully.")

if __name__ == "__main__":
    main()