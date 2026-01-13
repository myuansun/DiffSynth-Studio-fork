#!/usr/bin/env python3
"""
Convert high_quality_partition.csv + captions.json -> FilteredYoutubeDataset compatible format.

This script creates:
1. A segment list JSON file compatible with FilteredYoutubeDataset
2. info.json files in each segment folder with captions (optional)

Usage:
    python convert_partition_to_segment_list.py \
        --partition metadata/high_quality_partition.csv \
        --captions metadata/captions.json \
        --output metadata/filtered_segments.json
        
    # Then use with FilteredYoutubeDataset:
    from diffsynth.core.data.youtube_dataset import FilteredYoutubeDataset
    dataset = FilteredYoutubeDataset(
        base_folder="/home/azureuser/data",
        segment_list_file="metadata/filtered_segments.json"
    )
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional, Set


def load_partition_csv(csv_path: str) -> list[dict]:
    """Load high_quality_partition.csv and return list of segment dicts."""
    segments = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                'segment_key': row['segment_key'],
                'num_frames': int(row['num_frames']),
                'mean_quality': float(row['mean_quality']),
                'quality_assessment': row['quality_assessment'],
                'batch': row['batch'],
                'video_id': row['video_id'],
                'segment_name': row['segment_name'],
            })
    return segments


def load_captions_json(json_path: str) -> Dict[str, dict]:
    """Load captions.json and return dict mapping segment_key -> caption data."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle nested format with metadata
    if 'captions' in data:
        return data['captions']
    return data


def create_segment_list(
    segments: list[dict],
    captions: Dict[str, dict],
    min_frames: int = 81,
    min_quality: Optional[float] = None,
    quality_assessment: Optional[str] = None,
) -> dict:
    """
    Create a segment list compatible with FilteredYoutubeDataset.
    
    Returns dict with:
    - metadata: generation info
    - segment_keys: list of segment keys
    - segment_captions: dict mapping segment_key -> caption text
    """
    filtered_keys = []
    segment_captions = {}
    
    for seg in segments:
        key = seg['segment_key']
        
        # Apply filters
        if seg['num_frames'] < min_frames:
            continue
        if min_quality is not None and seg['mean_quality'] < min_quality:
            continue
        if quality_assessment is not None and seg['quality_assessment'] != quality_assessment:
            continue
        
        filtered_keys.append(key)
        
        # Get caption if available
        if key in captions:
            caption_data = captions[key]
            if isinstance(caption_data, dict):
                segment_captions[key] = caption_data.get('caption', '')
            else:
                segment_captions[key] = str(caption_data)
    
    return {
        'metadata': {
            'total_segments': len(filtered_keys),
            'min_frames': min_frames,
            'min_quality': min_quality,
            'quality_assessment': quality_assessment,
        },
        'segment_keys': filtered_keys,
        'segment_captions': segment_captions,
    }


def write_info_json_files(
    segment_list: dict,
    data_root: str,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Write info.json files to each segment folder with the caption.
    
    Returns number of files written.
    """
    written = 0
    segment_captions = segment_list.get('segment_captions', {})
    
    for key in segment_list['segment_keys']:
        if key not in segment_captions:
            continue
        
        # segment_key format: "processed_videos11-15/video_id/segment1"
        segment_path = Path(data_root) / key
        info_path = segment_path / "info.json"
        
        if info_path.exists() and not overwrite:
            continue
        
        caption = segment_captions[key]
        info_data = {
            "prompt": caption,
            "caption": caption,
            "source": "captions.json",
        }
        
        if dry_run:
            print(f"[dry-run] Would write: {info_path}")
        else:
            try:
                segment_path.mkdir(parents=True, exist_ok=True)
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(info_data, f, ensure_ascii=False, indent=2)
                written += 1
            except Exception as e:
                print(f"[error] Failed to write {info_path}: {e}")
    
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Convert partition CSV + captions JSON to FilteredYoutubeDataset format"
    )
    parser.add_argument("--partition", type=str, required=True,
                        help="Path to high_quality_partition.csv")
    parser.add_argument("--captions", type=str, required=True,
                        help="Path to captions.json")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for filtered_segments.json")
    parser.add_argument("--min-frames", type=int, default=81,
                        help="Minimum frames per segment (default: 81)")
    parser.add_argument("--min-quality", type=float, default=None,
                        help="Minimum mean_quality threshold")
    parser.add_argument("--quality-assessment", type=str, default=None,
                        choices=["GOOD", "MODERATE", "POOR"],
                        help="Required quality_assessment value")
    parser.add_argument("--write-info-json", action="store_true",
                        help="Write info.json files to segment folders")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Data root path (required if --write-info-json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing info.json files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")
    args = parser.parse_args()
    
    # Load inputs
    print(f"Loading partition from: {args.partition}")
    segments = load_partition_csv(args.partition)
    print(f"  Found {len(segments)} segments")
    
    print(f"Loading captions from: {args.captions}")
    captions = load_captions_json(args.captions)
    print(f"  Found {len(captions)} captions")
    
    # Create segment list
    segment_list = create_segment_list(
        segments,
        captions,
        min_frames=args.min_frames,
        min_quality=args.min_quality,
        quality_assessment=args.quality_assessment,
    )
    print(f"After filtering: {len(segment_list['segment_keys'])} segments")
    print(f"  With captions: {len(segment_list['segment_captions'])}")
    
    # Write output
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(segment_list, f, ensure_ascii=False, indent=2)
        print(f"Written to: {args.output}")
    else:
        print(f"[dry-run] Would write to: {args.output}")
    
    # Optionally write info.json files
    if args.write_info_json:
        if not args.data_root:
            print("Error: --data-root required when using --write-info-json")
            return 1
        
        written = write_info_json_files(
            segment_list,
            args.data_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(f"Written {written} info.json files")
    
    return 0


if __name__ == "__main__":
    exit(main())
