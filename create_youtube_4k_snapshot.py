from __future__ import annotations

from metadata_db_client.db import DBManager


query = """
WITH
    youtube_clips AS (
        SELECT
            CAST(synthesia_id AS VARCHAR) AS synthesia_id,
            CAST(clip_id AS VARCHAR) AS clip_id,
            clip_type,
            dataset_name,
            domain,
            video_clips_start_frame AS c_start_frame,
            video_clips_end_frame AS c_end_frame,
            video_clips_duration_s AS c_duration_s,
            video_clips_frame_rate AS c_frame_rate,
            video_clips_video_width AS c_video_width,
            video_clips_video_height AS c_video_height,
            training_video_s3_path AS video_s3_path,
            skeleton_detection_landmarks_s3_path AS landmarks_s3_path,
            video_compact_audio_audio_s3_path AS audio_s3_path,
            audio_fp_16_embedding_s3_path AS audio_embedding_s3_path,
            clip_annotation_camera_framing AS camera_framing,
            clip_annotation_dynamicity_hands AS dynamicity_hands,
            clip_annotation_dynamicity_head AS dynamicity_head,
            clip_annotation_frames_with_visible_hand_frac AS frames_with_visible_hand_frac,
            technical_quality AS clip_technical_quality_score,
            aesthetic_quality AS clip_aesthetic_quality_score,
            clip_text_annotation_frac_frames_with_text
        FROM delta.prd_consume_snapshots_gold.obt_video_clips
    ),
    youtube_filtered_clips AS (
        SELECT *
        FROM youtube_clips
        WHERE (
                clip_type = 'foundation_human'
                AND dynamicity_hands >= 0.33
                AND dynamicity_head >= 0.33
                AND frames_with_visible_hand_frac >= 0.1
                AND camera_framing NOT IN ('face_only', 'other')
                AND clip_aesthetic_quality_score >= 0.95
                AND clip_text_annotation_frac_frames_with_text <= 0.0
                AND (c_video_height = 2160 AND c_video_width = 3840) OR dataset_name = 'youtube_4k_cc'
        )
    ),
    youtube_4k AS (
        SELECT
            -- Clip metadata
            synthesia_id,
            clip_id,
            clip_type,
            dataset_name,
            domain,
            c_start_frame,
            c_end_frame,
            c_duration_s,
            c_frame_rate,
            c_video_width,
            c_video_height,
            camera_framing,
            -- S3 paths
            video_s3_path,
            audio_s3_path,
            audio_embedding_s3_path,
            landmarks_s3_path
        FROM youtube_filtered_clips
    )
    SELECT * from youtube_4k
"""


db_manager = DBManager(db_type="trino")

print("\n=== SELECTED CLIPS ===")
results = db_manager.query_to_polars(query=query)

if len(results) == 0:
    print("No rows returned.")
else:
    total = len(results)
    print(f"Total clips selected: {total:,}")
    if "c_duration_s" in results.columns:
        total_hours = float(results["c_duration_s"].sum()) / 3600.0
        print(f"Total hours selected: {total_hours:.2f}")
