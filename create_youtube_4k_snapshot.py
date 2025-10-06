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
            video_metadata_approx_avg_frame_rate AS c_frame_rate,
            video_clips_video_width AS c_video_width,
            video_clips_video_height AS c_video_height,
            FORMAT('%dx%d', video_clips_video_width, video_clips_video_height) AS resolution,
            training_video_s3_path AS video_s3_path,
            skeleton_detection_landmarks_s3_path AS landmarks_s3_path,
            video_compact_audio_audio_s3_path AS audio_s3_path,
            audio_fp_16_embedding_s3_path AS audio_embedding_s3_path,
            clip_annotation_average_person_width AS average_person_width,
            clip_annotation_average_person_height AS average_person_height,
            clip_annotation_camera_framing AS camera_framing,
            clip_annotation_dynamicity_hands AS dynamicity_hands,
            clip_annotation_dynamicity_head AS dynamicity_head,
            clip_annotation_frames_with_visible_hand_frac AS frames_with_visible_hand_frac,
            overall_score AS clip_quality_score,
            technical_quality AS clip_technical_quality_score,
            aesthetic_quality AS clip_aesthetic_quality_score,
            clip_text_annotation_avg_text_detection_rel_area AS avg_text_detection_rel_area,
            short_description,
            dense_description_subject,
            dense_description_scene,
            dense_description_action
        FROM delta.prd_consume_snapshots_gold.obt_video_clips
    ),
    youtube_filtered_clips AS (
        SELECT *
        FROM youtube_clips
        WHERE (
                clip_type = 'foundation_human'
                AND c_duration_s >= 3.0
                AND c_duration_s <= 7200.0
                AND average_person_width >= 0.0
                AND average_person_height >= 0.5
                AND (dynamicity_hands >= 0.33 OR dynamicity_head >= 0.33)
                AND frames_with_visible_hand_frac >= 0.1
                AND c_frame_rate >= 29 AND c_frame_rate <= 61
                AND (resolution IN ('3840x2160') OR dataset_name = 'youtube_4k_cc')
                AND camera_framing IN ('chest_up','waist_up','full_body')
                AND clip_aesthetic_quality_score >= 0.8
                AND avg_text_detection_rel_area <= 0.05
                AND short_description IS NOT NULL
                AND dense_description_subject IS NOT NULL
                AND dense_description_scene IS NOT NULL
                AND dense_description_action IS NOT NULL
        )
    ),
    youtube AS (
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
            CAST(NULL AS VARCHAR) AS camera,
            -- S3 paths
            video_s3_path,
            audio_s3_path,
            audio_embedding_s3_path,
            landmarks_s3_path,
            CAST(NULL AS VARCHAR) AS landmarks_3d_s3_path,
            CAST(NULL AS VARCHAR) AS calibration_s3_path,
            CAST(NULL AS VARCHAR) AS hand_tracking_s3_path,
            CAST(NULL AS VARCHAR) AS eyelandmarks_s3_path,
            -- WEKA paths
            CAST(NULL AS VARCHAR) AS video_weka_path,
            CAST(NULL AS VARCHAR) AS landmarks_weka_path,
            CAST(NULL AS VARCHAR) AS audio_embedding_weka_path,
            CAST(NULL AS VARCHAR) AS landmarks_3d_weka_path,
            CAST(NULL AS VARCHAR) AS calibration_weka_path,
            CAST(NULL AS VARCHAR) AS hand_tracking_weka_path,
            CAST(NULL AS VARCHAR) AS eyelandmarks_weka_path,
            -- Dataset split
            (CASE WHEN (FALSE) THEN 'test' ELSE 'train' END) AS split,
            -- Dense captioning
            short_description,
            dense_description_subject,
            dense_description_scene,
            dense_description_action
        FROM youtube_filtered_clips
        -- Sort to ensure that the test clips are included when the limit is applied
        ORDER BY
            CASE split
                WHEN 'test' THEN 1
                WHEN 'train' THEN 2
            END
    )
    SELECT * FROM youtube
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
