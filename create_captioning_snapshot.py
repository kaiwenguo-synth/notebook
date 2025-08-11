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
            FORMAT('%dx%d', video_clips_video_width, video_clips_video_height) AS resolution,
            training_video_s3_path AS video_s3_path,
            skeleton_detection_landmarks_s3_path AS landmarks_s3_path,
            CAST(NULL AS VARCHAR) AS audio_embedding_s3_path,
            clip_annotation_average_person_width AS average_person_width,
            clip_annotation_average_person_height AS average_person_height,
            clip_annotation_camera_framing AS camera_framing,
            clip_annotation_dynamicity_full_body AS dynamicity_full_body,
            clip_annotation_frames_with_visible_hand_frac AS frames_with_visible_hand_frac,
            overall_score AS clip_quality_score,
            technical_quality AS clip_technical_quality_score,
            clip_text_annotation_max_text_detection_rel_area AS max_text_detection_rel_area,
            short_description,
            dense_description_subject,
            dense_description_scene,
            dense_description_action
        FROM delta.prd_consume_snapshots_gold.obt_video_clips
        WHERE
            clip_type = 'foundation_human'
    ),
    youtube_filtered_clips AS (
        SELECT *
        FROM youtube_clips
        WHERE (
            c_duration_s >= 3.0
            AND c_duration_s <= 7200.0
            AND average_person_width >= 0.2
            AND average_person_width <= 0.6
            AND average_person_height >= 0.4
            AND dynamicity_full_body >= 0.0
            AND dynamicity_full_body <= 0.6
            AND frames_with_visible_hand_frac >= 1.0
            AND clip_quality_score >= 0.7
            AND clip_technical_quality_score >= 0.9
            AND max_text_detection_rel_area <= 0.0
            AND c_frame_rate IN ('30','30000/1001','29917/1000','29833/1000','60','2997/50','25','24000/1001')
            AND resolution IN ('1920x1080','2560x1440','3840x2160','4320x7680')
            AND camera_framing IN ('chest_up','waist_up','full_body')
            AND (
                short_description IS NOT NULL AND
                dense_description_subject IS NOT NULL AND
                dense_description_scene IS NOT NULL AND
                dense_description_action IS NOT NULL
            )
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
    ),
    ava_6 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_984a9629_1ff7_47d7_9361_ce7d55209cf1
        WHERE split = 'test'
        LIMIT 1
    ),
    ava_2 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_62f29a4a_864b_4e0e_8bba_e62b8803be13
        WHERE split = 'test'
        LIMIT 1
    ),
    anna_m_5 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_fdbe4939_3f07_43b9_9b49_a313d66db983
        WHERE split = 'test'
        LIMIT 1
    ),
    anna_m_1 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_05c2a33f_c468_4a84_be06_57a937d968d3
        WHERE split = 'test'
        LIMIT 1
    ),
    anna_m_3 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_7d71f945_d313_44cf_aa25_7ca9f41e80f6
        WHERE split = 'test'
        LIMIT 1
    ),
    anna_m_4 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_7b1eef6d_510a_4ff0_88f7_2419c0d8cf2a
        WHERE split = 'test'
        LIMIT 1
    ),
    ava_1_v2 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_8b5d9df7_ad4b_4b16_8a7b_6f3541362805
        WHERE split = 'test'
        LIMIT 1
    ),
    dan_r_2 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_45b8596b_a78b_429a_90a5_26f1d751d311
        WHERE split = 'test'
        LIMIT 1
    ),
    tat_ash_9 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_839d73e1_ff2c_4356_8b53_a72948d0a9a4
        WHERE split = 'test'
        LIMIT 1
    ),
    tat_ash_2_v3 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_3de594d4_c1d8_4565_88ee_f8657831e0d5
        WHERE split = 'test'
        LIMIT 1
    ),
    tat_ash_6 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_6e8fcae9_5e02_4376_835e_370dcc356867
        WHERE split = 'test'
        LIMIT 1
    ),
    tat_ash_5 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_cf2e5bf4_efcc_4adc_a50f_81a888eb9182
        WHERE split = 'test'
        LIMIT 1
    ),
    tat_ash_4 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_2a3bc014_32c0_4de0_9ad3_df1cd82c33cf
        WHERE split = 'test'
        LIMIT 1
    ),
    tat_ash_7 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_9f552356_850e_46d5_ba4a_3b17b52c9d1b
        WHERE split = 'test'
        LIMIT 1
    ),
    tat_ash_8 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_b4b8bf58_9373_4492_9c17_4e50f179302c
        WHERE split = 'test'
        LIMIT 1
    ),
    aaron_4 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_eadaec5f_0f58_48bb_a109_69e9733be310
        WHERE split = 'test'
        LIMIT 1
    ),
    aaron_2 AS (
        SELECT *
        FROM delta.landing_prd.snapshot_4deda56c_f62a_40b6_9ecf_eb91234b45fe
        WHERE split = 'test'
        LIMIT 1
    ),
    spa_test_suite_intermediary AS (
        SELECT * FROM ava_6
        UNION ALL
        SELECT * FROM ava_2
        UNION ALL
        SELECT * FROM anna_m_5
        UNION ALL
        SELECT * FROM anna_m_1
        UNION ALL
        SELECT * FROM anna_m_3
        UNION ALL
        SELECT * FROM anna_m_4
        UNION ALL
        SELECT * FROM ava_1_v2
        UNION ALL
        SELECT * FROM dan_r_2
        UNION ALL
        SELECT * FROM tat_ash_9
        UNION ALL
        SELECT * FROM tat_ash_2_v3
        UNION ALL
        SELECT * FROM tat_ash_6
        UNION ALL
        SELECT * FROM tat_ash_5
        UNION ALL
        SELECT * FROM tat_ash_4
        UNION ALL
        SELECT * FROM tat_ash_7
        UNION ALL
        SELECT * FROM tat_ash_8
        UNION ALL
        SELECT * FROM aaron_4
        UNION ALL
        SELECT * FROM aaron_2
    ),
    spa_test_suite AS (
        SELECT
            -- Clip metadata
            synthesia_id,
            clip_id,
            clip_type,
            CAST(NULL AS VARCHAR) AS dataset_name,
            CAST(NULL AS VARCHAR) AS domain,
            c_start_frame,
            c_end_frame,
            c_duration_s,
            c_frame_rate,
            c_video_width,
            c_video_height,
            CAST(NULL AS VARCHAR) AS camera,
            -- S3 paths
            video_s3_path,
            audio_embedding_s3_path,
            landmarks_s3_path,
            CAST(NULL AS VARCHAR) AS landmarks_3d_s3_path,
            CAST(NULL AS VARCHAR) AS calibration_s3_path,
            CAST(NULL AS VARCHAR) AS hand_tracking_s3_path,
            CAST(NULL AS VARCHAR) AS eyelandmarks_s3_path,
            -- WEKA paths
            video_weka_path,
            landmarks_weka_path,
            audio_embedding_weka_path,
            landmarks_3d_weka_path,
            calibration_weka_path,
            hand_tracking_weka_path,
            eyelandmarks_weka_path,
            -- Dataset split
            'test' AS split
        FROM spa_test_suite_intermediary
    ),
    counts AS (
        SELECT
            dataset_name,
            COUNT(DISTINCT video_s3_path) AS total_videos,
            COUNT(*) AS total_clips,
            SUM(c_duration_s) / 3600.0 AS total_hours
        FROM youtube
        GROUP BY dataset_name
    )
    SELECT * FROM counts
    ORDER BY total_hours DESC
"""

dense_query = """
SELECT
    dataset_name,
    COUNT(DISTINCT CAST(synthesia_id AS VARCHAR)) AS total_videos,
    COUNT(*) AS total_clips,
    SUM(video_clips_duration_s) / 3600.0 AS total_hours
FROM delta.prd_consume_snapshots_gold.obt_clips
WHERE clip_dense_captions_exist = 1
GROUP BY dataset_name
ORDER BY total_hours DESC
"""


db_manager = DBManager(db_type="trino")

print("\n=== COUNT SUMMARY ===")
results = db_manager.query_to_polars(query=query)

if len(results) == 0:
    print("No rows returned.")
else:
    # If grouped by dataset_name, print per-dataset rows
    if "dataset_name" in results.columns and all(
        col in results.columns for col in ["total_videos", "total_clips", "total_hours"]
    ):
        print("\nBy dataset:")
        for row in results.iter_rows(named=True):
            print(
                f"  {row['dataset_name']}: "
                f"videos={int(row['total_videos']):,}, "
                f"clips={int(row['total_clips']):,}, "
                f"hours={float(row['total_hours']):.2f}"
            )
