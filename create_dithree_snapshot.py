from metadata_db_client.db import DBManager

# Query to get sample S3 paths and extract bucket names
query = """
SELECT
    dataset_name,
    training_video_s3_path AS video_s3_path,
    SPLIT_PART(training_video_s3_path, '/', 1) AS bucket_name,
    SUBSTRING(training_video_s3_path, LENGTH(SPLIT_PART(training_video_s3_path, '/', 1)) + 2, 50) AS path_after_bucket
FROM delta.prd_consume_snapshots_gold.obt_video_clips
WHERE dataset_name = 'youtube_4k_cc'
    AND training_video_s3_path IS NOT NULL
    AND SPLIT_PART(training_video_s3_path, '/', 1) != 's3'
LIMIT 10
"""


db_manager = DBManager(db_type="trino")

print("\n=== COUNT SUMMARY ===")
results = db_manager.query_to_polars(query=query)

if len(results) == 0:
    print("No rows returned.")
else:
    print("\n=== SAMPLE S3 PATHS FOR YOUTUBE_4K_CC ===")
    print("Sample video S3 paths to identify bucket names:")
    print("-" * 100)
    print(f"{'Dataset':<15} {'Bucket':<25} {'Path After Bucket':<50}")
    print("-" * 100)

    for row in results.iter_rows(named=True):
        dataset = row['dataset_name'] or 'Unknown'
        bucket = row['bucket_name'] or 'Unknown'
        path = row['path_after_bucket'] or 'Unknown'

        print(f"{dataset:<15} {bucket:<25} {path:<50}")

    print("-" * 100)
    print("Note: 's3:' appears to be a placeholder or anonymized bucket name.")
    print("The actual S3 bucket names are not visible in this data.")
