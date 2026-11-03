-- How long does each track last? Shortest first.
SELECT track_id, COUNT(*) AS frames, ROUND(MAX(t) - MIN(t), 1) AS seconds
FROM read_parquet('data/rides/2026-11-03_2/video_tracks.parquet')
GROUP BY track_id
ORDER BY seconds;

-- How many tracks are shorter than 0.5 s?
SELECT COUNT(*) FROM (
    SELECT track_id FROM read_parquet('data/rides/2026-11-03_2/video_tracks.parquet')
    GROUP BY track_id HAVING MAX(t) - MIN(t) < 0.5
);

-- Which tracks changed class?
SELECT track_id, STRING_AGG(DISTINCT class, ', ') AS classes
FROM read_parquet('data/rides/2026-11-03_2/video_tracks.parquet')
GROUP BY track_id
HAVING COUNT(DISTINCT class) > 1;