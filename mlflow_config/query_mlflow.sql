-- List all experiments
SELECT experiment_id, name, artifact_location, lifecycle_stage
FROM experiments;

-- List all runs for a specific experiment
SELECT r.run_uuid, r.status, r.start_time, r.end_time,
       t.key as metric_key, t.value as metric_value
FROM runs r
JOIN metrics t ON r.run_uuid = t.run_uuid
WHERE r.experiment_id = (
    SELECT experiment_id 
    FROM experiments 
    WHERE name = 'logistic_regression_experiment'
);

-- Get all metrics for a specific run
SELECT key, value, step, timestamp
FROM metrics
WHERE run_uuid = :run_uuid
ORDER BY timestamp;

-- Get all parameters for a specific run
SELECT key, value
FROM params
WHERE run_uuid = :run_uuid;

-- Get model performance metrics across all runs
SELECT e.name as experiment_name,
       r.run_uuid,
       MAX(CASE WHEN m.key = 'accuracy' THEN m.value END) as accuracy,
       MAX(CASE WHEN m.key = 'precision' THEN m.value END) as precision,
       MAX(CASE WHEN m.key = 'recall' THEN m.value END) as recall,
       MAX(CASE WHEN m.key = 'f1' THEN m.value END) as f1,
       MAX(CASE WHEN m.key = 'roc_auc' THEN m.value END) as roc_auc
FROM experiments e
JOIN runs r ON e.experiment_id = r.experiment_id
JOIN metrics m ON r.run_uuid = m.run_uuid
GROUP BY e.name, r.run_uuid
ORDER BY e.name, r.run_uuid;

-- Get all tags for runs
SELECT r.run_uuid, t.key, t.value
FROM runs r
JOIN tags t ON r.run_uuid = t.run_uuid;

-- Get artifact locations for all runs
SELECT e.name as experiment_name,
       r.run_uuid,
       r.artifact_uri
FROM experiments e
JOIN runs r ON e.experiment_id = r.experiment_id;

-- Compare parameters across runs
SELECT e.name as experiment_name,
       r.run_uuid,
       p.key as parameter,
       p.value as value
FROM experiments e
JOIN runs r ON e.experiment_id = r.experiment_id
JOIN params p ON r.run_uuid = p.run_uuid
ORDER BY e.name, r.run_uuid, p.key; 