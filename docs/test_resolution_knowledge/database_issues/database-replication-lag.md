# Database Replication Lag

## Metadata
- **ID**: 497b7b77-b27f-41c1-9486-51bd950d9dc4
- **Category**: database_issues
- **Severity**: medium

## Symptoms
- Read replicas showing outdated data
- Monitoring alerts for replication lag exceeding thresholds
- Increasing lag metrics over time
- Data inconsistency reports from users

## Resolution Steps

1. Check current replication lag: `SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;`

2. Identify heavy write operations: `SELECT query, calls, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;`

3. Check disk I/O on primary and replicas

4. Ensure replicas have adequate resources (CPU, memory, disk I/O)

5. Reduce write load on primary if possible

6. Add more replicas to distribute read load

7. Consider enhancing network bandwidth between primary and replicas

8. For immediate relief, restart slow replicas if replication is severely lagging

## Prevention
- Monitor replication lag continuously
- Set up alerts for increasing lag trends
- Schedule heavy write operations during low-traffic periods
- Ensure replicas have equivalent or better hardware than primary
- Implement read/write splitting in application to reduce primary load
- Use synchronous replication for critical data if acceptable performance impact
