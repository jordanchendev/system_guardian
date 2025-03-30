# Database Index Corruption

## Metadata
- **ID**: 1070e157-dacf-44e9-9605-24f227a5a2b3
- **Category**: database_issues
- **Severity**: high

## Symptoms
- Slow queries that were previously fast
- Error messages about corrupt indexes
- Unexpected query results
- Database logs showing index-related errors

## Resolution Steps

1. Identify the corrupt index: Check database logs for error messages

2. Verify index corruption: `SELECT pg_stat_user_indexes.schemaname, pg_stat_user_indexes.relname, pg_stat_user_indexes.indexrelname FROM pg_stat_user_indexes, pg_index WHERE pg_index.indisvalid = false AND pg_stat_user_indexes.indexrelid = pg_index.indexrelid;`

3. Drop and recreate the corrupt index: `DROP INDEX <index_name>; CREATE INDEX <index_name> ON <table> (<columns>);`

4. If multiple indexes are corrupt, consider `REINDEX DATABASE <database_name>;`

5. Check system for underlying hardware issues (disk errors, memory problems)

6. Verify database consistency: `ANALYZE VERBOSE;`

## Prevention
- Regular database maintenance (VACUUM, ANALYZE)
- Monitor index usage and performance
- Schedule regular index rebuilds for critical tables
- Ensure proper database shutdown procedures are followed
- Implement database health checks in monitoring system
