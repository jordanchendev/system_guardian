# Database Connection Pool Exhaustion

## Metadata
- **ID**: 50329f81-6c7d-449d-b637-f1e5a95b57b6
- **Category**: database_issues
- **Severity**: high

## Symptoms
- Application timeout errors when connecting to database
- Error logs showing 'connection limit exceeded'
- Increasing latency in database operations
- Application crashes with database connection errors

## Resolution Steps

1. Check current connection count: `SELECT count(*) FROM pg_stat_activity;`

2. Identify connection sources: `SELECT client_addr, count(*) FROM pg_stat_activity GROUP BY client_addr ORDER BY count(*) DESC;`

3. Check for idle connections: `SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';`

4. Terminate long-running idle connections: `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND current_timestamp - state_change > interval '10 minutes';`

5. Increase max_connections parameter in postgresql.conf

6. Implement connection pooling using PgBouncer or similar tool

7. Review application code for connection leaks

8. Add proper connection release in application exception handlers

## Prevention
- Implement connection pooling at application level
- Set appropriate connection timeouts and idle connection limits
- Monitor connection utilization and set up alerts
- Use connection leak detection tools in development and testing
- Implement circuit breaker pattern for database access
