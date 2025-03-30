# High CPU Usage on Application Servers

## Metadata
- **ID**: a7f3aca6-0981-4991-b396-b89959b730fd
- **Category**: performance_issues
- **Severity**: medium

## Symptoms
- Server responding slowly to requests
- Monitoring alerts for high CPU usage
- Increased response times across multiple services
- Process using abnormally high CPU resources

## Resolution Steps

1. Identify high CPU processes: `top` or `htop`

2. Check system load: `uptime`

3. Get process details: `ps aux | grep <process-id>`

4. For Java applications, take thread dumps: `jstack <pid>`

5. For other applications, use language-specific profiling tools

6. Check for CPU throttling in cloud environments

7. Identify specific endpoints causing high load with APM tools

8. For immediate relief, scale horizontally by adding more servers

9. If single-threaded process, consider increasing instance size (vertical scaling)

## Prevention
- Regular performance testing under load
- Implement auto-scaling based on CPU metrics
- Code profiling during development and testing
- Set up monitoring and alerting for CPU usage trends
- Optimize critical code paths and database queries
