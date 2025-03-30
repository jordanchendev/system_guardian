# Network Connectivity Issues Between Services

## Metadata
- **ID**: de3e2a4c-bcef-49d3-afc3-34bd4081e657
- **Category**: network_problems
- **Severity**: high

## Symptoms
- Timeout errors in service-to-service communication
- Increased latency in API responses
- Connection refused errors in logs
- Intermittent service availability

## Resolution Steps

1. Verify basic connectivity: `ping <service_hostname>`

2. Check if port is open: `telnet <service_hostname> <port>` or `nc -zv <service_hostname> <port>`

3. Verify DNS resolution: `nslookup <service_hostname>`

4. Check network routes: `traceroute <service_hostname>`

5. Inspect firewall rules and security groups

6. Check network ACLs in cloud environment

7. Verify service health status in load balancer

8. Check for network saturation or packet loss: `iperf` between services

## Prevention
- Implement service mesh for better network visibility
- Use health checks and circuit breakers
- Monitor network metrics (latency, packet loss, throughput)
- Document network topology and firewall rules
- Implement retry with exponential backoff for network operations
- Use connection pooling where appropriate
