# DNS Resolution Failures

## Metadata
- **ID**: 386f2d8a-2656-4836-8f54-512146aef351
- **Category**: network_problems
- **Severity**: critical

## Symptoms
- Services unable to resolve hostnames
- Error logs showing 'unknown host'
- Intermittent connectivity issues
- Some services working while others fail

## Resolution Steps

1. Verify DNS configuration: `cat /etc/resolv.conf`

2. Test DNS resolution: `dig <hostname>` or `nslookup <hostname>`

3. Check DNS server health: `dig @<dns_server> <hostname>`

4. Verify DNS service is running: `systemctl status named` or equivalent

5. Check for DNS cache poisoning: clear DNS cache

6. If using cloud DNS, check cloud provider status

7. Temporarily add hostname entries to /etc/hosts as emergency fix

8. Restart DNS services if necessary

## Prevention
- Use multiple DNS servers for redundancy
- Implement local DNS caching
- Monitor DNS query performance and success rates
- Maintain backup of DNS records
- Consider using service discovery solutions
