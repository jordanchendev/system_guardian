# Pod Stuck in CrashLoopBackOff

## Metadata
- **ID**: a79bc952-c7c7-4acf-899a-953b866721c0
- **Category**: kubernetes_problems
- **Severity**: medium

## Symptoms
- Pod continuously restarting
- Pod status showing CrashLoopBackOff
- Service unavailability despite pod presence
- Application error logs before crash

## Resolution Steps

1. Identify affected pods: `kubectl get pods -A | grep CrashLoop`

2. Check pod details: `kubectl describe pod <pod-name> -n <namespace>`

3. Examine container logs: `kubectl logs <pod-name> -n <namespace>`

4. Check previous container logs if current container won't start: `kubectl logs <pod-name> -n <namespace> --previous`

5. Verify resource limits are appropriate

6. Check if liveness probe is properly configured

7. Verify container image and tag are correct

8. Check for configuration issues in ConfigMaps or Secrets

9. Temporarily increase initialDelaySeconds for liveness probe as emergency fix

## Prevention
- Thorough testing of containers before deployment
- Implement health checks in application code
- Set appropriate resource requests and limits
- Use startup probes for slow-starting applications
- Implement proper logging for faster debugging
