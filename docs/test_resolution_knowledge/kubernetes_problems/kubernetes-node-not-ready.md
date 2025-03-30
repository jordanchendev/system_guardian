# Kubernetes Node Not Ready

## Metadata
- **ID**: 8490e81d-0fc2-4a71-ade5-ab832e7789fa
- **Category**: kubernetes_problems
- **Severity**: high

## Symptoms
- Pods stuck in Pending state
- Kubectl shows node(s) in NotReady state
- Application deployments incomplete
- Kubelet service failures

## Resolution Steps

1. Identify affected nodes: `kubectl get nodes`

2. Check node conditions: `kubectl describe node <node-name>`

3. Check kubelet status on node: `systemctl status kubelet`

4. Examine kubelet logs: `journalctl -u kubelet`

5. Check for resource exhaustion: `top`, `df -h`, `free -m`

6. Check container runtime: `systemctl status docker` (or containerd, etc.)

7. Restart kubelet if necessary: `systemctl restart kubelet`

8. Cordon and drain node if persistent issues: `kubectl cordon <node-name>` followed by `kubectl drain <node-name>`

9. For cloud environments, check if node has been terminated by provider

## Prevention
- Regular node maintenance and updates
- Implement resource requests and limits for all pods
- Set up monitoring for node resource usage
- Use Pod Disruption Budgets for critical applications
- Implement automated node health checks
