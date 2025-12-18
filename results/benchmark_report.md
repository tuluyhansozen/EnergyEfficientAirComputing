# AirCompSim Benchmark Report

**Generated:** 2025-12-18 22:28:03

## Summary

Ran 10 different configurations to compare performance metrics.

## Results Table

| Configuration | Users | UAVs | Edges | Tasks | Success Rate | Avg Latency (s) | Avg QoS | Energy (J) |
|--------------|-------|------|-------|-------|--------------|-----------------|---------|------------|
| Baseline | 10 | 3 | 4 | 288 | 94.8% | 0.3976 | 63.5 | 2290.00 |
| Low Users (5) | 5 | 3 | 4 | 116 | 77.6% | 0.5466 | 63.8 | 1268.00 |
| Medium Users (20) | 20 | 3 | 4 | 521 | 93.1% | 0.2793 | 63.0 | 2910.00 |
| High Users (30) | 30 | 3 | 4 | 690 | 90.0% | 0.2936 | 64.4 | 4076.00 |
| No UAVs | 10 | 0 | 4 | 230 | 100.0% | 0.2696 | 60.0 | 1240.00 |
| Many UAVs (5) | 10 | 5 | 4 | 232 | 75.9% | 0.4694 | 53.7 | 2190.00 |
| Many UAVs (8) | 10 | 8 | 4 | 412 | 86.7% | 0.3150 | 54.0 | 2596.00 |
| Few Edges (2) | 10 | 3 | 2 | 154 | 84.4% | 0.3922 | 51.0 | 1208.00 |
| Many Edges (6) | 10 | 3 | 6 | 452 | 95.4% | 0.2004 | 65.0 | 1812.00 |
| High Load | 30 | 5 | 6 | 688 | 95.9% | 0.2382 | 62.6 | 3278.00 |

## Key Observations

### Success Rate
- **Best:** No UAVs (100.0%)
- **Worst:** Many UAVs (5) (75.9%)

### Energy Consumption
- **Highest:** High Users (30) (4076.00 J)
- **Lowest:** Few Edges (2) (1208.00 J)

### Task Throughput
- **Most Tasks:** High Users (30) (690 tasks)

## Configuration Details

Each simulation ran for 500 simulation time units with varying:
- **User Count:** 5-30 users
- **UAV Count:** 0-8 UAVs
- **Edge Server Count:** 2-6 edge servers



## Visualizations

### Success Rate Comparison

![Success Rate Comparison](charts/basic_success_rate.png)

*Task success rate across different infrastructure configurations.*

### Energy Consumption

![Energy Consumption](charts/basic_energy.png)

*Total energy consumed by each configuration.*

### Multi-Metric Comparison

![Multi-Metric Comparison](charts/basic_metrics.png)

*Normalized comparison of success rate, QoS, and throughput.*

### Energy-Latency Trade-off

![Energy-Latency Trade-off](charts/basic_tradeoff.png)

*Scatter plot showing the relationship between energy and latency.*

### Configuration Radar

![Configuration Radar](charts/basic_radar.png)

*Radar chart comparing top configurations across all metrics.*

## Conclusion

The benchmark demonstrates how different infrastructure configurations affect:
1. Task completion success rate
2. Average latency
3. Quality of Service (QoS)
4. Total energy consumption

Increasing UAV count generally improves coverage but increases energy usage.
More edge servers provide better load balancing and lower latency.
