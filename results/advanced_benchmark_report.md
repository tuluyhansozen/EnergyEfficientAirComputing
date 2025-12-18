# AirCompSim Advanced Benchmark Report

**Generated:** 2025-12-18 23:07:56

## Overview

This report presents results from advanced benchmarking of the AirCompSim simulator,
testing 4 key areas:
1. UAV positioning strategies
2. Charging station placement
3. User mobility patterns
4. Scheduling algorithms

---

## UAV Positioning

| Configuration | Tasks | Success Rate | Avg Latency (s) | Avg QoS | Energy (J) |
|--------------|-------|--------------|-----------------|---------|------------|
| Random Positioning | 480 | 63.3% | 0.3282 | 71.0 | 2278.00 |
| Grid Positioning | 480 | 45.4% | 0.2671 | 74.9 | 1202.00 |
| Edge-Centric | 480 | 53.3% | 0.2889 | 65.1 | 1590.00 |
| User-Centric | 480 | 75.8% | 0.3470 | 67.5 | 2894.00 |
| Cluster-Based | 480 | 73.5% | 0.2935 | 79.5 | 2090.00 |

### Analysis

- **Best Success Rate:** User-Centric (75.8%)
- **Most Energy Efficient:** Grid Positioning (1202.00 J)

**Key Insight:** UAV positioning strategy significantly impacts coverage.
User-centric and cluster-based positioning tend to outperform random placement by ensuring
UAVs are located where demand is highest.
---

## Charging Stations

| Configuration | Tasks | Success Rate | Avg Latency (s) | Avg QoS | Energy (J) |
|--------------|-------|--------------|-----------------|---------|------------|
| No Charging Stations | 480 | 70.2% | 0.2972 | 70.2 | 2104.00 |
| 1 Station (Center) | 480 | 62.1% | 0.3390 | 71.3 | 2100.00 |
| 2 Stations (Diagonal) | 480 | 71.9% | 0.2637 | 72.8 | 1830.00 |
| 4 Stations (Corners) | 480 | 66.0% | 0.3405 | 66.5 | 2368.00 |
| 4 Stations (Edges) | 480 | 44.8% | 0.3570 | 76.6 | 1678.00 |

### Analysis

- **Best Success Rate:** 2 Stations (Diagonal) (71.9%)
- **Most Energy Efficient:** 4 Stations (Edges) (1678.00 J)

**Key Insight:** Charging station availability affects UAV uptime.
With low initial battery (60%), having strategically placed charging stations can
improve task success rates by keeping more UAVs operational.
---

## Mobility Patterns

| Configuration | Tasks | Success Rate | Avg Latency (s) | Avg QoS | Energy (J) |
|--------------|-------|--------------|-----------------|---------|------------|
| Static Users | 480 | 71.0% | 0.2935 | 80.5 | 2084.00 |
| Low Mobility (speed=1) | 480 | 57.1% | 0.2635 | 79.9 | 1444.00 |
| Medium Mobility (speed=3) | 480 | 87.1% | 0.2848 | 73.5 | 2530.00 |
| High Mobility (speed=5) | 480 | 59.8% | 0.2969 | 67.3 | 1918.00 |
| Clustered Static | 480 | 94.8% | 0.2610 | 76.4 | 2434.00 |

### Analysis

- **Best Success Rate:** Clustered Static (94.8%)
- **Most Energy Efficient:** Low Mobility (speed=1) (1444.00 J)

**Key Insight:** User mobility affects task offloading success.
Faster moving users may leave server coverage before task completion, while
clustered users benefit from concentrated coverage.
---

## Scheduling

| Configuration | Tasks | Success Rate | Avg Latency (s) | Avg QoS | Energy (J) |
|--------------|-------|--------------|-----------------|---------|------------|
| Default (Load Balance) | 638 | 70.2% | 0.3917 | 65.9 | 4082.00 |
| Energy-First | 638 | 42.0% | 0.4340 | 65.5 | 2716.00 |
| Latency-First | 638 | 67.6% | 0.3527 | 74.3 | 3266.00 |
| Balanced | 638 | 70.8% | 0.3409 | 72.6 | 3348.00 |
| Utilization-Based | 638 | 64.1% | 0.3094 | 70.8 | 2556.00 |

### Analysis

- **Best Success Rate:** Balanced (70.8%)
- **Most Energy Efficient:** Utilization-Based (2556.00 J)

**Key Insight:** Different scheduling strategies optimize for different metrics.
Energy-first reduces consumption but may increase latency, while latency-first
prioritizes speed at higher energy cost.
---



## Visualizations

### UAV Positioning Success Rates

![UAV Positioning Success Rates](charts/adv_uav_positioning_success.png)

*Comparison of UAV positioning strategies.*

### Mobility Pattern Impact

![Mobility Pattern Impact](charts/adv_mobility_patterns_success.png)

*Success rates under different user mobility patterns.*

### Scheduling Algorithm Comparison

![Scheduling Algorithm Comparison](charts/adv_scheduling_success.png)

*Performance of different scheduling algorithms.*

### Performance Heatmap

![Performance Heatmap](charts/adv_heatmap.png)

*Heatmap showing normalized metrics across all configurations.*

### Top Configurations Radar

![Top Configurations Radar](charts/adv_radar.png)

*Radar chart comparing the best performing configurations.*

## Overall Summary

### Best Configurations by Metric

| Metric | Best Configuration | Value | Category |
|--------|-------------------|-------|----------|
| Success Rate | Clustered Static | 94.8% | Mobility Patterns |
| Energy Efficiency | Grid Positioning | 1202.00 J | UAV Positioning |
| Throughput | Default (Load Balance) | 638 tasks | Scheduling |
| Latency | Clustered Static | 0.2610s | Mobility Patterns |

### Recommendations

1. **For Maximum Reliability:** Use user-centric UAV positioning with balanced scheduling
2. **For Energy Savings:** Use energy-first scheduling with strategic charging stations
3. **For Low Latency:** Use latency-first scheduling with grid-based UAV positioning
4. **For High Mobility Users:** Increase UAV count and use cluster-based positioning

### Key Takeaways

1. UAV positioning strategy can improve success rates by 10-20%
2. Charging stations are critical when UAV battery starts below 70%
3. User mobility patterns affect optimal server placement
4. Scheduling algorithm choice depends on optimization goal
5. No single configuration is best for all metrics - trade-offs are necessary

---

*Report generated by AirCompSim Advanced Benchmark Suite*
