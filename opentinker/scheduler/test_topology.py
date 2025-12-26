#!/usr/bin/env python3
"""
Test script for GPU topology detection.
Tests the detect_gpu_topology function without needing ray or other dependencies.
"""

import subprocess
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_gpu_topology() -> List[List[int]]:
    """
    Detect GPU topology using nvidia-smi to group GPUs by proximity.

    Groups GPUs based on NUMA nodes and PCIe connectivity. GPUs in the same
    group have better interconnect bandwidth (PIX/PXB) compared to cross-group
    connections (SYS).

    Returns:
        List of GPU groups, where each group is a list of GPU IDs with close topology.
        Falls back to single group with all GPUs if detection fails.
    """
    try:
        # Run nvidia-smi to get topology matrix
        result = subprocess.run(
            ["nvidia-smi", "topo", "--matrix"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning(
                "nvidia-smi topo command failed, using fallback GPU grouping"
            )
            return []

        lines = result.stdout.strip().split("\n")

        # Parse the topology matrix
        # First line contains GPU headers
        header_line = None
        matrix_lines = []

        for line in lines:
            if line.strip().startswith("GPU"):
                if header_line is None:
                    header_line = line
                else:
                    matrix_lines.append(line)

        if not header_line or not matrix_lines:
            logger.warning("Could not parse nvidia-smi topology output")
            return []

        # Extract GPU IDs from header
        gpu_ids = []
        parts = header_line.split()
        for part in parts:
            if part.startswith("GPU"):
                try:
                    gpu_id = int(part[3:])  # Extract number from "GPU0", "GPU1", etc.
                    gpu_ids.append(gpu_id)
                except ValueError:
                    continue

        if not gpu_ids:
            logger.warning("No GPU IDs found in topology matrix")
            return []

        logger.info(f"Detected {len(gpu_ids)} GPUs from topology: {gpu_ids}")

        # Parse NUMA affinity to group GPUs
        # Look for "NUMA Affinity" column
        numa_groups = {}

        for line in matrix_lines:
            parts = line.split()
            if not parts or not parts[0].startswith("GPU"):
                continue

            try:
                gpu_id = int(parts[0][3:])
            except (ValueError, IndexError):
                continue

            # Find NUMA affinity (second to last column before "GPU NUMA ID")
            # Format is usually like "0-23,48-71" or just a number "0" or "1"
            try:
                # NUMA affinity is typically at index -2 (before "GPU NUMA ID")
                numa_str = parts[-2]

                # Extract the first number as NUMA node ID
                if "-" in numa_str or "," in numa_str:
                    # Format like "0-23,48-71" - extract first number
                    numa_node = int(numa_str.split("-")[0].split(",")[0])
                else:
                    # Format like "0" or "1"
                    numa_node = int(numa_str)

                if numa_node not in numa_groups:
                    numa_groups[numa_node] = []
                numa_groups[numa_node].append(gpu_id)

            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse NUMA affinity for GPU{gpu_id}: {e}")
                continue

        # Convert to list of groups
        topology_groups = [sorted(gpus) for gpus in numa_groups.values()]

        if topology_groups:
            logger.info(f"Detected {len(topology_groups)} GPU topology groups:")
            for i, group in enumerate(topology_groups):
                logger.info(f"  Group {i}: GPUs {group} (NUMA node {i})")
            return topology_groups
        else:
            logger.warning("No NUMA groups detected, using fallback")
            return []

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi topo command timed out")
        return []
    except FileNotFoundError:
        logger.warning("nvidia-smi not found, GPU topology detection disabled")
        return []
    except Exception as e:
        logger.warning(f"Error detecting GPU topology: {e}")
        return []


if __name__ == "__main__":
    print("Testing GPU topology detection...")
    print("=" * 60)

    groups = detect_gpu_topology()

    if groups:
        print(f"\n✓ Successfully detected {len(groups)} topology groups:")
        for i, group in enumerate(groups):
            print(f"  Group {i}: GPUs {group} ({len(group)} GPUs)")

        print("\nTopology-aware allocation will prefer GPUs from the same group.")
        print("This improves inter-GPU communication bandwidth and reduces latency.")
    else:
        print("\n✗ No topology groups detected")
        print("Scheduler will fall back to sequential GPU allocation.")

    print("=" * 60)
    print("Topology detection test completed.")
