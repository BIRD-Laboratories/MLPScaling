import argparse
import csv
import math
import random

def generate_ratios(max_layers, max_width):
    ratios = []
    for i in range(1, 5):  # Adjusted to generate more ratios
        ratio = i / 16
        if ratio <= 8:  # Ensure the ratio is not greater than 8
            ratios.append(ratio)
    return ratios

def generate_experiments(max_layers, max_width, min_layers=1, min_width=1):
    experiments = []
    ratios = generate_ratios(max_layers, max_width)

    for ratio in ratios:
        for layers in range(min_layers, max_layers + 1):
            width = max(int(max_width * ratio), min_width)
            experiments.append((layers, width))

    return experiments

def generate_experiments_linear(max_layers, max_width, min_layers=1, min_width=1):
    experiments = []
    step_layers = (max_layers - min_layers) // 10  # Divide the range into 10 steps
    step_width = (max_width - min_width) // 10

    for i in range(11):  # 0 to 10
        layers = min_layers + i * step_layers
        width = min_width + i * step_width
        experiments.append((layers, width))

    return experiments

def estimate_vram(layer_count, width, input_size, output_size):
    param_count = 0
    for i in range(layer_count):
        if i == 0:
            param_count += (input_size * width) + width
        else:
            param_count += (width * width) + width
    param_count += (width * output_size) + output_size

    vram_usage = param_count * 4 + input_size * 4

    return vram_usage, param_count

def calculate_batch_size(vram_usage, memory_gb=20):
    memory_bytes = memory_gb * (1024 ** 3)
    available_memory_bytes = memory_bytes * 0.75
    batch_memory_bytes = available_memory_bytes / 4

    max_batch_size = batch_memory_bytes // vram_usage

    if max_batch_size <= 0:
        return 0

    batch_size = 2 ** int(math.log2(max_batch_size))

    return batch_size

def write_csv(experiments, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['layer_count', 'width', 'vram_usage', 'batch_size'])
        for experiment in experiments:
            writer.writerow(experiment)

def main():
    parser = argparse.ArgumentParser(description='Generate CSV files with experiments.')
    parser.add_argument('--max_layers', type=int, default=224, help='Maximum number of layers (default: 224)')
    parser.add_argument('--max_width', type=int, default=int(4096*1.5), help='Maximum width (default: 6144)')
    parser.add_argument('--min_layers', type=int, default=12, help='Minimum number of layers (default: 12)')
    parser.add_argument('--min_width', type=int, default=256, help='Minimum width (default: 256)')
    parser.add_argument('--input_size', type=int, default=64*64, help='Input size (default: 4096)')
    parser.add_argument('--output_size', type=int, default=10, help='Output size (default: 10)')
    parser.add_argument('--memory_gb', type=int, default=80, help='Total memory in GiB (default: 80)')
    args = parser.parse_args()

    # Generate linearly scaling experiments
    experiments_linear = generate_experiments_linear(args.max_layers, args.max_width, args.min_layers, args.min_width)
    experiments_linear_with_vram_and_batch = []

    for experiment in experiments_linear:
        layer_count, width = experiment
        vram_usage, param_count = estimate_vram(layer_count, width, args.input_size, args.output_size)
        batch_size = calculate_batch_size(vram_usage, args.memory_gb)
        
        if batch_size > 0:
            experiments_linear_with_vram_and_batch.append((layer_count, width, vram_usage, batch_size))

    # Sort linear experiments by layer_count
    experiments_linear_with_vram_and_batch.sort(key=lambda x: x[0])

    # Write linear experiments to experiments.csv
    write_csv(experiments_linear_with_vram_and_batch, 'experiments.csv')
    print(f'Generated {len(experiments_linear_with_vram_and_batch)} linear experiments and saved to experiments.csv')

    # Generate ratio-based experiments
    experiments_ratio = generate_experiments(args.max_layers, args.max_width, args.min_layers, args.min_width)
    experiments_ratio_with_vram_and_batch = []

    for experiment in experiments_ratio:
        layer_count, width = experiment
        vram_usage, param_count = estimate_vram(layer_count, width, args.input_size, args.output_size)
        batch_size = calculate_batch_size(vram_usage, args.memory_gb)
        
        if batch_size > 0:
            experiments_ratio_with_vram_and_batch.append((layer_count, width, vram_usage, batch_size))

    # Sort ratio experiments by param_count
    experiments_ratio_with_vram_and_batch.sort(key=lambda x: x[2])

    # Write ratio-based experiments to experiments_expanded.csv
    write_csv(experiments_ratio_with_vram_and_batch, 'experiments_expanded.csv')
    print(f'Generated {len(experiments_ratio_with_vram_and_batch)} ratio-based experiments and saved to experiments_expanded.csv')

if __name__ == '__main__':
    main()
