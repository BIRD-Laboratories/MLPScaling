import argparse
import csv
import math

def generate_ratios(max_layers, max_width):
    ratios = []
    for i in range(1, 5):  # 1/16 steps from 1:1 to 1:8 and vice versa
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

def estimate_vram(layer_count, width, input_size, output_size):
    # Calculate the number of parameters
    # Each layer has (input_size * width) + width parameters (weights + biases)
    # The last layer has (width * output_size) + output_size parameters
    param_count = 0
    for i in range(layer_count):
        if i == 0:
            param_count += (input_size * width) + width
        else:
            param_count += (width * width) + width
    param_count += (width * output_size) + output_size

    # Estimate the VRAM usage
    # Parameters: 4 bytes per parameter (FP32)
    # Activations: Assume the size of the activations is the same as the input size
    vram_usage = param_count * 4 + input_size * 4

    return vram_usage

def calculate_batch_size(vram_usage, memory_gb=20):
    memory_bytes = memory_gb * (1024 ** 3)  # Convert GiB to bytes
    batch_memory_bytes = memory_bytes / 4  # Divide by 4
    
    # Calculate the maximum batch size that fits within the available memory
    max_batch_size = batch_memory_bytes // vram_usage
    
    # Find the nearest power of 2
    batch_size = 2 ** int(math.log2(max_batch_size))
    
    return batch_size

def write_csv(experiments, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['layer_count', 'width', 'vram_usage', 'batch_size'])
        for experiment in experiments:
            writer.writerow(experiment)

def main():
    parser = argparse.ArgumentParser(description='Generate a CSV file with a variety of layer counts and widths.')
    parser.add_argument('--max_layers', type=int, default=176, help='Maximum number of layers (default: 72)')
    parser.add_argument('--max_width', type=int, default=int(4096*1.5), help='Maximum width (default: 4096)')
    parser.add_argument('--min_layers', type=int, default=12, help='Minimum number of layers (default: 1)')
    parser.add_argument('--min_width', type=int, default=256, help='Minimum width (default: 1)')
    parser.add_argument('--output_file', type=str, default='experiments.csv', help='Output CSV file (default: experiments.csv)')
    parser.add_argument('--input_size', type=int, default=64*64, help='Input size (default: 64*64*3)')
    parser.add_argument('--output_size', type=int, default=10, help='Output size (default: 10)')
    parser.add_argument('--memory_gb', type=int, default=80, help='Total memory in GiB (default: 20)')
    args = parser.parse_args()

    experiments = generate_experiments(args.max_layers, args.max_width, args.min_layers, args.min_width)
    experiments_with_vram_and_batch = []

    for experiment in experiments:
        layer_count, width = experiment
        vram_usage = estimate_vram(layer_count, width, args.input_size, args.output_size)
        batch_size = calculate_batch_size(vram_usage, args.memory_gb)
        experiments_with_vram_and_batch.append((layer_count, width, vram_usage, batch_size))
        print(f'Layer Count: {layer_count}, Width: {width}, Estimated VRAM Usage: {vram_usage} bytes, Batch Size: {batch_size}')

    # Sort experiments by VRAM usage in descending order
    experiments_with_vram_and_batch.sort(key=lambda x: x[2], reverse=True)

    write_csv(experiments_with_vram_and_batch, args.output_file)
    print(f'Generated {len(experiments_with_vram_and_batch)} experiments and saved to {args.output_file}')

if __name__ == '__main__':
    main()
