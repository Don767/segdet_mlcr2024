import time
import torch

def measure_gpu_throughput(model, inputs, batch_size):
    inputs = inputs.to('cuda')
    model = model.to('cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for i in range(0, inputs.size(0), batch_size):
            output = model(inputs[i:i + batch_size])
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end)
    throughput = inputs.size(0) * batch_size / latency
    return throughput

# Example usage:
# throughput = measure_gpu_throughput(model, inputs, batch_size)