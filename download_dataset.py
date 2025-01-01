import datasets as ds
import tiktoken
import torch
from torch.utils.data import TensorDataset
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

CHUNK_LENGTH = 1030
PADDING_TOKEN = 50256
BATCH_SIZE = 10000
TOTAL_EXAMPLES = 10_000_000
tokenizer = tiktoken.encoding_for_model('gpt2')
dataset = ds.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

def process_batch(text_batch):
    # Tokenize the batch with 16 threads
    tokenized = tokenizer.encode_ordinary_batch(text_batch, num_threads=16)
    processed_chunks = []
    for tokens in tokenized:
        # Split tokens into chunks of CHUNK_LENGTH
        for i in range(0, len(tokens), CHUNK_LENGTH):
            chunk = tokens[i:i + CHUNK_LENGTH]
            # Pad the chunk on the right if it's shorter than CHUNK_LENGTH
            if len(chunk) < CHUNK_LENGTH:
                chunk += [PADDING_TOKEN] * (CHUNK_LENGTH - len(chunk))
            processed_chunks.append(chunk)
    return processed_chunks

def batch_iterator(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example['text'])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    all_chunks = []
    results = []
    with Pool(processes=8) as pool:  # 4 workers * 16 threads each = 64 threads
        with tqdm(total=TOTAL_EXAMPLES, desc="Processing examples") as pbar:
            # Submit all batches to the pool
            for text_batch in batch_iterator(dataset, BATCH_SIZE):
                results.append(pool.apply_async(process_batch, args=(text_batch,)))
                pbar.update(len(text_batch))
        
        # Collect all results without an additional progress bar
        for result in results:
            all_chunks.extend(result.get())
    
    # Convert the list of chunks to a tensor
    print("Converting to tensor...")
    tensor_data = torch.tensor(np.array(all_chunks), dtype=torch.long)
    
    # Create a TensorDataset
    tensor_dataset = TensorDataset(tensor_data)
    
    # Save the TensorDataset to disk
    print("Saving dataset...")
    torch.save(tensor_dataset, 'fineweb_edu_10BT.pt')
    print("Done!")

if __name__ == "__main__":
    main()
