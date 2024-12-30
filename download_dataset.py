import datasets as ds

dataset = ds.load_dataset("cerebras/SlimPajama-627B",
                                    split="train/chunk1",
                                    streaming=False)

print("Done")