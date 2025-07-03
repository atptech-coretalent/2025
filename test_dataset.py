from datasets import load_dataset

# Hugging Face'den CameraBench veri setini indir
ds = load_dataset("syCen/CameraBench")
print(ds)