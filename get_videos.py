from datasets import load_dataset

# Dataset'i yükle
ds = load_dataset("syCen/CameraBench")
test_data = ds["test"]

# İlk 3 video bilgisini yazdır
for i in range(3):
    print(f"Caption: {test_data[i]['caption']}")
    print(f"Label: {test_data[i]['labels']}")
    print(f"Path: {test_data[i]['path']}")
    print()
