import requests

# Dataset path (elde ettiğin path)
video_path = "videos/0TqQje61Hoo.3.1.mp4"

# Hugging Face URL
base_url = "https://huggingface.co/datasets/syCen/CameraBench/resolve/main/"
video_url = base_url + video_path

# Kaydedilecek dosya adı
output_filename = video_path.split("/")[-1]

# İndirme işlemi
response = requests.get(video_url)
with open(output_filename, "wb") as f:
    f.write(response.content)

print(f"✅ Video indirildi: {output_filename}")
