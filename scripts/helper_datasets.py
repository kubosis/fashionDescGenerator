from huggingface_hub  import snapshot_download
import os

dataset = snapshot_download(
    repo_id="hieupth/fashiongen",
    repo_type="dataset",
    local_dir="fashiongen_data",
    token=""
)

print(dataset)
