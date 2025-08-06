import torch

input_video = 'input_videos/shop_test.mp4'
output_video = 'output_videos/yolo11test_unskelet.mp4'
json_file = 'track_data.json'

conf_threshold = 0.5
keypoint_threshpoint = 0.3

device = "cuda" if torch.cuda.is_available() else "cpu"