import torch

input_video = 'competition_1_1_xvid.avi'
output_video = 'output_analysisv11_2.mp4'
json_file = 'track_data.json'

conf_threshold = 0.3
keypoint_threshpoint = 0.3

device = "cuda" if torch.cuda.is_available() else "cpu"