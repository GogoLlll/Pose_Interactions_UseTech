import torch

input_video = 'input_videos/competition_1_3_xvid.avi'
output_video = 'output_videos/testes_3_isolate.mp4'
json_file = 'track_data.json'

conf_threshold = 0.5
keypoint_threshpoint = 0.3

device = "cuda" if torch.cuda.is_available() else "cpu"