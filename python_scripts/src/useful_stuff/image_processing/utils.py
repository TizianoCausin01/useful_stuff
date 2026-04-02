import sys, os
import cv2
import numpy as np
from torchvision import models, transforms
import timm
from scipy.io import loadmat
import torch
import torch.nn.functional as F
sys.path.append("../..")
from useful_stuff.general_utils.utils import print_wise, get_upsampling_indices, is_empty, get_device

def get_video_dimensions(cap):
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return round(height), round(width), round(n_frames) # round to make them int

"""
read_video
Load a video from disk and return its frames as a NumPy array.

INPUT:
    - paths: dict -> must contain the key 'livingstone_lab' pointing to the base directory
    - rank: int -> process rank used for logging
    - fn: str -> filename of the video to load
    - vid_duration: float (default=0) -> duration (in seconds) of the video to read.
        If 0, the full video is loaded; otherwise, only frames up to the specified duration are read.

OUTPUT:
    - video: np.ndarray, shape (n_frames, H, W, C) -> video frames in RGB format

NOTES:
    - Frames are read using OpenCV and converted from BGR to RGB.
    - If vid_duration is specified, the number of frames is computed as:
          min(round(vid_duration * fps), total_frames)
    - Raises FileNotFoundError if the video cannot be opened.
    - Raises RuntimeError if a frame cannot be read.
"""
def read_video(paths, file_name, folder_name=None, cap=None, start=0, end=-1, rank=None, to_array=None, conversion=cv2.COLOR_BGR2RGB, device='cpu', verbose=True, release=True):
    stimuli_path = f"{paths['data_path']}/stimuli/"
    if not os.path.isdir(stimuli_path): # in the livingstone lab the case is upper
        stimuli_path = f"{paths['data_path']}/Stimuli/"
    # end if not os.path.isdir(stimuli_path):
    if folder_name is not None:
        stimuli_path = f"{stimuli_path}{folder_name}/"
    # end if folder_name is not None:
    fn_path = f"{stimuli_path}{file_name}"
    if cap is None:
        cap = cv2.VideoCapture(fn_path)
    # end if cap is None: 
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {fn_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width, frames_n = get_video_dimensions(cap)
    video_duration_s = frames_n / fps  
    start_frame = int(round(start * fps))
    end_frame   = int(round(end * fps)) if end != -1 else frames_n
    if start_frame >= frames_n:
        raise IndexError(f"{start=} is beyond video length")
    # end if start_frame >= frames_n:
    if end_frame > frames_n:
        raise IndexError(
            f"The selected video {end=} is larger than the video duration={round(video_duration_s, 3)}"
        )
    # end if end_frame > frames_n:
    if end_frame <= start_frame:
        raise ValueError(f"{end=} must be greater than {start=}")
    # end if end_frame <= start_frame:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    actual_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if start_frame != actual_start: # checks the actual time of the cap in frames
        raise IndexError(f"Requested {start_frame}, got {actual_start} - Open-cv not handling cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) correctly, likely it is landing on the previous keyframe")
    # end if start_frame != actual_start:
    actual_time = actual_start/fps
    if not np.isclose(start, actual_time, atol=1/fps): # checks the actual time of the cap in seconds
        raise IndexError(f"Requested {start}, got {actual_time} - Open-cv not handling cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) correctly, likely it is landing on the previous keyframe")
    # end if start_frame != actual_start:
    frames_to_loop = int(end_frame - start_frame)

    video = []
    for _ in range(frames_to_loop):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {round(cap.get(cv2.CAP_PROP_POS_MSEC)*fps/1000)} from {fn_path}")
        
        if conversion is not None:
            frame = cv2.cvtColor(frame, conversion) # can I do something fancier here?
        # end if conversion is not None:
        video.append(frame)
    # end for _ in range(frames_to_loop):
    if to_array is not None:
        if to_array == 'numpy':
            video = np.stack(video)
        elif to_array == 'torch':
            video = [torch.from_numpy(frame).float() / 255.0 for frame in video]
            video = torch.stack(video).to(device) # (N,H,W,...)
            if video.ndim == 4:
                video = video.permute(0, 3, 1, 2) # (N,C,H,W,...)
            # end if video.ndim == 4:
        # end if to_array == 'numpy':
    # end if to_array is not None:
    if release:
        cap.release()  
    if verbose:
        print_wise(f"finished reading video {fn_path} \nfps={round(fps, 2)}, {height=}, {width=}, n_frames={len(video)}", rank=rank)
    return video
# EOF


def load_stimuli_models(paths, model_name, file_names, resolution_Hz):
    all_models = {}
    models_path = f"{paths['livingstone_lab']}/tiziano/models"
    for fn in file_names:
        video_path = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos/{fn}" 
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        h, w, _ = get_video_dimensions(cap)
        if model_name=="human_face_detection":
            if (h != 1080) or (w != 1920):
                raise ValueError("The size of the movie is different to the way eye-tracking were prerpocessed")
            curr_model = loadmat(f"{models_path}/{model_name}_{fn[:-4]}.mat")['coords']
        else:
            curr_model = np.load(f"{models_path}/{model_name}_{fn[:-4]}.npz")['data']
        # end model_name=="human_face_detection":
        indices = get_upsampling_indices(curr_model.shape[1], fps, resolution_Hz)
        curr_model = curr_model[:, indices]
        all_models[fn] = curr_model
    return all_models


"""
resize_video_array
Resize a video stored as a NumPy array of shape (n_frames, H, W, C).

INPUT:
    - video: np.ndarray -> (n_frames, H, W, C)
    - new_height: int -> desired output height
    - new_width: int -> desired output width
    - interpolation: cv2 interpolation method (default: bilinear)

OUTPUT:
    - resized_video: np.ndarray -> (n_frames, new_height, new_width, C)    
"""
def resize_video_array(video, new_height, new_width, interpolation=cv2.INTER_LINEAR, normalize=True):
    resized_video = np.stack([cv2.resize(frame, (new_width, new_height), interpolation=interpolation) for frame in video])
    if normalize:
        mean = resized_video.mean(axis=(0,1,2))
        std = resized_video.std(axis=(0,1,2)) + 1e-8
        resized_video = (resized_video - mean) / std
    # end if normalize:
    return resized_video
# EOF

"""
concatenate_frames_batch
Concatenate a batch of video frames from multiple videos, optionally processing only
the first part of long videos and keeping leftover frames from previous batches.

INPUT:
    - paths: dict -> paths to the video files
    - rank: int -> worker rank
    - frames_batch: list or np.ndarray -> leftover frames from previous batch
    - curr_video_idx: int -> current video index in fn_list
    - idx: int -> current batch index
    - batches_to_proc_togeth: int -> number of batches to process consecutively
    - batch_sizes: list/array -> sizes of each batch in frames
    - new_h, new_w: int -> target frame height and width
    - long_vids: list of bools -> flags indicating whether each video is long
    - vid_duration_lim: int (default 20) -> max duration in seconds for long videos
    - normalize: bool (default True) -> whether to normalize frames

OUTPUT:
    - frames_batch: np.ndarray -> concatenated frames for the batch
    - progression: int -> updated video index after processing

"""
def concatenate_frames_batch(paths, rank, fn_list, frames_batch, curr_video_idx, curr_batch_idx, batches_to_proc_togeth, batch_sizes, new_h, new_w, long_vids, vid_duration_lim=20, normalize=True):
    n_batches = len(batch_sizes)
    idx_tot = [curr_batch_idx + i for i in range(batches_to_proc_togeth) if curr_batch_idx + i < n_batches] # takes the next $batches_to_proc frames filtering for out of range indices 
    curr_tot_batch_size = np.sum(batch_sizes[idx_tot])
    cumulative_frames_sum = 0
    if is_empty(frames_batch):
        frames_batch = [] # otherwise we have arrays of inconsistent size to concatenate
    else:
        cumulative_frames_sum += frames_batch.shape[0]
        frames_batch = [frames_batch] # makes it a list with all the frames remained from the previous batch (ideally we read 3 batches and shuffle)
    # end if frames_batch:
    while cumulative_frames_sum < curr_tot_batch_size:
        fn = fn_list[curr_video_idx]
        if long_vids[curr_video_idx]: # if the video is marked as long
            video = read_video(paths, rank, fn, vid_duration=vid_duration_lim) # if the video is too long, we just process the beginning (vid_duration_lim is in sec)
        else:
            video = read_video(paths, rank, fn, vid_duration=0)
        # end if long_vids[progression]: 
        video = resize_video_array(video, new_h, new_w, normalize=False)
        curr_video_idx += 1
        curr_frames_n = video.shape[0] 
        cumulative_frames_sum += curr_frames_n
        frames_batch.append(video)
    # end while cumulative_frames_sum < curr_tot_batch_size:
    frames_batch = np.concatenate(frames_batch, axis=0)
    return frames_batch, curr_video_idx
# EOF


"""
shuffle_frames
Randomly shuffle the frames of a video array along the 0th dimension.

INPUT:
    - video: np.ndarray, shape (n_frames, H, W, C) -> video to shuffle

OUTPUT:
    - shuffled_video: np.ndarray, same shape as input -> frames randomly permuted
"""
def shuffle_frames(video):
    n_frames = video.shape[0] 
    indices = np.arange(n_frames)
    np.random.shuffle(indices)
    shuffled_video = video[indices, :, :, :]
    return shuffled_video
# EOF



"""
list_videos
Creates a list with all videos with the required characteristics.
INPUT:
    - paths: dict -> paths to the video files
    - video_type: str -> the type of videos to index
OUTPUT:
    - fn_list: list{str} -> a list with the file names of the required paths (or all the videos if video_type is None)
"""
def list_videos(paths: dict, video_type: str):
    videos_dir = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos"
    all_files = os.listdir(videos_dir) 
    if video_type:
        if video_type == 'YDX':
            fn_list = [f for f in all_files if "YDX" in f]
        elif video_type == 'IMG':
            fn_list = [f for f in all_files if "IMG" in f]
        elif video_type == 'faceswap':
            fn_list = [f for f in all_files if "YDX" not in f and "IMG" not in f]
        else:
            raise ValueError("video_type must be 'YDX', 'IMG' or 'faceswap'")
        return fn_list
        # end if video_type == 'YDX':
    else:
        return all_files
    # end if video_type:
# EOF

"""
get_frames_number
Computes the number of frames for each video in fn_list, marking which videos exceed a maximum duration and should be truncated.

INPUT:
    - paths: dict -> dictionary containing base paths (expects key 'livingstone_lab')
    - fn_list: list -> list of video filenames to process
    - max_duration: float -> maximum allowed duration in seconds; videos longer than
                               this are marked as long and truncated
OUTPUT:
    - frames_per_vid: list of floats -> number of frames for each video (truncated if long)
    - long_vids: list of bools -> True for long videos exceeding max_duration, else False
"""
def get_frames_number(paths: dict, fn_list: list, max_duration: float):
    videos_dir = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos"
    frames_per_vid = []
    long_vids = []
    for fn in fn_list:
        video_path = os.path.join(videos_dir, fn)
        cap = cv2.VideoCapture(video_path)
        _, _, n_frames = get_video_dimensions(cap)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if n_frames/fps > max_duration:
            n_frames = fps*max_duration
            long_vids.append(True)
        else:
            long_vids.append(False)
        frames_per_vid.append(n_frames)
    # end for fn in fn_list:
    return frames_per_vid, long_vids
# EOF


"""
split_in_batches
Splits a total number of frames into batches of approximately equal size.
Useful for distributing frames across workers or for chunked processing.
PROCESS:
    1. Computes the total number of frames across all videos.
    2. Computes how many batches are needed (rounded).
    3. Splits the frame indices into n_batches approximately equal parts.
    4. Extracts and stores the size of each batch.

INPUT:
    - frames_per_vid: list/array of ints -> number of frames for each video
    - batch_size: int -> desired approximate size of each batch
OUTPUT:
    - batch_size_list: list of ints -> sizes of each batch
    - splits: list of np.ndarray -> the actual index splits (each array contains the indices for that batch)
"""
def split_in_batches(frames_per_vid, batch_size):
    tot_frame_num = round(np.sum(frames_per_vid))
    n_batches = round(tot_frame_num/batch_size)
    splits = np.array_split(np.arange(tot_frame_num), n_batches)
    batch_size_list = []
    for batch_idx in splits:
        batch_size_list.append(len(batch_idx)) # stores the current batch size
    return np.array(batch_size_list) 

def map_anns_names(model_name, pkg='torchvision'):
    if model_name=='alexnet':
        return 'AlexNet'
    elif model_name== 'resnet50':
        return 'ResNet50'    
    elif model_name== 'resnet18':
        return 'ResNet18'
    elif model_name == 'vit_b_16':
        return 'ViT_B_16'
    elif model_name == 'vit_l_16':
        if pkg=='torchvision':
            return 'ViT_L_16'
        elif pkg=='timm':
            return 'vit_large_patch16'
        # end if pkg=='torchvision':
    elif model_name == 'vgg16':
        return 'VGG16'

"""
get_usual_transform
Returns a standard preprocessing pipeline commonly used for 
pretrained convolutional neural networks (e.g., ResNet, AlexNet) 
trained on ImageNet. This includes resizing, cropping, conversion 
to tensor, and normalization.

Inputs:
    None

Outputs:
    transform (torchvision.transforms.Compose):
        A composition of torchvision transforms to apply to input images.
        The transformations include:
            - Resize to 256 pixels (shorter side)
            - Center crop to 224x224 pixels
            - Convert to PyTorch tensor (values in [0,1])
            - Normalize using ImageNet mean and std:
                mean = [0.485, 0.456, 0.406]
                std  = [0.229, 0.224, 0.225]

Example usage:
    transform = get_usual_transform()
    image = PIL.Image.open("example.jpg")
    input_tensor = transform(image)
"""
def get_usual_transform(resize_size=224, center_crop_size=None, normalize=True):
    transform_list = [transforms.Resize((resize_size, resize_size))]
    if center_crop_size is not None:
        transform_list.append(transforms.CenterCrop(center_crop_size))
    # end if center_crop_size is not None:
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))

    # end if normalize:
    transform = transforms.Compose(transform_list)
    return transform
# EOF





"""
preprocess_batch
Convert a batch of images to model-ready format: channel-first, resized, and normalized.

INPUT:
    - batch: torch.Tensor -> (B, H, W, C)
    - input_size: int -> target spatial size (e.g. 224, 384)
    - m: list[float] -> mean for normalization (default: ImageNet)
    - std: list[float] -> std for normalization (default: ImageNet)

OUTPUT:
    - batch: torch.Tensor -> (B, 3, input_size, input_size)
"""
def preprocess_batch(batch, input_size, m=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    # 1. Convert to float and scale to [0,1] if needed
    batch = batch.to(device)
    batch = batch.permute(0,3,1,2)
    if batch.dtype != torch.float32:
        batch = batch.float()
    if batch.max() > 1.0:
        batch = batch / 255.0
    # 2. Resize (keeps it simple: direct resize)
    batch = F.interpolate(
        batch,
        size=(input_size, input_size),
        mode='bilinear',
        align_corners=False
    )
    # 3. Normalize with ImageNet stats
    mean = torch.tensor(m, device=batch.device)[None, :, None, None]
    std  = torch.tensor(std, device=batch.device)[None, :, None, None]
    batch = (batch - mean) / std
    return batch
# EOF


