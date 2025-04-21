from diffusers_helper.hf_login import login

import os
import json
import traceback
from dataclasses import dataclass, asdict
from typing import Optional
import uuid

# Path to the quick prompts JSON file
PROMPT_FILE = os.path.join(os.getcwd(), 'quick_prompts.json')

# Queue file path
QUEUE_FILE = os.path.join(os.getcwd(), 'job_queue.json')

# Temp directory for queue images
temp_queue_images = os.path.join(os.getcwd(), 'temp_queue_images')
os.makedirs(temp_queue_images, exist_ok=True)

# Default prompts
DEFAULT_PROMPTS = [
    {'prompt': 'The girl dances gracefully, with clear movements, full of charm.', 'length': 5.0},
    {'prompt': 'A character doing some simple body movements.', 'length': 5.0},
]

# Load existing prompts or create the file with defaults
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, 'r') as f:
        quick_prompts = json.load(f)
else:
    quick_prompts = DEFAULT_PROMPTS.copy()
    with open(PROMPT_FILE, 'w') as f:
        json.dump(quick_prompts, f, indent=2)

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

@dataclass
class QueuedJob:
    prompt: str
    image_path: str
    video_length: float
    job_id: str  # Changed to string for hex ID
    seed: int
    use_teacache: bool
    gpu_memory_preservation: float
    steps: int
    cfg: float
    gs: float
    rs: float
    status: str = "pending"
    thumbnail: str = ""

    def to_dict(self):
        try:
            return {
                'prompt': self.prompt,
                'image_path': self.image_path,
                'video_length': self.video_length,
                'job_id': self.job_id,
                'seed': self.seed,
                'use_teacache': self.use_teacache,
                'gpu_memory_preservation': self.gpu_memory_preservation,
                'steps': self.steps,
                'cfg': self.cfg,
                'gs': self.gs,
                'rs': self.rs,
                'status': self.status,
                'thumbnail': self.thumbnail
            }
        except Exception as e:
            print(f"Error converting job to dict: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                prompt=data['prompt'],
                image_path=data['image_path'],
                video_length=data['video_length'],
                job_id=data['job_id'],
                seed=data['seed'],
                use_teacache=data['use_teacache'],
                gpu_memory_preservation=data['gpu_memory_preservation'],
                steps=data['steps'],
                cfg=data['cfg'],
                gs=data['gs'],
                rs=data['rs'],
                status=data['status'],
                thumbnail=data['thumbnail']
            )
        except Exception as e:
            print(f"Error creating job from dict: {str(e)}")
            return None

# Initialize job queue as a list
job_queue = []

def save_queue():
    try:
        jobs = []
        for job in job_queue:
            job_dict = job.to_dict()
            if job_dict is not None:
                jobs.append(job_dict)
        
        file_path = os.path.abspath(QUEUE_FILE)
        with open(file_path, 'w') as f:
            json.dump(jobs, f, indent=2)
        print(f"Queue saved with {len(jobs)} jobs")
        return True
    except Exception as e:
        print(f"Error saving queue: {str(e)}")
        traceback.print_exc()
        return False

def load_queue():
    try:
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, 'r') as f:
                jobs = json.load(f)
            print(f"Loading {len(jobs)} jobs from queue file")
            # Clear existing queue and load jobs from file
            job_queue.clear()
            for job_data in jobs:
                job = QueuedJob.from_dict(job_data)
                if job is not None:
                    job_queue.append(job)
            print(f"Successfully loaded {len(job_queue)} jobs into memory")
            return job_queue
        print("No queue file found, starting with empty queue")
        return []
    except Exception as e:
        print(f"Error loading queue: {str(e)}")
        traceback.print_exc()
        return []

# Load existing queue on startup
job_queue = load_queue()
print(f"Initial queue loaded with {len(job_queue)} jobs")

def save_image_to_temp(image: np.ndarray, job_id: str) -> str:
    """Save image to temp directory and return the path"""
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        # Create unique filename using hex ID
        filename = f"queue_image_{job_id}.png"
        filepath = os.path.join(temp_queue_images, filename)
        # Save image
        pil_image.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return ""

def add_to_queue(prompt, image, video_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, status="pending"):
    try:
        # Generate a unique hex ID for the job
        job_id = uuid.uuid4().hex[:8]
        # Save image to temp directory and get path
        image_path = save_image_to_temp(image, job_id)
        if not image_path:
            return None
            
        job = QueuedJob(
            prompt=prompt,
            image_path=image_path,
            video_length=video_length,
            job_id=job_id,
            seed=seed,
            use_teacache=use_teacache,
            gpu_memory_preservation=gpu_memory_preservation,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            status=status
        )
        job_queue.append(job)
        save_queue()  # Save immediately after adding
        return job_id
    except Exception as e:
        print(f"Error adding job to queue: {str(e)}")
        traceback.print_exc()
        return None

def get_next_job():
    try:
        if job_queue:
            job = job_queue.pop(0)  # Remove and return first job
            save_queue()  # Save after removing job
            return job
        return None
    except Exception as e:
        print(f"Error getting next job: {str(e)}")
        traceback.print_exc()
        return None

def move_job(job_id: str, direction: str):
    """Move a job in the queue based on direction: 'up', 'down', 'top', 'bottom'"""
    try:
        # Find the job index
        job_index = next((i for i, job in enumerate(job_queue) if job.job_id == job_id), None)
        if job_index is None:
            return False

        if direction == 'up' and job_index > 0:
            # Swap with previous job
            job_queue[job_index], job_queue[job_index - 1] = job_queue[job_index - 1], job_queue[job_index]
        elif direction == 'down' and job_index < len(job_queue) - 1:
            # Swap with next job
            job_queue[job_index], job_queue[job_index + 1] = job_queue[job_index + 1], job_queue[job_index]
        elif direction == 'top':
            # Move to top
            job = job_queue.pop(job_index)
            job_queue.insert(0, job)
        elif direction == 'bottom':
            # Move to bottom
            job = job_queue.pop(job_index)
            job_queue.append(job)

        save_queue()  # Save after moving
        return True
    except Exception as e:
        print(f"Error moving job: {str(e)}")
        traceback.print_exc()
        return False

def delete_job(job_id):
    """Delete a job from the queue"""
    try:
        print(f"\n=== DEBUG: Deleting Job {job_id} ===")
        # Find and remove the job
        for i, job in enumerate(job_queue):
            if job.job_id == job_id:
                print(f"Removing job {job_id} from queue")
                # Delete associated files
                if job.image_path and os.path.exists(job.image_path):
                    os.remove(job.image_path)
                if job.thumbnail and os.path.exists(job.thumbnail):
                    os.remove(job.thumbnail)
                # Remove from queue
                job_queue.pop(i)
                save_queue()
                print("Job removed and queue saved")
                return (
                    update_queue_display(),  # queue_display
                    gr.update(interactive=True)  # queue_button (always enabled)
                )
        print(f"Job {job_id} not found in queue")
        return update_queue_display(), gr.update(interactive=True)  # queue_button (always enabled)
    except Exception as e:
        print(f"Error deleting job: {str(e)}")
        traceback.print_exc()
        return update_queue_display(), gr.update(interactive=True)  # queue_button (always enabled)

def update_queue_display():
    try:
        print(f"Updating queue display with {len(job_queue)} jobs")
        queue_data = []
        for job in job_queue:
            # Create thumbnail if it doesn't exist
            if not job.thumbnail and job.image_path:
                try:
                    # Load and resize image
                    img = Image.open(job.image_path)
                    width, height = img.size
                    new_height = 200
                    new_width = int((new_height / height) * width)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    thumb_path = os.path.join(temp_queue_images, f"thumb_{job.job_id}.png")
                    img.save(thumb_path)
                    job.thumbnail = thumb_path
                    save_queue()
                except Exception as e:
                    print(f"Error creating thumbnail: {str(e)}")
                    job.thumbnail = ""

            # Add job data to display
            if job.thumbnail:
                # Create caption with status above and prompt below
                caption = f"Status: {job.status}\n\nPrompt: {job.prompt[:30]}...\nLength: {job.video_length}s"
                # Add delete button HTML for pending jobs
                if job.status.lower() == "pending":
                    caption += f"\n\n<button class='delete-button' data-job-id='{job.job_id}'>Delete</button>"
                queue_data.append((job.thumbnail, caption))
        print(f"Returning {len(queue_data)} jobs for display")
        return queue_data
    except Exception as e:
        print(f"Error updating queue display: {str(e)}")
        traceback.print_exc()
        return []

def move_job_handler(job_id: str, direction: str):
    """Handle job movement and return updated queue display"""
    if move_job(job_id, direction):
        return update_queue_display()
    return []

# Quick prompts management functions
def get_default_prompt():
    try:
        if quick_prompts and len(quick_prompts) > 0:
            return quick_prompts[0]['prompt'], quick_prompts[0]['length']
        return "", 5.0
    except Exception as e:
        print(f"Error getting default prompt: {str(e)}")
        return "", 5.0

def save_quick_prompt(prompt_text, video_length):
    global quick_prompts
    if prompt_text:
        # Check if prompt already exists
        for item in quick_prompts:
            if item['prompt'] == prompt_text:
                item['length'] = video_length
                break
        else:
            quick_prompts.append({'prompt': prompt_text, 'length': video_length})
        
        with open(PROMPT_FILE, 'w') as f:
            json.dump(quick_prompts, f, indent=2)
    # Keep the text in the prompt box and set it as selected in quick list
    return prompt_text, gr.update(choices=[item['prompt'] for item in quick_prompts], value=prompt_text), video_length

def delete_quick_prompt(prompt_text):
    global quick_prompts
    if prompt_text:
        quick_prompts = [item for item in quick_prompts if item['prompt'] != prompt_text]
        with open(PROMPT_FILE, 'w') as f:
            json.dump(quick_prompts, f, indent=2)
    # Clear the prompt box and quick list selection
    return "", gr.update(choices=[item['prompt'] for item in quick_prompts], value=None), 5.0

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    global stream
    
    print("\n=== DEBUG: Process Function Start ===")
    print(f"Input Image: {input_image is not None}")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    print(f"Total Second Length: {total_second_length}")
    print(f"Steps: {steps}")
    print(f"Queue Status: {len(job_queue)} jobs")
    print(f"All Jobs Status: {[job.status for job in job_queue]}")
    print(f"All Jobs Details:")
    for job in job_queue:
        print(f"  - ID: {job.job_id}, Status: {job.status}, Path: {job.image_path}")
    
    # Initialize variables
    output_filename = None
    job_id = None
    
    # Determine which job to process
    if input_image is not None:
        print("\n=== DEBUG: Checking for just_added jobs ===")
        # Check for just_added jobs first
        just_added_jobs = [job for job in job_queue if job.status == "just_added"]
        if just_added_jobs:
            print(f"Found {len(just_added_jobs)} just_added jobs, processing the first one")
            next_job = just_added_jobs[0]
            print(f"\n=== DEBUG: Processing Just Added Job ===")
            print(f"Job ID: {next_job.job_id}")
            print(f"Image Path: {next_job.image_path}")
            print(f"Prompt: {next_job.prompt}")
            print(f"Status: {next_job.status}")
            
            next_job.status = "processing"
            save_queue()
            job_id = next_job.job_id
            
            try:
                print("\n=== DEBUG: Loading Image from Path ===")
                print(f"Attempting to load image from: {next_job.image_path}")
                process_image = np.array(Image.open(next_job.image_path))
                print(f"Image loaded successfully, shape: {process_image.shape}")
            except Exception as e:
                print(f"ERROR loading image: {str(e)}")
                traceback.print_exc()
                raise
            
            process_prompt = next_job.prompt
            process_seed = next_job.seed
            process_length = next_job.video_length
            process_steps = next_job.steps
            process_cfg = next_job.cfg
            process_gs = next_job.gs
            process_rs = next_job.rs
            process_preservation = next_job.gpu_memory_preservation
            process_teacache = next_job.use_teacache
        else:
            print("\n=== DEBUG: No just_added jobs found, processing input image ===")
            # Process input image
            job_id = add_to_queue(
                prompt=prompt,
                image=input_image,
                video_length=total_second_length,
                seed=seed,
                use_teacache=use_teacache,
                gpu_memory_preservation=gpu_memory_preservation,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                status="processing"
            )
            print(f"Added new job to queue with ID: {job_id}")
            process_image = input_image
            process_prompt = prompt
            process_seed = seed
            process_length = total_second_length
            process_steps = steps
            process_cfg = cfg
            process_gs = gs
            process_rs = rs
            process_preservation = gpu_memory_preservation
            process_teacache = use_teacache
    else:
        print("\n=== DEBUG: Checking Pending Jobs ===")
        # Check for pending jobs - using case-insensitive comparison
        pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
        print(f"Found {len(pending_jobs)} pending jobs")
        
        if not pending_jobs:
            print("No pending jobs found, asserting input image requirement")
            assert input_image is not None, 'No input image!'
        
        # Process first pending job
        next_job = pending_jobs[0]
        print(f"\n=== DEBUG: Processing Pending Job ===")
        print(f"Job ID: {next_job.job_id}")
        print(f"Image Path: {next_job.image_path}")
        print(f"Prompt: {next_job.prompt}")
        print(f"Status: {next_job.status}")
        
        next_job.status = "processing"
        save_queue()
        job_id = next_job.job_id
        
        try:
            print("\n=== DEBUG: Loading Image from Path ===")
            print(f"Attempting to load image from: {next_job.image_path}")
            process_image = np.array(Image.open(next_job.image_path))
            print(f"Image loaded successfully, shape: {process_image.shape}")
        except Exception as e:
            print(f"ERROR loading image: {str(e)}")
            traceback.print_exc()
            raise
        
        process_prompt = next_job.prompt
        process_seed = next_job.seed
        process_length = next_job.video_length
        process_steps = next_job.steps
        process_cfg = next_job.cfg
        process_gs = next_job.gs
        process_rs = next_job.rs
        process_preservation = next_job.gpu_memory_preservation
        process_teacache = next_job.use_teacache
    
    print("\n=== DEBUG: Starting Processing ===")
    print(f"Using Job ID: {job_id}")
    print(f"Process Image Shape: {process_image.shape if process_image is not None else 'None'}")
    print(f"Process Prompt: {process_prompt}")
    print(f"Process Seed: {process_seed}")
    
    # Start processing
    stream = AsyncStream()
    async_run(worker, process_image, process_prompt, n_prompt, process_seed, 
             process_length, latent_window_size, process_steps, 
             process_cfg, process_gs, process_rs, 
             process_preservation, process_teacache)
    
    # Initial yield with updated queue display and button states
    yield (
        None,  # result_video
        None,  # preview_image
        '',    # progress_desc
        '',    # progress_bar
        gr.update(interactive=False),  # start_button
        gr.update(interactive=True),   # end_button
        gr.update(interactive=True),   # queue_button (always enabled)
        update_queue_display()         # queue_display
    )

    print("\n=== DEBUG: Starting Output Queue Processing ===")
    # Process output queue
    while True:
        flag, data = stream.output_queue.next()
        print(f"\n=== DEBUG: Queue Flag: {flag} ===")

        if flag == 'file':
            output_filename = data
            print(f"Received file: {output_filename}")
            yield (
                output_filename,  # result_video
                gr.update(),  # preview_image
                gr.update(),  # progress_desc
                gr.update(),  # progress_bar
                gr.update(interactive=False),  # start_button
                gr.update(interactive=True),   # end_button
                gr.update(interactive=True),   # queue_button (always enabled)
                update_queue_display()         # queue_display
            )

        if flag == 'progress':
            preview, desc, html = data
            print(f"Progress update: {desc}")
            yield (
                gr.update(),  # result_video
                gr.update(visible=True, value=preview),  # preview_image
                desc,  # progress_desc
                html,  # progress_bar
                gr.update(interactive=False),  # start_button
                gr.update(interactive=True),   # end_button
                gr.update(interactive=True),   # queue_button (always enabled)
                update_queue_display()         # queue_display
            )

        if flag == 'end':
            print("\n=== DEBUG: Job Completed ===")
            # Only remove the job if it completed normally (not ended by user)
            if not stream.input_queue.top() == 'end':
                # Mark current job as completed and remove it
                for job in job_queue:
                    if job.job_id == job_id:
                        print(f"Marking job {job_id} as completed and removing from queue")
                        job.status = "completed"
                        job_queue.remove(job)
                        save_queue()
                        break
            else:
                # Job was ended by user, just update its status (which should already be pending from end_process)
                print("\n=== DEBUG: Job Ended by User ===")
                for job in job_queue:
                    if job.job_id == job_id:
                        print(f"Job {job_id} was ended by user, status should be pending")
                        break

            # Check if we should continue processing (only if end button wasn't clicked)
            if not stream.input_queue.top() == 'end':
                # Find next job to process
                next_job = None
                
                # First check for pending jobs
                pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
                if pending_jobs:
                    print(f"\n=== DEBUG: Found {len(pending_jobs)} pending jobs")
                    next_job = pending_jobs[0]
                else:
                    # If no pending jobs, check for just_added jobs
                    just_added_jobs = [job for job in job_queue if job.status == "just_added"]
                    if just_added_jobs:
                        print(f"\n=== DEBUG: Found {len(just_added_jobs)} just_added jobs")
                        next_job = just_added_jobs[0]

                if next_job:
                    print(f"\n=== DEBUG: Starting Next Job ===")
                    print(f"Next Job ID: {next_job.job_id}")
                    print(f"Next Job Status: {next_job.status}")
                    # Update next job status to processing
                    next_job.status = "processing"
                    save_queue()
                    
                    try:
                        print(f"Loading next image from: {next_job.image_path}")
                        next_image = np.array(Image.open(next_job.image_path))
                        print(f"Next image loaded successfully, shape: {next_image.shape}")
                    except Exception as e:
                        print(f"ERROR loading next image: {str(e)}")
                        traceback.print_exc()
                        raise
                    
                    # Process next job
                    async_run(worker, next_image, next_job.prompt, n_prompt, next_job.seed, 
                             next_job.video_length, latent_window_size, next_job.steps, 
                             next_job.cfg, next_job.gs, next_job.rs, 
                             next_job.gpu_memory_preservation, next_job.use_teacache)
                else:
                    print("\n=== DEBUG: No More Jobs ===")
                    # No more jobs, return to initial state
                    yield (
                        output_filename,  # result_video
                        gr.update(visible=False),  # preview_image
                        gr.update(),  # progress_desc
                        '',  # progress_bar
                        gr.update(interactive=True),  # start_button
                        gr.update(interactive=False),  # end_button
                        gr.update(interactive=True),  # queue_button
                        update_queue_display()  # queue_display
                    )
                    break
            else:
                # End button was clicked, stop processing
                print("\n=== DEBUG: End Button Clicked, Stopping Processing ===")
                yield (
                    output_filename,  # result_video
                    gr.update(visible=False),  # preview_image
                    gr.update(),  # progress_desc
                    '',  # progress_bar
                    gr.update(interactive=True),  # start_button
                    gr.update(interactive=False),  # end_button
                    gr.update(interactive=True),  # queue_button
                    update_queue_display()  # queue_display
                )
                break

def end_process():
    """Handle end generation button click - stop all processes and change all processing/just_added jobs to pending"""
    try:
        print("\n=== DEBUG: End Process Called ===")
        # First send the end signal to stop all processes
        stream.input_queue.push('end')
        
        # Find and update all processing and just_added jobs
        jobs_changed = 0
        processing_job = None
        
        # First find the processing job
        for job in job_queue:
            if job.status == "processing":
                processing_job = job
                break
        
        # Then process all jobs
        for job in job_queue:
            if job.status in ("processing", "just_added"):
                print(f"Changing job {job.job_id} status from {job.status} to pending")
                job.status = "pending"
                jobs_changed += 1
        
        # If we found a processing job, move it to the top
        if processing_job:
            print(f"Moving job {processing_job.job_id} to top of queue")
            job_queue.remove(processing_job)
            job_queue.insert(0, processing_job)
        
        save_queue()
        print(f"Queue saved with {jobs_changed} jobs changed to pending")
        return (
            update_queue_display(),  # queue_display
            gr.update(interactive=True)  # queue_button (always enabled)
        )
    except Exception as e:
        print(f"Error in end_process: {str(e)}")
        traceback.print_exc()
        return [], gr.update(interactive=True)  # queue_button (always enabled)

def add_to_queue_handler(input_image, prompt, total_second_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs):
    """Handle adding a new job to the queue"""
    if input_image is None or not prompt:
        return [], gr.update(interactive=True)  # queue_button (always enabled)
    
    try:
        print("\n=== DEBUG: Adding Job to Queue ===")
        # Change any existing just_added jobs to pending
        for job in job_queue:
            if job.status == "just_added":
                print(f"Changing job {job.job_id} status from just_added to pending")
                job.status = "pending"
        
        # Add new job with just_added status
        job_id = add_to_queue(
            prompt=prompt,
            image=input_image,
            video_length=total_second_length,
            seed=seed,
            use_teacache=use_teacache,
            gpu_memory_preservation=gpu_memory_preservation,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            status="just_added"  # Set status as just_added
        )
        
        if job_id is not None:
            print(f"Job {job_id} added to queue successfully with status: just_added")
            save_queue()  # Save after changing statuses
            return update_queue_display(), gr.update(interactive=True)  # queue_button (always enabled)
        else:
            print("Failed to add job to queue")
            return [], gr.update(interactive=True)  # queue_button (always enabled)
    except Exception as e:
        print(f"Error in add_to_queue_handler: {str(e)}")
        traceback.print_exc()
        return [], gr.update(interactive=True)  # queue_button (always enabled)

def cleanup_orphaned_files():
    """Clean up any temp files that don't correspond to jobs in the queue"""
    try:
        # Get all job files from queue
        job_files = set()
        for job in job_queue:
            if job.image_path:
                job_files.add(job.image_path)
            if job.thumbnail:
                job_files.add(job.thumbnail)
        
        # Get all files in temp directory
        temp_files = set()
        for root, _, files in os.walk(temp_queue_images):
            for file in files:
                temp_files.add(os.path.join(root, file))
        
        # Find orphaned files (in temp but not in queue)
        orphaned_files = temp_files - job_files
        
        # Delete orphaned files
        for file in orphaned_files:
            try:
                os.remove(file)
                print(f"Deleted orphaned file: {file}")
            except Exception as e:
                print(f"Error deleting file {file}: {str(e)}")
    except Exception as e:
        print(f"Error in cleanup_orphaned_files: {str(e)}")
        traceback.print_exc()

def reset_processing_jobs():
    """Reset any processing or just_added jobs to pending and move them to top of queue"""
    try:
        print("\n=== DEBUG: Reset Processing Jobs Called ===")
        # Find all processing and just_added jobs
        jobs_to_move = []
        for job in job_queue:
            if job.status in ("processing", "just_added"):
                print(f"Found job {job.job_id} with status {job.status}")
                jobs_to_move.append(job)
        
        # Remove these jobs from their current positions
        for job in jobs_to_move:
            job_queue.remove(job)
            job.status = "pending"
            print(f"Changed job {job.job_id} status to pending")
        
        # Add them back at the top in reverse order (so they maintain their relative order)
        for job in reversed(jobs_to_move):
            job_queue.insert(0, job)
            print(f"Moved job {job.job_id} to top of queue")
        
        save_queue()
        print(f"Queue saved with {len(jobs_to_move)} jobs moved to top")
    except Exception as e:
        print(f"Error in reset_processing_jobs: {str(e)}")
        traceback.print_exc()

# Add these calls at startup
reset_processing_jobs()
cleanup_orphaned_files()

# Update the gallery component
queue_display = gr.Gallery(
    label="Job Queue",
    show_label=True,
    columns=3,
    height=600,  # Increased height
    object_fit="contain",
    elem_classes=["queue-gallery"]  # Add class for custom styling
)

# Add custom CSS for the queue display
block = gr.Blocks(css="""
.queue-gallery {
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
    display: flex;
    flex-direction: column;
}
.queue-gallery .gallery-container {
    overflow-y: auto !important;
    max-height: 600px !important;
}
.queue-gallery .gallery-item {
    margin: 5px;
    padding: 5px;
    border: 1px solid #eee;
    border-radius: 3px;
    flex-shrink: 0;
}
.queue-gallery .gallery-item img {
    max-height: 200px;
    object-fit: contain;
}
.delete-button {
    background-color: #ff4444;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
    cursor: pointer;
    margin-top: 5px;
}
.delete-button:hover {
    background-color: #cc0000;
}
""").queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            save_button = gr.Button("Save Prompt")
            delete_button = gr.Button("Delete Selected Prompt")
            quick_list = gr.Dropdown(
                label="Quick List",
                choices=[item['prompt'] for item in quick_prompts],
                value=quick_prompts[0]['prompt'] if quick_prompts else None,
                allow_custom_value=True
            )

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

            # Set default prompt and length
            default_prompt, default_length = get_default_prompt()
            prompt.value = default_prompt
            total_second_length.value = default_length

            save_button.click(
                save_quick_prompt,
                inputs=[prompt, total_second_length],
                outputs=[prompt, quick_list, total_second_length],
                queue=False
            )
            delete_button.click(
                delete_quick_prompt,
                inputs=[quick_list],
                outputs=[prompt, quick_list, total_second_length],
                queue=False
            )
            quick_list.change(
                lambda x: (x, next((item['length'] for item in quick_prompts if item['prompt'] == x), 5.0)) if x else ("", 5.0),
                inputs=[quick_list],
                outputs=[prompt, total_second_length],
                queue=False
            )

            # Add JavaScript to set default prompt on page load
            block.load(
                fn=lambda: (default_prompt, default_length),
                inputs=None,
                outputs=[prompt, total_second_length],
                queue=False
            )

        with gr.Column():
            start_button = gr.Button(value="Start Generation", interactive=True)
            end_button = gr.Button(value="End Generation", interactive=False)
            queue_button = gr.Button(value="Add to Queue", interactive=True)
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            queue_display = gr.Gallery(
                label="Job Queue",
                show_label=True,
                columns=3,
                height=600,  # Increased height
                object_fit="contain",
                elem_classes=["queue-gallery"]  # Add class for custom styling
            )

            # Load queue on startup and page refresh
            block.load(
                fn=update_queue_display,
                inputs=None,
                outputs=[queue_display],
                queue=False
            )

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache]
    start_button.click(
        fn=process, 
        inputs=ips, 
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, queue_button, queue_display]
    )
    end_button.click(
        fn=end_process,
        outputs=[queue_display, queue_button]
    )
    queue_button.click(
        fn=add_to_queue_handler,
        inputs=[input_image, prompt, total_second_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs],
        outputs=[queue_display, queue_button]
    )

    # Add JavaScript for delete button functionality
    block.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        function setupDeleteButtons() {
            document.querySelectorAll('.delete-button').forEach(button => {
                button.onclick = function() {
                    const jobId = this.getAttribute('data-job-id');
                    fetch('/delete_job/' + jobId, {
                        method: 'POST'
                    }).then(response => {
                        if (response.ok) {
                            window.location.reload();
                        }
                    });
                };
            });
        }
        setupDeleteButtons();
        """
    )
    
    # Add route for delete job
    @block.app.post("/delete_job/{job_id}")
    def delete_job_route(job_id: str):
        return delete_job(job_id)

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
