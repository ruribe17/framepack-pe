# demo_gradio_cpu.py - FramePack en CPU con FP16 storage, MKL, oneDNN

# === CONFIGURACIÓN DE RENDIMIENTO CPU ===
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"  # Cambiar a AVX2 si no soporta AVX512

import torch
import numpy as np

def cpu_supports_f16c():
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("flags"):
                if "f16c" in line.split():
                    return True
    return False

# Verificar F16C (soporte de hardware para half precision)
cap = torch.backends.cpu.get_cpu_capability()
print(f"Detected CPU capabilities: {cap}")
if not cpu_supports_f16c():
    print("⚠️  Advertencia: CPU no soporta F16C. Rendimiento en FP16 puede ser bajo.")
else:
    print("✅ F16C soportado: conversión FP16/FP32 eficiente")

# Activar multi-threading
torch.set_num_threads(32)
torch.set_num_interop_threads(4)
torch.set_grad_enabled(False)

# === MONKEY PATCH CORREGIDO: HunyuanVideoCausalConv3d para CPU + FP16 storage + FP32 compute ===
def patch_hunyuan_causal_conv3d_for_cpu():
    import torch.nn.functional as F

    try:
        # ✅ Nombre correcto: HunyuanVideoCausalConv3d
        from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import HunyuanVideoCausalConv3d
    except ImportError:
        try:
            # Si usas diffusers local o modificado
            from autoencoder_kl_hunyuan_video import HunyuanVideoCausalConv3d
        except:
            print("⚠️ No se encontró HunyuanVideoCausalConv3d. Asegúrate de que el módulo esté disponible.")
            return

    if hasattr(HunyuanVideoCausalConv3d, '_patched_for_cpu'):
        return  # Ya fue parchado

    # Guardar el forward original
    original_forward = HunyuanVideoCausalConv3d.forward

    def cpu_compatible_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Guardar el dtype original (probablemente torch.float16)
        input_dtype = hidden_states.dtype

        # Convertir a float32 para operaciones críticas en CPU
        hidden_states = hidden_states.to(torch.float32)

        # Aplicar padding con modo 'replicate' → ahora seguro en float32
        # Nota: self.time_causal_padding ya está definido (e.g., (k//2, k//2, k//2, k//2, k-1, 0))
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)

        # Aplicar convolución
        hidden_states = self.conv(hidden_states)

        # Devolver al tipo original (float16)
        return hidden_states.to(input_dtype)

    # Reemplazar el método forward
    HunyuanVideoCausalConv3d.forward = cpu_compatible_forward
    HunyuanVideoCausalConv3d._patched_for_cpu = True
    print("✅ Monkey patch aplicado: HunyuanVideoCausalConv3d ahora es compatible con CPU + FP16 storage")

patch_hunyuan_causal_conv3d_for_cpu()  # ← ¡Aquí!

# Mostrar estado de MKL y oneDNN
print(f"✅ MKL está activo: {torch.backends.mkl.is_available()}")
print(f"✅ oneDNN (MKLDNN) está activo: {torch.backends.mkldnn.is_available()}")

# === PATH y LOGIN HuggingFace ===
from diffusers_helper.hf_login import login
import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import einops
import safetensors.torch as sf
from PIL import Image
import argparse
import math
import traceback

# === MODELOS ===
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# === ARGUMENTOS ===
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

print(args)

# Simular memoria GPU alta para activar high_vram = True
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Simulated VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# === CARGA DE MODELOS en FP16 (almacenamiento), con map_location=cpu ===
dtype_storage = torch.float16  # Almacenamiento en FP16
dtype_compute = torch.float32  # Cálculo en FP32 solo cuando sea necesario

text_encoder = LlamaModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='text_encoder',
    torch_dtype=dtype_storage
).eval().requires_grad_(False)

text_encoder_2 = CLIPTextModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='text_encoder_2',
    torch_dtype=dtype_storage
).eval().requires_grad_(False)

tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')

vae = AutoencoderKLHunyuanVideo.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='vae',
    torch_dtype=dtype_storage
).eval().requires_grad_(False)

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl",
    subfolder='image_encoder',
    torch_dtype=dtype_storage
).eval().requires_grad_(False)

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
    'lllyasviel/FramePackI2V_HY',
    torch_dtype=dtype_storage
).eval().requires_grad_(False)


# === CONFIGURACIÓN DE PRECISIÓN ===
# Asegurar que LLaMA use atención segura en CPU
for block in text_encoder.layers:
    block.self_attn._use_sdpa = True  # Usa SDP pero de forma controlada
    # O, si falla:
    # block.self_attn._use_sdpa = False  # Fuerza atención manual

transformer.to(dtype_storage)
vae.to(dtype_storage)
image_encoder.to(dtype_storage)
text_encoder.to(dtype_storage)
text_encoder_2.to(dtype_storage)

# Activar alta calidad en salida final si es necesario
transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# Slicing y tiling no necesarios en CPU, pero no afecta
if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

# === STREAMING Y SALIDA ===
stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # --- Text encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Convertir a dtype_storage
        llama_vec = llama_vec.to(dtype_storage)
        llama_vec_n = llama_vec_n.to(dtype_storage)
        clip_l_pooler = clip_l_pooler.to(dtype_storage)
        clip_l_pooler_n = clip_l_pooler_n.to(dtype_storage)

        # --- Procesar imagen ---
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None].to(dtype_storage)

        # --- VAE encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        start_latent = vae_encode(input_image_pt, vae).to(dtype_storage)

        # --- CLIP Vision ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(dtype_storage)

        # --- Sampling ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
            dtype=torch.float32  # Histórico en FP32 para estabilidad
        ).to(cpu)
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2).to(dtype_storage)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised'].float()  # Para preview, convertir a FP32
                preview = vae_decode_fake(preview)
                preview = (preview * 255.0).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt()

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds'
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
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=cpu,
                dtype=dtype_storage,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x.to(dtype_storage),
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x.to(dtype_storage),
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break

    except Exception as e:
        traceback.print_exc()
    finally:
        stream.output_queue.push(('end', None))

def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    global stream
    assert input_image is not None, 'No input image!'
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    stream = AsyncStream()
    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf)

    output_filename = None
    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break

def end_process():
    stream.input_queue.push('end')

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]

css = make_progress_bar_css()
with gr.Blocks(css=css).queue() as block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but may affect hands.')
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                seed = gr.Number(label="Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1)
                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1)

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results at <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X)</a></div>')

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)

block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser)
