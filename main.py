import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import sys

# Add pipeline directory to path
pipeline_dir = os.path.join(os.path.dirname(__file__), 'pipeline')
sys.path.insert(0, pipeline_dir)

from pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    """Retrieve latents from VAE encoder output"""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_image(image_path: str) -> Image.Image:
    """Load image from path"""
    return Image.open(image_path).convert("RGB")


def apply_mask_to_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply mask to image - mask out the masked regions"""
    image_np = np.array(image)
    mask_np = np.array(mask.convert("L"))
    
    # Normalize mask to 0-1
    mask_np = mask_np.astype(np.float32) / 255.0
    
    # Expand mask to 3 channels
    mask_3d = np.stack([mask_np] * 3, axis=-1)
    
    # Apply mask: where mask is 1 (white), set to 0 (black)
    masked_image = image_np * (1 - mask_3d)
    
    return Image.fromarray(masked_image.astype(np.uint8))


def encode_image_to_latents(pipeline, image: Image.Image, device: str) -> torch.Tensor:
    """Encode image to latents using VAE"""
    with torch.no_grad():
        # Preprocess image
        image_tensor = pipeline.image_processor.preprocess(image).to(device=device, dtype=pipeline.vae.dtype)
        
        # Encode to latents
        latents = retrieve_latents(
            pipeline.vae.encode(image_tensor), 
            generator=None, 
            sample_mode="argmax"
        )
        
        # Clean up
        del image_tensor
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Apply scaling and shift
        latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    
    return latents


def decode_latents_to_image(pipeline, latents: torch.Tensor, height: int, width: int) -> Image.Image:
    """Decode latents to image using VAE"""
    with torch.no_grad():
        # Unpack latents
        unpacked_latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        
        # Denormalize
        denormalized_latents = (unpacked_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        
        # Clean up
        del unpacked_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decode
        image = pipeline.vae.decode(denormalized_latents, return_dict=False)[0]
        
        # Clean up
        del denormalized_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Postprocess
        image = pipeline.image_processor.postprocess(image, output_type="pil")
    
    return image[0] if isinstance(image, list) else image


def invert_image_to_noise(
    pipeline: FluxPipeline,
    image_latents: torch.Tensor,
    prompt: str,
    device: str,
    latent_height: int,
    latent_width: int,
    num_inference_steps: int = 28,
) -> torch.Tensor:
    """Invert image latents to noise using inversion process"""
    with torch.no_grad():
        # Prepare timesteps (reverse order for inversion)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(pipeline.scheduler.config, "use_flow_sigmas") and pipeline.scheduler.config.use_flow_sigmas:
            sigmas = None
        
        image_seq_len = image_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipeline.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        
        # Reverse timesteps for inversion (from clean to noisy)
        timesteps = timesteps.flip(0)
        
        # Encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
        )
        
        # Prepare latent image ids
        batch_size = 1
        latent_image_ids = pipeline._prepare_latent_image_ids(
            batch_size, 
            latent_height // 2, 
            latent_width // 2, 
            device, 
            image_latents.dtype
        )
        
        # Prepare guidance
        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([1], 3.5, device=device, dtype=torch.float32)
            guidance = guidance.expand(image_latents.shape[0])
        else:
            guidance = None
        
        # Start from clean image latents
        current_latents = image_latents.clone()
        
        # Generate random noise for inversion
        generator = torch.Generator(device=device)
        generator.manual_seed(0)  # Fixed seed for reproducibility
        noise = torch.randn(
            current_latents.shape,
            generator=generator,
            device=device,
            dtype=current_latents.dtype
        )
        
        # Inversion loop: go from clean image to noise using Flow Matching
        # For Flow Matching, we use scale_noise to progressively add noise
        pipeline.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            # For inversion, we add noise progressively
            # Flow matching: x_t = (1 - alpha_t) * x_0 + alpha_t * noise
            # where alpha_t increases from 0 to 1
            
            # Normalize timestep (Flow Matching uses 0-1 range)
            t_normalized = float(t) / 1000.0 if t > 0 else 0.0
            
            # Use scale_noise if available to properly scale the noise
            if hasattr(pipeline.scheduler, 'scale_noise') and i < len(timesteps) - 1:
                # Get the next timestep for scaling
                next_t = timesteps[i + 1] if i < len(timesteps) - 1 else t
                # Scale noise according to the scheduler
                scaled_noise = pipeline.scheduler.scale_noise(
                    noise, torch.tensor([next_t], device=device), noise
                )
            else:
                scaled_noise = noise
            
            # Interpolate between image and noise based on timestep
            # For inversion: start with image (t=0) and end with noise (t=1)
            current_latents = (1 - t_normalized) * image_latents + t_normalized * scaled_noise
            
            # Clean up
            if i % 5 == 0 and device == "cuda":  # Periodically clear cache
                torch.cuda.empty_cache()
        
        # Clear cache after inversion
        if device == "cuda":
            torch.cuda.empty_cache()
            # Clear transformer cache if available
            if hasattr(pipeline.transformer, 'clear_cache'):
                pipeline.transformer.clear_cache()
    
    # Return the inverted noise
    return current_latents


def get_inverted_noise(
    pipeline: FluxPipeline,
    image: Image.Image,
    prompt: str,
    device: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    """Get inverted noise from image - no denoising, just inversion"""
    if height is None:
        height = image.height
    if width is None:
        width = image.width
    
    with torch.no_grad():
        # Encode image to latents
        image_latents = encode_image_to_latents(pipeline, image, device)
        
        # Prepare latents for pipeline (pack them)
        batch_size = 1
        num_channels_latents = pipeline.transformer.config.in_channels // 4
        
        # Adjust height and width for VAE scale factor
        latent_height = 2 * (int(height) // (pipeline.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (pipeline.vae_scale_factor * 2))
        
        # Resize latents if needed
        if image_latents.shape[2] != latent_height or image_latents.shape[3] != latent_width:
            image_latents = torch.nn.functional.interpolate(
                image_latents, 
                size=(latent_height, latent_width), 
                mode="bilinear", 
                align_corners=False
            )
        
        # Pack latents
        packed_latents = pipeline._pack_latents(
            image_latents, 
            batch_size, 
            num_channels_latents, 
            latent_height, 
            latent_width
        )
        
        # Clean up intermediate tensors
        del image_latents
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Invert image to noise (this gives us the inverted noise)
        inverted_noise = invert_image_to_noise(
            pipeline,
            packed_latents,
            prompt,
            device,
            latent_height,
            latent_width,
            num_inference_steps=28,
        )
        
        # Clean up
        del packed_latents
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return inverted_noise


def invert_image_to_noise_with_steps(
    pipeline: FluxPipeline,
    image_latents: torch.Tensor,
    prompt: str,
    device: str,
    latent_height: int,
    latent_width: int,
    num_inference_steps: int = 28,
) -> list:
    """Invert image latents to noise and return all intermediate steps"""
    step_latents = []
    
    with torch.no_grad():
        # Prepare timesteps (reverse order for inversion)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(pipeline.scheduler.config, "use_flow_sigmas") and pipeline.scheduler.config.use_flow_sigmas:
            sigmas = None
        
        image_seq_len = image_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipeline.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        
        # Reverse timesteps for inversion (from clean to noisy)
        timesteps = timesteps.flip(0)
        
        # Encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
        )
        
        # Prepare latent image ids
        batch_size = 1
        latent_image_ids = pipeline._prepare_latent_image_ids(
            batch_size, 
            latent_height // 2, 
            latent_width // 2, 
            device, 
            image_latents.dtype
        )
        
        # Prepare guidance
        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([1], 3.5, device=device, dtype=torch.float32)
            guidance = guidance.expand(image_latents.shape[0])
        else:
            guidance = None
        
        # Start from clean image latents
        current_latents = image_latents.clone()
        
        # Save initial step (clean image)
        step_latents.append(current_latents.clone().cpu())
        
        # Generate random noise for inversion
        generator = torch.Generator(device=device)
        generator.manual_seed(0)  # Fixed seed for reproducibility
        noise = torch.randn(
            current_latents.shape,
            generator=generator,
            device=device,
            dtype=current_latents.dtype
        )
        
        # Inversion loop: go from clean image to noise using Flow Matching
        pipeline.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            # For inversion, we add noise progressively
            # Flow matching: x_t = (1 - alpha_t) * x_0 + alpha_t * noise
            # where alpha_t increases from 0 to 1
            
            # Normalize timestep (Flow Matching uses 0-1 range)
            t_normalized = float(t) / 1000.0 if t > 0 else 0.0
            
            # Use scale_noise if available to properly scale the noise
            if hasattr(pipeline.scheduler, 'scale_noise') and i < len(timesteps) - 1:
                # Get the next timestep for scaling
                next_t = timesteps[i + 1] if i < len(timesteps) - 1 else t
                # Scale noise according to the scheduler
                scaled_noise = pipeline.scheduler.scale_noise(
                    noise, torch.tensor([next_t], device=device), noise
                )
            else:
                scaled_noise = noise
            
            # Interpolate between image and noise based on timestep
            # For inversion: start with image (t=0) and end with noise (t=1)
            current_latents = (1 - t_normalized) * image_latents + t_normalized * scaled_noise
            
            # Save this step
            step_latents.append(current_latents.clone().cpu())
            
            # Clean up
            if i % 5 == 0 and device == "cuda":  # Periodically clear cache
                torch.cuda.empty_cache()
        
        # Clear cache after inversion
        if device == "cuda":
            torch.cuda.empty_cache()
            # Clear transformer cache if available
            if hasattr(pipeline.transformer, 'clear_cache'):
                pipeline.transformer.clear_cache()
    
    return step_latents


def process_images_with_inversion_steps(
    pipeline: FluxPipeline,
    image_paths: list,
    output_base_dir: str,
    device: str,
    prompt: str = "a person",
    num_inference_steps: int = 28,
):
    """
    Process a group of images and save inversion steps for each image.
    
    Args:
        pipeline: FluxPipeline instance
        image_paths: List of image file paths to process
        output_base_dir: Base directory to save output steps
        device: Device to run on
        prompt: Prompt to use for inversion
        num_inference_steps: Number of inversion steps
    """
    # Create output directory structure
    steps_output_dir = os.path.join(output_base_dir, "inversion_steps")
    os.makedirs(steps_output_dir, exist_ok=True)
    
    print(f"Processing {len(image_paths)} images with inversion steps...")
    
    for idx, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Warning: Image not found {image_path}, skipping...")
            continue
        
        # Get filename without extension
        filename = Path(image_path).stem
        
        # Create output directory for this image
        image_steps_dir = os.path.join(steps_output_dir, filename)
        os.makedirs(image_steps_dir, exist_ok=True)
        
        print(f"Processing {filename} ({idx+1}/{len(image_paths)})...")
        
        try:
            # Load image
            image = load_image(image_path)
            height, width = image.height, image.width
            
            with torch.no_grad():
                # Encode image to latents
                image_latents = encode_image_to_latents(pipeline, image, device)
                
                # Prepare latents for pipeline (pack them)
                batch_size = 1
                num_channels_latents = pipeline.transformer.config.in_channels // 4
                
                # Adjust height and width for VAE scale factor
                latent_height = 2 * (int(height) // (pipeline.vae_scale_factor * 2))
                latent_width = 2 * (int(width) // (pipeline.vae_scale_factor * 2))
                
                # Resize latents if needed
                if image_latents.shape[2] != latent_height or image_latents.shape[3] != latent_width:
                    image_latents = torch.nn.functional.interpolate(
                        image_latents, 
                        size=(latent_height, latent_width), 
                        mode="bilinear", 
                        align_corners=False
                    )
                
                # Pack latents
                packed_latents = pipeline._pack_latents(
                    image_latents, 
                    batch_size, 
                    num_channels_latents, 
                    latent_height, 
                    latent_width
                )
                
                # Clean up
                del image_latents
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                # Get inversion steps
                step_latents = invert_image_to_noise_with_steps(
                    pipeline,
                    packed_latents,
                    prompt,
                    device,
                    latent_height,
                    latent_width,
                    num_inference_steps=num_inference_steps,
                )
                
                # Clean up
                del packed_latents
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            # Decode and save each step
            print(f"  Saving {len(step_latents)} inversion steps...")
            for step_idx, step_latent in enumerate(step_latents):
                # Move back to device for decoding
                step_latent_gpu = step_latent.to(device)
                
                with torch.no_grad():
                    # Decode latents to image
                    step_image = decode_latents_to_image(
                        pipeline,
                        step_latent_gpu,
                        height,
                        width,
                    )
                
                # Save step image
                step_filename = f"step_{step_idx:04d}.png"
                step_path = os.path.join(image_steps_dir, step_filename)
                step_image.save(step_path)
                
                # Clean up
                del step_latent_gpu, step_image
                if device == "cuda" and step_idx % 5 == 0:
                    torch.cuda.empty_cache()
            
            print(f"  Saved all steps to {image_steps_dir}")
            
            # Clean up
            del step_latents, image
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            if device == "cuda":
                torch.cuda.empty_cache()
    
    print(f"Completed processing {len(image_paths)} images. Results saved to {steps_output_dir}")


def process_single_image(
    pipeline: FluxPipeline,
    image_path: str,
    mask_path: str,
    label: str,
    output_img_ori_dir: str,
    output_img_mask_ori_dir: str,
    output_img_with_dir: str,
    device: str,
):
    """Process a single image and generate three inverted noise images"""
    # Load image and mask
    image = load_image(image_path)
    mask = load_image(mask_path)
    
    # Get filename without extension
    filename = Path(image_path).stem
    
    print(f"Processing {filename}...")
    
    # 1. Original image + prompt "a person" → img_ori
    print(f"  Generating inverted noise for original image (prompt: 'a person')...")
    try:
        with torch.no_grad():
            inverted_noise_ori = get_inverted_noise(
                pipeline,
                image,
                prompt="a person",
                device=device,
                height=image.height,
                width=image.width,
            )
            
            # Decode inverted noise latents to image
            noise_image_ori = decode_latents_to_image(
                pipeline,
                inverted_noise_ori,
                image.height,
                image.width,
            )
        
        # Save to img_ori
        output_path_ori = os.path.join(output_img_ori_dir, f"{filename}.png")
        noise_image_ori.save(output_path_ori)
        print(f"  Saved to {output_path_ori}")
        
        # Clean up
        del inverted_noise_ori, noise_image_ori
        if device == "cuda":
            torch.cuda.empty_cache()
            # Clear transformer cache if available
            if hasattr(pipeline.transformer, 'clear_cache'):
                pipeline.transformer.clear_cache()
    except Exception as e:
        print(f"  Error processing original image with 'a person': {e}")
        import traceback
        traceback.print_exc()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # 2. Masked image + prompt "a person" → img_mask_ori
    print(f"  Generating inverted noise for masked image (prompt: 'a person')...")
    masked_image = apply_mask_to_image(image, mask)
    
    try:
        with torch.no_grad():
            inverted_noise_mask_ori = get_inverted_noise(
                pipeline,
                masked_image,
                prompt="a person",
                device=device,
                height=image.height,
                width=image.width,
            )
            
            # Decode inverted noise latents to image
            noise_image_mask_ori = decode_latents_to_image(
                pipeline,
                inverted_noise_mask_ori,
                image.height,
                image.width,
            )
        
        # Save to img_mask_ori
        output_path_mask_ori = os.path.join(output_img_mask_ori_dir, f"{filename}.png")
        noise_image_mask_ori.save(output_path_mask_ori)
        print(f"  Saved to {output_path_mask_ori}")
        
        # Clean up
        del inverted_noise_mask_ori, noise_image_mask_ori, masked_image
        if device == "cuda":
            torch.cuda.empty_cache()
            # Clear transformer cache if available
            if hasattr(pipeline.transformer, 'clear_cache'):
                pipeline.transformer.clear_cache()
    except Exception as e:
        print(f"  Error processing masked image with 'a person': {e}")
        import traceback
        traceback.print_exc()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # 3. Original image + prompt "a person with {label}" → img_with
    print(f"  Generating inverted noise for original image (prompt: 'a person with {label}')...")
    prompt_with = f"a person with {label}"
    
    try:
        with torch.no_grad():
            inverted_noise_with = get_inverted_noise(
                pipeline,
                image,
                prompt=prompt_with,
                device=device,
                height=image.height,
                width=image.width,
            )
            
            # Decode inverted noise latents to image
            noise_image_with = decode_latents_to_image(
                pipeline,
                inverted_noise_with,
                image.height,
                image.width,
            )
        
        # Save to img_with
        output_path_with = os.path.join(output_img_with_dir, f"{filename}.png")
        noise_image_with.save(output_path_with)
        print(f"  Saved to {output_path_with}")
        
        # Clean up
        del inverted_noise_with, noise_image_with
        if device == "cuda":
            torch.cuda.empty_cache()
            # Clear transformer cache if available
            if hasattr(pipeline.transformer, 'clear_cache'):
                pipeline.transformer.clear_cache()
    except Exception as e:
        print(f"  Error processing original image with 'a person with {label}': {e}")
        import traceback
        traceback.print_exc()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Final cleanup
    del image, mask
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    # Configuration
    data_dir = "/root/workspace/train/ztrain/pre_test/data"
    image_dir = os.path.join(data_dir, "image")
    mask_dir = os.path.join(data_dir, "mask")
    labels_file = os.path.join(data_dir, "labels.json")
    output_base_dir = os.path.join(data_dir, "output")
    output_img_ori_dir = os.path.join(output_base_dir, "img_ori")
    output_img_mask_ori_dir = os.path.join(output_base_dir, "img_mask_ori")
    output_img_with_dir = os.path.join(output_base_dir, "img_with")
    
    # Create output directories
    os.makedirs(output_img_ori_dir, exist_ok=True)
    os.makedirs(output_img_mask_ori_dir, exist_ok=True)
    os.makedirs(output_img_with_dir, exist_ok=True)
    
    # Load labels
    print("Loading labels...")
    with open(labels_file, 'r') as f:
        labels_dict = json.load(f)
    print(f"Loaded {len(labels_dict)} labels")
    
    # Load pipeline
    print("Loading pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/data0/FLUX.1-dev"  # You may need to change this
    
    try:
        pipeline = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        print(f"Pipeline loaded on {device}")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Please make sure the model path is correct and the pipeline is available")
        return
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file)
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {image_file}, skipping...")
            continue
        
        # Get label
        label = labels_dict.get(image_file, "unknown")
        
        # Process image
        process_single_image(
            pipeline,
            image_path,
            mask_path,
            label,
            output_img_ori_dir,
            output_img_mask_ori_dir,
            output_img_with_dir,
            device,
        )
        
        # Aggressive memory cleanup after each image
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Print memory usage for monitoring
            if idx % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    print("Processing complete!")


def main_with_steps():
    """Main function for processing images with inversion steps saved"""
    # Configuration
    data_dir = "/root/workspace/train/ztrain/pre_test/data"
    image_dir = os.path.join(data_dir, "image")
    output_base_dir = os.path.join(data_dir, "output")
    
    # Load pipeline
    print("Loading pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/data0/FLUX.1-dev"  # You may need to change this
    
    try:
        pipeline = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        print(f"Pipeline loaded on {device}")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Please make sure the model path is correct and the pipeline is available")
        return
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} images to process")
    
    # Prepare image paths
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    
    # Process images with inversion steps
    process_images_with_inversion_steps(
        pipeline=pipeline,
        image_paths=image_paths,
        output_base_dir=output_base_dir,
        device=device,
        prompt="a person",
        num_inference_steps=28,
    )
    
    print("Processing complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images with FLUX inversion')
    parser.add_argument('--mode', type=str, default='batch', 
                       choices=['batch', 'steps'],
                       help='Processing mode: batch (default) or steps (save each inversion step)')
    args = parser.parse_args()
    
    if args.mode == 'steps':
        main_with_steps()
    else:
        main()

