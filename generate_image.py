import os
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

class PromptOptimizer:
    """Optimizes prompts for better cross-modal coherence"""
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    def enhance_prompt(self, base_prompt: str) -> str:
        """Add details and modifiers to improve prompt clarity"""
        enhancements = {
            'quality': ['high quality', 'detailed', 'professional', '4k', 'sharp focus'],
            'lighting': ['well-lit', 'studio lighting', 'dramatic lighting'],
            'composition': ['professional photography', 'award winning', 'centered composition']
        }
        
        # Add quality enhancement
        enhanced = f"{np.random.choice(enhancements['quality'])}, {base_prompt}"
        # Add lighting if not present
        if not any(light in base_prompt.lower() for light in enhancements['lighting']):
            enhanced = f"{enhanced}, {np.random.choice(enhancements['lighting'])}"
        # Add composition if not present
        if not any(comp in base_prompt.lower() for comp in enhancements['composition']):
            enhanced = f"{enhanced}, {np.random.choice(enhancements['composition'])}"
            
        return enhanced

class CoherenceOptimizer:
    """Optimizes image generation for cross-modal coherence"""
    def __init__(self, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        
        # Initialize models
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        self.prompt_optimizer = PromptOptimizer()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
    def compute_clip_score(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP similarity score between image and prompt"""
        with torch.no_grad():
            inputs = self.processor(
                images=image,
                text=[prompt],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            outputs = self.clip_model(**inputs)
            similarity = torch.nn.functional.cosine_similarity(
                outputs.image_embeds,
                outputs.text_embeds
            )
            return similarity.item()
    
    def analyze_components(self, image: Image.Image, prompt: str) -> Dict[str, float]:
        """Analyze different aspects of the image-text alignment"""
        aspects = {
            'objects': self.compute_clip_score(image, f"objects: {prompt}"),
            'style': self.compute_clip_score(image, f"artistic style of {prompt}"),
            'composition': self.compute_clip_score(image, f"composition of {prompt}"),
            'lighting': self.compute_clip_score(image, f"lighting in {prompt}"),
            'detail': self.compute_clip_score(image, f"details in {prompt}")
        }
        return aspects
    
    def generate_optimal_image(
        self,
        prompt: str,
        num_iterations: int = 3,
        num_samples: int = 2,
        guidance_range: Tuple[float, float] = (7.0, 9.0)
    ) -> Tuple[Image.Image, Dict[str, float]]:
        """Generate image with optimized cross-modal coherence"""
        best_score = -1
        best_image = None
        best_metrics = None
        
        print(f"\nOptimizing image generation for: '{prompt}'")
        print("=" * 50)
        
        # Enhanced prompt
        enhanced_prompt = self.prompt_optimizer.enhance_prompt(prompt)
        print(f"Enhanced prompt: '{enhanced_prompt}'")
        
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # Try different guidance scales
            for sample in range(num_samples):
                # Dynamic guidance scale
                guidance_scale = np.random.uniform(*guidance_range)
                guidance_scale *= (1 + iteration * 0.1)  # Gradually increase
                
                print(f"Sample {sample + 1}, Guidance: {guidance_scale:.2f}")
                
                # Generate image
                image = self.pipe(
                    enhanced_prompt,
                    num_inference_steps=50,
                    guidance_scale=guidance_scale,
                    negative_prompt="blurry, bad quality, distorted, deformed, ugly, bad anatomy"
                ).images[0]
                
                # Compute comprehensive metrics
                metrics = self.analyze_components(image, prompt)
                current_score = np.mean(list(metrics.values()))
                
                print(f"Score: {current_score:.4f}")
                
                if current_score > best_score:
                    best_score = current_score
                    best_image = image
                    best_metrics = metrics
                    print("→ New best score!")
        
        print("\nOptimization complete!")
        print(f"Best overall score: {best_score:.4f}")
        print("\nComponent scores:")
        for component, score in best_metrics.items():
            print(f"- {component}: {score:.4f}")
        
        return best_image, best_metrics

def main():
    parser = argparse.ArgumentParser(description="Generate images with optimized cross-modal coherence")
    parser.add_argument("--prompt", type=str, default="a photograph of a car",
                      help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output2.png",
                      help="Output file path")
    parser.add_argument("--iterations", type=int, default=3,
                      help="Number of optimization iterations")
    parser.add_argument("--samples", type=int, default=2,
                      help="Number of samples per iteration")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = CoherenceOptimizer()
    
    # Generate optimized image
    image, metrics = optimizer.generate_optimal_image(
        args.prompt,
        num_iterations=args.iterations,
        num_samples=args.samples
    )
    
    # Save image and metrics
    image.save(args.output)
    metrics_file = os.path.splitext(args.output)[0] + "_metrics.txt"
    
    with open(metrics_file, "w") as f:
        f.write(f"Prompt: {args.prompt}\n\n")
        f.write("Component Scores:\n")
        for component, score in metrics.items():
            f.write(f"{component}: {score:.4f}\n")
    
    print(f"\nImage saved as: {args.output}")
    print(f"Metrics saved as: {metrics_file}")

if __name__ == "__main__":
    main()