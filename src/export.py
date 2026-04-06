"""Convert merged model to GGUF format for Ollama."""
import os
import subprocess
import sys

import yaml


def run_export(config_path: str = "configs/training_config.yaml"):
    """Convert the merged model to GGUF format using llama.cpp."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    merged_dir = cfg["export"]["merged_dir"]
    gguf_dir = cfg["export"]["gguf_dir"]
    quantization = cfg["export"]["gguf_quantization"]
    model_name = cfg["export"]["ollama_model_name"]

    if not os.path.exists(merged_dir):
        print(f"ERROR: No merged model found at {merged_dir}")
        print("Run 'python cli.py merge' first.")
        return

    print("=== Narrative Forge — Export to GGUF ===\n")

    os.makedirs(gguf_dir, exist_ok=True)

    output_path = os.path.join(gguf_dir, f"{model_name}.gguf")

    llama_cpp_convert = _find_llama_cpp_convert()

    if llama_cpp_convert:
        print("Using llama.cpp for conversion...")
        _convert_with_llama_cpp(llama_cpp_convert, merged_dir, gguf_dir, model_name, quantization)
    else:
        print("Using llama-cpp-python for conversion...")
        _convert_with_python(merged_dir, output_path, quantization)

    if os.path.exists(output_path):
        size_gb = os.path.getsize(output_path) / 1024**3
        print(f"\nExport complete! GGUF model saved to {output_path} ({size_gb:.1f} GB)")
        print("Next step: python cli.py register")
    else:
        print("\nERROR: GGUF file was not created. Check the output above for errors.")


def _find_llama_cpp_convert():
    """Check if llama.cpp convert script is available."""
    try:
        result = subprocess.run(
            ["python", "-c", "import llama_cpp; print(llama_cpp.__file__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return "llama_cpp"
    except FileNotFoundError:
        pass
    return None


def _convert_with_python(merged_dir: str, output_path: str, quantization: str):
    """Convert using the transformers/huggingface_hub GGUF export."""
    try:
        subprocess.run(
            [
                sys.executable, "-m", "llama_cpp.llama_cpp",
                "--convert", merged_dir,
                "--outfile", output_path,
                "--outtype", quantization,
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing llama.cpp conversion tools...")
        llama_cpp_dir = os.path.join("models", "llama.cpp")
        if not os.path.exists(llama_cpp_dir):
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r",
                 os.path.join(llama_cpp_dir, "requirements.txt")],
                check=True,
            )

        f16_path = output_path.replace(".gguf", "-f16.gguf")
        subprocess.run(
            [sys.executable, os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
             merged_dir, "--outfile", f16_path, "--outtype", "f16"],
            check=True,
        )

        quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
        if os.path.exists(quantize_bin):
            subprocess.run([quantize_bin, f16_path, output_path, quantization], check=True)
        else:
            os.rename(f16_path, output_path)
            print(f"NOTE: Could not quantize to {quantization}. Using f16 instead.")
            print("For quantization, build llama.cpp: cd models/llama.cpp && cmake -B build && cmake --build build")


def _convert_with_llama_cpp(converter, merged_dir, gguf_dir, model_name, quantization):
    """Convert using llama.cpp tools."""
    output_path = os.path.join(gguf_dir, f"{model_name}.gguf")
    _convert_with_python(merged_dir, output_path, quantization)
