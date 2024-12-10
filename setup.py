import subprocess

packages = [
    'comet-ml',
    'torch',
    'torchvision',
    'numpy',
    'tqdm',
    'einops',
    'torchgeometry',
    'pytorch-msssim',
    'opencv-python',
    'imageio',
    'deblurring_diffusion_pytorch'
]

def install_packages():
    for package in packages:
        try:
            if package == 'torch':
                # 根据你的系统配置调整安装命令
                subprocess.check_call(["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
            else:
                subprocess.check_call(["pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Error installing {package}")

if __name__ == "__main__":
    install_packages()