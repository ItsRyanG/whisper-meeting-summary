#!/usr/bin/env python3
"""
Diagnostic script to check PyTorch XPU support and debug integration issues
"""
import sys
import re
import traceback

def check_pytorch():
    """Basic check for PyTorch and XPU support"""
    try:
        print("Checking for PyTorch...")
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch path: {torch.__file__}")
        
        # Check for XPU
        if hasattr(torch, 'xpu'):
            print("\nXPU module exists")
            print(f"torch.xpu.is_available(): {torch.xpu.is_available()}")
            
            if torch.xpu.is_available():
                print(f"XPU device count: {torch.xpu.device_count()}")
                print(f"XPU device name: {torch.xpu.get_device_name()}")
                print(f"Current XPU device: {torch.xpu.current_device()}")
                print(f"Has set_memory_allocator: {hasattr(torch.xpu, 'set_memory_allocator')}")
                print(f"Has enable_auto_mixed_precision: {hasattr(torch.xpu, 'enable_auto_mixed_precision')}")
                    
        else:
            print("\nXPU module does not exist in this PyTorch build")
        
        # Check PyTorch version to determine compatibility
        try:
            torch_version = torch.__version__
            major_minor = re.match(r'(\d+\.\d+)', torch_version)
            
            if major_minor:
                version = float(major_minor.group(1))
                print(f"\nParsed version: {version}")
                
                if version >= 2.6:
                    print("This is PyTorch 2.6+, should use built-in XPU support")
                    print("intel_extension_for_pytorch should NOT be imported")
                    
                    try:
                        # Try to import IPEX without importing it directly in code
                        ipex_module = __import__('intel_extension_for_pytorch')
                        print("WARNING: intel_extension_for_pytorch is installed")
                        print("This can cause conflicts with PyTorch 2.6+ built-in XPU support")
                    except ImportError:
                        print("Good: intel_extension_for_pytorch is not installed")
                else:
                    print("This is PyTorch < 2.6, might need intel_extension_for_pytorch")
                    
                    try:
                        # Try to import IPEX without importing it directly in code
                        ipex_module = __import__('intel_extension_for_pytorch')
                        print("Good: intel_extension_for_pytorch is installed")
                    except ImportError:
                        print("WARNING: intel_extension_for_pytorch is not installed")
                        print("This might limit XPU functionality")
        except Exception as e:
            print(f"Error checking PyTorch version compatibility: {e}")
    
    except ImportError:
        print("PyTorch is not installed")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    check_pytorch()