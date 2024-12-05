import torch
import intel_extension_for_pytorch as ipex



if torch.xpu.is_available():
    device = torch.device("xpu")
    print("Using Intel GPU")
else:
    device = torch.device("cpu")
    print("Falling back to CPU")


# platforms = cl.get_platforms()
# for platform in platforms:
#     print(f"Platform: {platform.name}")
#     for device in platform.get_devices():
#         print(f"  Device: {device.name}")
