
import numpy as np
import torch

# CUDA extension
import _raytracing as _backend

def _cuda_device(*tensors, device=None):
    devices = [tensor.device for tensor in tensors if torch.is_tensor(tensor) and tensor.is_cuda]
    assert all(device == devices[0] for device in devices), "CUDA tensors must be on the same device"
    if device is not None:
        device = torch.device(device)
        assert device.type == "cuda" and device.index is not None, device
        if devices:
            assert devices[0] == device, f"CUDA tensors must be on {device}, got {devices[0]}"
        return device
    assert devices, "CPU-only RayTracer construction requires an explicit CUDA device"
    return devices[0]

def _to_device(tensor, device, dtype):
    assert torch.is_tensor(tensor)
    assert not tensor.is_cuda or tensor.device == device
    if not tensor.is_cuda:
        tensor = tensor.to(device)
    return tensor.to(dtype=dtype).contiguous()

class RayTracer():
    def __init__(self, vertices, triangles, device=None):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]

        self.device = _cuda_device(vertices, triangles, device=device)
        if torch.is_tensor(vertices): vertices = vertices.cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.cpu().numpy()

        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        # implementation
        with torch.cuda.device(self.device):
            self.impl = _backend.create_raytracer(vertices, triangles)

    def trace(self, rays_o, rays_d, inplace=False):
        # rays_o: torch.Tensor, cuda, float, [N, 3]
        # rays_d: torch.Tensor, cuda, float, [N, 3]
        # inplace: write positions to rays_o, face_normals to rays_d

        device = _cuda_device(rays_o, rays_d, device=self.device)
        rays_o = _to_device(rays_o, device, torch.float32)
        rays_d = _to_device(rays_d, device, torch.float32)

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        if not inplace:
            # allocate
            positions = torch.empty_like(rays_o)
            face_normals = torch.empty_like(rays_d)
        else:
            positions = rays_o
            face_normals = rays_d

        depth = torch.empty_like(rays_o[:, 0])
        
        # inplace write intersections back to rays_o
        with torch.cuda.device(device):
            self.impl.trace(rays_o, rays_d, positions, face_normals, depth) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_normals = face_normals.view(*prefix, 3)
        depth = depth.view(*prefix)

        return positions, face_normals, depth
