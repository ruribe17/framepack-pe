# diffusers_helper/memory.py - CPU-only version, no CUDA dependency
import torch

# --- Dispositivos ---
cpu = torch.device('cpu')
gpu = torch.device('cpu')  # Alias para mantener compatibilidad con código original

# Lista de modelos completamente cargados (no usado en CPU, pero mantenemos API)
gpu_complete_modules = []

# --- DynamicSwapInstaller (no-op en CPU) ---
# Esta clase es para offload en GPU. En CPU, no se usa.
class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        pass

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        pass

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        # No-op: no hay swap en CPU
        pass

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        pass


# --- fake_diffusers_current_device ---
# Solo asegura que algunos buffers estén en el dispositivo
def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
    # No mover capas completas: ya están en CPU


# --- get_cuda_free_memory_gb: Simulación para CPU ---
def get_cuda_free_memory_gb(device=None):
    """
    Simula memoria libre de GPU para forzar 'high_vram = True' en CPU.
    Evita cualquier llamada a torch.cuda.memory_stats.
    """
    print("Running in CPU-only mode: faking high 'VRAM' (60.0 GB) to enable full model loading.")
    return 60.0  # Simula más de 60 GB libres → activa high_vram = True


# --- Funciones de gestión de memoria (no-op en CPU) ---
def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    """
    En CPU, simplemente mueve el modelo al dispositivo.
    No hay memoria dinámica de GPU que preservar.
    """
    model.to(target_device)


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    """
    No aplicable en CPU. No hacer nada.
    """
    pass


def unload_complete_models(*args):
    """
    No aplicable. Todos los modelos permanecen en CPU.
    """
    pass


def load_model_as_complete(model, target_device, unload=True):
    """
    Cargar modelo completo en dispositivo (CPU).
    """
    model.to(target_device)
