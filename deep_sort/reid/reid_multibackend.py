"""
Re-Identification Multi-Backend

"""

from pathlib import Path
from collections import OrderedDict, namedtuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from deep_sort.reid.models import build_model, get_model_name
from tools.general import check_version, check_requirements, check_suffix
from tools.load import load_pretrained_weights

class ReIDDetectMultiBackend(nn.Module):
    # ReID models multi-backend class inference on various backends:
    def __init__(self, weights = 'osnet_x0_25_msmt17.pt', device = torch.device('cpu'), fp16 = False):
        
        super().__init__()

        w = weights[0] if isinstance(weights, list) else weights
        self.pt, self.jit, self.onnx, self.engine = self.model_type(w)  # get backend
        self.fp16 = fp16
        self.fp16 &= self.pt or self.jit or self.engine  # FP16
        self.device = device

        # Build transform functions
        self.image_size = (256, 128)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.transforms = []
        self.transforms += [T.Resize(self.image_size)]
        self.transforms += [T.ToTensor()]
        self.transforms += [T.Normalize(mean = self.pixel_mean, std = self.pixel_std)]
        self.preprocess = T.Compose(self.transforms)
        self.to_pil = T.ToPILImage()

        # Build model:
        model_name = get_model_name(w)
        use_gpu = True if str(self.device) != 'cpu' else False
        self.model = build_model(model_name, num_classes = 1, pretrained = not (w and w.is_file()), use_gpu = use_gpu)

        if self.pt:  # PyTorch
            if w and w.is_file() and w.suffix == '.pt':
                load_pretrained_weights(self.model, w)
            self.model.to(device).eval()
            self.model.half() if self.fp16 else self.model.float()
        elif self.jit:
            print(f'Loading {w} for TorchScript inference...')
            self.model = torch.jit.load(w)
            self.model.half() if self.fp16 else self.model.float()
        elif self.onnx:  # ONNX Runtime
            print(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available() and device.type != 'cpu'
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(str(w), providers = providers)
        elif self.engine:  # TensorRT
            print(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                self.model_ = runtime.deserialize_cuda_engine(f.read())
            self.context = self.model_.create_execution_context()
            self.bindings = OrderedDict()
            self.fp16 = False  # default updated below
            dynamic = False
            for index in range(self.model_.num_bindings):
                name = self.model_.get_binding_name(index)
                dtype = trt.nptype(self.model_.get_binding_dtype(index))
                if self.model_.binding_is_input(index):
                    if -1 in tuple(self.model_.get_binding_shape(index)):  # dynamic
                        dynamic = True
                        self.context.set_binding_shape(index, tuple(self.model_.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        self.fp16 = True
                shape = tuple(self.context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            batch_size = self.bindings['images'].shape[0]  # if dynamic, this is instead max batch size 
        else:
            print("Framework not implemented...")
        
    def model_type(self, p = 'path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from deep_sort.reid.reid_export import export_formats
        sf = list(export_formats().Suffix)  # export suffixes
        check_suffix(p, sf)  # checks
        p = Path(p).name  # eliminate trailing separators
        types = [s in Path(p).name for s in sf]

        return types

    def pre_process(self, im_batch):
        # Pre-process image:
        images = []
        for element in im_batch:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)
        images = torch.stack(images, dim = 0)
        images = images.to(self.device)

        return images
    
    def forward(self, im_batch):
        
        # Pre-process batch:
        im_batch = self.pre_process(im_batch)

        # Batch to half:
        if self.fp16 and im_batch.dtype != torch.float16:
           im_batch = im_batch.half()

        # Inference:
        features = []
        if self.pt:
            features = self.model(im_batch)
        elif self.jit:  # TorchScript
            features = self.model(im_batch)
        elif self.onnx:  # ONNX Runtime
            im_batch = im_batch.cpu().numpy()  # torch to numpy
            features = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im_batch})[0]
        elif self.engine:  # TensorRT
            if True and im_batch.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model_.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, im_batch.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im_batch.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert im_batch.shape == s, f"input size {im_batch.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings['output'].data
        else:
            print("Framework not implemented...")

        if isinstance(features, (list, tuple)):
            return self.from_numpy(features[0]) if len(features) == 1 else [self.from_numpy(x) for x in features]
        else:
            return self.from_numpy(features)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz = [(256, 128, 3)]):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb
        if any(warmup_types) and self.device.type != 'cpu':
            im = [np.empty(*imgsz).astype(np.uint8)]  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup