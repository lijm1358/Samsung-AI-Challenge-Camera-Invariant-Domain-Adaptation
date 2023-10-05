from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
from torch import nn


class SegFormer(nn.Module):
    def __init__(self, num_classes, pretrained: bool=True):
        super(SegFormer, self).__init__()
        
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
        if num_classes != 19:
            self.model.decode_head.classifier = nn.Conv2d(768, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x, mask=None):
        # inputs = self.processor(images=x, return_tensors="pt")
        if mask is None:
            outputs = self.model(x)
            return outputs.logits
        else:
            outputs = self.model(x, mask)
            return outputs.loss, outputs.logits  # shape (batch_size, num_labels, height/4, width/4)