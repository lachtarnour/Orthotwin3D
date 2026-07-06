import unittest

import torch

from src.datasets.teeth3ds_processed import create_segmentation_dataloader
from src.models.dgcnn import DGCNNSegmentation


class TestDGCNNForward(unittest.TestCase):
    def test_forward_shape_from_dataloader_batch(self) -> None:
        loader = create_segmentation_dataloader("configs/data.yaml", split="train", limit=2)
        batch = next(iter(loader))
        x = batch["x"][:, :1024, :]

        model = DGCNNSegmentation(input_channels=6, num_classes=33, k=20)
        model.eval()

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (2, 1024, 33))
        self.assertEqual(logits.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
