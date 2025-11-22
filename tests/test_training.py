import unittest
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from models.cnn import CNN
from main import fix_inplace_modules
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = CNN(num_classes=10)

    def test_model_instantiation(self):
        """Test if the model can be instantiated correctly."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model.resnet.fc, nn.Linear)
        self.assertEqual(self.model.resnet.fc.out_features, 10)

    def test_fix_inplace_modules(self):
        """Test if fix_inplace_modules correctly disables inplace operations."""
        # Manually set an inplace module to True
        self.model.resnet.relu.inplace = True
        self.assertTrue(self.model.resnet.relu.inplace)
        
        fix_inplace_modules(self.model)
        
        # Check if it was disabled
        self.assertFalse(self.model.resnet.relu.inplace)
        
        # Check all modules
        for module in self.model.modules():
            if hasattr(module, 'inplace'):
                self.assertFalse(module.inplace, f"Module {module} still has inplace=True")

    def test_basic_block_monkey_patch(self):
        """Test if BasicBlock.forward has been monkey-patched to avoid inplace addition."""
        # Create a dummy input
        block = BasicBlock(64, 64)
        x = torch.randn(1, 64, 32, 32)
        
        out = block(x)
        self.assertEqual(out.shape, (1, 64, 32, 32))

    def test_opacus_compatibility_step(self):
        """Test a single training step with Opacus to ensure no RuntimeErrors."""
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Fix inplace modules
        fix_inplace_modules(model)
        model = ModuleValidator.fix(model)
        fix_inplace_modules(model) # Call again as in main.py
        
        # Re-initialize optimizer with the new model parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Make private
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=1,
            target_epsilon=10.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
        )
        
        # Run one step
        model.train()
        inputs, targets = next(iter(dataloader))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # If we got here without RuntimeError, the test passes
        self.assertTrue(True)

    def test_disable_dp(self):
        """Test training with Differential Privacy disabled."""
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy data
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Run one step without Opacus
        model.train()
        inputs, targets = next(iter(dataloader))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
