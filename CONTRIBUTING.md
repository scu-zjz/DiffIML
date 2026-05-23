# Contributing to DiffIML

Thanks for your interest in contributing!

## Reporting bugs

* Open a [GitHub Issue](../../issues) and include:
  - A minimal reproducible snippet
  - Your environment (`python -c "import torch, diffusers; print(torch.__version__, diffusers.__version__)"`)
  - Full stack trace / log
  - Dataset configuration

## Submitting changes

1. Fork the repo and create a feature branch:
   ```bash
   git checkout -b feature/your-change
   ```
2. Make your changes. Please respect the existing code style:
   - 4-space indentation, no tabs
   - Run `python -m pyflakes IMDLBenCo` before committing
3. Add tests under `tests/` if applicable.
4. Update `README.md` if you change the public API or scripts.
5. Push and open a Pull Request describing **what** changed and **why**.

## Adding a new model

The framework follows a registry pattern. To add a new model:

```python
# IMDLBenCo/model_zoo/your_model/your_model.py
from IMDLBenCo.registry import MODELS

@MODELS.register_module()
class YourModel(nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, image, mask=None, edge_mask=None, *args, **kwargs):
        if self.training:
            return {
                "backward_loss": loss,
                "pred_mask": pred,
                "pred_label": None,
                "visual_loss":  {"loss": loss},
                "visual_image": {"pred_mask": pred},
            }
        else:
            return {
                "backward_loss": None,
                "pred_mask": pred,
                "pred_label": None,
                "visual_loss":  {},
                "visual_image": {"pred_mask": pred},
            }
```

Then add an `__init__.py` import in `IMDLBenCo/model_zoo/__init__.py`.

## License

By contributing, you agree that your contributions will be licensed under the
same **CC-BY-4.0** license that covers this project.
