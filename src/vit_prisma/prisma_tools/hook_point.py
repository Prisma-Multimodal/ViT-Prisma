"""
Prisma Repo
By firstname Joseph

Copyright (c) firstname Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""

from typing import List, Union, Dict, Callable, Tuple, Optional, Any
from vit_prisma.prisma_tools.lens_handle import LensHandle
import torch.nn as nn


class HookPoint(nn.Module):
    """
    A helper class to access intermediate activations in a PyTorch model (adapted from TransformerLens, which was inspired by Garcon).

    HookPoint is a dummy module that acts as an identity function by default. By wrapping any
    intermediate activation in a HookPoint, it provides a convenient way to add PyTorch hooks.
    """
    def __init__(self):
        super().__init__()
        self.fwd_hooks: List[LensHandle] = []
        self.bwd_hooks: List[LensHandle] = []

        self.ctx = {} # what is this?

        self.name = None
    
    def add_perma_hook(self, hook, dir="fwd") -> None:
        self.add_hook(hook, dir, is_permanent=True)
    
    def add_hook(
            self, hook, dir="fwd", is_permanent=False, level=None, prepend=False
    ) -> None:
        """
        If prepend is True, add this hook before all other hooks.
        """

        if dir == "fwd":

            def full_hook(module, module_input, module_output):
                return hook(module_output, hook=self)

            full_hook.__name__ = (
                hook.__repr__()
            )

            handle = self.register_forward_hook(full_hook)
            handle = LensHandle(handle, is_permanent, level)

            if prepend:
                self._forward_hooks.move_to_end(handle.hook.id, last=False)
                self.fwd_hooks.insert(0, handle)
            else:
                self.fwd_hooks.append(handle)

        elif dir == "bwd":

            def full_hook(module, module_input, module_output):
                return hook(module_output[0], hook=self)
            
            full_hook.__name__ = (
                hook.__repr__()
            )

            handle = self.register_backward_hook(full_hook)
            handle = LensHandle(handle, is_permanent, level)
        
            if prepend:
                self._backward_hooks.move_to_end(handle.hook.id, last=False)
                self.bwd_hooks.insert(0, handle)
            else:
                self.bwd_hooks.append(handle)
        
        else :
            raise ValueError(f"Invalid dir {dir}. dir must be 'fwd' or 'bwd'")

    def remove_hooks(self, dir="fwd", including_permanent=False, level=None) -> None:
        def _remove_hooks(handles: List[LensHandle]) -> List[LensHandle]:
            output_handles = []
            for handle in handles:
                if including_permanent:
                    handle.hook.remove()
                elif (not handle.is_permanent) and (level is None or handle.context_level == level):
                    handle.hook.remove()
                else:
                    output_handles.append(handle)
            return output_handles
        
        if dir == "fwd" or dir == "both":
            self.fwd_hooks = _remove_hooks(self.fwd_hooks)
        elif dir == "bwd" or dir == "both":
            self.bwd_hooks = _remove_hooks(self.bwd_hooks)
        else:
            raise ValueError(f"Invalid direction {dir}. dir must be 'fwd', 'bwd', or 'both'")

    def clear_context(self):
        del self.ctx
        self.ctx = {}

    def forward(self, x):
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on HookedTransformer
        # If it doesn't have this form, raises an error -
        split_name = self.name.split(".")
        return int(split_name[1])

