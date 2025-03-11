"""
Prisma Repo
By firstname Joseph

Copyright (c) firstname Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""

import logging
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union, Callable, Sequence, List

import torch.nn as nn
from vit_prisma.prisma_tools.hook_point import HookPoint


NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str]]]

class HookedRootModule(nn.Module):

    def __init__(self, *args):
        super().__init__()

        self.is_caching = False
        self.context_level = 0
    
    def setup(self):
        """
        Sets up model.

        This function must be called in the model's `__init__` method AFTER defining all layers. It
        adds a parameter to each module containing its name, and builds a dictionary mapping module
        names to the module instances. It also initializes a hook dictionary for modules of type
        "HookPoint".
        """
        self.mod_dict = {}
        self.hook_dict: Dict[str, HookPoint] = {}
        for name, module in self.named_modules():
            if name == "":
                continue
            module.name = name
            self.mod_dict[name] = module
            if isinstance(module, HookPoint):
                self.hook_dict[name] = module
        
    def hook_points(self):
        return self.hook_dict.values()
    
    def remove_all_hook_fns(
        self, dir="both", including_permanent=False, level=None
    ):
        for hp in self.hook_points():
            hp.remove_hooks(dir, including_permanent, level)
    
    def clear_context(self):
        for hp in self.hook_points():
            hp.clear_context()
    
    def reset_hooks(
            self, clear_contexts=True, direction="both", including_permanent=False, level=None
    ) -> None:
        if clear_contexts:
            self.clear_context()
        self.remove_all_hook_fns(direction, including_permanent, level)
        self.is_caching = False
    
    def check_and_add_hook(self, hook_point, hook_point_name, hook, dir="fwd", is_permanent=False, level=None, prepend=False) -> None:
        self.check_hooks_to_add(
            hook_point,
            hook_point_name,
            hook,
            dir = dir,
            is_permanent = is_permanent,
            prepend = prepend,
        )
        hook_point.add_hook(hook, dir=dir, is_permanent = is_permanent, level = level, prepend = prepend)
    
    def check_hooks_to_add(
        self, hook_point, hook_point_name, hook, dir="fwd", is_permanent=False, prepend=False) -> None:
        """Override this function to add checks on which hooks should be added (ex: see base_vit.py)"""
        pass 
    
    def add_hook(self, name, hook, dir="fwd", is_permanent=False, level=None, prepend=False) -> None:
        
        if type(name) == str:
            self.check_and_add_hook(
                self.mod_dict[name],
                name,
                hook,
                dir = dir,
                is_permanent = is_permanent,
                level = level,
                prepend = prepend,
            )
        else:
            for hook_point_name, hp in self.hook_dict.items():
                if name(hook_point_name):
                    self.check_and_add_hook(
                        hp,
                        hook_point_name,
                        hook,
                        dir = dir,
                        is_permanent = is_permanent,
                        level = level,
                        prepend = prepend,
                    )
    
    def add_perma_hook(self, name, hook, dir="fwd") -> None:
        self.add_hook(name, hook, dir, is_permanent=True)
    
    @contextmanager
    def hooks(
        self,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = True
    ):
        try:
            self.context_level += 1

            for name, hook in fwd_hooks:
                if type(name) == str:
                    self.mod_dict[name].add_hook(
                        hook, dir="fwd", level=self.context_level
                    )
                else:
                    for hook_point_name, hp in self.hook_dict.items():
                        if name(hook_point_name):
                            hp.add_hook(hook, dir="fwd", level=self.context_level)

            for name, hook in bwd_hooks:
                if type(name) == str:
                    self.mod_dict[name].add_hook(
                        hook, dir="bwd", level=self.context_level
                    )
                else:
                    for hook_point_name, hp in self.hook_dict.items():
                        if name(hook_point_name):
                            hp.add_hook(hook, dir="bwd", level=self.context_level)
            yield self
        finally:
            if reset_hooks_end:
                self.reset_hooks(clear_contexts=clear_contexts, including_permanent=False, level=self.context_level)
            self.context_level -= 1
    
    def run_with_hooks(
        self,
        *model_args,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        """
        Executes the model with specified forward and backward hooks.

        Args:
            model_args: The arguments to be passed to the model's forward method.
            fwd_hooks (List[Tuple[Union[str, Callable], Callable]], optional): A list of tuples specifying forward hooks.
                Each tuple contains a string (layer name) or a callable to match layers, and a callable hook function.
            bwd_hooks (List[Tuple[Union[str, Callable], Callable]], optional): A list of tuples specifying backward hooks.
                Each tuple contains a string (layer name) or a callable to match layers, and a callable hook function.
            reset_hooks_end (bool, optional): Whether to reset the hooks at the end of the run. Default is True.
            clear_contexts (bool, optional): Whether to clear contexts at the end of the run. Default is False.

        Returns:
            The output of the model's forward method.

        Raises:
            Warning: If backward hooks are provided and reset_hooks_end is True, a warning is logged that hooks will be reset before a backward pass can occur.
        """
        
        if len(bwd_hooks) > 0 and reset_hooks_end:
            logging.warning(
                "WARNING: Hooks will be reset at the end of run_with_hooks. This removes the backward hooks before a backward pass can occur."
            )
        with self.hooks(
            fwd_hooks, bwd_hooks,reset_hooks_end, clear_contexts
        ) as hooked_model:
            return hooked_model.forward(*model_args)

    def add_caching_hooks(
            self,
            names_filter: NamesFilter = None,
            incl_bwd: bool = False,
            device = None,
            remove_batch_dim: bool = False,
            cache: Optional[dict] = None,
    ) -> dict:
        
        if cache is None:
            cache = {}
        
        if names_filter is None:
            names_filter = lambda name: True
        elif type(names_filter) == str:
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif type(names_filter) == list:
            filter_list = names_filter
            names_filter = lambda name: name in filter_list

        self.is_caching = True

        def _save_hook(tensor, hook):
            if remove_batch_dim:
                cache[hook.name] = tensor.detach().to(device)[0]
            else:
                cache[hook.name] = tensor.detach().to(device)
        
        def _save_hook_back(tensor, hook):
            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor.detach().to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor.detach().to(device)
        
        for name, hp in self.hook_dict.items():
            if names_filter(name):
                hp.add_hook(_save_hook, dir="fwd")
                if incl_bwd:
                    hp.add_hook(_save_hook_back, dir="bwd")

        return cache

    def run_with_cache(
            self,
            *model_args,
            names_filter: NamesFilter = None,
            device = None,
            remove_batch_dim: bool = False,
            incl_bwd = False,
            reset_hooks_end = True,
            clear_contexts = False,
            **model_kwargs,

    ):
        cache_dict, fwd, bwd = self.get_caching_hooks(
            names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim
        )
        with self.hooks(fwd_hooks=fwd, bwd_hooks=bwd, reset_hooks_end=reset_hooks_end, clear_contexts=clear_contexts):
            model_out = self(*model_args, **model_kwargs)
            if incl_bwd:
                model_out.backward()
        return model_out, cache_dict

    def get_caching_hooks(
            self,
            names_filter: NamesFilter = None,
            incl_bwd: bool = False,
            device = None,
            remove_batch_dim: bool = False,
            cache: Optional[dict] = None,
    ) -> Tuple[dict, list, list]:
        
        if cache is None:
            cache = {}
        
        if names_filter is None:
            names_filter = lambda name: True
        elif type(names_filter) == str:
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif type(names_filter) == list:
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        
        self.is_caching = True

        def _save_hook(tensor, hook):
            if remove_batch_dim:
                cache[hook.name] = tensor.detach().to(device)[0]
            else:
                cache[hook.name] = tensor.detach().to(device)
        
        def _save_hook_back(tensor, hook):
            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor.detach().to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor.detach().to(device)
        
        fwd_hooks = []
        bwd_hooks = []
        for name, hp in self.hook_dict.items():
            if names_filter(name):
                fwd_hooks.append((name, _save_hook))
                if incl_bwd:
                    bwd_hooks.append((name, _save_hook_back))
        
        return cache, fwd_hooks, bwd_hooks
