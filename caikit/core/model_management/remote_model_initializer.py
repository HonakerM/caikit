# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The RemoteModelInitializer loads a RemoteModuleConfig as an empty Module that
sends all requests to an external runtime server

Configuration for RemoteModelInitializer lives under the config as follows:

model_management:
    initializers:
        <initializer name>:
            type: REMOTE
"""
# Standard
from contextlib import contextmanager
from functools import cached_property, partial
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple
import copy
import inspect
import uuid

# Third Party
from grpc._channel import Channel

# First Party
import aconfig
import alog

# Local
from ..data_model import DataBase, DataObjectBase
from ..exceptions import error_handler
from ..module_backends import BackendBase, backend_types
from ..modules import ModuleBase, ModuleConfig, module
from ..modules.decorator import SUPPORTED_LOAD_BACKENDS_VAR_NAME
from ..modules.meta import _ModuleBaseMeta
from ..registries import (
    module_backend_classes,
    module_backend_registry,
    module_backend_types,
)
from ..signature_parsing import CaikitMethodSignature
from ..task import TaskBase
from .model_initializer_base import ModelInitializerBase
from .remote_model_finder import RemoteMethodRpc, RemoteModuleConfig

log = alog.use_channel("RLOAD")
error = error_handler.get(log)


class RemoteModelInitializer(ModelInitializerBase):
    __doc__ = __doc__

    name = "REMOTE"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Construct with the config"""
        self._instance_name = instance_name
        self._module_class_map = {}

    def init(self, model_config: RemoteModuleConfig, **kwargs) -> Optional[ModuleBase]:
        """Given a RemoteModuleConfig, initialize a RemoteModule instance"""

        # Ensure the module config was produced by a RemoteModelFinder
        error.type_check(
            "<COR47750753E>", RemoteModuleConfig, model_config=model_config
        )

        if model_config.module_id not in self._module_class_map:
            self._module_class_map[model_config.module_id] = self._init_module_class(
                model_config
            )

        remote_module_class = self._module_class_map[model_config.module_id]
        return remote_module_class(
            model_config.connection, model_config.protocol, model_config.model_path
        )

    def _init_module_class(self, model_config: RemoteModuleConfig) -> type[ModuleBase]:
        """Helper class to create Module"""

        return construct_remote_module_class(
            model_config.inference_methods,
            model_config.train_method,
            model_config.module_id,
            model_config.model_name,
        )


def construct_remote_module_class(
    inference_methods: List[Tuple[type[TaskBase], List[RemoteMethodRpc]]],
    train_method: RemoteMethodRpc,
    source_module_id: str = None,
    source_module_name: str = None,
) -> type[ModuleBase]:
    # Don't default to uuid.uuid4 in args to ensure value is recomputed.
    remote_module_id = (
        f"{source_module_id}_remote" if source_module_id else uuid.uuid4()
    )
    remote_module_name = (
        f"{source_module_name} Remote" if source_module_name else "Remote Module"
    )

    class _RemoteModelInstance(_RemoteModelBaseClass):
        pass

    # Add the method signatures for train and each task
    if train_method:
        train_func = _RemoteModelInstance.generate_train_function(train_method)
        setattr(
            _RemoteModelInstance,
            train_method.signature.method_name,
            train_func,
        )

    task_list = []
    for task, inference_methods in inference_methods:
        task_list.append(task)
        for infer_method in inference_methods:
            infer_func = _RemoteModelInstance.generate_inference_function(
                task, infer_method
            )
            setattr(
                _RemoteModelInstance, infer_method.signature.method_name, infer_func
            )

    _RemoteModelInstance = module(
        id=remote_module_id,
        name=remote_module_name,
        version="0.0.0",
        tasks=task_list,
        # We should make a remote backend that just stores signatures
        backend_type="LOCAL",
    )(_RemoteModelInstance)

    return _RemoteModelInstance


class _RemoteModelBaseClass(ModuleBase):
    """Private class to act as the base for remote modules. This class will be subclassed and mutated by the
    RemoteModelInitializer to make it have the same functions and parameters as the source module."""

    def __init__(self, connection_info: Dict[str, Any], protocol: str, model_name: str):
        self.connection_info = connection_info
        self.protocol = protocol
        self.model_name = model_name

    @classmethod
    def generate_train_function(cls, method: RemoteMethodRpc):
        """Helper function to generate a train function that will then be set as an attribute"""

        def train_func(self, *args, **kwargs) -> method.signature.return_type:
            return self.remote_train(*args, method=method, **kwargs)

        # Override infer function name attributes
        train_func.__name__ = method.signature.method_name
        train_func.__qualname__ = method.signature._method_pointer.__qualname__
        train_func.__signature__ = signature(method.signature._method_pointer)
        return train_func

    @classmethod
    def generate_inference_function(cls, task: type[TaskBase], method: RemoteMethodRpc):
        """Helper function to generate inference functions that will be set as attributes"""

        def infer_func(self, *args, **kwargs) -> method.signature.return_type:
            return self.remote_infer(
                *args,
                method=method,
                **kwargs,
            )

        # Override infer function name attributes
        infer_func.__name__ = method.signature.method_name
        infer_func.__qualname__ = method.signature._method_pointer.__qualname__
        infer_func.__signature__ = signature(method.signature._method_pointer)

        task_wrapped_infer_func = task.taskmethod(
            method.input_streaming, method.output_streaming
        )(infer_func)
        return task_wrapped_infer_func

    # Cache the service-package to avoid computation. Don't do this in init as
    # it has to be called after the runtime server has started
    @cached_property
    def inference_service_package(self):
        # Local
        from ...runtime.service_factory import ServicePackageFactory

        return ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.INFERENCE
        )

    @cached_property
    def training_service_package(self):
        # Local
        from ...runtime.service_factory import ServicePackageFactory

        return ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.TRAINING
        )

    def remote_train(self, method: RemoteMethodRpc, *args, **kwargs):
        if self.protocol == "grpc":
            self._request_via_grpc(
                self.training_service_package, method, *args, **kwargs
            )

        raise NotImplementedError(f"Unknown protocol {self.protocol}")

    def remote_infer(
        self,
        method: RemoteMethodRpc,
        *args,
        **kwargs,
    ):
        if self.protocol == "grpc":
            return self._request_via_grpc(
                self.inference_service_package, method, *args, **kwargs
            )

        raise NotImplementedError(f"Unknown protocol {self.protocol}")

    def _request_via_grpc(
        self,
        service_package: "ServicePackage",
        method: RemoteMethodRpc,
        *args,
        **kwargs,
    ):
        channel_target = self._get_channel_target()
        grpc_options = self._get_grpc_options()
        with Channel(channel_target, grpc_options, None, None) as channel:
            # Construct the request object
            request_dm = method.request_dm(*args, **kwargs)

            # Send RPC request and close channel once completed
            stub = service_package.stub_class(channel)
            rpc = getattr(stub, method.rpc_name)
            response_protobuf = rpc(
                request_dm.to_proto(), metadata=[("mm-model-id", self.model_name)]
            )

            return method.response_dm.from_proto(response_protobuf)

    def _get_grpc_options(self) -> List[Tuple]:
        return list(self.connection_info.get("options", {}).items())

    def _get_channel_target(self) -> str:
        host = self.connection_info.get("host")
        port = self.connection_info.get("port")
        return f"{host}:{port}"
