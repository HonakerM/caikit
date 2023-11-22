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
from ..task import TaskBase, task
from .model_initializer_base import ModelInitializerBase
from .remote_model_finder import RemoteModuleConfig

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

        module_class = model_config.module_class
        if module_class.MODULE_ID not in self._module_class_map:
            self._module_class_map[module_class.MODULE_ID] = self._init_module_class(
                module_class
            )

        remote_module_class = self._module_class_map[module_class.MODULE_ID]
        return remote_module_class(
            model_config.connection, model_config.protocol, model_config.model_path
        )

    def _init_module_class(self, module_class: type[ModuleBase]) -> type[ModuleBase]:
        """Helper class to create Module"""

        # Gather inference and train signatures from module class
        inference_signatures = {}
        for task in module_class.tasks:
            inference_signatures[task] = module_class.get_inference_signatures(task)

        # Local
        from ...runtime.service_generation.rpcs import ModuleClassTrainRPC

        train_rpc_name = ModuleClassTrainRPC.module_class_to_rpc_name(module_class)
        train_signature = (train_rpc_name, module_class.TRAIN_SIGNATURE)

        return construct_remote_module_class(
            inference_signatures,
            train_signature,
            module_class.MODULE_ID,
            module_class.MODULE_NAME,
        )


def construct_remote_module_class(
    inference_signatures: Dict[
        type[TaskBase], List[Tuple[bool, bool, CaikitMethodSignature]]
    ],
    train_signature: Optional[Tuple[str, CaikitMethodSignature]],
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
    if train_signature:
        rpc_name, signature = train_signature
        partial(_RemoteModelInstance.remote_train, rpc_name=rpc_name)
        setattr(
            _RemoteModelInstance,
            signature.method_name,
            _RemoteModelInstance.remote_train,
        )

    for task in inference_signatures:
        for input, output, signature in inference_signatures[task]:
            infer_func = _RemoteModelInstance.generate_inference_function(
                task, input, output, signature
            )
            setattr(_RemoteModelInstance, signature.method_name, infer_func)

    _RemoteModelInstance = module(
        id=remote_module_id,
        name=remote_module_name,
        version="0.0.0",
        tasks=list(inference_signatures.keys()),
        # We should make a remote backend that just stores signatures
        backend_type="LOCAL",
    )(_RemoteModelInstance)

    return _RemoteModelInstance


class _RemoteModelBaseClass(ModuleBase):
    def __init__(self, connection_info: Dict[str, Any], protocol: str, model_name: str):
        self.connection_info = connection_info
        self.protocol = protocol
        self.model_name = model_name

    @classmethod
    def generate_inference_function(
        cls,
        task: type[TaskBase],
        input,
        output,
        method_signature: CaikitMethodSignature,
    ):
        """Helper function to generate inference functions that will be set as attributes"""

        def infer_func(self, *args, **kwargs) -> method_signature.return_type:
            return self.remote_infer(
                *args,
                task_class=task,
                input_streaming=input,
                output_streaming=output,
                **kwargs,
            )

        # Override infer function name attributes
        infer_func.__name__ = method_signature.method_name
        infer_func.__qualname__ = method_signature._method_pointer.__qualname__
        infer_func.__signature__ = signature(method_signature._method_pointer)

        task_wrapped_infer_func = task.taskmethod(input, output)(infer_func)
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

    def remote_train(self, rpc_name: str, *args, **kwargs):
        # Local
        from ...runtime.service_generation.rpcs import ModuleClassTrainRPC

        if self.protocol == "grpc":
            train_rpc = self.training_service_package.caikit_rpcs.get(rpc_name)
            self._request_via_grpc(
                self.training_service_package, train_rpc, *args, **kwargs
            )

        raise NotImplementedError(f"Unknown protocol {self.protocol}")

    def remote_infer(
        self,
        task_class: type[TaskBase],
        *args,
        input_streaming=False,
        output_streaming=False,
        **kwargs,
    ):
        if self.protocol == "grpc":
            infer_rpc = self.inference_service_package.caikit_rpcs.get(
                f"{task_class.__name__}Predict"
            )
            return self._request_via_grpc(
                self.inference_service_package, infer_rpc, *args, **kwargs
            )

        raise NotImplementedError(f"Unknown protocol {self.protocol}")

    def _request_via_grpc(
        self,
        service_package: "ServicePackage",
        request_rpc: "CaikitRPCBase",
        *args,
        **kwargs,
    ):
        channel_target = self._get_channel_target()
        grpc_options = self._get_grpc_options()
        with Channel(channel_target, grpc_options, None, None) as channel:
            # Construct the request object
            request_data_model_class = DataBase.get_class_for_name(
                request_rpc.request.name
            )
            request_dm = request_data_model_class(*args, **kwargs)

            # Send RPC request and close channel once completed
            stub = service_package.stub_class(channel)
            rpc = getattr(stub, request_rpc.name)
            response_protobuf = rpc(
                request_dm.to_proto(), metadata=[("mm-model-id", self.model_name)]
            )

            return request_rpc.return_type.from_proto(response_protobuf)

    def _get_grpc_options(self) -> List[Tuple]:
        return list(self.connection_info.get("options", {}).items())

    def _get_channel_target(self) -> str:
        host = self.connection_info.get("host")
        port = self.connection_info.get("port")
        return f"{host}:{port}"
