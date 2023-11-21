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
from typing import Tuple, List, Optional, Dict, Any
import copy
from functools import partial
import inspect
from functools import cached_property
from contextlib import contextmanager

# Third Party
from grpc._channel import Channel

# First Party
import aconfig
import alog

# Local
from ..exceptions import error_handler
from ..task import TaskBase, task
from ..module_backends import BackendBase, backend_types
from ..modules import ModuleBase, ModuleConfig, module
from ..data_model import DataObjectBase, DataBase
from ..modules.decorator import SUPPORTED_LOAD_BACKENDS_VAR_NAME
from ..registries import (
    module_backend_classes,
    module_backend_registry,
    module_backend_types,
)
from ...runtime.service_generation.rpcs import CaikitRPCBase
from ...runtime.service_factory import ServicePackage, ServicePackageFactory

from .remote_model_finder import RemoteModuleConfig
from .model_initializer_base import ModelInitializerBase

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
        """Given a RemoteModuleConfig, initialize a RemoteModule instance
        """

        # Ensure the module config was produced by a RemoteModelFinder
        error.type_check("<COR47750753E>", RemoteModuleConfig, model_config=model_config)

        module_class = model_config.module_class
        if module_class.MODULE_ID not in self._module_class_map:
            self._module_class_map[module_class.MODULE_ID] = self._init_module_class(module_class)

        remote_module_class = self._module_class_map[module_class.MODULE_ID]
        return remote_module_class(model_config.connection, model_config.protocol)

    def _init_module_class(self, module_class: type[ModuleBase])->ModuleBase:
        """"""
        @module(
            id=f"{module_class.MODULE_ID}_remote",
            name=f"{module_class.MODULE_NAME} Remote",
            version=module_class.MODULE_VERSION,
            # We should make a remote backend that just stores signatures
            backend_type="LOCAL"
        )
        class _RemoteModelInstance(_RemoteModelBaseClass):
            pass

        if hasattr(module_class.TRAIN_SIGNATURE,"_method_pointer"):
            setattr(_RemoteModelInstance, module_class.TRAIN_SIGNATURE.method_name, _RemoteModelInstance._train)

        for task in module_class.tasks:
            # This should add a run arg? I'm not positive though
            for input, output, signature in module_class.get_inference_signatures(task):
                infer_func = partial(
                    _RemoteModelInstance._infer,
                    input_streaming=input,
                    output_streaming=output
                )
                setattr(_RemoteModelInstance,signature.method_name,infer_func)

        return _RemoteModelInstance


class _RemoteModelBaseClass(ModuleBase):
    def __init__(self, connection_info: Dict[str,Any], protocol: str):
        self.connection_info = connection_info
        self.protocol = protocol

    # Cache the service-package to avoid computation. Don't do this in init as
    # it has to be called after the runtime server has started
    @cached_property
    def service_package(self) -> ServicePackage:
        return ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.INFERENCE
        )

    def _train(self, *args, **kwargs):
        pass

    def _infer(self, task_class: type[TaskBase], *args, input_streaming=False, output_streaming=False, **kwargs):
        if self.protocol == "grpc":
            return self._infer_via_grpc(task_class, *args, output_streaming=output_streaming, **kwargs)
        raise NotImplementedError(f"Unknown protocol {self.protocol}")
    def _infer_via_grpc(self, task_class: type[TaskBase],*args,output_streaming=False, **kwargs) -> DataObjectBase:
        with self._grpc_channel() as channel:
            # Get the prediction name and service rpc
            task_prediction_name = f"{task_class.__name__}Predict"
            runtime_rpc = self.service_package.caikit_rpcs.get(task_prediction_name)

            # Construct request datamodel
            request_data_model_class = DataBase.get_class_for_name(runtime_rpc.name)
            request_dm = request_data_model_class(*args, **kwargs)

            # Send RPC request and close channel once completed
            stub = self.service_package.stub_class(channel)
            rpc = getattr(stub, task_prediction_name)
            response_protobuf = rpc(request_dm.to_proto(), metadata=self._meta_data)

            # Parse the GRPC protobuf into a DM
            response_dm_class = task_class.get_output_type(output_streaming=output_streaming)
            response = response_dm_class.from_proto(response_protobuf)
            return response

    @contextmanager
    def _grpc_channel(self)->Channel:
        channel_target = self._get_channel_target()
        grpc_options = self._get_grpc_options()
        with Channel(channel_target, grpc_options, None, None) as channel:
            yield channel

    def _get_grpc_options(self) -> List[Tuple]:
        return list(self.connection_info.get("options", {}).items())

    def _get_channel_target(self) -> str:
        host = self.connection_info.get("host")
        port = self.connection_info.get("port")
        return f"{host}:{port}"

