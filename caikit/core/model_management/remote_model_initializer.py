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
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple
import uuid

# Third Party
from grpc._channel import Channel

# First Party
import aconfig
import alog

# Local
from ...config import get_config
from ..data_model import DataBase
from ..exceptions import error_handler
from ..modules import ModuleBase, module
from ..task import TaskBase
from .model_initializer_base import ModelInitializerBase
from .remote_model_finder import RemoteMethodRpc, RemoteModuleConfig

log = alog.use_channel("RINIT")
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

        # Construct remote module class if one has not already been created
        if model_config.module_id not in self._module_class_map:
            self._module_class_map[
                model_config.module_id
            ] = construct_remote_module_class(
                model_config.inference_methods,
                model_config.train_method,
                model_config.module_id,
                model_config.model_name,
            )

        remote_module_class = self._module_class_map[model_config.module_id]
        return remote_module_class(
            model_config.connection, model_config.protocol, model_config.model_path
        )


def construct_remote_module_class(
    inference_methods: List[Tuple[type[TaskBase], List[RemoteMethodRpc]]],
    train_method: RemoteMethodRpc,
    source_module_id: str = None,
    source_module_name: str = None,
) -> type[ModuleBase]:
    """Helper function to construct unique Remote Module Class."""

    # Construct new module id and name
    remote_module_id = (
        f"{source_module_id}_remote" if source_module_id else f"{uuid.uuid4()}_remote"
    )
    remote_module_name = (
        f"{source_module_name} Remote" if source_module_name else "Remote Module"
    )

    # Construct unique class which will have functions attached to it
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

    # Wrap Module with decorator to ensure attributes are properly set
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
    """Private class to act as the base for remote modules. This class will be subclassed and mutated by
    construct_remote_module_class to make it have the same functions and parameters as the source module."""

    def __init__(self, connection_info: Dict[str, Any], protocol: str, model_name: str):
        self.connection_info = connection_info
        self.protocol = protocol
        self.model_name = model_name

    ### Method Generation Helpers

    @classmethod
    def generate_train_function(cls, method: RemoteMethodRpc):
        """Helper function to construct a train function that will then be set as an attribute"""

        def train_func(self, *args, **kwargs) -> method.signature.return_type:
            return self.remote_method_request(*args, method=method, **kwargs)

        # Override infer function name attributes and signature
        train_func.__name__ = method.signature.method_name
        train_func.__qualname__ = method.signature._method_pointer.__qualname__
        train_func.__signature__ = signature(method.signature._method_pointer)
        return train_func

    @classmethod
    def generate_inference_function(cls, task: type[TaskBase], method: RemoteMethodRpc):
        """Helper function to construct inference functions that will be set as an attribute."""

        def infer_func(self, *args, **kwargs) -> method.signature.return_type:
            return self.remote_method_request(
                *args,
                method=method,
                **kwargs,
            )

        # Override infer function name attributes and signature
        infer_func.__name__ = method.signature.method_name
        infer_func.__qualname__ = method.signature._method_pointer.__qualname__
        infer_func.__signature__ = signature(method.signature._method_pointer)

        # Wrap infer function with task method to ensure internal attributes are properly
        # set
        task_wrapped_infer_func = task.taskmethod(
            method.input_streaming, method.output_streaming
        )(infer_func)
        return task_wrapped_infer_func

    ### Remote Interface

    def remote_method_request(self, method: RemoteMethodRpc, *args, **kwargs) -> Any:
        """Function to run a remote request based on the data stored in RemoteMethodRpc"""
        if self.protocol == "grpc":
            return self._request_via_grpc(method, *args, **kwargs)

        raise NotImplementedError(f"Unknown protocol {self.protocol}")

    ### GRPC Helper Functions

    def _request_via_grpc(
        self,
        method: RemoteMethodRpc,
        *args,
        **kwargs,
    ) -> Any:
        """Helper function to send a grpc request"""

        channel_target = self._get_channel_target()
        grpc_options = self._get_grpc_options()
        with Channel(channel_target, grpc_options, None, None) as channel:
            # Construct the request object
            request_dm = DataBase.get_class_for_name(method.request_dm_name)(
                *args, **kwargs
            )
            request_protobuf_message = request_dm.to_proto()

            # Construct the response types
            response_dm_class = DataBase.get_class_for_name(method.response_dm_name)
            response_protobuf_class = response_dm_class.get_proto_class()

            # Construct the service_rpc and serializers
            if method.input_streaming and method.output_streaming:
                service_rpc = channel.stream_stream(
                    method.rpc_name,
                    request_serializer=request_protobuf_message.SerializeToString,
                    response_deserializer=response_protobuf_class.FromString,
                )
            elif method.input_streaming:
                service_rpc = channel.stream_unary(
                    method.rpc_name,
                    request_serializer=request_protobuf_message.SerializeToString,
                    response_deserializer=response_protobuf_class.FromString,
                )
            elif method.output_streaming:
                service_rpc = channel.unary_stream(
                    method.rpc_name,
                    request_serializer=request_protobuf_message.SerializeToString,
                    response_deserializer=response_protobuf_class.FromString,
                )
            else:
                service_rpc = channel.unary_unary(
                    method.rpc_name,
                    request_serializer=request_protobuf_message.SerializeToString,
                    response_deserializer=response_protobuf_class.FromString,
                )

            # Send RPC request and close channel once completed
            response_protobuf = service_rpc(
                request_protobuf_message, metadata=[("mm-model-id", self.model_name)]
            )

            return response_dm_class.from_proto(response_protobuf)

    def _get_grpc_options(self) -> List[Tuple]:
        """Helper function to get list of grpc options"""
        return list(self.connection_info.get("options", {}).items())

    def _get_channel_target(self) -> str:
        """Get the current cha"""
        host = self.connection_info.get("host")
        port = self.connection_info.get("port")
        return f"{host}:{port}"
