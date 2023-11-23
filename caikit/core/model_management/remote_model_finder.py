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
The RemoteModelFinder locates models that loaded by a remote runtime.

Configuration for RemoteModelFinder lives under the config as follows:

model_management:
    finders:
        <finder name>:
            type: REMOTE
            config:
                connection:
                    host: str <remote host>
                    port: int <remote port>
                    tls:
                        ca: Optional[str] <path to remote ca file>
                        cert: Optional[str] <path to MTLS cert>
                    options: Optional[Dict[str,str]] <optional dict of grpc or rest options>
                protocol: Optional[str]="grpc" <protocol the remote server is using>
                default_module: Optional[str] <the default module_class to use if one is not provided>
                supported_models: Optional[Dict[str, str]] <mapping of models to modules_class>
                    <model_name>:<module_id>

"""
# Standard
from dataclasses import dataclass
from typing import Optional
import os

# First Party
import aconfig
import alog

# Local
from ..data_model import DataBase, DataObjectBase
from ..exceptions import error_handler
from ..modules import ModuleBase, ModuleConfig
from ..registries import module_registry
from ..signature_parsing import CaikitMethodSignature
from .model_finder_base import ModelFinderBase

log = alog.use_channel("RFIND")
error = error_handler.get(log)


class RemoteModuleConfig(ModuleConfig):
    """Helper class to differentiate a local ModuleConfig and a RemoteModuleConfig. The structure
    should be as follows:
    {
        # Remote information copied from Finder config
        connection: Dict[str, Any]
        protocol: str

        # Method information
        # use list and tuples instead of a dictionary to avoid aconfig.Config error
        inference_methods: List[Tuple[type[TaskBase], List[RemoteMethodRpc]]]
        train_method: RemoteMethodRpc,

        # Source Module Information
        module_id: str
        module_name: str
        model_path: str
    }
    """

    # Reset reserved_keys, so we can manually add module_path
    reserved_keys = []


@dataclass
class RemoteMethodRpc:
    """Helper dataclass to store information about a RPC. This includes the service function"""

    # full signature for this RPC
    signature: CaikitMethodSignature
    # Request and response objects for this RPC
    request_dm: DataBase
    response_dm: DataBase
    # The function name on the GRPC Servicer
    rpc_name: str

    # Only used for infer RPC types
    input_streaming: bool
    output_streaming: bool


class RemoteModelFinder(ModelFinderBase):
    __doc__ = __doc__

    name = "REMOTE"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with an optional path prefix"""
        self._connection_info = config.connection
        self._protocol = config.get("protocol", "grpc")
        self._all_models = config.get("supported_models") is None
        self._supported_models = config.get("supported_models", {})
        self._default_module = config.get("default_module")
        self._instance_name = instance_name

        # Type check parameters
        error.type_check("<COR72281545E>", str, protocol=self._protocol)
        error.type_check(
            "<COR74343245E>",
            dict,
            allow_none=True,
            supported_models=self._supported_models,
        )
        error.type_check("<COR72281587E>", str, host=self._connection_info.host)
        error.type_check("<COR73381567E>", int, host=self._connection_info.port)

        tls_info = self._connection_info.get("tls", {})
        error.type_check(
            "<COR74321567E>",
            str,
            allow_none=True,
            tls_ca=tls_info.get("ca"),
            tls_cert=tls_info.get("cert"),
        )

        # Replace the string references in supported_models with the real representation
        for model_name in self._supported_models:
            self._supported_models[model_name] = module_registry().get(
                self._supported_models[model_name]
            )

    def find_model(
        self,
        model_path: str,
        **__,
    ) -> Optional[RemoteModuleConfig]:
        """Check if the remote runtime supports the model_path"""

        # If supported_models was supplied and model_path is not present then raise an error
        if not self._all_models and model_path not in self._supported_models:
            raise KeyError(
                f"Model {model_path} is not supported by finder {self._instance_name}"
            )

        module_class = self._supported_models.get(model_path, self._default_module)

        remote_config_dict = {
            # Connection info
            "connection": self._connection_info,
            "protocol": self._protocol,
            # Method info
            "inference_methods": [],
            "train_method": None,
            # Source module info
            "model_path": model_path,
            "module_id": module_class.MODULE_ID,
            "module_name": module_class.MODULE_NAME,
        }

        # These are a really bad imports!! This is required because I wanted to reuse the string formatting from
        # service_generation
        # Local
        from ...runtime.service_factory import get_inference_request, get_train_request
        from ...runtime.service_generation.rpcs import ModuleClassTrainRPC

        # Parse inference methods signatures
        for task_class in module_class.tasks:
            task_methods = []
            for input, output, signature in module_class.get_inference_signatures(
                task_class
            ):

                rpc_name = f"{task_class.__name__}Predict"

                request_dm = get_inference_request(task_class, input, output)
                task_methods.append(
                    RemoteMethodRpc(
                        signature=signature,
                        request_dm=request_dm,
                        response_dm=signature.return_type,
                        rpc_name=rpc_name,
                        input_streaming=input,
                        output_streaming=output,
                    )
                )

            remote_config_dict["inference_methods"].append((task_class, task_methods))

        # parse train method signature if there is one
        if module_class.TRAIN_SIGNATURE and (
            module_class.TRAIN_SIGNATURE.return_type is not None
            and module_class.TRAIN_SIGNATURE.parameters is not None
        ):

            rpc_name = ModuleClassTrainRPC.module_class_to_rpc_name(module_class)
            request_dm = get_train_request(module_class)

            remote_config_dict["train_method"] = RemoteMethodRpc(
                signature=module_class.TRAIN_SIGNATURE,
                request_dm=request_dm,
                response_dm=module_class.TRAIN_SIGNATURE.return_type,
                rpc_name=rpc_name,
            )

        return RemoteModuleConfig(remote_config_dict)
