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

"""Unit tests for the service factory"""
# Standard
from pathlib import Path
import tempfile

# Third Party
from google.protobuf.message import Message
import pytest

# Local
from caikit.core.data_model import render_dataobject_protos
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from sample_lib import SampleModule
from sample_lib.data_model import SampleInputType, SampleOutputType
from sample_lib.modules.sample_task import ListModule
from tests.conftest import temp_config
from tests.core.helpers import MockBackend
from tests.data_model_helpers import reset_global_protobuf_registry, temp_dpool
from tests.runtime.conftest import sample_inference_service, sample_train_service
import caikit

## Helpers #####################################################################


@pytest.fixture
def clean_data_model(sample_inference_service, sample_train_service):
    """Set up a temporary descriptor pool that inherits all explicitly created
    dataobjects from the global, but skips dynamically created ones from
    datastreamsource.
    """
    with reset_global_protobuf_registry():
        with temp_dpool(
            inherit_global=True,
            skip_inherit=[".*sampletask.*\.proto"],
        ) as dpool:
            yield dpool


def validate_package_with_override(
    service_type: ServicePackageFactory.ServiceType,
    domain_override: str,
    package_override: str,
) -> ServicePackage:
    """Construct and validate train or infer service package with domain and package path override

    The config should be set up for domain and/or package override, before this function is called.

    Args:
        service_type (ServicePackageFactory.ServiceType) : The type of service to build,
            only TRAINING and INFERENCE are supported
        domain_override (str): The domain to validate (maybe overridden value, or the default)
        package_override (str): The package path to validate (maybe overridden value, or the default)

    Returns: The service package that was created, for potential subsequent reuse
    """
    assert service_type in {
        ServicePackageFactory.ServiceType.TRAINING,
        ServicePackageFactory.ServiceType.INFERENCE,
    }
    svc = ServicePackageFactory.get_service_package(service_type)

    service_name = (
        f"{domain_override}TrainingService"
        if service_type == ServicePackageFactory.ServiceType.TRAINING
        else f"{domain_override}Service"
    )
    assert svc.service.__name__ == service_name
    assert svc.descriptor.full_name == f"{package_override}.{service_name}"
    for message_name in [
        x for x in svc.messages.__dict__.keys() if not x.startswith("_")
    ]:
        message: Message = getattr(svc.messages, message_name)
        assert message.DESCRIPTOR.full_name == f"{package_override}.{message_name}"

    return svc


### Private method tests #############################################################


MODULE_LIST = [
    module_class
    for module_class in caikit.core.registries.module_registry().values()
    if module_class.__module__.partition(".")[0] == "sample_lib"
]


### Test ServicePackageFactory._get_and_filter_modules function
def test_get_and_filter_modules_respects_excluded_task_type():
    with temp_config(
        {
            "runtime": {
                "service_generation": {"task_types": {"excluded": ["SampleTask"]}}
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_excluded_modules():
    assert "InnerModule" in str(MODULE_LIST)
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"excluded": [ListModule.MODULE_ID]}
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "ListModule" not in str(clean_modules)
        assert "SampleModule" in str(clean_modules)
        assert "OtherModule" in str(clean_modules)


def test_get_and_filter_modules_respects_excluded_modules_and_excluded_task_type():
    assert "InnerModule" in str(MODULE_LIST)
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"excluded": [ListModule.MODULE_ID]},
                    "task_types": {"excluded": ["OtherTask"]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "ListModule" not in str(clean_modules)
        assert "OtherModule" not in str(clean_modules)
        assert "SampleModule" in str(clean_modules)


def test_get_and_filter_modules_respects_included_modules_and_included_task_types():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"included": [ListModule.MODULE_ID]},
                    "task_types": {"included": ["OtherTask"]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert len(clean_modules) == 2
        assert "OtherModule" in str(clean_modules)
        assert "ListModule" in str(clean_modules)


def test_get_and_filter_modules_respects_included_modules():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"included": [ListModule.MODULE_ID]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert len(clean_modules) == 1
        assert "ListModule" in str(clean_modules)
        assert "SampleModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_included_task_types():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)
        assert "OtherModule" not in str(clean_modules)
        # InnerModule has no task
        assert "InnerModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_included_task_types_and_excluded_modules():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                    "module_guids": {"excluded": [ListModule.MODULE_ID]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)
        assert "ListModule" not in str(clean_modules)


def test_override_domain(clean_data_model):
    """
    Test override of gRPC domain generation from config.
    The feature allows achieving backwards compatibility with earlier gRPC client.
    """
    domain_override = "OverrideDomainName"
    with temp_config(
        {
            "runtime": {
                "library": "sample_lib",
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                    "domain": domain_override,
                },
            }
        }
    ) as cfg:
        # Changing the domain still effects the default package name.
        # But if you override the package, it overrides the package name completely (see next test).
        validate_package_with_override(
            ServicePackageFactory.ServiceType.INFERENCE,
            domain_override,
            f"caikit.runtime.{domain_override}",
        )
        validate_package_with_override(
            ServicePackageFactory.ServiceType.TRAINING,
            domain_override,
            f"caikit.runtime.{domain_override}",
        )

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)


def test_override_package(clean_data_model):
    """
    Test override of gRPC package generation from config.
    The feature allows achieving backwards compatibility with earlier gRPC client.
    """
    package_override = "foo.runtime.yada.v0"
    with temp_config(
        {
            "runtime": {
                "library": "sample_lib",
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                    "package": package_override,
                },
            }
        }
    ) as cfg:
        validate_package_with_override(
            ServicePackageFactory.ServiceType.INFERENCE, "SampleLib", package_override
        )
        validate_package_with_override(
            ServicePackageFactory.ServiceType.TRAINING, "SampleLib", package_override
        )

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)


def test_override_package_and_domain_with_proto_gen(clean_data_model):
    """
    Test override of both package and domain, to make sure they work together, and
    additionally test the proto generation.
    The feature allows achieving backwards compatibility with earlier gRPC client.
    """
    package_override = "foo.runtime.yada.v0"
    domain_override = "OverrideDomainName"
    with temp_config(
        {
            "runtime": {
                "library": "sample_lib",
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                    "package": package_override,
                    "domain": domain_override,
                },
            }
        }
    ) as cfg:
        inf_svc = validate_package_with_override(
            ServicePackageFactory.ServiceType.INFERENCE,
            domain_override,
            package_override,
        )
        train_svc = validate_package_with_override(
            ServicePackageFactory.ServiceType.TRAINING,
            domain_override,
            package_override,
        )

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)

        # Full check on proto generation
        with tempfile.TemporaryDirectory() as output_dir:
            render_dataobject_protos(output_dir)
            inf_svc.service.write_proto_file(output_dir)
            train_svc.service.write_proto_file(output_dir)

            output_dir_path = Path(output_dir)
            for proto_file in output_dir_path.glob("*.proto"):
                # print(proto_file)
                with open(proto_file, "rb") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().decode("utf-8")
                        if line.startswith("package "):
                            package_name = line.split()[1]
                            assert package_name[-1] == ";"
                            package_name = package_name[0:-1]
                            root_package_name = package_name.split(".")[0]
                            if root_package_name not in {"caikit_data_model", "caikit"}:
                                assert package_name == package_override
                        elif line.startswith("service "):
                            service_name = line.split()[1]
                            domain_override_lower = domain_override.lower()
                            if proto_file.name.startswith(domain_override_lower):
                                if proto_file.stem.endswith("trainingservice"):
                                    assert (
                                        service_name
                                        == f"{domain_override}TrainingService"
                                    )
                                else:
                                    assert service_name == f"{domain_override}Service"


def test_backend_modules_included_in_service_generation(
    clean_data_model, reset_globals
):
    # Add a new backend module for the good ol' `SampleModule`
    @caikit.module(backend_type=MockBackend.backend_type, base_module=SampleModule)
    class NewBackendModule(caikit.core.ModuleBase):
        def run(
            self, sample_input: SampleInputType, backend_param: str
        ) -> SampleOutputType:
            pass

    inference_service = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE
    )
    sample_task_request = inference_service.messages.SampleTaskRequest

    # Check that the new parameter defined in this backend module exists in the service
    assert "backend_param" in sample_task_request.DESCRIPTOR.fields_by_name.keys()
