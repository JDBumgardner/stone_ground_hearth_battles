# Copyright 2020 Google Inc. All Rights Reserved.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard_vega_embed import metadata
from torch.utils.tensorboard import SummaryWriter


def vega_embed(writer: SummaryWriter, tag, vega_json, step=None, description=None):
    smd = _create_summary_metadata(description)
    tensor = TensorProto(dtype='DT_STRING',
                         string_val=[vega_json.encode(encoding='utf_8')],
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    summary = Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])
    writer._get_file_writer().add_summary(summary, step)


def _create_summary_metadata(description):
    return summary_pb2.SummaryMetadata(
        summary_description=description,
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=metadata.PLUGIN_NAME,
            content=b"",
        ),
    )
