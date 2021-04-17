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

"""A Plugin for displaying Vega/Vega lite embedded charts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import six
import werkzeug
from tensorboard.plugins import base_plugin
from tensorboard.util import tensor_util
from tensorboard_vega_embed import metadata
from werkzeug import wrappers
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.shared_data import SharedDataMiddleware


class VegaEmbedXPlugin(base_plugin.TBPlugin):
    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates VegaEmbedXPlugin.

        Args:
        context: A base_plugin.TBContext instance.
        """
        self._multiplexer = context.multiplexer
        self._static_path = os.path.join((os.path.dirname(__file__)), "svelte_frontend", "public")
        self._svelte_path = f'/data/plugin/{self.plugin_name}/static'

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process.

        When there are no runs with greeting data, TensorBoard will hide the
        plugin from the main navigation bar.
        """
        return bool(
            self._multiplexer.PluginRunToTagToContent(metadata.PLUGIN_NAME)
        )

    def get_plugin_apps(self):
        return {
            "/*": DispatcherMiddleware(
                self.not_found,
                {
                    self._svelte_path: SharedDataMiddleware(
                        self.not_found, {
                            "/": self._static_path
                        }),
                    self._svelte_path + "/tags": self._serve_tags,
                    self._svelte_path + "/plot_specs": self._serve_plot_specs,
                })
        }

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/static/index.js")

    @wrappers.Request.application
    def _serve_tags(self, request):
        del request  # unused
        mapping = self._multiplexer.PluginRunToTagToContent(
            metadata.PLUGIN_NAME
        )
        result = {run: {} for run in self._multiplexer.Runs()}
        for (run, tag_to_content) in six.iteritems(mapping):
            for tag in tag_to_content:
                summary_metadata = self._multiplexer.SummaryMetadata(run, tag)
                result[run][tag] = {
                    u"description": summary_metadata.summary_description,
                }
        contents = json.dumps(result, sort_keys=True)
        return werkzeug.Response(contents, content_type="application/json")

    @wrappers.Request.application
    def _serve_plot_specs(self, request):
        run = request.args.get("run")
        tag = request.args.get("tag")
        if run is None or tag is None:
            raise werkzeug.exceptions.BadRequest("Must specify run and tag")
        try:
            data = [
                {
                    "step": event.step,
                    "vega_spec": tensor_util.make_ndarray(event.tensor_proto).item().decode("utf-8")
                }
                for event in self._multiplexer.Tensors(run, tag)
            ]
        except KeyError:
            raise werkzeug.exceptions.BadRequest("Invalid run or tag")
        contents = json.dumps(data, sort_keys=True)
        return werkzeug.Response(contents, content_type="application/json")

    @wrappers.Request.application
    def not_found(self, request):
        print(request.full_path)
        del request
        return werkzeug.exceptions.NotFound()
