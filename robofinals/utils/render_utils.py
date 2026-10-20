# Copyright 2025 Lightwheel Team
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

def optimize_rendering(env, args_cli):
    import carb

    settings = carb.settings.get_settings()
    max_bounces = int(getattr(args_cli, "pathtracing_max_bounces", 4))
    bounce_settings = (
        "/rtx/pathtracing/maxBounces",
        "/rtx-defaults/pathtracing/maxBounces",
        "/rtx/rtpt/maxBounces",
    )
    for setting_path in bounce_settings:
        settings.set_int(setting_path, max_bounces)
    readback = {setting_path: settings.get(setting_path) for setting_path in bounce_settings}
    print(f"[rendering] max bounces configured: {readback}", flush=True)
