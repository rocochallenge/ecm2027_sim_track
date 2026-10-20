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

from robofinals.core.checks.checker_factory import get_checkers_from_cfg


def _resolve_main_viewport_window(sim):
    """Resolve the Kit main viewport window used for overlay UI.

    Isaac Lab 2 exposed ``sim._viewport_window``; Isaac Lab 3 ``SimulationContext`` does not.
    Prefer an actual viewport window; ``get_viewport_from_window_name()`` can return a
    ``ViewportAPI`` object that does not expose the window Overlay frame.
    """
    vp = getattr(sim, "_viewport_window", None)
    if vp is not None:
        return vp
    import omni.kit.viewport.utility as vp_utils

    get_active = getattr(vp_utils, "get_active_viewport_window", None)
    if get_active is not None:
        vp = get_active()
        if vp is not None:
            return vp

    vp = vp_utils.get_viewport_from_window_name()
    if vp is not None:
        return vp

    return None


def _get_viewport_overlay_frame(main_viewport):
    """Return a fixed, non-draggable overlay frame on the main viewport."""
    if main_viewport is None:
        return None

    get_frame = getattr(main_viewport, "get_frame", None)
    if get_frame is not None:
        try:
            return get_frame("Overlay")
        except Exception:
            pass

    return getattr(main_viewport, "frame", None)


def _resolve_scene_name(env_cfg):
    """Resolve scene display name from scene object (not task ep_meta)."""
    scene = getattr(env_cfg, "scene", None)
    if scene is not None:
        scene_name = getattr(scene, "scene_name", None)
        if scene_name:
            return scene_name
        if hasattr(scene, "get_ep_meta"):
            scene_name = scene.get_ep_meta().get("scene_name", "")
            if scene_name:
                return scene_name

    orchestrator = getattr(env_cfg, "orchestrator", None)
    if orchestrator is not None and hasattr(orchestrator, "get_ep_meta"):
        return orchestrator.get_ep_meta().get("scene_name", "")

    return ""


def _get_success_stages(env_cfg):
    stages = getattr(env_cfg.task, "success_stages", ())
    if stages:
        return tuple(stages)
    if hasattr(env_cfg.task, "get_ep_meta"):
        return tuple(env_cfg.task.get_ep_meta().get("success_stages", ()))
    return ()


def get_task_desc(env_cfg):
    """
    Get the task description from the environment configuration.
    """
    if not hasattr(env_cfg.task, "get_ep_meta"):
        return ""

    ep_meta = env_cfg.task.get_ep_meta()
    task_name = getattr(env_cfg.task, "task_name", "")
    scene_name = _resolve_scene_name(env_cfg)
    lang = ep_meta.get("lang", "")

    if scene_name:
        return f"Task name: {task_name}\nScene: {scene_name}\nDesc: {lang}"

    scene = getattr(env_cfg, "scene", None)
    usd_path = getattr(scene, "scene_usd_path", None) if scene is not None else None
    if usd_path:
        return f"Task name: {task_name}\nUSD path: {usd_path}\nDesc: {lang}"

    if hasattr(env_cfg.task, "usd_path"):
        return f"Task name: {task_name}\nUSD path: {env_cfg.task.usd_path}\nDesc: {lang}"

    return f"Task name: {task_name}\nDesc: {lang}"


def setup_task_description_ui(env, viewports=None):
    """
    Set up UI for displaying task description as a fixed subtitle overlay.

    Args:
        env: Environment object.

    Returns:
        The main viewport overlay frame, or ``None`` if there is no description or no viewport.
    """
    import omni.ui as ui

    env_cfg = env.cfg.isaaclab_arena_env
    desc = get_task_desc(env_cfg)
    success_stages = _get_success_stages(env_cfg)

    if not desc and not success_stages:
        return None

    main_viewport = getattr(env, "_main_viewport_window", None)
    if main_viewport is None:
        main_viewport = _resolve_main_viewport_window(env.sim)
    overlay_viewports = [main_viewport]
    for viewport in (viewports or {}).values():
        if viewport is not None and viewport not in overlay_viewports:
            overlay_viewports.append(viewport)

    overlay_frames = [
        _get_viewport_overlay_frame(viewport)
        for viewport in overlay_viewports
    ]
    overlay_frames = [frame for frame in overlay_frames if frame is not None]
    if not overlay_frames:
        print(
            "Warning: could not resolve viewport overlay frame for task description "
            "(headless or unsupported sim context); skipping UI."
        )
        return None
    overlay_frame = overlay_frames[0]

    if hasattr(main_viewport, "dock_tab_bar_visible"):
        main_viewport.dock_tab_bar_visible = False
    env.sim.render()

    checkers = None

    if hasattr(env_cfg, "checkers_cfg") and get_checkers_from_cfg is not None:
        try:
            filtered_cfg = {
                name: cfg
                for name, cfg in env_cfg.checkers_cfg.items()
                if cfg.get("warning_on_screen", False)
            }
            if filtered_cfg:
                checkers = get_checkers_from_cfg(filtered_cfg)
            else:
                checkers = None
        except Exception as e:
            print(f"Error getting checkers: {e}")
            checkers = None

    label_style = {
        "color": 0xB300FF00,
        "font_size": 14,
        "background_color": 0x00000066,
        "padding": 1,
        "border_radius": 2,
    }

    task_desc_label = None
    warning_label = None
    yellow_warning_label = None
    checker_labels = {}
    success_stage_label_panels = []
    stage_label_style = {
        **label_style,
        "color": 0xB3FFFFFF,
        "font_size": 13,
        "padding": 0,
        "margin": 0,
    }

    for viewport_index, current_overlay_frame in enumerate(overlay_frames):
        is_main_overlay = viewport_index == 0
        with current_overlay_frame:
            with ui.VStack(
                style={"margin": 8},
                spacing=0,
                alignment=ui.Alignment.LEFT_BOTTOM,
                width=ui.Percent(100),
                height=ui.Percent(100),
            ):
                ui.Spacer()
                if is_main_overlay and desc:
                    task_desc_label = ui.Label(
                        desc,
                        alignment=ui.Alignment.LEFT_BOTTOM,
                        style=label_style,
                        width=ui.Pixel(360),
                        height=0,
                        word_wrap=True,
                    )

                    success_stage_labels = {
                        stage_name: ui.Label(
                            stage_name,
                            alignment=ui.Alignment.LEFT_BOTTOM,
                            style=stage_label_style,
                            width=0,
                            height=ui.Pixel(16),
                        )
                        for stage_name in success_stages
                    }
                if success_stage_labels:
                    success_stage_label_panels.append(success_stage_labels)

                if is_main_overlay:
                    warning_label = ui.Label(
                        "",
                        alignment=ui.Alignment.LEFT_BOTTOM,
                        style={**label_style, "color": 0xFF0000FF},
                        width=0,
                        height=ui.Pixel(18),
                        visible=False,
                    )
                    yellow_warning_label = ui.Label(
                        "",
                        alignment=ui.Alignment.LEFT_BOTTOM,
                        style={**label_style, "color": 0xFF00FFFF},
                        width=0,
                        height=ui.Pixel(18),
                        visible=False,
                    )

                    if checkers:
                        for checker in checkers:
                            label = ui.Label(
                                checker.type,
                                alignment=ui.Alignment.LEFT_BOTTOM,
                                style=label_style,
                                width=0,
                                height=ui.Pixel(18),
                            )
                            checker_labels[checker.type] = label

    env.sim.render()

    try:
        setattr(env, "_task_desc_label", task_desc_label)
        setattr(env, "_warning_label", warning_label)
        setattr(env, "_yellow_warning_label", yellow_warning_label)
        setattr(env, "_checker_labels", checker_labels)
        setattr(env, "_success_stage_labels", success_stage_label_panels)
        if success_stage_label_panels:
            viewport_names = [
                getattr(viewport, "name", str(viewport))
                for viewport in overlay_viewports
            ]
            print(
                "[TaskStageUI] displaying "
                f"{len(success_stage_label_panels[0])} success stages "
                f"in {len(success_stage_label_panels)} viewports: {viewport_names}",
                flush=True,
            )
    except Exception as e:
        print(f"Error setting attributes on env: {e}")

    return overlay_frame


def update_task_desc(env):
    """
    Update the motion warning label with new warning text.

    Args:
        warning_label: The UI label to update
        warning_text: The warning text to display
    """
    env_cfg = env.cfg.isaaclab_arena_env
    if getattr(env, "_task_desc_label", None):
        env._task_desc_label.text = get_task_desc(env_cfg)


def update_success_stage_status(env):
    """Refresh task success-stage label colors from the active task state."""
    stage_label_panels = getattr(env, "_success_stage_labels", None)
    if not stage_label_panels:
        return

    task = getattr(getattr(getattr(env, "cfg", None), "isaaclab_arena_env", None), "task", None)
    if task is None or not callable(getattr(task, "get_success_stage_flags", None)):
        return

    flags = list(task.get_success_stage_flags())
    if any(len(flags) != len(stage_labels) for stage_labels in stage_label_panels):
        print(
            "Warning: success stage flags and labels have different lengths: "
            f"{len(flags)} vs {[len(stage_labels) for stage_labels in stage_label_panels]}"
        )
        return

    for stage_labels in stage_label_panels:
        for completed_flag, label in zip(flags, stage_labels.values()):
            label.style = {
                **label.style,
                "color": 0xB300FF00 if bool(completed_flag) else 0xB3FFFFFF,
            }


def update_checkers_status(env, warning_text: str):
    """
    Updates the color of checker labels based on the warning text.

    Args:
        env: Environment object containing the UI elements.
        warning_text (str): A string containing the names of checkers that have warnings.
    """
    if not hasattr(env, "_checker_labels"):
        return
    warning_label = getattr(env, "_warning_label", None)
    yellow_warning_label = getattr(env, "_yellow_warning_label", None)

    for checker_name, label in env._checker_labels.items():
        if not warning_text:
            label.style = {**label.style, "color": 0xB300FF00}
            if yellow_warning_label:
                yellow_warning_label.text = ""
                yellow_warning_label.visible = False
            if warning_label:
                warning_label.text = ""
                warning_label.visible = False

        elif f"{checker_name}_1" in warning_text:
            label.style = {**label.style, "color": 0xFF00FFFF}
            yellow_warning_label.text = warning_text
            yellow_warning_label.visible = True
            if warning_label:
                warning_label.text = ""
                warning_label.visible = False

        elif checker_name in warning_text:
            label.style = {**label.style, "color": 0xFF0000FF}
            warning_label.text = warning_text
            warning_label.visible = True
            if yellow_warning_label:
                yellow_warning_label.text = ""
                yellow_warning_label.visible = False

        else:
            label.style = {**label.style, "color": 0xB300FF00}


def dock_window(space, name, location, ratio):
    """
    Dock a window in the specified space with the given name, location, and size ratio.

    Args:
        space: The workspace to dock the window in.
        name: The name of the window to dock.
        location: The docking position.
        ratio: Size ratio for the docked window.
    """
    import omni.ui as ui
    window = ui.Workspace.get_window(name)
    if window and space:
        window.dock_in(space, location, ratio=ratio)


def create_and_dock_viewport(
    env,
    parent_window_name,
    position,
    ratio,
    camera_path,
    viewport=None,
    texture_resolution=None,
):
    """
    Create and configure a viewport window.

    Args:
        env: Environment object
        parent_window: Parent window to dock this viewport to
        position: Docking position
        ratio: Size ratio for the docked window
        camera_path: Prim path to the camera to set as active

    Returns:
        The created viewport window
    """
    from omni.kit.viewport.utility import create_viewport_window
    import omni.ui as ui
    from robofinals.utils.viewport_capture import WRIST_VIEWPORT_SIZE

    if texture_resolution is None:
        texture_resolution = WRIST_VIEWPORT_SIZE
    if viewport is None:
        viewport = create_viewport_window()
    env.sim.render()

    parent_window = ui.Workspace.get_window(parent_window_name)
    dock_window(parent_window, viewport.name, position, ratio)
    env.sim.render()

    viewport.viewport_api.set_active_camera(camera_path)
    viewport.viewport_api.set_texture_resolution(texture_resolution)

    env.sim.render()

    return viewport


def _embodiment_viewport_camera_names(env, role: str, renderer_defaults: tuple[str, ...]) -> tuple[str, ...]:
    """Merge backend defaults with optional embodiment-specific USD camera prim names."""
    names = list(renderer_defaults)
    embodiment = getattr(getattr(env.cfg, "isaaclab_arena_env", None), "embodiment", None)
    mapping = getattr(embodiment, "camera_viewport_prim_candidates", None)
    if callable(mapping):
        mapping = mapping()
    if isinstance(mapping, dict) and role in mapping:
        for name in mapping[role]:
            if name not in names:
                names.append(name)
    return tuple(names)


def _find_camera_prim_on_stage(stage, name_candidates: tuple[str, ...]):
    from pxr import UsdGeom

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera) and prim.GetName().lower() in name_candidates:
            return prim
    return None


def set_viewport_to_first_person(env):
    """Activate the first-person camera on the Kit main viewport.

    Returns ``True`` when both the camera prim and the main viewport are found.
    The explicit result/log is important for Newton scenes, where camera sensors
    and Kit viewport cameras are separate objects.
    """
    from pxr import UsdGeom
    import omni.kit.viewport.utility as vp_utils
    from robofinals.utils.viewport_capture import FIRST_PERSON_VIEWPORT_SIZE

    viewport = vp_utils.get_viewport_from_window_name()
    renderer = env.backend_runtime.renderer
    first_person_names = _embodiment_viewport_camera_names(
        env, "first_person", ("first_person_camera",)
    )
    first_person_camera = _find_camera_prim_on_stage(env.sim.stage, first_person_names)
    if first_person_camera is not None:
        if viewport is None:
            print("[teleop] first-person viewport unavailable; camera=%s" % first_person_camera.GetPath(), flush=True)
            return False
        camera_path = first_person_camera.GetPath()
        viewport.set_active_camera(camera_path)
        viewport.set_texture_resolution(FIRST_PERSON_VIEWPORT_SIZE)
        print("[teleop] first-person viewport activated: %s" % camera_path, flush=True)
        return True

    print("[teleop] first-person camera prim not found in stage", flush=True)
    return False


# TODO to optimize this function
#  1. enable setup cameras with config
#  2. try to optimize camera config to increase performance
def setup_cameras(env, viewports=None):
    """
    Set up mulitiple viewports for the teleoperation.

    Args:
        env: Environment object
    Returns:
        viewports: Dictionary of created viewports with their names as keys
    """
    from pxr import UsdGeom
    import omni.ui as ui
    from robofinals.utils.viewport_capture import FIRST_PERSON_VIEWPORT_SIZE

    if viewports is None:
        viewports = {}

    main_viewport = _resolve_main_viewport_window(env.sim)
    if main_viewport is not None:
        env._main_viewport_window = main_viewport

    renderer = env.backend_runtime.renderer
    left_hand_names = _embodiment_viewport_camera_names(
        env, "left", renderer.camera_name_candidates("left")
    )
    right_hand_names = _embodiment_viewport_camera_names(
        env, "right", renderer.camera_name_candidates("right")
    )

    left_hand_camera = _find_camera_prim_on_stage(env.sim.stage, left_hand_names)
    right_hand_camera = _find_camera_prim_on_stage(env.sim.stage, right_hand_names)
    left_shoulder_camera = _find_camera_prim_on_stage(env.sim.stage, ("left_shoulder_camera",))
    right_shoulder_camera = _find_camera_prim_on_stage(env.sim.stage, ("right_shoulder_camera",))
    eye_in_hand_camera = _find_camera_prim_on_stage(env.sim.stage, ("eye_in_hand_camera",))
    first_person_names = _embodiment_viewport_camera_names(
        env, "first_person", ("first_person_camera",)
    )
    first_person_camera = _find_camera_prim_on_stage(env.sim.stage, first_person_names)
    if first_person_camera is not None:
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_viewport_from_window_name()
        viewport.set_active_camera(first_person_camera.GetPath())
        viewport.set_texture_resolution(FIRST_PERSON_VIEWPORT_SIZE)
    if eye_in_hand_camera is not None:
        viewport_eye_in_hand = viewports.get("eye_in_hand", None)
        viewport_eye_in_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.BOTTOM,
            0.25,
            eye_in_hand_camera.GetPath(),
            viewport=viewport_eye_in_hand

        )
        viewports["eye_in_hand"] = viewport_eye_in_hand
    if left_hand_camera is not None:
        viewport_left_hand = viewports.get("left_hand", None)
        viewport_left_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.LEFT,
            0.25,
            left_hand_camera.GetPath(),
            viewport=viewport_left_hand,
        )
        viewports["left_hand"] = viewport_left_hand
    if right_hand_camera is not None:
        viewport_right_hand = viewports.get("right_hand", None)
        viewport_right_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.RIGHT,
            0.25,
            right_hand_camera.GetPath(),
            viewport=viewport_right_hand,
        )
        viewports["right_hand"] = viewport_right_hand
    if left_shoulder_camera is not None:
        viewport_left_shoulder = viewports.get("left_shoulder", None)
        viewport_left_shoulder = create_and_dock_viewport(
            env,
            viewport_left_hand.name,
            ui.DockPosition.BOTTOM,
            0.5,
            left_shoulder_camera.GetPath(),
            viewport=viewport_left_shoulder
        )
        viewports["left_shoulder"] = viewport_left_shoulder
    if right_shoulder_camera is not None:
        viewport_right_shoulder = viewports.get("right_shoulder", None)
        viewport_right_shoulder = create_and_dock_viewport(
            env,
            viewport_right_hand.name,
            ui.DockPosition.BOTTOM,
            0.5,
            right_shoulder_camera.GetPath(),
            viewport=viewport_right_shoulder
        )
        viewports["right_shoulder"] = viewport_right_shoulder
    if left_hand_camera is None:
        print("[setup_cameras] left_hand_camera prim not found in USD stage", flush=True)
    if right_hand_camera is None:
        print("[setup_cameras] right_hand_camera prim not found in USD stage", flush=True)
    return viewports


def spawn_cylinder_with_xform(
    parent_prim_path,
    xform_name,
    cylinder_name,
    cfg,
    env,
):
    from pxr import UsdGeom, Sdf, Gf, UsdShade
    stage = env.sim.stage

    xform_path = f"{parent_prim_path}/{xform_name}"
    xform_prim = stage.GetPrimAtPath(xform_path)
    if xform_prim and xform_prim.IsValid():
        return xform_prim

    xform = UsdGeom.Xform.Define(stage, Sdf.Path(xform_path))

    xform.AddTranslateOp().Set(Gf.Vec3f(*cfg["translation"]))
    xform.AddOrientOp().Set(Gf.Quatf(*cfg["orientation"]))

    cyl_path = f"{xform_path}/{cylinder_name}"
    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(cyl_path))
    cyl.CreateRadiusAttr(cfg["spawn"].radius)
    cyl.CreateHeightAttr(cfg["spawn"].height)
    cyl.CreateAxisAttr(cfg["spawn"].axis)

    material_path = f"{xform_path}/{cylinder_name}_Material"
    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*cfg["color"]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    surface_output = material.CreateSurfaceOutput()
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    surface_output.ConnectToSource(shader_output)

    UsdShade.MaterialBindingAPI(cyl).Bind(material)

    return xform


def spawn_robot_vis_helper_general(env):
    # check if the robot_vis_helper_cfg is available
    if not hasattr(env.cfg, "robot_vis_helper_cfg"):
        return

    vis_helper_prims = []

    for prim in env.sim.stage.Traverse():
        if prim.GetName().lower() == "robot":
            robot_prim_path = prim.GetPath()
    for key, cfg in env.cfg.robot_vis_helper_cfg.items():
        prim_path = robot_prim_path.AppendPath(cfg["relative_prim_path"])
        cylinder_prim = spawn_cylinder_with_xform(
            parent_prim_path=prim_path,
            xform_name=key,
            cylinder_name="mesh",
            cfg=cfg,
            env=env,
        )
        vis_helper_prims.append(cylinder_prim)
    return vis_helper_prims


def spawn_robot_vis_helper(env):
    # Have problems with Isaaclab/IsaacSim 4.5, works fine with Isaaclab/IsaacSim 5.0
    # check if the robot_vis_helper_cfg is available
    if not hasattr(env.cfg, "robot_vis_helper_cfg"):
        return
    import isaaclab.sim as sim_utils

    robot_prim = None
    vis_helper_prims = []

    for prim in env.sim.stage.Traverse():
        if prim.GetName().lower() == "robot":
            robot_prim = prim
            robot_prim_path = prim.GetPath()
    for key, cfg in env.cfg.robot_vis_helper_cfg.items():
        prim_path = robot_prim_path.AppendPath(cfg["relative_prim_path"])
        prim = sim_utils.spawn_cylinder(prim_path, cfg['spawn'], translation=cfg['translation'], orientation=cfg['orientation'])
        vis_helper_prims.append(prim)
    return vis_helper_prims


def destroy_robot_vis_helper(prim_list, env):
    if not prim_list:
        return
    for prim in prim_list:
        if prim.GetPrim().IsValid():
            env.sim.stage.RemovePrim(prim.GetPath())


def hide_ui_windows(sim_app):
    hide_window_names = []
    hide_window_names.extend(
        [
            "Console",
            "Main ToolBar",
            "Stage",
            "Layer",
            "Property",
            "Render Settings",
            "Content",
            "Flow",
            "Semantics Schema Editor",
            "VR",
            "Isaac Sim Assets [Beta]",
        ]
    )
    import omni.ui as ui
    for name in hide_window_names:
        window = ui.Workspace.get_window(name)
        if window is not None:
            window.visible = False


def setup_batch_name_gui(initial_batch_name='default-batch'):
    """Setup GUI window for displaying and editing batch name"""
    import omni.ui as ui

    global batch_name_gui

    class BatchNameGUI:
        def __init__(self, initial_batch_name):
            self.batch_name = initial_batch_name
            self.original_batch_name = initial_batch_name
            self.window = None
            self.batch_name_input = None
            self.batch_name_label = None
            self.create_window()

        def create_window(self):
            """Create the batch name GUI window"""
            self.window = ui.Window(
                "Batch Name Control",
                width=400,
                height=150,
                flags=ui.WINDOW_FLAGS_NO_SCROLLBAR
            )

            with self.window.frame:
                with ui.VStack(spacing=10):
                    # Current batch name display
                    with ui.HStack():
                        ui.Label("Current Batch Name:", width=150)
                        self.batch_name_label = ui.Label(
                            self.batch_name,
                            style={"color": 0xFF00FF00}
                        )

                    # Batch name input
                    with ui.HStack():
                        ui.Label("New Batch Name:", width=150)
                        self.batch_name_input = ui.StringField()
                        self.batch_name_input.model.set_value(self.batch_name)

                    # Buttons
                    with ui.HStack():
                        update_btn = ui.Button("Update Batch Name")
                        update_btn.set_clicked_fn(self.update_batch_name)

                        reset_btn = ui.Button("Reset")
                        reset_btn.set_clicked_fn(self.reset_batch_name)

        def update_batch_name(self):
            """Update the batch name"""
            new_batch_name = self.batch_name_input.model.get_value_as_string()
            if new_batch_name and new_batch_name != self.batch_name:
                self.batch_name = new_batch_name
                self.batch_name_label.text = self.batch_name
                print(f"Batch name updated to: {new_batch_name}")

        def reset_batch_name(self):
            """Reset batch name to original value"""
            self.batch_name_input.model.set_value(self.original_batch_name)
            self.batch_name = self.original_batch_name
            self.batch_name_label.text = self.original_batch_name
            print(f"Batch name reset to: {self.original_batch_name}")

    batch_name_gui = BatchNameGUI(initial_batch_name)
    return batch_name_gui


def generate_box_edges(center, extents):
    """
    Generate the edges of a bounding box given its center and extents.

    Parameters:
    - center: Tuple of (x, y, z) coordinates for the box's center
    - extents: Tuple of (width, depth, height) extents of the box

    Returns:
    - A list of tuples, each representing an edge of the box
    """
    x_c, y_c, z_c = center
    w, d, h = extents

    corners = [
        (x_c - w, y_c - h, z_c - d),
        (x_c - w, y_c - h, z_c + d),
        (x_c - w, y_c + h, z_c - d),
        (x_c - w, y_c + h, z_c + d),
        (x_c + w, y_c - h, z_c - d),
        (x_c + w, y_c - h, z_c + d),
        (x_c + w, y_c + h, z_c - d),
        (x_c + w, y_c + h, z_c + d),
    ]

    edges = [
        (corners[0], corners[1]),
        (corners[0], corners[2]),
        (corners[1], corners[3]),
        (corners[2], corners[3]),
        (corners[4], corners[5]),
        (corners[4], corners[6]),
        (corners[5], corners[7]),
        (corners[6], corners[7]),
        (corners[0], corners[4]),
        (corners[1], corners[5]),
        (corners[2], corners[6]),
        (corners[3], corners[7]),
    ]

    return edges


def draw_line(start, end, color=(1.0, 0.0, 0.0, 1.0), size=1.0):
    """
    Draws a single line between two points.
    """
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.draw_lines([start], [end], [color], [size])


def draw_box(center, extents, color=(1.0, 0.0, 0.0, 1.0), size=1.0):
    """
    Draws a box defined by its center and extents.
    """
    edges = generate_box_edges(center, extents)
    for start, end in edges:
        draw_line(start, end, color, size)


def draw_aabb_from_bbox(bbox):
    """
    Draws the axis-aligned bounding box of a given object.
    """
    ctr = bbox.GetMidpoint()
    ext = bbox.GetSize() / 2.0
    draw_box(ctr, ext)


def clear_debug_drawing():
    """
    Clears all debug drawings.
    """
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()


def show_upload_dialog(upload_dialog_state):
    """
    Show upload dialog to user (non-blocking)

    Args:
        upload_dialog_state (dict): Global state dictionary for dialog management
    """
    import omni.ui as ui

    def on_upload_click():
        upload_dialog_state["result"] = True
        if upload_dialog_state["window"]:
            # Hide the window instead of close
            upload_dialog_state["window"].visible = False

    def on_skip_click():
        upload_dialog_state["result"] = False
        if upload_dialog_state["window"]:
            # Hide the window instead of close
            upload_dialog_state["window"].visible = False

    # Create dialog window
    dialog_window = ui.Window(
        "Upload Choice",
        width=400,
        height=200,
        flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_MODAL
    )
    upload_dialog_state["window"] = dialog_window

    with dialog_window.frame:
        with ui.VStack(spacing=20):
            # Title
            ui.Label("Task Completed Successfully!", style={"fontSize": 18, "color": 0xFF00FF00})

            # Message
            ui.Label(f"1 demonstrations recorded successfully.", style={"fontSize": 14})
            ui.Label("Do you want to upload this data?", style={"fontSize": 14})

            # Buttons
            with ui.HStack():
                upload_btn = ui.Button("Upload Data", width=150, height=40)
                upload_btn.set_clicked_fn(on_upload_click)

                skip_btn = ui.Button("Skip Upload", width=150, height=40)
                skip_btn.set_clicked_fn(on_skip_click)

    upload_dialog_state["shown"] = True
