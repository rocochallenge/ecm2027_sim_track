"""
Semantic Scene Generalization Agent (Qwen tool-calling, MVP).

Goal:
Given a natural-language instruction, modify a USD scene to generalize assets.

This file implements a *real* agent loop:
1) Call Qwen API.
2) Parse assistant JSON that requests a tool call.
3) Execute the tool (USD analyze / USD edit / asset acquisition).
4) Feed results back to Qwen.
5) Stop when Qwen returns a final JSON.

Notes / current MVP limitations:
- Replacement/add/remove focuses on "fixture-like placeholders" supported by FixtureGeneralization.
- Texture modification uses OpenUsd.set_rgb (diffuse_tint) for the chosen prim root.
- Physics friction/damping is not fully wired yet because USD material schema for friction/damping
  is project-specific; you can extend tools later.
"""

from __future__ import annotations

import json
import os
import random
import re
import socket
import time
import uuid
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from robofinals.utils.generalization_utils import FixtureGeneralization
from robofinals.utils.usd_utils import OpenUsd

try:
    from pxr import Gf, Sdf, UsdGeom
except ImportError as e:  # pragma: no cover
    raise RuntimeError("pxr USD python bindings are required") from e


# -----------------------------
# Qwen client (OpenAI-compatible)
# -----------------------------
class QwenChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def _build_chat_url(self) -> str:
        """
        Build OpenAI-compatible chat completions URL safely.
        Supports both:
        - base_url="https://.../compatible-mode"
        - base_url="https://.../compatible-mode/v1"
        """
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        url = self._build_chat_url()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                content = parsed["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise RuntimeError("Invalid Qwen response format: content is not a string")
                return content
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = "<failed to read error body>"
                # HTTP-level errors (auth/permission/model) generally should not be retried blindly.
                raise RuntimeError(
                    f"Qwen API HTTP error: status={e.code}, url={url}, body={err_body}"
                ) from e
            except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * (2 ** attempt))
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * (2 ** attempt))

        raise RuntimeError(
            f"Qwen API request failed after retries "
            f"(timeout_s={self.timeout_s}, max_retries={self.max_retries}). "
            f"last_error={last_err}"
        )


# -----------------------------
# Planned JSON schema
# -----------------------------
TOOL_CALL_SCHEMA_EXAMPLE = {
    "tool_call": {
        "name": "replace_placeholder_assets",
        "arguments": {
            "source_asset_type": "Microwave",
            "target_asset_type": "Oven"
        },
    }
}

FINAL_SCHEMA_EXAMPLE = {
    "final": {
        "status": "ok",
        "summary": "Replaced microwaves with ovens and saved the output USD.",
        "applied": [{"op": "replace_placeholder_assets", "count": 3}],
        "saved_to": "/abs/path/to/output.usd",
    }
}

# Common alias mapping for more robust LLM tool args.
# key: normalized alias (lowercase), value: canonical FixtureGeneralization asset_type
ASSET_TYPE_ALIASES: Dict[str, str] = {
    "fridge": "Fridge",
    "refrigerator": "Fridge",
    "microwave": "Microwave",
    "oven": "Oven",
    "dishwasher": "Dishwasher",
    "stove": "Stove",
    "sink": "Sink",
    "counter": "Counter",
    "cabinet": "SingleCabinet",
    # Pop-up bread toaster (often prim name contains toaster_main; USD type attr is "Toaster")
    "toaster": "Toaster",
    "面包机": "Toaster",
    "烤面包机": "Toaster",
    # Countertop toaster oven (USD type attr is often "ToasterOven")
    "toasteroven": "ToasterOven",
    "toaster_oven": "ToasterOven",
    "面包机烤箱": "ToasterOven",
}


# -----------------------------
# Tools context
# -----------------------------
class SemanticTools:
    def __init__(self, *, stage: Any, rng_seed: Optional[int] = None, preserve_size: bool = True, size_decimal_places: int = 4):
        self.stage = stage
        self.rng_seed = rng_seed
        self.preserve_size = preserve_size
        self.size_decimal_places = size_decimal_places

        if rng_seed is not None:
            random.seed(rng_seed)

        self._gen = FixtureGeneralization(stage)
        self._placeholders_cache: Optional[List[Dict[str, Any]]] = None
        # placeholder id -> placeholder info
        self._ph_by_id: Dict[str, Dict[str, Any]] = {}

    def _resolve_scene_root_path(self) -> str:
        """
        Resolve a robust root path for placeholder search.
        Priority:
        1) "/World"
        2) "/world"
        3) stage default prim path
        4) "/"
        """
        for p in ["/World", "/world"]:
            prim = self.stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                return p

        default_prim = self.stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            return str(default_prim.GetPath())

        return "/"

    # -----------------------------
    # Scene analyze tools
    # -----------------------------
    def list_placeholders(self) -> List[Dict[str, Any]]:
        root_path = self._resolve_scene_root_path()
        placeholders = self._gen.find_placeholders(root_path=root_path, fixture_types=None)
        # Rebuild id mapping each time to keep ids consistent with current stage contents.
        self._ph_by_id = {}
        # Assign stable ids for tool calls
        for i, ph in enumerate(placeholders):
            ph_id = f"ph_{i}"
            ph["id"] = ph_id
            self._ph_by_id[ph_id] = ph
        self._placeholders_cache = placeholders
        return [
            {
                "id": ph["id"],
                "prim_path": ph["path"],
                "name": ph["name"],
                "asset_type": ph["asset_type"],
                "size": ph.get("size"),
                "position": ph.get("position"),
                "rotation": ph.get("rotation"),
                # Note: placeholders returned by FixtureGeneralization include a scale vector if present,
                # but the replacement logic computes scale based on size/asset bbox.
                "scale": ph.get("scale"),
            }
            for ph in placeholders
        ]

    def get_placeholder_by_id(self, placeholder_id: str) -> Dict[str, Any]:
        if placeholder_id in self._ph_by_id:
            return self._ph_by_id[placeholder_id]
        # fallback: rebuild cache once
        if self._placeholders_cache is None:
            self.list_placeholders()
        if placeholder_id in self._ph_by_id:
            return self._ph_by_id[placeholder_id]
        raise ValueError(f"Unknown placeholder id: {placeholder_id}")

    # -----------------------------
    # Asset tools
    # -----------------------------
    def list_supported_asset_types(self) -> List[str]:
        # Keys are canonical types used by FixtureGeneralization.TYPE_TO_REGISTRIES
        return sorted(list(FixtureGeneralization.TYPE_TO_REGISTRIES.keys()))

    def _normalize_asset_type(self, raw_asset_type: str) -> str:
        """
        Convert model-provided aliases (e.g. 'refrigerator') to canonical types
        expected by FixtureGeneralization (e.g. 'Fridge').
        """
        raw = (raw_asset_type or "").strip()
        if not raw:
            return raw_asset_type

        # 1) exact canonical match
        canonical = self.list_supported_asset_types()
        if raw in canonical:
            return raw

        # 2) case-insensitive canonical match
        lower_raw = raw.lower()
        for c in canonical:
            if c.lower() == lower_raw:
                return c

        # 3) alias mapping (keys include Chinese phrases; use stripped raw for those)
        if raw in ASSET_TYPE_ALIASES:
            return ASSET_TYPE_ALIASES[raw]
        if lower_raw in ASSET_TYPE_ALIASES:
            return ASSET_TYPE_ALIASES[lower_raw]
        spaced = lower_raw.replace("_", " ")
        if spaced in ASSET_TYPE_ALIASES:
            return ASSET_TYPE_ALIASES[spaced]

        # 4) fuzzy: if it starts with "fridge", fallback to Fridge
        if lower_raw.startswith("fridge") or lower_raw.startswith("refrigerator"):
            return "Fridge"

        # 5) keep original for visibility (may still fail downstream)
        return raw_asset_type

    @staticmethod
    def _asset_type_match(placeholder_type: str, query_type: str) -> bool:
        """
        Matching rule between placeholder asset type and requested source type.
        - Exact match OR
        - Generic Fridge matches any Fridge* subtype.
        """
        if placeholder_type == query_type:
            return True
        if query_type == "Fridge" and placeholder_type.startswith("Fridge"):
            return True
        return False

    def acquire_asset_path(self, asset_type: str) -> Optional[str]:
        canonical = self._normalize_asset_type(asset_type)
        return self._gen.acquire_random_asset(canonical)

    def get_asset_bbox(self, asset_path: str) -> Optional[List[float]]:
        return self._gen.get_asset_bbox(asset_path)

    def list_type_attr_objects(self, *, source_asset_type: Optional[str] = None) -> Dict[str, Any]:
        """
        List Xform prims with a string `type` attribute.
        If source_asset_type is provided, filter by semantic type match.
        """
        query_type = self._normalize_asset_type(source_asset_type) if source_asset_type else None
        objects: List[Dict[str, Any]] = []
        for prim in self.stage.Traverse():
            if not prim or not prim.IsValid() or prim.GetTypeName() != "Xform":
                continue
            attr = prim.GetAttribute("type")
            if not attr:
                continue
            try:
                t_val = attr.Get()
            except Exception:
                t_val = None
            if not isinstance(t_val, str) or not t_val:
                continue
            if query_type is not None and not self._asset_type_match(t_val, query_type):
                continue
            objects.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "name": prim.GetName(),
                    "type_attr": t_val,
                }
            )
        return {"count": len(objects), "objects": objects}

    # -----------------------------
    # USD edit tools
    # -----------------------------
    def replace_placeholder_assets(
        self,
        *,
        source_asset_type: str,
        target_asset_type: str,
        placeholder_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        source_asset_type = self._normalize_asset_type(source_asset_type)
        target_asset_type = self._normalize_asset_type(target_asset_type)
        # Make sure placeholders are loaded
        if self._placeholders_cache is None:
            self.list_placeholders()

        # Select placeholders
        if placeholder_ids:
            targets = [self.get_placeholder_by_id(pid) for pid in placeholder_ids]
        else:
            targets = [
                ph
                for ph in self._placeholders_cache or []
                if self._asset_type_match(str(ph["asset_type"]), source_asset_type)
            ]

        replaced: List[str] = []
        failed: List[str] = []

        for ph in targets:
            asset_path = self._gen.acquire_random_asset(target_asset_type)
            if not asset_path:
                failed.append(ph["id"])
                continue

            asset_bbox = self._gen.get_asset_bbox(asset_path)
            if not asset_bbox:
                failed.append(ph["id"])
                continue

            scale = [1.0, 1.0, 1.0]
            if self.preserve_size and ph.get("size") is not None:
                # scale = placeholder_size / asset_bbox
                scale = self._gen.calculate_scale(ph.get("size"), asset_bbox)
            else:
                # If we don't preserve size, keep scale from replacement calc if any
                scale = ph.get("scale") or [1.0, 1.0, 1.0]

            ok = self._gen.replace_placeholder(
                prim_path=ph["path"],
                asset_path=asset_path,
                position=ph["position"],
                rotation=ph["rotation"],
                scale=scale,
                size=ph.get("size"),
                size_decimal_places=self.size_decimal_places,
            )
            if ok:
                replaced.append(ph["id"])
            else:
                failed.append(ph["id"])

        return {
            "replaced_placeholder_ids": replaced,
            "failed_placeholder_ids": failed,
            "count": len(replaced),
        }

    def remove_placeholder_assets(
        self,
        *,
        source_asset_type: str,
        placeholder_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        source_asset_type = self._normalize_asset_type(source_asset_type)
        if self._placeholders_cache is None:
            self.list_placeholders()

        if placeholder_ids:
            targets = [self.get_placeholder_by_id(pid) for pid in placeholder_ids]
        else:
            targets = [
                ph
                for ph in self._placeholders_cache or []
                if self._asset_type_match(str(ph["asset_type"]), source_asset_type)
            ]

        removed: List[str] = []
        failed: List[str] = []
        for ph in targets:
            try:
                stage_prim = self.stage.GetPrimAtPath(ph["path"])
                if not stage_prim:
                    failed.append(ph["id"])
                    continue
                self.stage.RemovePrim(ph["path"])
                removed.append(ph["id"])
            except Exception:
                failed.append(ph["id"])

        return {"removed_placeholder_ids": removed, "failed_placeholder_ids": failed, "count": len(removed)}

    def set_rgb_for_prim(self, *, prim_path: str, rgb: List[float]) -> Dict[str, Any]:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            return {"ok": False, "error": f"prim not found: {prim_path}"}
        try:
            OpenUsd.set_rgb(prim, tuple(rgb))
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def add_object_above_anchor(
        self,
        *,
        asset_type: str,
        anchor_keyword: str,
        new_prim_base_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        MVP: add a new prim with a USD reference, positioned above the first anchor prim
        whose name contains anchor_keyword (case-insensitive).
        """
        asset_type = self._normalize_asset_type(asset_type)
        # Acquire asset
        asset_path = self._gen.acquire_random_asset(asset_type)
        if not asset_path:
            raise RuntimeError(f"Failed to acquire asset for asset_type={asset_type}")

        asset_bbox = self._gen.get_asset_bbox(asset_path)
        if not asset_bbox:
            raise RuntimeError(f"Failed to get asset bbox for asset_type={asset_type}")

        anchor_keyword_norm = anchor_keyword.strip().lower()
        anchor_prim = None
        # Proper anchor search (first Xform prim whose name contains the keyword).
        for prim in self.stage.Traverse():
            if not prim or not prim.IsValid():
                continue
            if prim.GetTypeName() != "Xform":
                continue
            if anchor_keyword_norm in prim.GetName().lower():
                anchor_prim = prim
                break
        if anchor_prim is None:
            raise RuntimeError(f"Anchor prim not found for keyword: {anchor_keyword}")

        bbox = OpenUsd.get_prim_aabb_bounding_box(anchor_prim)
        anchor_min = bbox.GetMin()
        anchor_max = bbox.GetMax()

        x = 0.5 * (anchor_min[0] + anchor_max[0])
        y = 0.5 * (anchor_min[1] + anchor_max[1])
        z = anchor_max[2] + 0.5 * asset_bbox[2]

        # Follow scene root style to keep hierarchy consistent (e.g. /world vs /World).
        if not new_prim_base_path:
            new_prim_base_path = self._resolve_scene_root_path()

        # Use a group-like naming style, closer to existing scene assets.
        asset_slug = re.sub(r"[^a-z0-9]+", "_", asset_type.lower()).strip("_") or "asset"
        new_name = f"{asset_slug}_semgen_group_{uuid.uuid4().hex[:6]}"
        prim_path = f"{new_prim_base_path.rstrip('/')}/{new_name}"

        new_prim = self.stage.DefinePrim(prim_path, "Xform")
        xformable = UsdGeom.Xformable(new_prim)
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(x), float(y), float(z)))
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
        new_prim.GetReferences().AddReference(asset_path)

        # Set attrs for downstream parsing
        new_prim.CreateAttribute("type", Sdf.ValueTypeNames.String).Set(asset_type)
        new_prim.CreateAttribute("size", Sdf.ValueTypeNames.String).Set(
            f"{asset_bbox[0]},{asset_bbox[1]},{asset_bbox[2]}"
        )

        return {"ok": True, "prim_path": prim_path, "asset_path": asset_path}

    def replace_by_type_attr(
        self,
        *,
        source_asset_type: str,
        target_asset_type: str,
        limit: Optional[int] = 1,
    ) -> Dict[str, Any]:
        """
        Fallback replacement for non-placeholder scenes.
        Match Xform prims by `type` attr and replace prim reference using OpenUsd.replace_prim.
        """
        source_asset_type = self._normalize_asset_type(source_asset_type)
        target_asset_type = self._normalize_asset_type(target_asset_type)

        # Find candidates by type attr
        candidates: List[Any] = []
        for prim in self.stage.Traverse():
            if not prim or not prim.IsValid() or prim.GetTypeName() != "Xform":
                continue
            type_attr = prim.GetAttribute("type")
            if not type_attr:
                continue
            try:
                t_val = type_attr.Get()
            except Exception:
                t_val = None
            if not isinstance(t_val, str):
                continue
            if self._asset_type_match(t_val, source_asset_type):
                candidates.append(prim)

        if limit is not None and limit > 0:
            candidates = candidates[:limit]

        replaced_paths: List[str] = []
        failed_paths: List[str] = []

        for prim in candidates:
            asset_path = self._gen.acquire_random_asset(target_asset_type)
            if not asset_path:
                failed_paths.append(str(prim.GetPath()))
                continue
            try:
                OpenUsd.replace_prim(self.stage, prim, asset_path)
                replaced_paths.append(str(prim.GetPath()))
            except Exception:
                failed_paths.append(str(prim.GetPath()))

        return {
            "source_asset_type": source_asset_type,
            "target_asset_type": target_asset_type,
            "candidate_count": len(candidates),
            "replaced_paths": replaced_paths,
            "failed_paths": failed_paths,
            "count": len(replaced_paths),
        }

    def remove_by_type_attr(
        self,
        *,
        source_asset_type: str,
        limit: Optional[int] = 1,
    ) -> Dict[str, Any]:
        """
        Fallback remove for non-placeholder scenes.
        Match Xform prims by `type` attr and remove prims directly.
        """
        source_asset_type = self._normalize_asset_type(source_asset_type)

        candidates: List[Any] = []
        for prim in self.stage.Traverse():
            if not prim or not prim.IsValid() or prim.GetTypeName() != "Xform":
                continue
            type_attr = prim.GetAttribute("type")
            if not type_attr:
                continue
            try:
                t_val = type_attr.Get()
            except Exception:
                t_val = None
            if not isinstance(t_val, str):
                continue
            if self._asset_type_match(t_val, source_asset_type):
                candidates.append(prim)

        if limit is not None and limit > 0:
            candidates = candidates[:limit]

        removed_paths: List[str] = []
        failed_paths: List[str] = []
        for prim in candidates:
            path = str(prim.GetPath())
            try:
                self.stage.RemovePrim(path)
                removed_paths.append(path)
            except Exception:
                failed_paths.append(path)

        return {
            "source_asset_type": source_asset_type,
            "candidate_count": len(candidates),
            "removed_paths": removed_paths,
            "failed_paths": failed_paths,
            "count": len(removed_paths),
        }

    def save_scene(self, *, output_path: str) -> Dict[str, Any]:
        # Ensure parent dir exists
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.stage.Export(output_path)
        return {"ok": True, "output_path": output_path}


# -----------------------------
# Agent main loop
# -----------------------------
@dataclass
class AgentConfig:
    qwen_api_key: str
    qwen_base_url: str
    qwen_model: str = "qwen-plus"
    max_steps: int = 10
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_s: float = 120.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0


class SemanticGeneralizationAgent:
    def __init__(self, *, config: AgentConfig):
        self.config = config
        self.client = QwenChatClient(
            api_key=config.qwen_api_key,
            base_url=config.qwen_base_url,
            model=config.qwen_model,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            retry_backoff_s=config.retry_backoff_s,
        )

    @staticmethod
    def _parse_first_json_object(text: str) -> Dict[str, Any]:
        """
        Parse the first valid JSON object from model output.
        Handles common cases:
        - pure JSON
        - markdown fenced block with JSON
        - extra explanation text before/after JSON
        - multiple JSON blocks; we take the first valid one
        """
        # 1) fast path: whole text is valid JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 2) strip markdown fences if present
        fenced = text.strip()
        if fenced.startswith("```"):
            # remove leading ```... and trailing ```
            fenced = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", fenced)
            fenced = re.sub(r"\s*```$", "", fenced)
            try:
                obj = json.loads(fenced)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # 3) scan for first decodable JSON object using raw_decode
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

        raise RuntimeError(f"Failed to parse agent response as JSON object. Text: {text[:3000]}")

    @staticmethod
    def _parse_all_json_objects(text: str) -> List[Dict[str, Any]]:
        """
        Parse ALL valid JSON objects from model output.
        Robust to:
        - pure JSON dict
        - markdown fenced blocks
        - extra explanation text before/after
        - multiple JSON objects in one message
        """
        # 1) fast path: whole text is a JSON dict
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            pass

        # 2) strip markdown fences if present (best-effort)
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t)

        # 3) scan and raw_decode sequentially
        decoder = json.JSONDecoder()
        objs: List[Dict[str, Any]] = []
        i = 0
        while i < len(t):
            # Find next '{'
            j = t.find("{", i)
            if j < 0:
                break
            try:
                obj, end = decoder.raw_decode(t[j:])
                if isinstance(obj, dict):
                    objs.append(obj)
                i = j + end
            except Exception:
                # advance one char and continue searching
                i = j + 1

        if not objs:
            # fallback to old behavior (single obj)
            return [SemanticGeneralizationAgent._parse_first_json_object(text)]
        return objs

    def run(
        self,
        *,
        stage: Any,
        instruction: str,
        output_path: str,
        rng_seed: Optional[int] = None,
        preserve_size: bool = True,
    ) -> Dict[str, Any]:
        tools = SemanticTools(stage=stage, rng_seed=rng_seed, preserve_size=preserve_size)

        system_prompt = (
            "You are a semantic scene generalization agent for USD simulation.\n"
            "You MUST respond with valid JSON only. No markdown.\n\n"
            "You have access to the following tools. "
            "You should call tools to understand and modify the scene.\n\n"
            "Tools (tool_call):\n"
            "1) list_placeholders() -> placeholders with ids\n"
            "2) list_supported_asset_types() -> supported asset types\n"
            "3) list_type_attr_objects(source_asset_type?) -> non-placeholder editable candidates\n"
            "4) replace_placeholder_assets(source_asset_type, target_asset_type, placeholder_ids?)\n"
            "5) replace_by_type_attr(source_asset_type, target_asset_type, limit?)\n"
            "6) remove_placeholder_assets(source_asset_type, placeholder_ids?)\n"
            "7) remove_by_type_attr(source_asset_type, limit?)\n"
            "8) add_object_above_anchor(asset_type, anchor_keyword, new_prim_base_path?)\n"
            "9) set_rgb_for_prim(prim_path, rgb)\n"
            "10) save_scene(output_path)\n\n"
            "Task: interpret the user instruction and generalize the scene by replacing/removing/adding assets.\n"
            "Constraints:\n"
            "- First call list_placeholders(). If empty, call list_type_attr_objects() and use replace_by_type_attr().\n"
            "- If user asks to delete/remove in non-placeholder scene, use remove_by_type_attr().\n"
            "- For multi-clause instructions, execute EVERY clause in order; do not merge delete and replace into one action.\n"
            "- For multi-clause instructions, emit tool_call JSONs in the SAME order as the user clauses "
            "(e.g. delete first, then replace fridge, then add oven). Do not reorder steps.\n"
            "- For replacement intent, prefer replacement tools; do not add new objects unless user explicitly asks add.\n"
            "- Asset type args SHOULD use list_supported_asset_types() when applying to random asset acquisition.\n"
            "- For remove_by_type_attr / replace_by_type_attr matching Xform prim `type`, the scene may contain "
            "`type` values NOT listed in list_supported_asset_types(), e.g. Toaster, ToasterOven. Use those exact strings.\n"
            "- IMPORTANT: Normalize natural-language using the alias table. Do NOT invent new types.\n"
            "  Alias table: fridge/refrigerator->Fridge; microwave->Microwave; oven->Oven; dishwasher->Dishwasher;\n"
            "  stove->Stove; sink->Sink; counter->Counter; cabinet->SingleCabinet;\n"
            "  toaster/面包机/烤面包机->Toaster; toaster_oven/toasteroven/面包机烤箱->ToasterOven.\n"
            "- If user asks to remove 面包机 (pop-up toaster), call remove_by_type_attr(source_asset_type=\"Toaster\", limit=1)\n"
            "  after list_type_attr_objects(source_asset_type=\"Toaster\") confirms candidates. "
            "ToasterOven is a different `type`; do not confuse it unless the user clearly means the toaster-oven appliance.\n"
            "- If user asks to replace a fridge (冰箱/fridge/refrigerator), you MUST set:\n"
            "  replace_by_type_attr(source_asset_type=\"Fridge\", target_asset_type=\"Fridge\", limit=1)\n"
            "  or replace_placeholder_assets(source_asset_type=\"Fridge\", target_asset_type=\"Fridge\", ...).\n"
            "  NEVER replace unrelated types (e.g. SingleCabinet) into Fridge when the user asks to replace the fridge.\n"
            "- When replacing a fridge in non-placeholder scene, first call list_type_attr_objects(source_asset_type=\"Fridge\")\n"
            "  to confirm at least one Fridge* candidate exists; then replace_by_type_attr(source_asset_type=\"Fridge\", ...).\n"
            "- For add_object_above_anchor, use anchor_keyword like 'counter'.\n"
            "- Prefer preserve_size behavior.\n\n"
            "Response format:\n"
            "- Tool call: {\"tool_call\": {\"name\": \"...\", \"arguments\": {...}}}\n"
            "- Final: {\"final\": {...}}\n"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": instruction,
                        "output_path": output_path,
                        "notes": {
                            "replace/remove/add operations": "Use tools to modify USD.",
                            "placeholder ids": "Use ids from list_placeholders() response.",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        # Tools registry for execution
        tool_map: Dict[str, Callable[..., Dict[str, Any]]] = {
            "list_placeholders": lambda: {"placeholders": tools.list_placeholders()},
            "list_supported_asset_types": lambda: {"asset_types": tools.list_supported_asset_types()},
            "list_type_attr_objects": lambda **kwargs: tools.list_type_attr_objects(**kwargs),
            "replace_placeholder_assets": lambda **kwargs: tools.replace_placeholder_assets(**kwargs),
            "replace_by_type_attr": lambda **kwargs: tools.replace_by_type_attr(**kwargs),
            "remove_placeholder_assets": lambda **kwargs: tools.remove_placeholder_assets(**kwargs),
            "remove_by_type_attr": lambda **kwargs: tools.remove_by_type_attr(**kwargs),
            "add_object_above_anchor": lambda **kwargs: tools.add_object_above_anchor(**kwargs),
            "set_rgb_for_prim": lambda **kwargs: tools.set_rgb_for_prim(**kwargs),
            "save_scene": lambda **kwargs: tools.save_scene(**kwargs),
        }

        applied_log: List[Dict[str, Any]] = []
        saved = False

        for step in range(self.config.max_steps):
            assistant_text = self.client.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            objs = self._parse_all_json_objects(assistant_text)

            # Execute tools for ALL JSON dicts in order.
            # We record assistant message once per step.
            messages.append({"role": "assistant", "content": assistant_text})

            for obj in objs:
                if "tool_call" in obj:
                    call = obj.get("tool_call") or {}
                    name = call.get("name")
                    arguments = call.get("arguments", {}) or {}

                    if name not in tool_map:
                        result = {"ok": False, "error": f"Unknown tool: {name}", "tool": name, "arguments": arguments}
                    else:
                        try:
                            result = tool_map[name](**arguments)
                        except Exception as e:
                            result = {"ok": False, "error": str(e), "tool": name, "arguments": arguments}

                    if name == "save_scene":
                        saved = True
                    if isinstance(result, dict) and "count" in result:
                        entry = {"op": name, "count": result.get("count")}
                        applied_log.append(entry)
                    else:
                        applied_log.append({"op": name, **result})

                    messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    continue

                if "final" in obj:
                    final_obj = obj["final"] or {}
                    if "applied" not in final_obj:
                        final_obj["applied"] = applied_log
                    if "saved_to" not in final_obj:
                        final_obj["saved_to"] = output_path
                    if not saved:
                        tools.save_scene(output_path=output_path)
                        saved = True
                        final_obj.setdefault("saved_to", output_path)
                    return {"status": "ok", **final_obj}

            # If save_scene was called but no final returned, still return.
            if saved:
                return {"status": "ok", "applied": applied_log, "saved_to": output_path}

        # If we exhaust steps, still attempt to save (best-effort)
        tools.save_scene(output_path=output_path)
        return {
            "status": "ok",
            "summary": "Reached max_steps; saved scene with best-effort modifications.",
            "applied": applied_log,
            "saved_to": output_path,
        }


def load_default_agent_config_from_env() -> AgentConfig:
    api_key = os.environ.get("QWEN_API_KEY") or ""
    base_url = os.environ.get("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not api_key:
        raise RuntimeError("Missing QWEN_API_KEY env var")
    model = os.environ.get("QWEN_MODEL") or "qwen-plus"
    return AgentConfig(qwen_api_key=api_key, qwen_base_url=base_url, qwen_model=model)
