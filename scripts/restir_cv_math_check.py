#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/restir_cv_math_check.py --shader-root shaders --case full
# 3. Or make executable and run:
#      chmod +x scripts/restir_cv_math_check.py && ./scripts/restir_cv_math_check.py --shader-root shaders --case full
# ──────────────────

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

Vec3 = tuple[float, float, float]
POOL_SIZES = (
    ("65536", "8388608"),
    ("131072", "16777216"),
    ("262144", "33554432"),
    ("524288", "67108864"),
)


@dataclass(frozen=True, slots=True)
class Sources:
    radiance_cache: str
    update: str
    sample: str
    props_source: str
    props_generated: str
    options_source: str
    options_generated: str


def load_sources(args: argparse.Namespace) -> Sources:
    root = args.shader_root
    repo = root.parent
    return Sources(
        radiance_cache=(args.override_radiance_cache or root / "techniques/gi/RadianceCache.glsl").read_text(
            encoding="utf-8"
        ),
        update=(args.override_radiance_cache_update or root / "techniques/gi/RadianceCacheUpdate.glsl").read_text(
            encoding="utf-8"
        ),
        sample=(args.override_radiance_cache_sample or root / "techniques/gi/RadianceCacheSample.glsl").read_text(
            encoding="utf-8"
        ),
        props_source=(repo / "scripts/shaders.properties").read_text(encoding="utf-8"),
        props_generated=(root / "shaders.properties").read_text(encoding="utf-8"),
        options_source=(repo / "scripts/options.main.kts").read_text(encoding="utf-8"),
        options_generated=(root / "base/Options.glsl").read_text(encoding="utf-8"),
    )


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def contains_expr(text: str, expr: str) -> bool:
    return compact(expr) in compact(text)


def block_body(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        return ""
    start = text.find("{", match.end())
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return ""


def function_body(text: str, name: str) -> str:
    return block_body(text, rf"\b{re.escape(name)}\s*\(")


def struct_body(text: str, name: str) -> str:
    return block_body(text, rf"\bstruct\s+{re.escape(name)}\b")


def require(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def require_exprs(failures: list[str], scope: str, prefix: str, checks: tuple[tuple[str, str], ...]) -> None:
    for expr, message in checks:
        require(failures, contains_expr(scope, expr), f"{prefix}: {message}")


def assert_vec_close(actual: Vec3, expected: Vec3) -> None:
    assert all(abs(a - e) < 1e-6 for a, e in zip(actual, expected)), (actual, expected)


def check_buffer_sizes(failures: list[str], name: str, text: str) -> None:
    require(failures, "4 uvec4 records" in text, f"{name}: binding 11 comment must name 4 uvec4 records")
    for pool_size, expected_size in POOL_SIZES:
        pattern = rf"#(?:if|elif)\s+SETTING_RC_POOL_SIZE\s*==\s*{pool_size}\s*\n\s*bufferObject\.11={expected_size}"
        require(
            failures,
            re.search(pattern, text) is not None,
            f"{name}: SETTING_RC_POOL_SIZE {pool_size} must size binding 11 to {expected_size}",
        )
    require(
        failures,
        re.search(r"#else\s*\n\s*bufferObject\.11=134217728", text) is not None,
        f"{name}: fallback binding 11 size must be 134217728",
    )
    for old_size in ("6291456", "12582912", "25165824", "50331648", "100663296"):
        require(failures, old_size not in text, f"{name}: old 3-record binding size {old_size} remains")


def check_layout(sources: Sources) -> list[str]:
    failures: list[str] = []
    text = sources.radiance_cache
    require(failures, "#define RC_RESERVOIR_RECORDS 4u" in text, "RadianceCache: missing 4-record layout")
    require_exprs(failures, struct_body(text, "RCReservoir"), "RadianceCache layout", (("vec3 estimate;", "missing Fi"),))
    require_exprs(
        failures,
        function_body(text, "rc_reservoirLoad"),
        "RadianceCache load",
        (
            ("uvec4 r3 = rc_reservoirs[recordIndex + 3u];", "missing record 3"),
            ("reservoir.estimate = uintBitsToFloat(r3.xyz);", "missing estimate unpack"),
        ),
    )
    require_exprs(
        failures,
        function_body(text, "rc_reservoirStore"),
        "RadianceCache store",
        (("rc_reservoirs[recordIndex + 3u] = uvec4(floatBitsToUint(reservoir.estimate), 0u);", "missing Fi pack"),),
    )
    estimate_body = function_body(text, "rc_reservoirEstimateRadiance")
    require(failures, contains_expr(estimate_body, "reservoir.estimate"), "RadianceCache: helper must return Fi")
    require(
        failures,
        not contains_expr(estimate_body, "reservoir.radiance * reservoir.avgWY"),
        "RadianceCache: helper still reconstructs sample * avgWY",
    )
    check_buffer_sizes(failures, "scripts/shaders.properties", sources.props_source)
    check_buffer_sizes(failures, "shaders/shaders.properties", sources.props_generated)
    return failures


def check_temporal(sources: Sources) -> list[str]:
    failures: list[str] = []
    text = sources.update
    require(failures, "const float RC_CV_ALPHA = 1.0;" in text, "RadianceCacheUpdate: missing CV alpha")
    require(failures, "const float RC_CV_M_CAP = 20.0;" in text, "RadianceCacheUpdate: missing CV M cap")
    require_exprs(failures, struct_body(text, "RCCVAccumulator"), "accumulator", (("bool invalid;", "missing poison field"),))
    for name, checks in (
        ("rc_cvAccumulatorInit", (("accumulator.invalid = false;", "must clear poison"),)),
        (
            "rc_cvAccumulatorAdd",
            (
                ("weight <= 0.0", "must ignore non-positive q"),
                ("isnan(weight) || any(isnan(estimate))", "must detect invalid q/estimate"),
                ("accumulator.invalid = true;", "must poison invalid state"),
            ),
        ),
        ("rc_cvAccumulatorValid", (("!accumulator.invalid", "must reject poisoned accumulators"),)),
        ("rc_cvSampleContribution", (("return reservoir.radiance * reservoir.avgWY;", "must compute y / p_hat"),)),
        ("rc_cvInitialEstimate", (("return candidate.valid ? candidate.radiance : vec3(0.0);", "must use direct sample"),)),
    ):
        require_exprs(failures, function_body(text, name), f"RadianceCacheUpdate {name}", checks)
    require_exprs(
        failures,
        function_body(text, "rc_updateFace"),
        "RadianceCacheUpdate temporal",
        (
            ("float qInit = candidate.valid ? 1.0 : 0.0;", "missing initial q"),
            ("float qHistory = historyValid ? min(historyM, RC_CV_M_CAP) : 0.0;", "missing capped history q"),
            ("vec3 oldHistorySample = rc_cvSampleContribution(historyBeforeRevalidate);", "missing old history sample"),
            ("vec3 currentHistorySample = rc_cvSampleContribution(reservoir);", "missing current history sample"),
            (
                "vec3 fromHistory = historyBeforeRevalidate.estimate + (currentHistorySample - RC_CV_ALPHA * oldHistorySample);",
                "missing temporal CV difference",
            ),
            ("rc_cvAccumulatorAdd(cvAccumulator, fromHistory, qHistory);", "history Fi must enter accumulator"),
            ("reservoir.estimate = rc_cvAccumulatorResolve(cvAccumulator);", "final reservoir must store Fi"),
        ),
    )
    return failures


def check_spatial(sources: Sources) -> list[str]:
    failures: list[str] = []
    text = sources.update
    require(failures, re.search(r"out\s+float\s+misWeight", text) is not None, "spatial: missing MIS output")
    require_exprs(
        failures,
        function_body(text, "rc_generateSpatialCandidate") + function_body(text, "rc_updateFace"),
        "RadianceCacheUpdate spatial",
        (
            ("misWeight = rc_pairwiseSpatialMIS_MAware", "must expose pairwise MIS"),
            ("float qSpatial = min(sourceM, RC_CV_M_CAP) * SETTING_RC_SPATIAL_STRENGTH;", "missing spatial q"),
            (
                "vec3 spatialDelta = misWeight * sourceCorrection * neighborReservoir.avgWY * (spatialCandidate.radiance - RC_CV_ALPHA * neighborReservoir.radiance);",
                "missing spatial CV delta",
            ),
            ("vec3 fromSpatial = neighborReservoir.estimate + spatialDelta;", "missing neighbor Fi CV"),
            ("rc_cvAccumulatorAdd(cvAccumulator, fromSpatial, qSpatial);", "spatial Fi must enter accumulator"),
        ),
    )
    return failures


def check_lookup(sources: Sources) -> list[str]:
    failures: list[str] = []
    require(
        failures,
        re.search(r"reservoir\.radiance\s*\*\s*reservoir\.avgWY", sources.sample) is None,
        "RadianceCacheSample: lookup still uses sample * avgWY",
    )
    require(
        failures,
        sources.sample.count("rc_reservoirEstimateRadiance(reservoir)") >= 2,
        "RadianceCacheSample: both lookup paths must call rc_reservoirEstimateRadiance",
    )
    require(
        failures,
        'slider("SETTING_DEBUG_RC_MODE", 0, 0..10)' in sources.options_source,
        "options.main.kts: RC debug slider must include mode 10",
    )
    require(
        failures,
        "#define SETTING_DEBUG_RC_MODE 0//[0 1 2 3 4 5 6 7 8 9 10]" in sources.options_generated,
        "Options.glsl: generated RC debug range must include mode 10",
    )
    return failures


CHECKS = {"layout": check_layout, "temporal": check_temporal, "spatial": check_spatial, "lookup": check_lookup}


def run_checks(case: str, sources: Sources) -> list[str]:
    checks = CHECKS.values() if case == "full" else (CHECKS[case],)
    return [failure for check in checks for failure in check(sources)]


def self_test() -> None:
    sample = "vec3 f() { if (true) { return a + (b - C * d); } return x; }"
    assert contains_expr(sample, "a + (b - C * d)")
    assert "return x;" in function_body(sample, "f")
    from_history = tuple(h + c - o for h, c, o in zip((2.0, 1.0, 0.5), (1.5, 0.25, 0.75), (1.0, 0.5, 0.25)))
    estimate_sum = tuple(i + 4.0 * h for i, h in zip((0.3, 0.6, 0.9), from_history))
    assert_vec_close(tuple(x / 5.0 for x in estimate_sum), (2.06, 0.72, 0.98))
    spatial_delta = tuple((c - n) * 0.25 * 0.8 * 2.0 for c, n in zip((2.0, 3.0, 5.0), (0.5, 0.25, 0.125)))
    assert_vec_close(tuple(n + d for n, d in zip((0.7, 1.1, 1.3), spatial_delta)), (1.3, 2.2, 3.25))
    nan = float("nan")
    assert any(x != x for x in (nan, 0.0, 0.0))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ReSTIR-CV radiance-cache shader contracts.")
    parser.add_argument("--shader-root", type=Path, default=Path("shaders"))
    parser.add_argument("--case", choices=("full", *CHECKS.keys()), default="full")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--override-radiance-cache", type=Path)
    parser.add_argument("--override-radiance-cache-update", type=Path)
    parser.add_argument("--override-radiance-cache-sample", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("PASS self-test")
        return 0
    try:
        failures = run_checks(args.case, load_sources(args))
    except OSError as exc:
        print(f"FAIL {exc}")
        return 1
    if failures:
        print(f"FAIL ReSTIR-CV contract ({args.case})")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PASS ReSTIR-CV contract ({args.case})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
