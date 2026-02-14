import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import shlex


def file_to_module(file_path: Path, workspace: Path) -> str:
    rel = file_path.resolve().relative_to(workspace.resolve())
    if rel.suffix != ".mojo":
        raise ValueError(f"Not a Mojo file: {file_path}")
    if rel.name == "__init__.mojo":
        raise ValueError(f"Skipping package init file: {rel}")
    return ".".join(rel.with_suffix("").parts)


def collect_modules(workspace: Path, package: str, explicit_files: list[str]) -> list[str]:
    modules: list[str] = []
    if explicit_files:
        for value in explicit_files:
            path = Path(value)
            if not path.is_absolute():
                path = workspace / path
            modules.append(file_to_module(path, workspace))
        return modules

    package_path = workspace / package.replace(".", "/")
    if not package_path.exists():
        raise FileNotFoundError(f"Package path not found: {package_path}")

    for file_path in sorted(package_path.rglob("*.mojo")):
        if file_path.name == "__init__.mojo":
            continue
        modules.append(file_to_module(file_path, workspace))
    return modules


def build_module(module_name: str, workspace: Path, mojo_cmd: str) -> tuple[bool, str]:
    cmd_prefix = shlex.split(mojo_cmd)
    with tempfile.TemporaryDirectory(prefix="mojo_module_check_") as tmp:
        tmp_path = Path(tmp)
        harness = tmp_path / "harness.mojo"
        output_bin = tmp_path / "harness_bin"
        harness.write_text(f"from {module_name} import *\n\n" "def main():\n    pass\n")

        cmd = cmd_prefix + [
            "build",
            "-I",
            str(workspace),
            str(harness),
            "-o",
            str(output_bin),
        ]
        run = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)

    output = (run.stdout or "") + (run.stderr or "")
    return run.returncode == 0, output.strip()


def package_root_from_module(module_name: str, workspace: Path) -> str:
    first = module_name.split(".", 1)[0]
    candidate = workspace / first
    if not (candidate / "__init__.mojo").exists():
        raise FileNotFoundError(f"Cannot find package root for module '{module_name}' at {candidate}")
    return first


def build_package(package_name: str, workspace: Path, mojo_cmd: str) -> tuple[bool, str]:
    cmd_prefix = shlex.split(mojo_cmd)
    with tempfile.TemporaryDirectory(prefix="mojo_package_check_") as tmp:
        out_pkg = Path(tmp) / f"{package_name}.mojopkg"
        cmd = cmd_prefix + [
            "package",
            "-I",
            str(workspace),
            str(workspace / package_name.replace(".", "/")),
            "-o",
            str(out_pkg),
        ]
        run = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)

    output = (run.stdout or "") + (run.stderr or "")
    return run.returncode == 0, output.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        help="Mojo file paths to validate (for example: mmm_audio/Oscillators.mojo)",
    )
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--package", default="mmm_audio", help="Package to scan when no files are provided")
    parser.add_argument(
        "--mojo-cmd",
        default="pixi run mojo",
        help='Mojo command prefix (for example: "pixi run mojo" or "mojo")',
    )
    parser.add_argument(
        "--mode",
        choices=["import", "package", "both"],
        default="both",
        help="Validation mode: import harness build, package compile, or both (default)",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()

    if shutil.which("pixi") is None and args.mojo_cmd == "pixi run mojo":
        print("error: pixi not found; use --mojo-cmd mojo if mojo is on PATH")
        return 2

    try:
        modules = collect_modules(workspace, args.package, args.files)
    except Exception as exc:
        print(f"error: {exc}")
        return 2

    if not modules:
        print("No modules to validate.")
        return 0

    failures = 0
    if args.mode in ("import", "both"):
        for module_name in modules:
            ok, output = build_module(module_name, workspace, args.mojo_cmd)
            if ok:
                print(f"PASS  import {module_name}")
            else:
                failures += 1
                print(f"FAIL  import {module_name}")
                if output:
                    print(output)
                print("-" * 72)

    if args.mode in ("package", "both"):
        package_names = sorted({package_root_from_module(module_name, workspace) for module_name in modules})
        for package_name in package_names:
            ok, output = build_package(package_name, workspace, args.mojo_cmd)
            if ok:
                print(f"PASS  package {package_name}")
            else:
                failures += 1
                print(f"FAIL  package {package_name}")
                if output:
                    print(output)
                print("-" * 72)

    total = len(modules)
    print(f"Summary: checked {total} module(s), failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())