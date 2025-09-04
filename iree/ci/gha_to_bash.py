#!/usr/bin/env python3
import sys
import yaml
import textwrap
import re


def substitute_vars(s: str, env: dict) -> str:
    """Replace ${{ ... }} with env values if known."""
    if not isinstance(s, str):
        return s
    for k, v in env.items():
        s = re.sub(rf"\${{{{\s*{k}\s*}}}}", str(v), s)
    return s


def translate_step(step, env_vars):
    lines = []

    if "run" in step:
        cmd = substitute_vars(step["run"], env_vars)
        lines.append(cmd)

    elif "uses" in step:
        uses = step["uses"]

        if uses.startswith("actions/checkout"):
            repo = step.get("with", {}).get("repository", "https://github.com/openxla/iree")
            ref = step.get("with", {}).get("ref", "")
            path = step.get("with", {}).get("path", ".")
            clone_cmd = f"git clone {repo} {path}"
            if ref:
                clone_cmd += f" && cd {path} && git checkout {ref} && cd -"
            lines.append(clone_cmd)

        elif uses.startswith("actions/setup-python"):
            pyver = step.get("with", {}).get("python-version", "3.11")
            lines.append(textwrap.dedent(f"""
                echo "Using Python {pyver}"
                python{pyver} -m venv $VENV_DIR
                source $VENV_DIR/bin/activate
            """).strip())

        elif uses.startswith("actions/download-artifact"):
            name = step.get("with", {}).get("name", "artifact")
            path = step.get("with", {}).get("path", "./artifacts")
            lines.append(textwrap.dedent(f"""
                echo "[stub] Would download artifact {name} -> {path}"
                mkdir -p {path}
            """).strip())

        else:
            lines.append(f"echo '[TODO] Unsupported uses: {uses}'")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 4:
        print("Usage: gha_to_bash.py workflow.yml job_name matrix_name")
        sys.exit(1)

    workflow_file, job_name, matrix_name = sys.argv[1:4]

    with open(workflow_file) as f:
        wf = yaml.safe_load(f)

    if job_name not in wf["jobs"]:
        print(f"Job '{job_name}' not found in workflow")
        sys.exit(1)

    job = wf["jobs"][job_name]

    # Find the right matrix entry
    matrix_entry = None
    for entry in job["strategy"]["matrix"]["include"]:
        if entry["name"] == matrix_name:
            matrix_entry = entry
            break
    if not matrix_entry:
        print(f"Matrix '{matrix_name}' not found in job '{job_name}'")
        sys.exit(1)

    # Build env (job env + matrix values)
    env_vars = {}
    if "env" in job:
        for k, v in job["env"].items():
            env_vars[k] = substitute_vars(v, {**env_vars, **matrix_entry})
    for k, v in matrix_entry.items():
        env_vars[f"matrix.{k}"] = v
        env_vars[k] = v  # also allow bare form

    script_lines = ["#!/bin/bash", "set -euo pipefail", ""]

    for step in job["steps"]:
        name = step.get("name", step.get("uses", step.get("run", "unnamed")))
        script_lines.append(f"echo '=== Step: {name} ==='")
        translated = translate_step(step, env_vars)
        if translated:
            script_lines.append(translated)
        script_lines.append("")

    out_file = f"run_{job_name}_{matrix_name}.sh"
    with open(out_file, "w") as f:
        f.write("\n".join(script_lines))
    print(f"Generated {out_file}")


if __name__ == "__main__":
    main()

