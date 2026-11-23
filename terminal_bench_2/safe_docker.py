"""Project-local Harbor Docker environment; never manages a Docker daemon."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import TaskConfig
from harbor.models.trial.paths import TrialPaths


ROOT = Path(__file__).resolve().parent
DOCKER_HOST = "unix:///var/run/docker.sock"
DAEMON_ID = "592b3c79-f74b-41b0-9105-070626f12648"
NETWORK = "tthe-tb2-cont-20261002"
SUBNET = "10.253.91.0/24"
OWNER_LABEL = "org.tthe.owner"
OWNER = "terminal-bench-cont-20261002"


def direct_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
                "https_proxy", "all_proxy", "DOCKER_CONTEXT", "DOCKER_HOST"):
        env.pop(key, None)
    env.update(DOCKER_HOST=DOCKER_HOST, NO_PROXY="*", no_proxy="*")
    return env


def docker(*args: str, host: str = DOCKER_HOST) -> str:
    return subprocess.run(
        ["docker", "--host", host, *args], env=direct_env(), check=True,
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def check_daemon() -> None:
    info = json.loads(docker("info", "--format", "{{json .}}"))
    if info["ID"] != DAEMON_ID or info["DockerRootDir"] != "/data/docker":
        raise RuntimeError("Unexpected system Docker identity; refusing to run")


def ensure_network() -> dict:
    """Use Docker's API only, after checking both daemons and host routes."""
    check_daemon()
    existing = docker("network", "ls", "--filter", f"name=^{NETWORK}$", "-q")
    if existing:
        network = json.loads(docker("network", "inspect", existing))[0]
        if (network.get("Labels", {}).get(OWNER_LABEL) != OWNER
                or network["Driver"] != "bridge"
                or network["IPAM"]["Config"][0]["Subnet"] != SUBNET):
            raise RuntimeError("Network name is already owned by another configuration")
        return network

    target = ipaddress.ip_network(SUBNET)
    routes = json.loads(subprocess.run(
        ["ip", "-j", "-4", "route", "show", "table", "all"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout)
    occupied = [row["dst"] for row in routes if row.get("dst") not in (None, "default")]
    for host in (DOCKER_HOST, "unix:///data/ygzhang/nj/dockerswe.sock"):
        ids = docker("network", "ls", "-q", host=host).split()
        for network in json.loads(docker("network", "inspect", *ids, host=host)):
            occupied.extend(config["Subnet"] for config in (network["IPAM"]["Config"] or [])
                            if "Subnet" in config and ":" not in config["Subnet"])
    if any(target.overlaps(ipaddress.ip_network(value, strict=False)) for value in occupied):
        raise RuntimeError(f"Proposed task subnet {SUBNET} overlaps existing networking")
    docker("network", "create", "--driver", "bridge", "--subnet", SUBNET,
           "--label", f"{OWNER_LABEL}={OWNER}", NETWORK)
    return json.loads(docker("network", "inspect", NETWORK))[0]


def remove_idle_network() -> None:
    existing = docker("network", "ls", "--filter", f"name=^{NETWORK}$", "-q")
    if existing:
        network = json.loads(docker("network", "inspect", existing))[0]
        if network.get("Labels", {}).get(OWNER_LABEL) == OWNER and not network["Containers"]:
            docker("network", "rm", network["Id"])


def validate_config(config: dict, trial_dir: Path) -> None:
    services = config.get("services", {})
    if set(services) != {"main"}:
        raise RuntimeError("Only the audited single-service tasks are supported")
    service = services["main"]
    if (service.get("network_mode") not in (None, "none")
            or service.get("privileged") or service.get("cap_add")
            or service.get("pid") or service.get("ipc") == "host"
            or service.get("ports") or service.get("devices")
            or service.get("container_name") or service.get("build")):
        raise RuntimeError("Unsafe host-sharing/privilege/build configuration")
    if service.get("network_mode") != "none":
        if set(service.get("networks", {})) != {"tthe"}:
            raise RuntimeError("Task is not exclusively attached to the TTHE network")
        network = config["networks"]["tthe"]
        if not network.get("external") or network.get("name") != NETWORK:
            raise RuntimeError("Unexpected task network")
    if service.get("pull_policy") != "never":
        raise RuntimeError("Automatic image pulls are forbidden")
    for volume in service.get("volumes", []):
        source = Path(volume.get("source", "")).resolve()
        if (volume.get("type") != "bind" or not source.is_relative_to(trial_dir.resolve())
                or volume.get("target") not in ("/logs/agent", "/logs/verifier", "/logs/artifacts")):
            raise RuntimeError("Bind mount outside this trial's log directories")
    limits = service.get("deploy", {}).get("resources", {}).get("limits", {})
    if not limits.get("cpus") or not limits.get("memory"):
        raise RuntimeError("CPU/memory limits are missing")


class SafeDockerEnvironment(DockerEnvironment):
    _DOCKER_COMPOSE_PREBUILT_PATH = ROOT / "safe_docker_compose.yaml"

    @classmethod
    def preflight(cls) -> None:
        check_daemon()

    def __init__(self, *args, **kwargs):
        kwargs["session_id"] = "tthe-safe-" + kwargs["session_id"]
        super().__init__(*args, **kwargs)

    async def start(self, force_build: bool):
        if force_build or not self.task_env_config.docker_image:
            raise RuntimeError("Safe continuation requires an already-cached prebuilt image")
        if self._uses_compose or self._mounts_json:
            raise RuntimeError("Unaudited task compose overrides/mounts are forbidden")
        if os.environ.get("DOCKER_HOST") != DOCKER_HOST or os.environ.get("DOCKER_CONTEXT"):
            raise RuntimeError("System Docker endpoint must be pinned")
        docker("image", "inspect", self.task_env_config.docker_image)
        network = json.loads(docker("network", "inspect", NETWORK))[0]
        if network.get("Labels", {}).get(OWNER_LABEL) != OWNER:
            raise RuntimeError("Task network ownership mismatch")
        self._use_prebuilt = True
        result = await self._run_docker_compose_command(["config", "--format", "json"])
        config = json.loads(result.stdout)
        validate_config(config, self.trial_paths.trial_dir)
        (self.trial_paths.trial_dir / "safe-compose.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")
        await super().start(force_build=False)


def host_state() -> dict:
    return {
        "default_route": subprocess.run(["ip", "-4", "route", "show", "default"],
                                        check=True, capture_output=True, text=True).stdout,
        "resolv_conf": Path("/etc/resolv.conf").read_text(),
        "dockerd_pids": subprocess.run(["pgrep", "-x", "dockerd"], check=True,
                                       capture_output=True, text=True).stdout.split(),
    }


async def smoke() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ.update(direct_env())
    os.environ.pop("DOCKER_CONTEXT", None)
    before = host_state()
    existing = bool(docker("network", "ls", "--filter", f"name=^{NETWORK}$", "-q"))
    network = ensure_network()
    trial_dir = ROOT / ".tthe_tmp" / f"safe-smoke-{time.time_ns()}"
    paths = TrialPaths(trial_dir)
    paths.mkdir()
    task_dir = ROOT / ".harbor-cache/tasks/QUdXqzZFwRrgRZ7SPHyxto/train-fasttext"
    config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
    environment = SafeDockerEnvironment(
        environment_dir=task_dir / "environment", environment_name="train-fasttext",
        session_id=trial_dir.name, trial_paths=paths, task_env_config=config.environment,
    )
    report = {"before": before, "network_id": network["Id"], "trial_dir": str(trial_dir)}
    try:
        await environment.start(force_build=False)
        ids = (await environment._run_docker_compose_command(["ps", "-q"])).stdout.split()
        if len(ids) != 1:
            raise RuntimeError("Expected exactly one smoke container")
        container = json.loads(docker("inspect", ids[0]))[0]
        host = container["HostConfig"]
        if (host["NetworkMode"] in ("host", "bridge") or host["Privileged"]
                or host["NanoCpus"] != config.environment.cpus * 1_000_000_000
                or host["Memory"] != config.environment.memory_mb * 1024 * 1024
                or set(container["NetworkSettings"]["Networks"]) != {NETWORK}):
            raise RuntimeError("Runtime isolation/resource limits do not match the task")
        probe = (
            "import urllib.request, urllib.error\n"
            "opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
            "try:\n"
            "    response = opener.open('https://api.deepseek.com/models', timeout=15)\n"
            "    print(response.status)\n"
            "except urllib.error.HTTPError as error:\n"
            "    print(error.code)\n"
        )
        result = await environment.exec("python -c " + shlex.quote(probe), timeout_sec=25)
        if result.return_code:
            raise RuntimeError(f"Container direct connectivity failed: {result.stdout}")
        report.update(container_id=ids[0], network_mode=host["NetworkMode"],
                      nano_cpus=host["NanoCpus"], memory=host["Memory"], http_status=result.stdout)
        tools = await environment.exec("command -v tmux || true")
        report["tmux_preinstalled"] = bool((tools.stdout or "").strip())
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        await environment.stop(delete=True)
        if not existing:
            owned = json.loads(docker("network", "inspect", network["Id"]))[0]
            if owned.get("Labels", {}).get(OWNER_LABEL) == OWNER and not owned["Containers"]:
                docker("network", "rm", network["Id"])
        report["after"] = host_state()
        report["host_state_unchanged"] = before == report["after"]
        report["remaining_smoke_containers"] = docker(
            "ps", "-aq", "--filter", f"label=com.docker.compose.project={environment.session_id}")
        (trial_dir / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ("before", "after")}, indent=2), flush=True)
    if not report["host_state_unchanged"] or report["remaining_smoke_containers"]:
        raise RuntimeError("Smoke cleanup/host-state verification failed")


if __name__ == "__main__":
    asyncio.run(smoke())
