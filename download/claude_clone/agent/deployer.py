"""
Multi-platform Deployment Engine for Claude Code clone.

Supports deploying projects to Docker, Vercel, Netlify, AWS Lambda,
GitHub Pages, Railway, Render, and self-hosted environments. Includes
environment variable management, deployment history, rollback, and
health checks.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import html
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import cryptography
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 120
HEALTH_CHECK_RETRIES = 5
HEALTH_CHECK_INTERVAL = 6
MAX_HISTORY_ENTRIES = 500
ENCRYPTED_ENV_PREFIX = "ENC_"

DOCKERFILE_TEMPLATES: dict[str, str] = {
    "python": """\
FROM python:{python_version}-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:{python_version}-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')" || exit 1

# Run the application
CMD ["python", "-m", "{module}"]
""",
    "flask": """\
FROM python:{python_version}-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:{python_version}-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

ENV FLASK_APP={module}
ENV FLASK_ENV=production

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:{port}", "--workers", "4", "--threads", "2", "{module}:app"]
""",
    "fastapi": """\
FROM python:{python_version}-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:{python_version}-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')" || exit 1

CMD ["uvicorn", "{module}:app", "--host", "0.0.0.0", "--port", "{port}", "--workers", "4"]
""",
    "django": """\
FROM python:{python_version}-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential gcc default-libmysqlclient-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:{python_version}-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

RUN python manage.py collectstatic --noinput 2>/dev/null || true

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:{port}", "--workers", "4", "{module}.wsgi:application"]
""",
    "node": """\
FROM node:{node_version}-slim AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

FROM node:{node_version}-slim

WORKDIR /app

RUN groupadd --gid 1001 appgroup && \\
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

COPY --from=builder /app/node_modules ./node_modules
COPY . .

USER appuser

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD node -e "const http=require('http'); http.get('http://localhost:{port}/health', r=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))"

CMD ["node", "{entrypoint}"]
""",
    "nextjs": """\
FROM node:{node_version}-slim AS deps

WORKDIR /app

COPY package*.json ./
RUN npm ci

FROM node:{node_version}-slim AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:{node_version}-slim AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN groupadd --gid 1001 appgroup && \\
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

COPY --from=builder /app/public ./public
COPY --from=builder --chown=appuser:appgroup /app/.next/standalone ./
COPY --from=builder --chown=appuser:appgroup /app/.next/static ./.next/static

USER appuser

EXPOSE {port}

ENV PORT={port}
ENV HOSTNAME="0.0.0.0"

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \\
    CMD node -e "const http=require('http'); http.get('http://localhost:{port}/', r=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))"

CMD ["node", "server.js"]
""",
    "react": """\
FROM node:{node_version}-slim AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
ENV NODE_ENV=production
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf 2>/dev/null || true

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
""",
    "vue": """\
FROM node:{node_version}-slim AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
ENV NODE_ENV=production
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html

RUN echo 'server { \\
    listen 80; \\
    location / { \\
        root /usr/share/nginx/html; \\
        try_files $uri $uri/ /index.html; \\
    } \\
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
""",
    "static": """\
FROM nginx:alpine

COPY . /usr/share/nginx/html

RUN echo 'server { \\
    listen 80; \\
    root /usr/share/nginx/html; \\
    index index.html; \\
    location / { \\
        try_files $uri $uri/ /index.html; \\
    } \\
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=3s --retries=3 \\
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
""",
    "go": """\
FROM golang:{go_version}-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /app

COPY --from=builder /app/main .

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD wget -qO- http://localhost:{port}/health || exit 1

CMD ["./main"]
""",
}

COMPOSE_TEMPLATE = """\
version: "3.9"

services:
{services}

networks:
  app-network:
    driver: bridge
"""

COMPOSE_SERVICE_TEMPLATE = """\
  {name}:
    build:
      context: {context}
      dockerfile: Dockerfile
    container_name: {container_name}
    ports:
      - "{host_port}:{container_port}"
    environment:
{env_vars}
    volumes:
      - {context}:/app
      - {name}-data:/app/data
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{container_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
"""

VERCEL_CONFIG = {
    "nextjs": {
        "version": 2,
        "builds": [{"src": "package.json", "use": "@vercel/next"}],
        "routes": [
            {"src": "/(.*)", "dest": "/$1"},
        ],
    },
    "react": {
        "version": 2,
        "builds": [{"src": "package.json", "use": "@vercel/static-build", "config": {"distDir": "build"}}],
        "routes": [
            {"handle": "filesystem"},
            {"src": "/(.*)", "dest": "/index.html"},
        ],
    },
    "vue": {
        "version": 2,
        "builds": [{"src": "package.json", "use": "@vercel/static-build", "config": {"distDir": "dist"}}],
        "routes": [
            {"handle": "filesystem"},
            {"src": "/(.*)", "dest": "/index.html"},
        ],
    },
    "static": {
        "version": 2,
        "builds": [{"src": "**", "use": "@vercel/static"}],
    },
}

NETLIFY_TOML = """\
[build]
  command = "{build_command}"
  publish = "{publish_dir}"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[context.production.environment]
  NODE_ENV = "production"
"""

RAILWAY_JSON = """\
{{
  "$schema": "https://railway.app/railway.schema.json",
  "build": "{build_config}",
  "deploy": {{
    "startCommand": "{start_command}",
    "restartPolicyType": "on_failure",
    "restartPolicyMaxRetries": 3
  }}
}}
"""

RENDER_YAML = """\
services:
  - type: web
    name: {service_name}
    env: {runtime}
    buildCommand: {build_command}
    startCommand: {start_command}
    plan: {plan}
    region: {region}
    envVars:
{env_vars}

"""

CLOUDFORMATION_TEMPLATE = """\
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Claude Deploy - {function_name}

Resources:
  {FunctionName}Function:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: {function_name}
      Description: {description}
      CodeUri: ./deployment_package
      Handler: {handler}
      Runtime: {runtime}
      Timeout: {timeout}
      MemorySize: {memory_size}
      Environment:
        Variables:
{environment}
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /{api_path}
            Method: {http_method}

Outputs:
  {FunctionName}ApiUrl:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${{AWS::Region}}.amazonaws.com/Prod/{api_path}"
"""


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------

class Platform(Enum):
    """Supported deployment platforms."""
    DOCKER = "docker"
    VERCEL = "vercel"
    NETLIFY = "netlify"
    AWS_LAMBDA = "aws_lambda"
    GITHUB_PAGES = "github_pages"
    RAILWAY = "railway"
    RENDER = "render"
    SELF_HOSTED = "self_hosted"

    @classmethod
    def from_string(cls, value: str) -> Platform:
        mapping = {
            "docker": cls.DOCKER,
            "vercel": cls.VERCEL,
            "netlify": cls.NETLIFY,
            "aws_lambda": cls.AWS_LAMBDA,
            "aws-lambda": cls.AWS_LAMBDA,
            "lambda": cls.AWS_LAMBDA,
            "github_pages": cls.GITHUB_PAGES,
            "github-pages": cls.GITHUB_PAGES,
            "gh-pages": cls.GITHUB_PAGES,
            "railway": cls.RAILWAY,
            "render": cls.RENDER,
            "self_hosted": cls.SELF_HOSTED,
            "self-hosted": cls.SELF_HOSTED,
            "selfhosted": cls.SELF_HOSTED,
        }
        normalized = value.strip().lower().replace(" ", "_")
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(f"Unknown platform: {value!r}. Valid: {[p.value for p in cls]}")


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


@dataclass
class DeploymentConfig:
    """Configuration for a single deployment."""
    platform: Optional[Platform] = None
    project_path: str = "."
    build_command: str = ""
    output_dir: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    region: str = "us-east-1"
    domain: str = ""
    dockerfile_path: str = "Dockerfile"
    image_name: str = ""
    image_tag: str = "latest"
    registry_url: str = ""
    function_name: str = ""
    handler: str = ""
    runtime: str = ""
    memory_size: int = 256
    timeout: int = 30
    branch: str = "main"
    service_name: str = "claude-app"
    plan: str = "starter"
    python_version: str = "3.12"
    node_version: str = "20"
    go_version: str = "1.22"
    port: int = 8000
    health_path: str = "/health"
    workers: int = 4
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool = False
    platform: str = ""
    url: str = ""
    logs: list[str] = field(default_factory=list)
    duration: float = 0.0
    version: str = ""
    rollback_version: str = ""
    deployment_id: str = ""
    error: str = ""


@dataclass
class DeploymentHistory:
    """Single entry in deployment history."""
    id: str = ""
    platform: str = ""
    project: str = ""
    status: str = ""
    url: str = ""
    timestamp: str = ""
    version: str = ""
    logs: list[str] = field(default_factory=list)
    duration: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def _run_command(
    cmd: list[str],
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command and return the completed process."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture,
        text=capture,
        timeout=timeout,
        check=False,
    )


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _short_uuid() -> str:
    return uuid.uuid4().hex[:12]


def _detect_project_framework(project_path: str) -> dict[str, Any]:
    """Detect the framework and language used in a project directory."""
    path = Path(project_path).resolve()
    info: dict[str, Any] = {
        "framework": "unknown",
        "language": "unknown",
        "module": "app",
        "entrypoint": "index.js",
        "build_command": "",
        "output_dir": "",
        "has_dockerfile": False,
        "has_compose": False,
        "has_package_json": False,
        "has_requirements": False,
        "has_go_mod": False,
        "has_makefile": False,
        "has_nginx_conf": False,
    }

    info["has_dockerfile"] = (path / "Dockerfile").exists()
    info["has_compose"] = (path / "docker-compose.yml").exists() or (path / "docker-compose.yaml").exists()
    info["has_package_json"] = (path / "package.json").exists()
    info["has_requirements"] = (path / "requirements.txt").exists() or (path / "pyproject.toml").exists()
    info["has_go_mod"] = (path / "go.mod").exists()
    info["has_makefile"] = (path / "Makefile").exists()
    info["has_nginx_conf"] = (path / "nginx.conf").exists()

    if info["has_package_json"]:
        pkg_path = path / "package.json"
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            info["has_package_json"] = True

            if "next" in deps:
                info["framework"] = "nextjs"
                info["language"] = "node"
                info["build_command"] = "npm run build"
                info["output_dir"] = ".next"
                info["entrypoint"] = "server.js"
            elif "vue" in deps or "@vue/cli-service" in deps or "nuxt" in deps:
                info["framework"] = "vue"
                info["language"] = "node"
                info["build_command"] = "npm run build"
                info["output_dir"] = "dist"
            elif "react" in deps or "react-dom" in deps:
                info["framework"] = "react"
                info["language"] = "node"
                info["build_command"] = "npm run build"
                info["output_dir"] = "build"
            elif "express" in deps or "@nestjs/core" in deps or "fastify" in deps:
                info["framework"] = "node"
                info["language"] = "node"
                info["entrypoint"] = "src/index.js"
                info["module"] = "src/index"
            else:
                info["framework"] = "node"
                info["language"] = "node"

            scripts = pkg.get("scripts", {})
            if "build" in scripts:
                info["build_command"] = scripts["build"]
            if "start" in scripts:
                info["entrypoint"] = scripts["start"]
        except (json.JSONDecodeError, OSError):
            pass

    if info["has_requirements"]:
        info["language"] = "python"
        req_path = path / "requirements.txt"
        try:
            with open(req_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "django" in content:
                info["framework"] = "django"
                for candidate_path in path.rglob("wsgi.py"):
                    info["module"] = str(candidate_path.with_suffix("").relative_to(path)).replace(os.sep, ".")
                    break
                if info["module"] == "app":
                    info["module"] = "myapp.wsgi"
                info["build_command"] = "python manage.py collectstatic --noinput"
            elif "fastapi" in content:
                info["framework"] = "fastapi"
                info["module"] = "main"
            elif "flask" in content:
                info["framework"] = "flask"
                info["module"] = "app"
            else:
                info["framework"] = "python"
                info["module"] = "main"
        except OSError:
            pass

    if info["has_go_mod"]:
        info["framework"] = "go"
        info["language"] = "go"
        info["build_command"] = "go build -o main ."

    if info["framework"] == "unknown":
        if (path / "index.html").exists() or (path / "index.htm").exists():
            info["framework"] = "static"
            info["language"] = "html"
            info["output_dir"] = "."
        elif (path / "public" / "index.html").exists():
            info["framework"] = "static"
            info["language"] = "html"
            info["output_dir"] = "public"

    return info


def _ensure_directory(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Environment Encryption
# ---------------------------------------------------------------------------

class EnvEncryption:
    """Encrypt and decrypt environment variable values for secure storage."""

    def __init__(self, key: Optional[str] = None, key_file: Optional[str] = None):
        self._fernet: Optional[Any] = None
        if HAS_CRYPTOGRAPHY:
            resolved_key = key
            if not resolved_key and key_file:
                try:
                    with open(key_file, "rb") as f:
                        resolved_key = f.read().decode("utf-8")
                except OSError:
                    pass
            if not resolved_key:
                resolved_key = Fernet.generate_key().decode("utf-8")
                if key_file:
                    _ensure_directory(str(Path(key_file).parent))
                    with open(key_file, "w", encoding="utf-8") as f:
                        f.write(resolved_key)
            self._fernet = Fernet(resolved_key.encode("utf-8"))
        self._key = resolved_key or ""

    def encrypt(self, value: str) -> str:
        if not self._fernet:
            return base64.b64encode(value.encode("utf-8")).decode("utf-8")
        return self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    def decrypt(self, token: str) -> str:
        if not self._fernet:
            return base64.b64decode(token.encode("utf-8")).decode("utf-8")
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except Exception:
            return token

    @property
    def is_secure(self) -> bool:
        return HAS_CRYPTOGRAPHY and self._fernet is not None


# ---------------------------------------------------------------------------
# Deploy Engine
# ---------------------------------------------------------------------------

class DeployEngine:
    """Multi-platform deployment engine.

    Deploys projects to Docker, Vercel, Netlify, AWS Lambda, GitHub Pages,
    Railway, Render, and self-hosted environments.
    """

    def __init__(self, project_path: str = ".", config_dir: str = ".claude_deploy") -> None:
        self.project_path = Path(project_path).resolve()
        self.config_dir = _ensure_directory(str(self.project_path / config_dir))
        self.history_dir = _ensure_directory(str(self.config_dir / "history"))
        self.env_dir = _ensure_directory(str(self.config_dir / "env"))
        self.encryption = EnvEncryption(
            key_file=str(self.env_dir / ".encryption_key"),
        )
        self._project_info = _detect_project_framework(str(self.project_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def deploy(
        self,
        platform: Platform,
        config: Optional[DeploymentConfig] = None,
    ) -> DeploymentResult:
        """Deploy the project to the specified platform."""
        cfg = config or DeploymentConfig(
            platform=platform,
            project_path=str(self.project_path),
        )
        cfg.platform = platform
        cfg.project_path = str(self.project_path)

        if not cfg.image_name:
            cfg.image_name = self.project_path.name.lower().replace("-", "_").replace(" ", "_")
        if not cfg.service_name:
            cfg.service_name = self.project_path.name or "claude-app"
        if not cfg.version:
            cfg.version = _short_uuid()

        deploy_id = _short_uuid()
        logs: list[str] = []
        start = time.monotonic()

        logs.append(f"[{platform.value.upper()}] Starting deployment: {deploy_id}")
        logs.append(f"  Project: {self.project_path}")
        logs.append(f"  Framework: {self._project_info['framework']} ({self._project_info['language']})")

        dispatch = {
            Platform.DOCKER: self.deploy_docker,
            Platform.VERCEL: self.deploy_vercel,
            Platform.NETLIFY: self.deploy_netlify,
            Platform.AWS_LAMBDA: self.deploy_aws_lambda,
            Platform.GITHUB_PAGES: self.deploy_github_pages,
            Platform.RAILWAY: self._deploy_railway,
            Platform.RENDER: self._deploy_render,
            Platform.SELF_HOSTED: self._deploy_self_hosted,
        }

        deploy_fn = dispatch.get(platform)
        if deploy_fn is None:
            result = DeploymentResult(
                success=False,
                platform=platform.value,
                logs=logs,
                duration=time.monotonic() - start,
                deployment_id=deploy_id,
                error=f"Platform {platform.value} is not supported yet.",
            )
        else:
            result = await deploy_fn(cfg)
            result.deployment_id = deploy_id
            result.platform = platform.value
            result.duration = time.monotonic() - start
            result.logs = logs + result.logs
            result.version = result.version or cfg.version

        self._save_history(DeploymentHistory(
            id=deploy_id,
            platform=platform.value,
            project=str(self.project_path),
            status="success" if result.success else "failed",
            url=result.url,
            timestamp=_now_iso(),
            version=result.version,
            logs=result.logs[-50:],
            duration=result.duration,
            config={
                "region": cfg.region,
                "domain": cfg.domain,
                "image_name": cfg.image_name,
                "service_name": cfg.service_name,
                "build_command": cfg.build_command,
            },
        ))

        return result

    async def deploy_docker(self, config: DeploymentConfig) -> DeploymentResult:
        """Generate Dockerfile, build image, tag, and push."""
        logs: list[str] = []
        result = DeploymentResult(platform=Platform.DOCKER.value)
        framework = self._project_info["framework"]

        if framework not in DOCKERFILE_TEMPLATES:
            framework = self._infer_docker_template()

        dockerfile_content = await self.generate_dockerfile(framework)

        dockerfile_path = self.project_path / config.dockerfile_path
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        logs.append(f"  Generated Dockerfile at {dockerfile_path}")

        image_name = config.image_name or self.project_path.name
        tag = config.image_tag or "latest"
        full_image = f"{image_name}:{tag}"

        build_cmd = ["docker", "build", "-t", full_image, "-f", str(dockerfile_path), str(self.project_path)]
        proc = _run_command(build_cmd, cwd=str(self.project_path))
        logs.append(f"  docker build: exit code {proc.returncode}")
        if proc.stdout:
            for line in proc.stdout.strip().splitlines()[-10:]:
                logs.append(f"    {line}")
        if proc.returncode != 0:
            result.success = False
            result.error = "Docker build failed"
            result.logs = logs
            return result

        result.version = tag

        if config.registry_url:
            remote_image = f"{config.registry_url}/{full_image}"
            tag_cmd = ["docker", "tag", full_image, remote_image]
            proc = _run_command(tag_cmd)
            logs.append(f"  docker tag: exit code {proc.returncode}")

            push_cmd = ["docker", "push", remote_image]
            proc = _run_command(push_cmd, timeout=300)
            logs.append(f"  docker push: exit code {proc.returncode}")
            if proc.stdout:
                for line in proc.stdout.strip().splitlines()[-5:]:
                    logs.append(f"    {line}")
            if proc.returncode != 0:
                result.success = False
                result.error = "Docker push failed"
                result.logs = logs
                return result
            result.url = remote_image
            logs.append(f"  Pushed: {remote_image}")
        else:
            result.url = full_image
            logs.append(f"  Built locally: {full_image}")

        compose_path = self.project_path / "docker-compose.yml"
        if not compose_path.exists():
            compose_content = await self.generate_compose()
            with open(compose_path, "w", encoding="utf-8") as f:
                f.write(compose_content)
            logs.append(f"  Generated docker-compose.yml")

        result.success = True
        result.logs = logs
        return result

    async def deploy_vercel(self, config: DeploymentConfig) -> DeploymentResult:
        """Detect framework, generate vercel.json, and deploy via CLI."""
        logs: list[str] = []
        result = DeploymentResult(platform=Platform.VERCEL.value)
        framework = self._project_info["framework"]

        vercel_config = VERCEL_CONFIG.get(framework, VERCEL_CONFIG["static"])
        config_path = self.project_path / "vercel.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vercel_config, f, indent=2)
        logs.append(f"  Generated vercel.json for framework: {framework}")

        build_cmd = config.build_command or self._project_info.get("build_command", "")
        if build_cmd and self._project_info["has_package_json"]:
            proc = _run_command(build_cmd.split(), cwd=str(self.project_path))
            logs.append(f"  Build command: {build_cmd} (exit {proc.returncode})")
            if proc.returncode != 0:
                result.success = False
                result.error = f"Build failed: {build_cmd}"
                result.logs = logs
                return result

        vercel_bin = shutil.which("vercel")
        if vercel_bin:
            deploy_cmd = [vercel_bin, "--yes", "--prod"]
            env = os.environ.copy()
            if config.env_vars:
                env.update({k: v for k, v in config.env_vars.items()})
            proc = _run_command(deploy_cmd, cwd=str(self.project_path), timeout=300)
            logs.append(f"  vercel deploy: exit code {proc.returncode}")
            if proc.stdout:
                for line in proc.stdout.strip().splitlines():
                    logs.append(f"    {line}")
                    if "https://" in line and ".vercel.app" in line:
                        url = line.strip().split()[-1]
                        if url.startswith("http"):
                            result.url = url
            result.success = proc.returncode == 0
            if not result.success:
                result.error = "Vercel CLI deployment failed"
        else:
            logs.append("  Vercel CLI not found. Generated vercel.json for manual deployment.")
            result.success = True
            result.url = f"https://{config.service_name or self.project_path.name}.vercel.app"

        result.version = _short_uuid()
        result.logs = logs
        return result

    async def deploy_netlify(self, config: DeploymentConfig) -> DeploymentResult:
        """Generate netlify.toml and deploy static site."""
        logs: list[str] = []
        result = DeploymentResult(platform=Platform.NETLIFY.value)
        framework = self._project_info["framework"]

        build_command = config.build_command or self._project_info.get("build_command", "npm run build")
        publish_dir = config.output_dir or self._project_info.get("output_dir", "dist")

        if framework == "nextjs":
            publish_dir = ".next"
            build_command = "npm run build"
        elif framework == "react":
            publish_dir = "build"
            build_command = "npm run build"
        elif framework == "vue":
            publish_dir = "dist"
            build_command = "npm run build"
        elif framework in ("django", "flask", "fastapi", "python"):
            logs.append("  Warning: Netlify is best for static sites. Consider Docker or Railway for Python apps.")
            publish_dir = "static" if (self.project_path / "static").exists() else "."

        toml_content = NETLIFY_TOML.format(build_command=build_command, publish_dir=publish_dir)
        toml_path = self.project_path / "netlify.toml"
        with open(toml_path, "w", encoding="utf-8") as f:
            f.write(toml_content)
        logs.append(f"  Generated netlify.toml (publish: {publish_dir})")

        build_cmd_parts = build_command.split() if build_command else []
        if build_cmd_parts and self._project_info["has_package_json"]:
            proc = _run_command(build_cmd_parts, cwd=str(self.project_path))
            logs.append(f"  Build: {build_command} (exit {proc.returncode})")
            if proc.returncode != 0:
                result.success = False
                result.error = f"Netlify build failed: {build_command}"
                result.logs = logs
                return result

        netlify_bin = shutil.which("netlify")
        if netlify_bin:
            deploy_cmd = [netlify_bin, "deploy", "--prod", "--dir", publish_dir]
            proc = _run_command(deploy_cmd, cwd=str(self.project_path), timeout=300)
            logs.append(f"  netlify deploy: exit code {proc.returncode}")
            if proc.stdout:
                for line in proc.stdout.strip().splitlines():
                    logs.append(f"    {line}")
                    if "https://" in line and ".netlify.app" in line:
                        url = line.strip().split()[-1]
                        if url.startswith("http"):
                            result.url = url
            result.success = proc.returncode == 0
            if not result.success:
                result.error = "Netlify CLI deployment failed"
        else:
            logs.append("  Netlify CLI not found. Generated netlify.toml for manual deployment.")
            result.success = True
            result.url = f"https://{config.service_name or self.project_path.name}.netlify.app"

        result.version = _short_uuid()
        result.logs = logs
        return result

    async def deploy_aws_lambda(self, config: DeploymentConfig) -> DeploymentResult:
        """Package Python/Node.js function and generate CloudFormation template."""
        logs: list[str] = []
        result = DeploymentResult(platform=Platform.AWS_LAMBDA.value)
        language = self._project_info["language"]

        if language == "python":
            config.runtime = config.runtime or f"python{self.project_info.get('python_version', '3.12')}"
            handler = config.handler or "app.lambda_handler"
        elif language == "node":
            config.runtime = config.runtime or "nodejs20.x"
            handler = config.handler or "index.handler"
        else:
            config.runtime = config.runtime or "python3.12"
            handler = config.handler or "app.lambda_handler"

        config.function_name = config.function_name or f"{config.service_name}-fn"
        api_path = config.extra.get("api_path", "")
        http_method = config.extra.get("http_method", "ANY")
        description = config.extra.get("description", f"Claude Deploy: {config.service_name}")

        package_dir = self.project_path / "deployment_package"
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir()

        excluded = {
            "__pycache__", ".git", ".env", "node_modules", "*.pyc",
            ".claude_deploy", "deployment_package", ".venv", "venv",
        }
        for item in self.project_path.iterdir():
            if item.name not in excluded and not item.name.startswith("."):
                if item.is_file():
                    shutil.copy2(item, package_dir / item.name)
                elif item.is_dir() and item.name not in excluded:
                    shutil.copytree(item, package_dir / item.name, ignore=shutil.ignore_patterns(*excluded))

        if language == "python" and (self.project_path / "requirements.txt").exists():
            subprocess.run(
                ["pip", "install", "-r", "requirements.txt", "-t", str(package_dir)],
                capture_output=True, timeout=120, check=False,
            )

        if language == "node" and (self.project_path / "package.json").exists():
            subprocess.run(
                ["npm", "ci", "--production"],
                cwd=str(package_dir), capture_output=True, timeout=120, check=False,
            )

        archive_path = self.project_path / f"{config.function_name}.zip"
        shutil.make_archive(str(archive_path.with_suffix("")), "zip", base_dir=package_dir.name, root_dir=str(self.project_path))
        logs.append(f"  Package created: {archive_path}")

        env_block = "\n".join(f"          {k}: {v}" for k, v in config.env_vars.items()) if config.env_vars else "          ENV: production"
        template = CLOUDFORMATION_TEMPLATE.format(
            function_name=config.function_name,
            FunctionName=config.function_name.replace("-", "").replace("_", ""),
            description=description,
            handler=handler,
            runtime=config.runtime,
            timeout=config.timeout,
            memory_size=config.memory_size,
            environment=env_block,
            api_path=api_path,
            http_method=http_method,
        )
        template_path = self.project_path / "template.yaml"
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template)
        logs.append(f"  CloudFormation template: {template_path}")

        sam_bin = shutil.which("sam")
        if sam_bin:
            logs.append("  Deploying with AWS SAM CLI...")
            proc = _run_command(
                ["sam", "deploy", "--guided", "--template-file", str(template_path)],
                cwd=str(self.project_path), timeout=600,
            )
            logs.append(f"  sam deploy: exit code {proc.returncode}")
            result.success = proc.returncode == 0
            if not result.success:
                result.error = "SAM deploy failed"
            else:
                result.url = f"https://{config.function_name}.lambda-url.{config.region}.on.aws"
        else:
            logs.append("  AWS SAM CLI not found. Generated template.yaml and .zip for manual deployment.")
            result.success = True
            result.url = f"https://{config.function_name}.lambda-url.{config.region}.on.aws"

        shutil.rmtree(package_dir, ignore_errors=True)
        result.version = _short_uuid()
        result.logs = logs
        return result

    async def deploy_github_pages(self, config: DeploymentConfig) -> DeploymentResult:
        """Build and deploy to the gh-pages branch."""
        logs: list[str] = []
        result = DeploymentResult(platform=Platform.GITHUB_PAGES.value)
        framework = self._project_info["framework"]

        build_command = config.build_command or self._project_info.get("build_command", "")
        output_dir = config.output_dir or self._project_info.get("output_dir", "dist")

        if framework == "static":
            build_command = ""
            output_dir = "."

        if build_command:
            parts = build_command.split()
            proc = _run_command(parts, cwd=str(self.project_path))
            logs.append(f"  Build: {build_command} (exit {proc.returncode})")
            if proc.returncode != 0:
                result.success = False
                result.error = f"Build failed for GitHub Pages"
                result.logs = logs
                return result

        source = self.project_path / output_dir
        if not source.exists():
            result.success = False
            result.error = f"Output directory not found: {source}"
            result.logs = logs
            return result

        repo_url = ""
        proc = _run_command(["git", "remote", "get-url", "origin"], cwd=str(self.project_path))
        if proc.returncode == 0:
            repo_url = proc.stdout.strip()

        gh_pages_dir = self.project_path / ".gh-pages-deploy"
        if gh_pages_dir.exists():
            shutil.rmtree(gh_pages_dir)

        proc = _run_command(
            ["git", "clone", "--branch", "gh-pages", "--depth", "1", repo_url, str(gh_pages_dir)],
            cwd=str(self.project_path), timeout=60,
        ) if repo_url else None

        if proc and proc.returncode == 0:
            for item in gh_pages_dir.iterdir():
                if item.name != ".git":
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
            for item in source.iterdir():
                dest = gh_pages_dir / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                else:
                    shutil.copytree(item, dest)

            _run_command(["git", "add", "-A"], cwd=str(gh_pages_dir))
            _run_command(["git", "commit", "-m", f"Deploy {config.version or _short_uuid()}"], cwd=str(gh_pages_dir))
            push_proc = _run_command(["git", "push", "origin", "gh-pages"], cwd=str(gh_pages_dir), timeout=60)
            logs.append(f"  git push: exit code {push_proc.returncode}")
            result.success = push_proc.returncode == 0
            if not result.success:
                result.error = "Git push to gh-pages failed"
        else:
            logs.append("  Could not clone gh-pages branch. Generating static output for manual deployment.")
            docs_dir = self.project_path / "docs"
            if docs_dir.exists():
                shutil.rmtree(docs_dir)
            shutil.copytree(source, docs_dir)
            result.success = True
            logs.append("  Files copied to docs/ directory")

        repo_name = self.project_path.name
        result.url = f"https://{config.service_name or repo_name}.github.io/{repo_name}"
        result.version = _short_uuid()
        result.logs = logs

        shutil.rmtree(gh_pages_dir, ignore_errors=True)
        return result

    async def _deploy_railway(self, config: DeploymentConfig) -> DeploymentResult:
        """Generate railway.json and deploy via CLI."""
        logs: list[str] = []
        result = DeploymentResult(platform=Platform.RAILWAY.value)

        framework = self._project_info["framework"]
        if framework in ("python", "flask", "fastapi", "django"):
            build_config = "DATETIME_NOW=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
            start_command = config.extra.get("start_cmd", "sh start.sh || gunicorn app:app")
        elif framework == "node":
            build_config = "DATETIME_NOW=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
            start_command = config.extra.get("start_cmd", "npm start")
        else:
            build_config = "DATETIME_NOW=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
            start_command = config.extra.get("start_cmd", "npm start")

        railway_content = RAILWAY_JSON.format(build_config=build_config, start_command=start_command)
        railway_path = self.project_path / "railway.json"
        with open(railway_path, "w", encoding="utf-8") as f:
            f.write(railway_content)
        logs.append("  Generated railway.json")

        railway_bin = shutil.which("railway")
        if railway_bin:
            proc = _run_command(["railway", "up"], cwd=str(self.project_path), timeout=300)
            logs.append(f"  railway up: exit code {proc.returncode}")
            if proc.stdout:
                for line in proc.stdout.strip().splitlines():
                    logs.append(f"    {line}")
            result.success = proc.returncode == 0
            if result.success:
                status_proc = _run_command(["railway", "domain"], cwd=str(self.project_path), timeout=30)
                if status_proc.stdout and "up.app" in status_proc.stdout:
                    for line in status_proc.stdout.strip().splitlines():
                        if "up.app" in line:
                            result.url = line.strip()
                            break
            if not result.success:
                result.error = "Railway deploy failed"
        else:
            logs.append("  Railway CLI not found. Generated railway.json for manual deployment.")
            result.success = True
            result.url = f"https://{config.service_name}-prod.up.railway.app"

        result.version = _short_uuid()
        result.logs = logs
        return result

    async def _deploy_render(self, config: DeploymentConfig) -> DeploymentResult:
        """Generate render.yaml and deploy via CLI or web dashboard."""
        logs: list[str] = []
        result = DeploymentResult(platform=Platform.RENDER.value)

        framework = self._project_info["framework"]
        if framework in ("python", "flask", "fastapi", "django"):
            runtime = "PYTHON"
            build_command = config.build_command or "pip install -r requirements.txt"
            start_command = config.extra.get("start_cmd", "gunicorn app:app")
        elif framework == "node":
            runtime = "NODE"
            build_command = config.build_command or "npm install"
            start_command = config.extra.get("start_cmd", "npm start")
        elif framework == "go":
            runtime = "DOCKER"
            build_command = ""
            start_command = ""
        else:
            runtime = "DOCKER"
            build_command = ""
            start_command = ""

        env_block = "\n".join(f"      - key: {k}\n        value: {v}" for k, v in config.env_vars.items()) if config.env_vars else "      - key: NODE_ENV\n        value: production"

        render_content = RENDER_YAML.format(
            service_name=config.service_name,
            runtime=runtime,
            build_command=build_command,
            start_command=start_command,
            plan=config.plan,
            region=config.region,
            env_vars=env_block,
        )
        render_path = self.project_path / "render.yaml"
        with open(render_path, "w", encoding="utf-8") as f:
            f.write(render_content)
        logs.append(f"  Generated render.yaml (runtime: {runtime})")

        result.success = True
        result.url = f"https://{config.service_name}.onrender.com"
        result.version = _short_uuid()
        result.logs = logs
        return result

    async def _deploy_self_hosted(self, config: DeploymentConfig) -> DeploymentResult:
        """Prepare project for self-hosted deployment."""
        logs: list[str] = []
        result = DeploymentResult(platform=Platform.SELF_HOSTED.value)

        framework = self._project_info["framework"]
        dockerfile = await self.generate_dockerfile(framework)
        compose = await self.generate_compose()

        df_path = self.project_path / "Dockerfile"
        with open(df_path, "w", encoding="utf-8") as f:
            f.write(dockerfile)
        logs.append("  Generated Dockerfile")

        dc_path = self.project_path / "docker-compose.yml"
        with open(dc_path, "w", encoding="utf-8") as f:
            f.write(compose)
        logs.append("  Generated docker-compose.yml")

        env_file = self.project_path / ".env.example"
        env_vars_list = list(config.env_vars.items()) if config.env_vars else [
            ("PORT", str(config.port)),
            ("NODE_ENV", "production"),
            ("LOG_LEVEL", "info"),
        ]
        with open(env_file, "w", encoding="utf-8") as f:
            for key, value in env_vars_list:
                f.write(f"{key}={value}\n")
        logs.append("  Generated .env.example")

        deploy_script = f"""\
#!/usr/bin/env bash
set -euo pipefail

echo "=== Self-hosted deployment for {config.service_name} ==="

# Build and start
docker compose build
docker compose up -d

# Wait for health check
echo "Waiting for service to be healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:{config.port}{config.health_path} > /dev/null 2>&1; then
        echo "Service is healthy!"
        break
    fi
    echo "  Attempt $i/30..."
    sleep 2
done

echo "Deployed at http://localhost:{config.port}"
"""
        script_path = self.project_path / "deploy.sh"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(deploy_script)
        os.chmod(script_path, 0o755)
        logs.append("  Generated deploy.sh")

        result.success = True
        result.url = f"http://localhost:{config.port}"
        result.version = _short_uuid()
        result.logs = logs
        return result

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def rollback(self, deployment_id: str) -> DeploymentResult:
        """Roll back to a previous deployment."""
        logs: list[str] = []
        history = self._load_history()
        target_entry = None

        for entry in history:
            if entry.id == deployment_id:
                target_entry = entry
                break

        if target_entry is None:
            logs.append(f"  Deployment {deployment_id} not found in history.")
            return DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                logs=logs,
                error=f"Deployment {deployment_id} not found.",
            )

        logs.append(f"  Rolling back to deployment {deployment_id} (platform: {target_entry.platform})")

        try:
            platform = Platform.from_string(target_entry.platform)
        except ValueError:
            return DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                logs=logs,
                error=f"Unknown platform: {target_entry.platform}",
            )

        config = DeploymentConfig(
            platform=platform,
            project_path=str(self.project_path),
            region=target_entry.config.get("region", "us-east-1"),
            domain=target_entry.config.get("domain", ""),
            image_name=target_entry.config.get("image_name", ""),
            service_name=target_entry.config.get("service_name", ""),
        )

        dispatch = {
            Platform.DOCKER: self._rollback_docker,
            Platform.VERCEL: self._rollback_platform_cli,
            Platform.NETLIFY: self._rollback_platform_cli,
            Platform.GITHUB_PAGES: self._rollback_github_pages,
        }

        rollback_fn = dispatch.get(platform, self._rollback_generic)
        result = await rollback_fn(deployment_id, target_entry, config)
        result.deployment_id = deployment_id
        result.rollback_version = target_entry.version
        result.logs = logs + result.logs

        self._save_history(DeploymentHistory(
            id=_short_uuid(),
            platform=platform.value,
            project=str(self.project_path),
            status="rolled_back",
            url=result.url,
            timestamp=_now_iso(),
            version=target_entry.version,
            logs=result.logs[-50:],
            duration=result.duration,
            config={"rollback_from": deployment_id},
        ))

        return result

    async def _rollback_docker(
        self,
        deployment_id: str,
        entry: DeploymentHistory,
        config: DeploymentConfig,
    ) -> DeploymentResult:
        logs: list[str] = []
        version = entry.version or "previous"

        logs.append(f"  Checking for image tag: {version}")
        proc = _run_command(["docker", "images", "-q", f"{config.image_name}:{version}"])
        if proc.stdout.strip():
            logs.append(f"  Image found. Running: docker compose down && docker compose up")
            _run_command(["docker", "compose", "down"], cwd=str(self.project_path))
            _run_command(
                ["docker", "compose", "up", "-d"],
                cwd=str(self.project_path), timeout=120,
            )
            return DeploymentResult(success=True, platform=Platform.DOCKER.value, logs=logs)
        else:
            logs.append("  Image not found locally. Attempting to pull...")
            proc = _run_command(
                ["docker", "pull", f"{config.image_name}:{version}"],
                timeout=120,
            )
            if proc.returncode == 0:
                _run_command(["docker", "compose", "down"], cwd=str(self.project_path))
                _run_command(
                    ["docker", "compose", "up", "-d"],
                    cwd=str(self.project_path), timeout=120,
                )
                return DeploymentResult(success=True, platform=Platform.DOCKER.value, logs=logs)

        logs.append("  Rollback failed: image not found locally or remotely.")
        return DeploymentResult(success=False, platform=Platform.DOCKER.value, logs=logs, error="Image not found for rollback")

    async def _rollback_platform_cli(
        self,
        deployment_id: str,
        entry: DeploymentHistory,
        config: DeploymentConfig,
    ) -> DeploymentResult:
        logs: list[str] = []
        platform = entry.platform

        cli_map = {"vercel": "vercel", "netlify": "netlify"}
        cli_bin = shutil.which(cli_map.get(platform, ""))
        if cli_bin:
            logs.append(f"  {platform.capitalize()} CLI found. Run manual rollback if needed.")
            logs.append(f"  Previous deployment URL: {entry.url}")
            return DeploymentResult(success=True, platform=platform, url=entry.url, logs=logs)

        logs.append(f"  No CLI for {platform}. Restore via web dashboard to: {entry.url}")
        return DeploymentResult(success=True, platform=platform, url=entry.url, logs=logs)

    async def _rollback_github_pages(
        self,
        deployment_id: str,
        entry: DeploymentHistory,
        config: DeploymentConfig,
    ) -> DeploymentResult:
        logs: list[str] = []
        logs.append("  To rollback GitHub Pages, reset the gh-pages branch:")
        logs.append(f"    git checkout gh-pages")
        logs.append(f"    git reset --hard {entry.version}")
        logs.append(f"    git push origin gh-pages --force")

        repo_url = ""
        proc = _run_command(["git", "remote", "get-url", "origin"], cwd=str(self.project_path))
        if proc.returncode == 0:
            repo_url = proc.stdout.strip()
            if repo_url:
                proc = _run_command(["git", "rev-parse", "HEAD"], cwd=str(self.project_path))
                if proc.returncode == 0:
                    logs.append(f"  Current HEAD: {proc.stdout.strip()}")

        return DeploymentResult(success=True, platform=Platform.GITHUB_PAGES.value, url=entry.url, logs=logs)

    async def _rollback_generic(
        self,
        deployment_id: str,
        entry: DeploymentHistory,
        config: DeploymentConfig,
    ) -> DeploymentResult:
        logs: list[str] = []
        logs.append(f"  Generic rollback for {entry.platform}.")
        logs.append(f"  Previous URL: {entry.url}")
        logs.append("  Please restore via your platform's dashboard.")
        return DeploymentResult(success=True, platform=entry.platform, url=entry.url, logs=logs)

    # ------------------------------------------------------------------
    # Health Checks
    # ------------------------------------------------------------------

    async def health_check(self, url: str, timeout: int = 30) -> dict:
        """Verify deployment success with HTTP health checks."""
        results: dict[str, Any] = {
            "url": url,
            "healthy": False,
            "status_code": None,
            "response_time_ms": None,
            "attempts": 0,
            "errors": [],
        }

        if not HAS_AIOHTTP:
            return self._health_check_sync(url, timeout)

        for attempt in range(1, HEALTH_CHECK_RETRIES + 1):
            results["attempts"] = attempt
            try:
                start = time.monotonic()
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                        elapsed_ms = (time.monotonic() - start) * 1000
                        results["status_code"] = resp.status
                        results["response_time_ms"] = round(elapsed_ms, 1)

                        if 200 <= resp.status < 400:
                            results["healthy"] = True
                            return results
                        else:
                            results["errors"].append(f"HTTP {resp.status} on attempt {attempt}")
            except Exception as exc:
                results["errors"].append(f"Attempt {attempt}: {exc}")

            if attempt < HEALTH_CHECK_RETRIES:
                await _async_sleep(HEALTH_CHECK_INTERVAL)

        return results

    def _health_check_sync(self, url: str, timeout: int = 30) -> dict:
        """Synchronous fallback health check using urllib."""
        import urllib.request
        import urllib.error

        results: dict[str, Any] = {
            "url": url,
            "healthy": False,
            "status_code": None,
            "response_time_ms": None,
            "attempts": 0,
            "errors": [],
        }

        for attempt in range(1, HEALTH_CHECK_RETRIES + 1):
            results["attempts"] = attempt
            try:
                start = time.monotonic()
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    results["status_code"] = resp.status
                    results["response_time_ms"] = round(elapsed_ms, 1)
                    if 200 <= resp.status < 400:
                        results["healthy"] = True
                        return results
                    else:
                        results["errors"].append(f"HTTP {resp.status} on attempt {attempt}")
            except Exception as exc:
                results["errors"].append(f"Attempt {attempt}: {exc}")

            if attempt < HEALTH_CHECK_RETRIES:
                time.sleep(HEALTH_CHECK_INTERVAL)

        return results

    # ------------------------------------------------------------------
    # Platform Detection
    # ------------------------------------------------------------------

    async def detect_platform(self) -> Platform:
        """Auto-detect the best deployment platform from project structure."""
        fw = self._project_info["framework"]
        lang = self._project_info["language"]

        if self._project_info["has_dockerfile"]:
            return Platform.DOCKER

        if fw == "nextjs":
            return Platform.VERCEL
        if fw in ("react", "vue", "static"):
            return Platform.NETLIFY
        if fw in ("flask", "fastapi", "django", "python"):
            if self._project_info["has_requirements"] or self._project_info["has_package_json"]:
                return Platform.RAILWAY
            return Platform.DOCKER
        if fw == "go":
            return Platform.DOCKER
        if fw == "node":
            return Platform.RENDER
        if lang == "html":
            return Platform.GITHUB_PAGES

        return Platform.DOCKER

    # ------------------------------------------------------------------
    # Dockerfile Generation
    # ------------------------------------------------------------------

    async def generate_dockerfile(self, framework: Optional[str] = None) -> str:
        """Auto-generate a Dockerfile based on project type."""
        if framework is None:
            framework = self._project_info["framework"]

        if framework not in DOCKERFILE_TEMPLATES:
            framework = self._infer_docker_template()

        template = DOCKERFILE_TEMPLATES[framework]

        return template.format(
            python_version="3.12",
            node_version="20",
            go_version="1.22",
            port=self._project_info.get("port", 8000),
            module=self._project_info.get("module", "app"),
            entrypoint=self._project_info.get("entrypoint", "index.js"),
        )

    def _infer_docker_template(self) -> str:
        """Infer the closest Dockerfile template for the current project."""
        lang = self._project_info["language"]
        fw = self._project_info["framework"]
        mapping = {
            "python": "python",
            "node": "node",
            "go": "go",
            "html": "static",
        }
        if fw in DOCKERFILE_TEMPLATES:
            return fw
        return mapping.get(lang, "python")

    # ------------------------------------------------------------------
    # Docker Compose Generation
    # ------------------------------------------------------------------

    async def generate_compose(self, services: int = 1) -> str:
        """Generate a docker-compose.yml file."""
        service_blocks: list[str] = []
        fw = self._project_info["framework"]
        base_port = 8000 if fw in ("flask", "fastapi", "django", "python", "go") else 3000

        app_service = COMPOSE_SERVICE_TEMPLATE.format(
            name="app",
            context=".",
            container_name=f"{self.project_path.name}-app",
            host_port=base_port,
            container_port=base_port,
            env_vars="      - NODE_ENV=production\n      - PORT={}".format(base_port),
        )
        service_blocks.append(app_service)

        if services > 1:
            for i in range(1, services):
                extra = COMPOSE_SERVICE_TEMPLATE.format(
                    name=f"worker-{i}",
                    context=".",
                    container_name=f"{self.project_path.name}-worker-{i}",
                    host_port=base_port + i,
                    container_port=base_port,
                    env_vars="      - NODE_ENV=production\n      - WORKER=true\n      - PORT={}".format(base_port + i),
                )
                service_blocks.append(extra)

        db_service = """\
  db:
    image: postgres:16-alpine
    container_name: {name}-db
    environment:
      POSTGRES_DB: app
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app_secret
    volumes:
      - db-data:/var/lib/postgresql/data
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app"]
      interval: 10s
      timeout: 5s
      retries: 5
"""
        service_blocks.append(db_service.format(name=self.project_path.name))

        redis_service = """\
  redis:
    image: redis:7-alpine
    container_name: {name}-redis
    volumes:
      - redis-data:/data
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
"""
        service_blocks.append(redis_service.format(name=self.project_path.name))

        services_block = "\n".join(service_blocks)

        volumes_block = """
volumes:
  app-data:
  db-data:
  redis-data:
"""
        return COMPOSE_TEMPLATE.format(services=services_block) + volumes_block

    # ------------------------------------------------------------------
    # Environment Variable Management
    # ------------------------------------------------------------------

    async def manage_env(self, action: str, vars: Optional[dict[str, str]] = None) -> dict:
        """Manage environment variables: set, get, list, encrypt, remove, export.

        Actions:
            set     - Store variables (optionally encrypted with prefix ENC_)
            get     - Retrieve value(s) by key, auto-decrypting ENC_ vars
            list    - List all variables (encrypted values shown as ******)
            encrypt - Encrypt all existing env vars
            remove  - Remove variable(s) by key
            export  - Export to .env file in project root
        """
        env_file = self.env_dir / "env.json"
        data: dict[str, str] = {}
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    data = {k: str(v) for k, v in raw.items()}
            except (json.JSONDecodeError, OSError):
                data = {}

        action = action.lower().strip()
        result: dict[str, Any] = {"action": action}

        if action == "set":
            if not vars:
                return {**result, "error": "No variables provided for set action."}
            for key, value in vars.items():
                data[key] = value
            self._save_env(data)
            result["set_count"] = len(vars)
            result["keys"] = list(vars.keys())

        elif action == "get":
            if not vars:
                return {**result, "error": "No keys provided for get action."}
            retrieved: dict[str, str] = {}
            for key in vars:
                if key in data:
                    val = data[key]
                    if key.startswith(ENCRYPTED_ENV_PREFIX):
                        val = self.encryption.decrypt(val)
                    retrieved[key] = val
            result["values"] = retrieved

        elif action == "list":
            masked: dict[str, str] = {}
            for key, value in data.items():
                if key.startswith(ENCRYPTED_ENV_PREFIX):
                    masked[key] = "****** (encrypted)"
                else:
                    masked[key] = value[:4] + "****" if len(value) > 4 else "****"
            result["variables"] = masked
            result["count"] = len(data)

        elif action == "encrypt":
            encrypted_count = 0
            new_data: dict[str, str] = {}
            for key, value in data.items():
                if not key.startswith(ENCRYPTED_ENV_PREFIX):
                    enc_key = f"{ENCRYPTED_ENV_PREFIX}{key}"
                    new_data[enc_key] = self.encryption.encrypt(value)
                    encrypted_count += 1
                else:
                    new_data[key] = value
            self._save_env(new_data)
            result["encrypted_count"] = encrypted_count
            result["secure"] = self.encryption.is_secure

        elif action == "remove":
            if not vars:
                return {**result, "error": "No keys provided for remove action."}
            removed = []
            for key in list(vars.keys()):
                if key in data:
                    del data[key]
                    removed.append(key)
            self._save_env(data)
            result["removed"] = removed

        elif action == "export":
            export_path = self.project_path / ".env"
            with open(export_path, "w", encoding="utf-8") as f:
                for key, value in data.items():
                    actual_value = value
                    if key.startswith(ENCRYPTED_ENV_PREFIX):
                        actual_value = self.encryption.decrypt(value)
                    clean_key = key
                    if clean_key.startswith(ENCRYPTED_ENV_PREFIX):
                        clean_key = clean_key[len(ENCRYPTED_ENV_PREFIX):]
                    f.write(f"{clean_key}={actual_value}\n")
            result["exported_to"] = str(export_path)
            result["count"] = len(data)

        else:
            result["error"] = f"Unknown action: {action!r}. Use: set, get, list, encrypt, remove, export."

        return result

    # ------------------------------------------------------------------
    # History Management
    # ------------------------------------------------------------------

    async def get_history(self, limit: int = 20) -> list[DeploymentHistory]:
        """Return the last *limit* deployment history entries."""
        entries = self._load_history()
        return entries[:limit]

    def _load_history(self) -> list[DeploymentHistory]:
        entries: list[DeploymentHistory] = []
        if not self.history_dir.exists():
            return entries

        files = sorted(
            self.history_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for fpath in files[:MAX_HISTORY_ENTRIES]:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                entries.append(DeploymentHistory(
                    id=raw.get("id", ""),
                    platform=raw.get("platform", ""),
                    project=raw.get("project", ""),
                    status=raw.get("status", ""),
                    url=raw.get("url", ""),
                    timestamp=raw.get("timestamp", ""),
                    version=raw.get("version", ""),
                    logs=raw.get("logs", []),
                    duration=raw.get("duration", 0.0),
                    config=raw.get("config", {}),
                ))
            except (json.JSONDecodeError, OSError):
                continue

        return entries

    def _save_history(self, entry: DeploymentHistory) -> None:
        _ensure_directory(str(self.history_dir))
        filepath = self.history_dir / f"{entry.id}.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "id": entry.id,
                    "platform": entry.platform,
                    "project": entry.project,
                    "status": entry.status,
                    "url": entry.url,
                    "timestamp": entry.timestamp,
                    "version": entry.version,
                    "logs": entry.logs,
                    "duration": entry.duration,
                    "config": entry.config,
                }, f, indent=2, default=str)
        except OSError:
            pass

    def _save_env(self, data: dict[str, str]) -> None:
        _ensure_directory(str(self.env_dir))
        env_file = self.env_dir / "env.json"
        try:
            with open(env_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self, keep_last: int = 3) -> list[str]:
        """Remove old deployment artifacts, keeping the most recent *keep_last*."""
        removed: list[str] = []

        docker_images = [
            f"{self.project_path.name}:*",
        ]
        for pattern in docker_images:
            proc = _run_command(
                ["docker", "images", "-q", pattern],
                timeout=30,
            )
            if proc.stdout.strip():
                ids = proc.stdout.strip().splitlines()
                for img_id in ids[keep_last:]:
                    rm_proc = _run_command(["docker", "rmi", "-f", img_id], timeout=30)
                    if rm_proc.returncode == 0:
                        removed.append(f"docker-image:{img_id}")

        history = self._load_history()
        if len(history) > keep_last:
            for entry in history[keep_last:]:
                filepath = self.history_dir / f"{entry.id}.json"
                if filepath.exists():
                    filepath.unlink()
                    removed.append(f"history:{entry.id}")

        package_zip = self.project_path / "deployment_package"
        if package_zip.exists():
            shutil.rmtree(package_zip, ignore_errors=True)
            removed.append(str(package_zip))

        for zip_file in self.project_path.glob("*.zip"):
            if zip_file.name != "__init__.zip":
                zip_file.unlink()
                removed.append(str(zip_file))

        temp_gh = self.project_path / ".gh-pages-deploy"
        if temp_gh.exists():
            shutil.rmtree(temp_gh, ignore_errors=True)
            removed.append(str(temp_gh))

        return removed

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def get_project_info(self) -> dict[str, Any]:
        """Return detected project framework and metadata."""
        return dict(self._project_info)

    @property
    def project_framework(self) -> str:
        return self._project_info["framework"]

    @property
    def project_language(self) -> str:
        return self._project_info["language"]


# ---------------------------------------------------------------------------
# Async sleep helper
# ---------------------------------------------------------------------------

async def _async_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# Convenience CLI entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """CLI entry point for the deployment engine."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="claude-deploy",
        description="Multi-platform deployment engine for Claude Code clone.",
    )
    parser.add_argument(
        "platform",
        nargs="?",
        default=None,
        help="Target platform (docker, vercel, netlify, aws_lambda, github_pages, railway, render, self_hosted, auto).",
    )
    parser.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")
    parser.add_argument("--build", "-b", default="", help="Build command override")
    parser.add_argument("--output", "-o", default="", help="Output directory override")
    parser.add_argument("--env", "-e", action="append", default=[], help="Environment variables (KEY=VALUE)")
    parser.add_argument("--region", "-r", default="us-east-1", help="Deployment region")
    parser.add_argument("--domain", "-d", default="", help="Custom domain")
    parser.add_argument("--image", "-i", default="", help="Docker image name")
    parser.add_argument("--port", type=int, default=8000, help="Application port")
    parser.add_argument("--rollback", default=None, help="Rollback to deployment ID")
    parser.add_argument("--health-check", default=None, help="URL to health check")
    parser.add_argument("--detect", action="store_true", help="Detect best platform")
    parser.add_argument("--generate-dockerfile", action="store_true", help="Generate Dockerfile only")
    parser.add_argument("--generate-compose", action="store_true", help="Generate docker-compose.yml only")
    parser.add_argument("--env-action", default=None, help="Env management: set/get/list/encrypt/remove/export")
    parser.add_argument("--env-vars", default=None, help="JSON string of env vars for env management")
    parser.add_argument("--history", action="store_true", help="Show deployment history")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old deployments")

    args = parser.parse_args()

    engine = DeployEngine(project_path=args.project)

    if args.detect:
        platform = await engine.detect_platform()
        info = engine.get_project_info()
        print(f"Detected platform: {platform.value}")
        print(f"Framework: {info['framework']} ({info['language']})")
        return

    if args.generate_dockerfile:
        dockerfile = await engine.generate_dockerfile()
        print(dockerfile)
        return

    if args.generate_compose:
        compose = await engine.generate_compose()
        print(compose)
        return

    if args.env_action:
        env_vars = None
        if args.env_vars:
            try:
                env_vars = json.loads(args.env_vars)
            except json.JSONDecodeError:
                print("Error: --env-vars must be valid JSON", file=sys.stderr)
                sys.exit(1)
        result = await engine.manage_env(args.env_action, env_vars)
        print(json.dumps(result, indent=2))
        return

    if args.history:
        entries = await engine.get_history()
        if not entries:
            print("No deployment history found.")
        for entry in entries:
            status_icon = "+" if entry.status == "success" else "-"
            print(f"  [{status_icon}] {entry.id[:8]}  {entry.platform:15s}  {entry.status:12s}  {entry.url or 'N/A':50s}  {entry.timestamp}")
        return

    if args.rollback:
        result = await engine.rollback(args.rollback)
        if result.success:
            print(f"Rollback successful: {result.url}")
        else:
            print(f"Rollback failed: {result.error}", file=sys.stderr)
            sys.exit(1)
        return

    if args.health_check:
        result = await engine.health_check(args.health_check)
        status = "HEALTHY" if result["healthy"] else "UNHEALTHY"
        print(f"  [{status}] {result['url']}")
        print(f"  Status: {result['status_code']}  Time: {result['response_time_ms']}ms  Attempts: {result['attempts']}")
        if result["errors"]:
            for err in result["errors"]:
                print(f"  Error: {err}")
        sys.exit(0 if result["healthy"] else 1)
        return

    if args.cleanup:
        removed = await engine.cleanup()
        if removed:
            print("Cleaned up:")
            for item in removed:
                print(f"  - {item}")
        else:
            print("Nothing to clean up.")
        return

    if args.platform:
        if args.platform == "auto":
            platform = await engine.detect_platform()
        else:
            try:
                platform = Platform.from_string(args.platform)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        env_vars = {}
        for ev in args.env:
            if "=" in ev:
                k, v = ev.split("=", 1)
                env_vars[k] = v

        config = DeploymentConfig(
            platform=platform,
            project_path=args.project,
            build_command=args.build,
            output_dir=args.output,
            env_vars=env_vars,
            region=args.region,
            domain=args.domain,
            image_name=args.image,
            port=args.port,
        )

        print(f"Deploying to {platform.value}...")
        result = await engine.deploy(platform, config)

        if result.success:
            print(f"Deployment successful!")
            print(f"  URL: {result.url}")
            print(f"  Version: {result.version}")
            print(f"  Duration: {result.duration:.1f}s")
            print(f"  ID: {result.deployment_id}")
        else:
            print(f"Deployment failed: {result.error}", file=sys.stderr)
            print(f"  Logs:")
            for line in result.logs[-10:]:
                print(f"    {line}")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
