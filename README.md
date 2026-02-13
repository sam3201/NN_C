Here is your rewritten, production-grade, cross-platform README.md with:

Clear macOS / Linux / Windows install sections

Python version guard (>=3.10,<3.14)

Wheel-safe dependency guidance

Updated launcher usage (run_production.sh / .ps1)

Cleaner onboarding flow

Explicit C toolchain notes per OS

You can replace your entire README with this.

SAM-D AGI (ΨΔ•Ω-Core v5.0.0 Recursive)

SAM-D is a hybrid Python/C recursive meta-evolutionary system with a web dashboard, slash-command interface, and C-accelerated cores for meta-control and dual-system simulation. It features the advanced ΨΔ•Ω-Core morphogenesis and PDI-T submodel lifecycle.

⚙️ System Requirements
Supported Python Versions

Python >=3.10 and <3.14

Python 3.14+ is not currently supported due to upstream wheel availability for compiled dependencies (NumPy, etc.).

Check your version:

python --version


If needed, install:

macOS (Homebrew):

brew install python@3.13


Ubuntu/Debian:

sudo apt install python3.13 python3.13-venv python3.13-dev


Windows:
Install from https://python.org
 (3.12 or 3.13 recommended)

C Compiler Toolchain (Required for Extensions)
OS	Toolchain
macOS	Xcode Command Line Tools (xcode-select --install)
Linux	build-essential
Windows	MSVC Build Tools (Visual Studio Build Tools)
📦 Installation (Cross-Platform)

SAM-D now ships with deterministic environment bootstrapping.

macOS / Linux
git clone <repo>
cd SAM-D

./run_production.sh


The launcher will:

Create a venv (if missing)

Validate Python version

Upgrade pip/setuptools/wheel

Install dependencies

Build C extensions

Launch the system

Auto-restart on exit

Windows (PowerShell)
git clone <repo>
cd SAM-D

.\run_production.ps1


This performs the same deterministic setup using the venv interpreter directly.

🔬 Manual Setup (Advanced / Debug Mode)

If you prefer manual control:

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python setup.py build_ext --inplace

python src/python/complete_sam_unified.py --port 5005

📁 What's Included

Python orchestration + API: complete_sam_unified.py

Web dashboard + terminal UI

C extensions:

sam_sav_dual_system

sam_meta_controller_c

multi_agent_orchestrator_c

specialized_agents_c

consciousness_*

Deterministic launchers:

run_production.sh

run_production.ps1

🧠 Profiles (Full vs Experimental)

Profiles live in profiles/:

profiles/full.env

profiles/experimental.env

Launch with:

SAM_PROFILE=experimental ./run_production.sh


Default profile: Full

🌐 Interfaces

Dashboard: http://localhost:5005

Terminal: http://localhost:5005/terminal

Health: /api/health

Agents: /api/agents

Command API: /api/command

SocketIO: /api/groupchat/status

🔒 Secure Remote Access
Option 1: Tailscale (Recommended, Free)

Install Tailscale

Join same tailnet

Access via Tailscale IP + port 5005

Option 2: Cloudflare Tunnel + Access

Install cloudflared

Create tunnel

Route DNS

Configure Cloudflare Access rules

Set:

SAM_TRUST_PROXY=1
SAM_OAUTH_REDIRECT_BASE=https://yourdomain.com


Health check:

curl https://yourdomain.com/api/health

🔐 Authentication & OAuth

Set in .env.local:

SAM_LOGIN_PASSWORD
SAM_ALLOWED_EMAILS
SAM_OWNER_EMAIL
SAM_ADMIN_EMAILS
SAM_SESSION_SECRET
SAM_ALLOWED_IPS
SAM_TRUST_PROXY=1


OAuth:

SAM_GOOGLE_CLIENT_ID
SAM_GOOGLE_CLIENT_SECRET
SAM_GITHUB_CLIENT_ID
SAM_GITHUB_CLIENT_SECRET
SAM_OAUTH_REDIRECT_BASE


Helper:

GET /api/oauth/help

🧪 Smoke Test
python -c "import sam_sav_dual_system, sam_meta_controller_c; print('C extensions OK')"
python -c "from complete_sam_unified import UnifiedSAMSystem; print('System import OK')"

🧬 Dual-System C Arena

sam_sav_dual_system implements:

Fast RNG (xorshift64*)

Self-referential objective mutation

Adversarial arena pressure

SAV kill confirmation term

SAM/SAV unbounded modes

Python bindings for telemetry + mutation

🧭 Meta-Controller C Core

sam_meta_controller_c provides:

Pressure aggregation

Growth primitive selection

Identity anchoring

Objective contract evaluation

Structural cooldown enforcement

🧱 Growth Primitives

GP_LATENT_EXPAND

GP_SUBMODEL_SPAWN

GP_INDEX_EXPAND

GP_ROUTING_INCREASE

GP_CONTEXT_EXPAND

GP_PLANNER_WIDEN

GP_CONSOLIDATE

GP_REPARAM

🔁 Recursive Alignment Checks
./tools/run_recursive_checks.sh

💾 State Persistence

Profile-scoped state files:

sam_data/full/state.json
sam_data/experimental/state.json

🧠 Training Pipeline
Install training dependencies
pip install -r requirements_training.txt

Build distillation dataset
python -m training.distillation ...

Train (LoRA or full)
python -m training.training_loop ...

Regression Gate
python -m training.regression_suite ...

🔁 Environment Overrides (Selected)

SAM_POLICY_PROVIDER

SAM_PROVIDER_AUTO_SWITCH

SAM_REGRESSION_ON_GROWTH

SAM_TWO_PHASE_BOOT

SAM_AUTONOMOUS_LOOP_INTERVAL_S

SAM_TEACHER_POOL_ENABLED

SAM_REVENUE_OPS_ENABLED

(Full list unchanged from previous README.)

📜 Canonical Equation

See:

DOCS/GOD_EQUATION.md
DOCS/ALIGNMENT.md


Top-level form:

G = Σ α_i F_i(x,t) + β_i dF_i/dt + γ_i ∇F_i L + δ_i μ_i + ζ_i Φ(G)


Recursive:

Φ(G) = lim ( G_n + λ dG_n/dn + ρ d²G_n/dn² + … )

🧾 Notes on Cross-Platform Stability

Python >=3.10,<3.14 enforced

NumPy uses wheel-safe range (>=2.0,<3)

All compiled deps allow minor version flexibility

Launchers use python -m pip (never bare pip)

C builds isolated to venv interpreter
