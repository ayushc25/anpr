# Rising IT ANPR — Client Installation & Deployment Guide

This guide covers step-by-step instructions for deploying the **Rising IT Automated Number Plate Recognition (ANPR)** system on a client edge machine (Windows 10/11 Pro or Windows Server).

---

## 1. System Requirements

### Hardware Requirements
* **Processor (CPU):** Intel Core i5 / i7 (8th Gen or newer) or AMD Ryzen 5 / 7 (Inference runs on CPU using OpenVINO / ONNX Runtime).
* **RAM:** 16 GB recommended (8 GB absolute minimum for 1–2 cameras).
* **Storage:** 128+ GB SSD (for OS, models, event logs, and plate crop image retention).
* **Network:** Dedicated LAN connection to the camera network switch (Gigabit Ethernet recommended).

### Software Requirements (Client Machine)
1. **Operating System:** Windows 10/11 Pro (64-bit) or Windows Server 2019/2022.
2. **Python:** Version **3.11.x** (64-bit).
   > **Important:** Check *"Add Python to PATH"* during Python installation.
3. **Database:** **PostgreSQL 14+** (can be installed locally on the edge box or hosted on the client's internal network).

---

## 2. Pre-Deployment Preparation (Before Heading to Site)

To ensure smooth on-site installation without requiring Node.js or high-speed internet on the client PC:

1. **Build the Web Frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```
   *(This compiles the React application into `frontend/dist/`. FastAPI will serve this directly on port `8002`, so Node.js is **not** required on the client machine).*

2. **Verify Detection Models:**
   Ensure the required ONNX models are present inside the `models/` folder:
   * `yolo26n.onnx` (vehicle detector)
   * `license-plate-nano.onnx` (plate detector)
   * OCR recognizer models / VLM weights

---

## 3. Client Machine Setup

### Step 1: Copy Project Folder
Copy the project folder to the target drive on the client machine, for example:
```
C:\RisingIT\anpr
```
*(Avoid placing it in temporary or Desktop folders).*

---

### Step 2: Set Up Python Virtual Environment
Open **PowerShell as Administrator** and navigate to the project directory:
```powershell
cd C:\RisingIT\anpr

# Create backend virtual environment
python -m venv backend\venv

# Activate and upgrade pip
backend\venv\Scripts\python -m pip install --upgrade pip

# Install dependencies
backend\venv\Scripts\pip install -r backend\requirements.txt
```

*(Optional — If running the deferred Qwen-VL model recognizer)*:
```powershell
python -m venv backend\venv-vlm
backend\venv-vlm\Scripts\pip install -r backend\requirements-vlm.txt
```

---

### Step 3: Verify Model Artifacts
Run the CLI model validation tool:
```powershell
backend\venv\Scripts\python -m backend.app.cli models check
```
If detector models need export:
```powershell
backend\venv\Scripts\python scripts\export_onnx.py --all
```

---

### Step 4: Ensure PostgreSQL is Running
Create a database for ANPR on the PostgreSQL server (locally or on client LAN):
```sql
CREATE DATABASE anpr;
```
Note the database connection string:
```
postgresql://username:password@localhost:5432/anpr
```

---

## 4. First-Run Setup Wizard (UI-Driven Onboarding)

1. **Start the API Server:**
   ```powershell
   backend\venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8002
   ```

2. **Open the Setup Wizard:**
   Open any web browser on the machine (or on the local network) and navigate to:
   ```
   http://localhost:8002
   ```
   *(Or `http://<client-machine-ip>:8002` from another LAN computer).*

   The system will automatically detect that it is unconfigured and redirect to:
   ```
   http://localhost:8002/setup
   ```

3. **Complete the 3-Step Setup Form:**
   * **Step 1: Database URL:** Enter the PostgreSQL connection string:
     ```
     postgresql://postgres:YourPassword@localhost:5432/anpr
     ```
     Click **"Test Connection"**. You will receive a green confirmation once verified.
   * **Step 2: Rising IT Super Admin:** Enter the master administrator credentials for Rising IT support (e.g. `risingit` + password).
   * **Step 3: Client Admin (Handover):** Enter the client administrator credentials (e.g. `admin` + password).

4. **Click "Initialize & Deploy System":**
   The backend will automatically:
   * Write connection settings safely to `.env`.
   * Apply all database schema migrations (`alembic upgrade head`).
   * Seed permission sets, Super Admin, Client Admin, and default entry/exit gates.
   * Lock the setup wizard from future tampering.
   * Redirect to `/login`.

---

## 5. Setting up as a Windows Background Service

To ensure the ANPR system starts automatically when the computer boots up (without needing someone to log into Windows), use **NSSM (Non-Sucking Service Manager)**.

### 1. Download NSSM
1. Download NSSM from [https://nssm.cc/download](https://nssm.cc/download).
2. Extract `nssm.exe` (from the `win64` folder) into `C:\Windows\System32\` or `C:\RisingIT\anpr\scripts\`.

### 2. Install FastAPI Backend Service
Run in PowerShell (Admin):
```powershell
nssm install RisingIT-ANPR-API "C:\RisingIT\anpr\backend\venv\Scripts\python.exe" "-m uvicorn backend.app.main:app --host 0.0.0.0 --port 8002"
nssm set RisingIT-ANPR-API AppDirectory "C:\RisingIT\anpr"
nssm set RisingIT-ANPR-API Start SERVICE_AUTO_START
nssm set RisingIT-ANPR-API AppStdout "C:\RisingIT\anpr\storage\api.log"
nssm set RisingIT-ANPR-API AppStderr "C:\RisingIT\anpr\storage\api_error.log"

# Start the service
nssm start RisingIT-ANPR-API
```

### 3. Install Camera Worker Supervisor Service
*(Runs camera streaming and real-time ONNX inference)*:
```powershell
nssm install RisingIT-ANPR-Supervisor "C:\RisingIT\anpr\backend\venv\Scripts\python.exe" "-m backend.app.cli supervisor"
nssm set RisingIT-ANPR-Supervisor AppDirectory "C:\RisingIT\anpr"
nssm set RisingIT-ANPR-Supervisor Start SERVICE_AUTO_START
nssm set RisingIT-ANPR-Supervisor AppStdout "C:\RisingIT\anpr\storage\supervisor.log"
nssm set RisingIT-ANPR-Supervisor AppStderr "C:\RisingIT\anpr\storage\supervisor_error.log"

# Start the service
nssm start RisingIT-ANPR-Supervisor
```

---

## 6. Camera Configuration & Handover

1. Open `http://localhost:8002/login` in the browser.
2. Log in with the **Client Admin** credentials created during setup.
3. Go to the **Cameras** menu:
   * Click **"Add Camera"**.
   * Enter Camera Name (e.g. `Entry Gate 1`).
   * Enter Location (e.g. `Main Gate`).
   * Select Direction (`Entry` or `Exit`).
   * Enter RTSP Stream URLs (Main Stream & Sub Stream).
4. Configure the **Detection Zone (ROI)**:
   * In the camera preview, draw the bounding polygon where vehicle plates pass.
   * Save camera settings.
5. Go to **Live View** and verify live stream rendering.
6. Test with a passing vehicle to confirm detection, plate OCR, and entry into the **Events** log.

---

## 7. Windows Firewall Configuration (For LAN Access)

If security guards or client operators need to access the dashboard from other PCs on the local network:

1. Open **PowerShell as Administrator**.
2. Run the following command to allow inbound traffic on port `8002`:
   ```powershell
   New-NetFirewallRule -DisplayName "Rising IT ANPR Dashboard" -Direction Inbound -LocalPort 8002 -Protocol TCP -Action Allow
   ```
3. Operators can now access `http://<EDGE_PC_IP>:8002` from any browser on the gate LAN.

---

## 8. Verification & Health Check

* **Web UI:** `http://localhost:8002`
* **API Health Check:** `http://localhost:8002/health`
  Expected JSON response:
  ```json
  {
    "status": "ok",
    "checks": {
      "database": true,
      "media_writable": true
    }
  }
  ```
* **Logs Directory:** `C:\RisingIT\anpr\storage\`

---

## 9. Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **Setup wizard not showing** | Database already initialized | If re-setup is needed, delete/clear `database_url` in `.env` and restart service. |
| **"Could not connect to database"** | Incorrect host/port/credentials or PostgreSQL service stopped | Verify PostgreSQL Windows service is running in `services.msc` and check credentials. |
| **Camera shows offline** | Camera RTSP stream unreachable or wrong IP | Ping camera IP; test RTSP link in VLC Media Player (`rtsp://user:pass@ip:554/...`). |
| **Port 8002 already in use** | Another app is using port 8002 | Run `netstat -ano \| findstr :8002` to find process ID, or change port in startup command. |

