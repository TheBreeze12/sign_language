# Docker Guide for Sign Language App

This guide explains how to run the frontend and backend using Docker and Docker Compose.

## 1. Setup Environment Variables (MySQL Password)

Since everyone has a different MySQL password, we use a `.env` file to manage it securely.

1. **Copy the Template**
   In the root directory, copy `.env.example` to `.env`.
   - Windows Command Prompt:
     ```cmd
     copy .env.example .env
     ```
   - PowerShell:
     ```powershell
     cp  .env
     ```

2. **Edit `.env`**
   Open `.env` in any text editor and change `DB_PASSWORD=your_password_here` to your actual MySQL root password.
   *(Since `.env` is ignored by Git, it won't be pushed to the repository.)*

## 2. Running with Docker Compose

1. **Build and Start**
   Open a terminal in the project root folder (`f:\sign_language`) and run:
   ```bash
   docker-compose up --build
   ```
   *The `--build` flag ensures it builds the images if you modified any code.*

2. **Access the Application**
   - **Frontend (UI):** [http://localhost:5173](http://localhost:5173)
   - **Backend (API):** [http://localhost:5000](http://localhost:5000)

3. **Stop the Application**
   Press `Ctrl+C` in the terminal where it's running, or open another terminal and run:
   ```bash
   docker-compose down
   ```

## 3. How It Works (Volumes)

The `docker-compose.yml` mounts several large folders from your Windows drive directly into the Linux containers:
- `f:/sign_language/data` -> `/data`
- `f:/sign_language/result` -> `/result`
- `f:/sign_language/result_3d/glb_models` -> `/result_3d`

This prevents the need to copy huge models into the Docker images, saving space and build time.

The backend container accesses your local Windows MySQL database using `host.docker.internal`.
