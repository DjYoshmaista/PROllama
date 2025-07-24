#!/bin/bash
# setup.sh

set -e # Exit immediately if a command exits with a non-zero status.

echo "=== RAG Database Setup Script ==="

# --- Helper Functions ---

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        DISTRO=$ID
    elif command -v lsb_release >/dev/null 2>&1; then
        DISTRO=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
    elif [ -f /etc/redhat-release ]; then
        if grep -q "CentOS" /etc/redhat-release; then
            DISTRO="centos"
        elif grep -q "Red Hat" /etc/redhat-release; then
            DISTRO="rhel"
        else
            DISTRO="redhat"
        fi
    else
        DISTRO="unknown"
    fi
    echo "$DISTRO"
}

# Function to install packages based on package manager
install_packages() {
    local packages=("$@")
    echo "Attempting to install packages: ${packages[*]}"

    case $DISTRO in
        arch)
            echo "Using pacman (Arch Linux)..."
            sudo pacman -Syu --noconfirm "${packages[@]}"
            ;;
        ubuntu|debian)
            echo "Using apt-get (${DISTRO^})..."
            sudo apt-get update
            sudo apt-get install -y "${packages[@]}"
            ;;
        fedora)
            echo "Using dnf (Fedora)..."
            sudo dnf install -y "${packages[@]}"
            ;;
        centos|rhel)
            echo "Using yum (${DISTRO^})..."
            sudo yum install -y "${packages[@]}"
            ;;
        opensuse*|suse)
            echo "Using zypper (openSUSE)..."
            sudo zypper install -y "${packages[@]}"
            ;;
        *)
            echo "Error: Unsupported or unknown distribution: $DISTRO"
            echo "Please install the following packages manually:"
            echo "  PostgreSQL, pgvector, Python3, pip, venv"
            exit 1
            ;;
    esac
}

# --- 1. Check for and Install System Dependencies ---
DISTRO=$(detect_distro)
echo "Detected distribution: ${DISTRO^}"

# Check for PostgreSQL Installation
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL 'psql' command could not be found. Attempting to install PostgreSQL..."
    case $DISTRO in
        arch)
            install_packages postgresql
            # Initialize database (Arch specific steps)
            echo "Initializing PostgreSQL database (Arch)..."
            sudo su - postgres -c "initdb -D /var/lib/postgres/data"
            sudo systemctl enable postgresql
            sudo systemctl start postgresql
            ;;
        ubuntu|debian)
            install_packages postgresql postgresql-contrib
            sudo systemctl enable postgresql
            sudo systemctl start postgresql
            ;;
        fedora)
            install_packages postgresql-server postgresql-contrib
            # Initialize database (Fedora specific steps)
            echo "Initializing PostgreSQL database (Fedora)..."
            sudo postgresql-setup --initdb
            sudo systemctl enable postgresql
            sudo systemctl start postgresql
            ;;
        centos|rhel)
            install_packages postgresql-server postgresql-contrib
            echo "Initializing PostgreSQL database (${DISTRO^})..."
            sudo postgresql-setup --initdb
            sudo systemctl enable postgresql
            sudo systemctl start postgresql
            ;;
        opensuse*|suse)
            install_packages postgresql postgresql-contrib
            sudo systemctl enable postgresql
            sudo systemctl start postgresql
            ;;
        *)
            echo "Error: Cannot automatically install PostgreSQL for $DISTRO. Please install manually."
            exit 1
            ;;
    esac
    echo "PostgreSQL installation and service start attempted."
else
    echo "PostgreSQL is already installed."
fi

# Check for Python, pip, and venv
MISSING_PYTHON_DEPS=()
if ! command -v python3 &> /dev/null; then MISSING_PYTHON_DEPS+=("python3"); fi
if ! command -v pip3 &> /dev/null; then MISSING_PYTHON_DEPS+=("pip3"); fi

# Check for venv module (usually part of a separate package)
if ! python3 -c "import venv" &> /dev/null; then
    case $DISTRO in
        arch) MISSING_PYTHON_DEPS+=("python-virtualenv") ;; # or python-pipenv?
        ubuntu|debian|fedora|centos|rhel|opensuse*|suse) MISSING_PYTHON_DEPS+=("python3-venv") ;;
        *) echo "Warning: Unable to determine venv package for $DISTRO. You might need to install it manually." ;;
    esac
fi

if [ ${#MISSING_PYTHON_DEPS[@]} -ne 0 ]; then
    echo "Missing Python dependencies: ${MISSING_PYTHON_DEPS[*]}. Attempting to install..."
    install_packages "${MISSING_PYTHON_DEPS[@]}"
    echo "Python dependencies installed."
else
    echo "Python3, pip3, and venv module are already installed."
fi

# --- 2. Ensure pgvector Extension is installed and available ---
# This step requires the extension to be available in the PostgreSQL installation.
# It's often part of the postgresql-contrib package installed above.
echo "Attempting to create pgvector extension in the default 'postgres' database (requires postgres user permissions)..."
# Use PGPASSWORD environment variable or .pgpass file for passwordless execution
# This might fail if the extension library isn't installed correctly.
if sudo -u postgres psql -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" &>/dev/null; then
    echo "pgvector extension enabled in 'postgres' database (or already existed)."
else
    echo "Warning: Could not create pgvector extension in 'postgres' database."
    echo "Ensure pgvector is installed (often via postgresql-contrib package)."
    echo "You might need to manually run 'CREATE EXTENSION vector;' in psql as the postgres user after ensuring the library is available."
fi


# --- 3. Setup Database/Table and .env ---
echo ""
read -p "Do you want to set up a new database and table? (y/N): " -n 1 -r
echo    # Move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Setting up database and table..."

    DEFAULT_DB_NAME="rag_db"
    DEFAULT_TABLE_NAME="documents"

    read -p "Enter the new database name (or press Enter for '$DEFAULT_DB_NAME'): " DB_NAME
    DB_NAME=${DB_NAME:-$DEFAULT_DB_NAME} # Use default if empty

    read -p "Enter the new table name (or press Enter for '$DEFAULT_TABLE_NAME'): " TABLE_NAME
    TABLE_NAME=${TABLE_NAME:-$DEFAULT_TABLE_NAME} # Use default if empty

    # Create Database
    echo "Creating database '$DB_NAME'..."
    if sudo -u postgres createdb "$DB_NAME"; then
        echo "Database '$DB_NAME' created successfully."
    else
        echo "Error: Failed to create database '$DB_NAME'. It might already exist or you lack permissions."
        # Decide whether to continue or exit. Let's warn and continue for now.
        echo "Warning: Proceeding, but database setup might be incomplete."
    fi

    # Connect to the new DB and create table/extension
    echo "Connecting to '$DB_NAME' and setting up table '$TABLE_NAME' and pgvector extension..."
    # Enable extension in the new database
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" || echo "Warning: Could not create vector extension in '$DB_NAME'. Ensure pgvector is properly installed."

    # Create table and index
    if sudo -u postgres psql -d "$DB_NAME" -c "
    CREATE TABLE IF NOT EXISTS $TABLE_NAME (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        tags TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        embedding VECTOR(1024)
    );
    CREATE INDEX IF NOT EXISTS ${TABLE_NAME}_embedding_idx
    ON $TABLE_NAME
    USING hnsw (embedding vector_cosine_ops);
    "; then
        echo "Table '$TABLE_NAME' and HNSW index created successfully in database '$DB_NAME'."
    else
        echo "Error: Failed to create table '$TABLE_NAME' or index in database '$DB_NAME'."
        echo "Warning: Proceeding, but database schema might be incomplete."
    fi


    # --- Write .env file ---
    echo "Writing database configuration to '.env' file..."
    cat > .env << EOF
DB_NAME=$DB_NAME
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
TABLE_NAME=$TABLE_NAME
EOF
    echo ".env file created with database settings."

else
   echo "Skipping database/table setup."
   # Check if .env exists, create default if not
   if [ ! -f ".env" ]; then
       echo "Warning: .env file not found. Creating a default one. Please edit it with your settings if they differ."
       cat > .env << EOF
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
TABLE_NAME=documents
EOF
       echo "Default .env file created."
   else
        echo ".env file already exists. Skipping creation."
   fi
fi

# --- 4. Install Python Dependencies ---
echo "Installing Python dependencies from requirements.txt..."
# Create virtual environment (recommended)
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "Virtual environment 'venv' created."
else
    echo "Virtual environment 'venv' already exists."
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies listed in requirements.txt..."
    pip install -r requirements.txt
    echo "Dependencies installed successfully within the virtual environment."
else
    echo "Warning: requirements.txt not found. Please create it or install dependencies manually."
    echo "A standard requirements.txt might include:"
    echo "psycopg2-binary>=2.9.0"
    echo "requests>=2.31.0"
    echo "aiohttp>=3.8.0"
    echo "numpy>=1.24.0"
    echo "torch>=2.0.0"
    echo "asyncio>=3.4.3"
    echo "aiomultiprocess>=0.9.0"
    echo "asyncpg>=0.27.0"
    echo "pgvector>=0.2.0"
    echo "tqdm>=4.64.0"
    echo "psutil>=5.9.0"
fi

# --- 5. Final Instructions ---
echo ""
echo "=== Setup Summary ==="
echo "1. System dependencies (PostgreSQL, Python, etc.) checked/installed for $DISTRO."
echo "2. pgvector extension creation attempted."
echo "3. Database '$DB_NAME' and table '$TABLE_NAME' setup attempted (if requested)."
echo "4. .env file created/checked."
echo "5. Python virtual environment 'venv' created and activated."
echo "6. Dependencies installed from requirements.txt (if found)."

echo ""
echo "To run the program, ensure the virtual environment is active and execute rag_db.py:"
echo "   source venv/bin/activate  # If not already active"
echo "   python rag_db.py"
echo ""
read -p "Do you want to run the program now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "rag_db.py" ]; then
        echo "Starting rag_db.py..."
        python rag_db.py
    else
        echo "Error: rag_db.py not found in the current directory."
    fi
fi

echo "Setup script finished."