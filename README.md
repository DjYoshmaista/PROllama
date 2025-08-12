# PROllama README.md v0.0.1a

## Explanation
This is the first version of the README for PROllama--PostgreSQL RAGDB Ollama indexing/inferencing and referencing using PostgreSQL Database with Vector Store as a Retrieval Augmented Generation Database.

The PROllama application is meant to utilize local Large Language Model AI via the Ollama platform to glean information from a RAG database and generate/structure a reply to the user's query or prompt.  This generated response should directly reference any subject matter the user specifies in their original prompt/query.  In order to accomplish this, we will have to install, set up, and configure multiple existing applications, chaining them together and leveraging the capabilities of each to accomplish our end goal.

Currently in early tests on Arch Linux, and while I can ensure that all of this works on the Arch Linux x86-64 platform, I cannot guarantee functionality/usability on any other platform.  For the time being, all tools must be local unless specifically configured in the code by the user to use remote services.  I do plan on adding support for remote Ollama servers as well as remote psql databases for future functionality.  For the time being, all programs must be installed on the host machine at the OS level.  If you insist on utilizing/using remote services/servers, your mileage may vary EXTREMELY and ease of configuration is not implemented to facilitate this functionality.

Let's move on to the initial setup

**`_____________________________________________________________________________________________________________________________________________________________________________________________________________`**

## SETUP

**`1. I would suggest either creating a conda environment or a virtualenv via Python to install all of the required Python packages.  Your mileage may vary, however, and feel free to let me know how it works out for you if you stray from this suggestion!`**

**`_____________________________________________________________________________________________________________________________________________________________________________________________________________`**

**Installation:**

**`- First`** create your condaenv or virtualenv.  Then, using the 'requirements.txt' file, run **`pip install -r requirements.txt`** which should install all required pip packages.

**`- Second`** run the setup.sh bash file using **`bash setup.sh`** which will run you through the generation of the .env file and creation of the PostgreSQL database you will be using for storing your documents/data.  This process should also install PostgreSQL and pgvector if you do not have them installed already, as well as ollama (if you're on Arch Linux that is) if it's not already installed/running.  This is all very early stages and could use a fair bit of refinement--if you're comfortable, I would suggest setting up Postgres and Ollama on your own, as well as pulling the models *dengcao/Qwen3-Embed-0.6B:Q8_0*, *qwen3:8b*, *dengcao/Qwen3-Reranker-4B:Q5_K_M* and optionally installing CUDA, CUDnn, nVidia drivers, or the ROCm equivalence depending on your specific GPU brand and environment.  Keep in mind, CPU inferencing will always work by default, but even an 8GB GPU of some sort that can be leveraged computationally will MASSIVELY speed up the various RAG operations provided by this program.

**`- Third`** now you should be ready to run the program assuming all tools are installed, models are pulled, and everything is up and running correctly!

**`- Optional`** the *debug_database.py* file can be ran to ensure that your Database and tables are correctly setup.

Currently, the program supports *.txt*, *.py*, *.csv*, and *.json* filetypes for data ingestion/organization.  There will be more filetypes added in the future, but these are definitely early Alpha days.

**`_______________________________________________________________________________________________________________________________________________________________________________________________________`**

**Operation:**

**`- Menu Option 1`** - Ask the inference model for information -- This option is actually going to be the final usage case of the program.  This manually queries the *qwen3:8b* model to construct a response to your prompt, query or question based on the contents of the RAG Database.  ***NOTE*** - If you do not have the database populated with any [relevant] information, then the model will NOT respond using its own knowledge base currently.  I'll work on integrating the model's current knowledge base as well as the RAG Database in a future version

**`- Menu Option 2`** - Add data to the database -- Legacy testing option which allowed for the manual adding of an entry to the database.  Not entirely useful or functional currently, but will get updated/fixed in future version(s)

**`- Menu Option 3`** - Load data from a single file -- Just like it sounds--the user specifies a filepath on the system.  The data in that file is then chunked, embedding(s) is/are generated, and the data is added to the RAG database.  Somewhat of a legacy option

**`- Menu Option 4`** - Load documents from folder -- If ran on the host machine (or with X forwarding via SSH), allows the user to select a folder from a tKinter GUI interface.  All files (of compatible filetypes/encodings) within that folder are then processed, the data is chunked, embeddings are generated, and the information is added to the RAG DB.  This all happens in parallel if you have GPU(s) available for inferencing.  With a decently powerful computer, the process is fairly speedy--the embedding generation needs a bit of optimization.  NOTE: Ollama is used for embedding generation in ALL aspects of the program--query, inference, data processing, and comparison.  This is why I chose the int8 version of the Qwen3 Embed model for embedding generation--I wanted the embedding generation to be as accurate as possible but for the gateway to usage to be as low as possible for the program (system specifications-wise).

**`- Menu Option 5`** - Query The Database Directly -- Is supposed to allow the user to search the database via keyword search--will be updated to have these keywords broken up into tokens, and to compare the keyword tokens to the tokens contained within the database.  Soon[TM]

**`- Menu Option 6`** - List database contents -- Lists the full contents of the table in the database specified in the *.env* file

**`- Menu Option 7`** - Configure system parameters -- Allows the user to configure parameters for Postgres and other aspects of the program.

**`- Menu Option 8`** - Database management - Allows the user to: 1) Initialize/Reset the Database Schema; 2) Run Maintenance (Not Implemented); 3) Backup Database (Not Implemented); 4) Show Database Statistics

**`- Menu Option 9`** - System information - Shows the current system specifications/information (Memory Usage *(%)*; Memory Used *(MB)*; Memory Available *(MB)*; GPU Memory (Not Implemented); Embedding Queue Size (Not Implemented); Active Embedding Workers *(Number of concurrent embedding workers querying the embedding model via Ollama)*; Database Documents *(# of entries in the database)*; Cache status (Not Implemented)

**`- Menu Option 0`** - Exit -- Exits the program

**`_____________________________________________________________________________________________________________________________________________________________________________________________________________`**

Ctrl + D is the easiest way to stop execution of the program

This is a simple README for now, and the scope and depth of the README will expand organically as new features and functions are added.  Currently, my focus is on optimizing the RAG DB performance and the response time of the LLM's.  My machine is fairly modest by development standards, so I am using the "If it takes too long for me, it'll take too long for the end user" approach to optimizing.  Obviously, on Enterprise/Professional hardware this program will execute MUCH faster and a lot of the small optimizations/improvements I've made/am making will not be felt nearly as much as I feel them on my system.

Feel free to comment, leave feature requests, or email me to get a hold of me directly!

**E-Mail:** djyoshmaista@gmail.com